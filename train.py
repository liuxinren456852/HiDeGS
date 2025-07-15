#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import os
import sys
import random
import time
import uuid
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pandas.core.array_algos.quantile import quantile_with_mask
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.app_model import AppModel
from scene.cameras import Camera
from scripts.frequency_regularization import *
from utils.general_utils import safe_state, get_expon_lr_func
from utils.graphics_utils import patch_offsets, patch_warp
from utils.image_utils import psnr, erode
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight


def direct_collate(x):
    return x


def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    """生成虚拟相机视角"""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    # 添加随机扰动
    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)

    # 构建旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)

    virtual_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                         cam.image_width, cam.image_height, cam.image_path, cam.image_name, 100000,
                         trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                         preload_img=False, data_device="cuda")
    return virtual_cam


def compute_depth_uncertainty(depth_map, normal_map, scaling, visibility_filter):
    """计算深度不确定性权重"""
    device = depth_map.device

    # 统一深度图维度处理
    try:
        if len(depth_map.shape) == 4:
            depth_squeeze = depth_map.squeeze(0).squeeze(0)
        elif len(depth_map.shape) == 3:
            depth_squeeze = depth_map.squeeze(0)
        elif len(depth_map.shape) == 2:
            depth_squeeze = depth_map
        else:
            H, W = 64, 64
            return (torch.zeros(H, W, device=device),
                    torch.zeros(H, W, device=device),
                    torch.zeros(H, W, device=device))

        if len(depth_squeeze.shape) != 2:
            H, W = 64, 64
            return (torch.zeros(H, W, device=device),
                    torch.zeros(H, W, device=device),
                    torch.zeros(H, W, device=device))

        H, W = depth_squeeze.shape

    except Exception as e:
        print(f"Warning: Error processing depth map shape: {e}")
        H, W = 64, 64
        return (torch.zeros(H, W, device=device),
                torch.zeros(H, W, device=device),
                torch.zeros(H, W, device=device))

    # 计算深度梯度
    try:
        depth_gradient_magnitude = torch.zeros(H, W, device=device)

        if H > 1 and W > 1:
            depth_grad_x_inner = torch.abs(depth_squeeze[:, 1:] - depth_squeeze[:, :-1])
            depth_grad_y_inner = torch.abs(depth_squeeze[1:, :] - depth_squeeze[:-1, :])

            depth_grad_x = torch.zeros(H, W, device=device)
            depth_grad_x[:, :-1] = depth_grad_x_inner
            depth_grad_x[:, -1] = depth_grad_x_inner[:, -1]

            depth_grad_y = torch.zeros(H, W, device=device)
            depth_grad_y[:-1, :] = depth_grad_y_inner
            depth_grad_y[-1, :] = depth_grad_y_inner[-1, :]

            depth_gradient_magnitude = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2 + 1e-8)

    except Exception as e:
        print(f"Warning: Error in depth gradient computation: {e}")
        depth_gradient_magnitude = torch.zeros(H, W, device=device)

    # 高斯体缩放不确定性
    try:
        if visibility_filter.sum() > 0:
            visible_scaling = scaling[visibility_filter]
            scale_variance = visible_scaling.var(dim=1)
            scale_uncertainty = scale_variance.mean()
        else:
            scale_uncertainty = torch.tensor(0.0, device=device)
    except Exception as e:
        print(f"Warning: Error in scale uncertainty: {e}")
        scale_uncertainty = torch.tensor(0.0, device=device)

    # 计算法线梯度
    try:
        if len(normal_map.shape) == 4:
            normal_squeeze = normal_map.squeeze(0)
        elif len(normal_map.shape) == 3:
            normal_squeeze = normal_map
        else:
            normal_squeeze = torch.zeros(3, H, W, device=device)
            normal_squeeze[2, :, :] = 1.0

        if normal_squeeze.shape[0] != 3:
            normal_squeeze = torch.zeros(3, H, W, device=device)
            normal_squeeze[2, :, :] = 1.0

        normal_gradient_magnitude = torch.zeros(H, W, device=device)

        if H > 1 and W > 1:
            normal_grad_x_inner = torch.norm(normal_squeeze[:, :, 1:] - normal_squeeze[:, :, :-1], dim=0)
            normal_grad_y_inner = torch.norm(normal_squeeze[:, 1:, :] - normal_squeeze[:, :-1, :], dim=0)

            normal_grad_x = torch.zeros(H, W, device=device)
            normal_grad_x[:, :-1] = normal_grad_x_inner
            normal_grad_x[:, -1] = normal_grad_x_inner[:, -1]

            normal_grad_y = torch.zeros(H, W, device=device)
            normal_grad_y[:-1, :] = normal_grad_y_inner
            normal_grad_y[-1, :] = normal_grad_y_inner[-1, :]

            normal_gradient_magnitude = torch.sqrt(normal_grad_x ** 2 + normal_grad_y ** 2 + 1e-8)

    except Exception as e:
        print(f"Warning: Error in normal gradient computation: {e}")
        normal_gradient_magnitude = torch.zeros(H, W, device=device)

    # 计算最终不确定性
    try:
        depth_uncertainty = torch.sigmoid(depth_gradient_magnitude * 10.0)
        normal_uncertainty = torch.sigmoid(normal_gradient_magnitude * 5.0)
        combined_uncertainty = 0.6 * depth_uncertainty + 0.4 * normal_uncertainty

        return combined_uncertainty, depth_uncertainty, normal_uncertainty

    except Exception as e:
        print(f"Warning: Error in final uncertainty computation: {e}")
        return (torch.zeros(H, W, device=device),
                torch.zeros(H, W, device=device),
                torch.zeros(H, W, device=device))


def compute_normal_uncertainty(normal_map, rendered_distance):
    """计算法线不确定性"""
    device = normal_map.device

    # 处理距离图
    try:
        if isinstance(rendered_distance, np.ndarray):
            rendered_distance = torch.from_numpy(rendered_distance).float().to(device)
        elif not isinstance(rendered_distance, torch.Tensor):
            rendered_distance = torch.tensor(rendered_distance).float().to(device)
        else:
            rendered_distance = rendered_distance.to(device)

        while len(rendered_distance.shape) > 2:
            rendered_distance = rendered_distance.squeeze(0)

        if len(rendered_distance.shape) != 2:
            H, W = normal_map.shape[-2:]
            rendered_distance = torch.zeros(H, W, device=device)

    except Exception as e:
        print(f"Warning: Error processing rendered_distance: {e}")
        H, W = normal_map.shape[-2:]
        rendered_distance = torch.zeros(H, W, device=device)

    # 处理法线图
    try:
        if len(normal_map.shape) == 4:
            normal_squeeze = normal_map.squeeze(0)
        elif len(normal_map.shape) == 3:
            normal_squeeze = normal_map
        else:
            H, W = rendered_distance.shape
            normal_squeeze = torch.zeros(3, H, W, device=device)
            normal_squeeze[2, :, :] = 1.0

        H, W = normal_squeeze.shape[1], normal_squeeze.shape[2]

    except Exception as e:
        print(f"Warning: Error processing normal_map: {e}")
        H, W = 64, 64
        return torch.zeros(H, W, device=device)

    # 法线一致性计算
    try:
        normal_inconsistency = torch.zeros(H, W, device=device)

        if H > 1 and W > 1:
            dot_right_inner = torch.sum(normal_squeeze[:, :, :-1] * normal_squeeze[:, :, 1:], dim=0)
            dot_right = torch.ones(H, W, device=device)
            dot_right[:, :-1] = torch.abs(dot_right_inner)
            dot_right[:, -1] = torch.abs(dot_right_inner[:, -1])

            dot_down_inner = torch.sum(normal_squeeze[:, :-1, :] * normal_squeeze[:, 1:, :], dim=0)
            dot_down = torch.ones(H, W, device=device)
            dot_down[:-1, :] = torch.abs(dot_down_inner)
            dot_down[-1, :] = torch.abs(dot_down_inner[-1, :])

            normal_consistency = (dot_right + dot_down) / 2.0
            normal_inconsistency = 1.0 - torch.clamp(normal_consistency, 0, 1)

    except Exception as e:
        print(f"Warning: Error in normal consistency: {e}")
        normal_inconsistency = torch.zeros(H, W, device=device)

    # 计算距离梯度
    try:
        distance_uncertainty = torch.zeros(H, W, device=device)

        if H > 1 and W > 1:
            distance_grad_x_inner = torch.abs(rendered_distance[:, 1:] - rendered_distance[:, :-1])
            distance_grad_y_inner = torch.abs(rendered_distance[1:, :] - rendered_distance[:-1, :])

            distance_grad_x = torch.zeros(H, W, device=device)
            distance_grad_x[:, :-1] = distance_grad_x_inner
            distance_grad_x[:, -1] = distance_grad_x_inner[:, -1]

            distance_grad_y = torch.zeros(H, W, device=device)
            distance_grad_y[:-1, :] = distance_grad_y_inner
            distance_grad_y[-1, :] = distance_grad_y_inner[-1, :]

            distance_gradient = torch.sqrt(distance_grad_x ** 2 + distance_grad_y ** 2 + 1e-8)
            distance_uncertainty = torch.sigmoid(distance_gradient * 3.0)

    except Exception as e:
        print(f"Warning: Error in distance uncertainty: {e}")
        distance_uncertainty = torch.zeros(H, W, device=device)

    # 组合不确定性
    try:
        combined_normal_uncertainty = 0.7 * normal_inconsistency + 0.3 * distance_uncertainty
        return combined_normal_uncertainty

    except Exception as e:
        print(f"Warning: Error in combining uncertainties: {e}")
        return torch.zeros(H, W, device=device)


def compute_enhanced_depth_consistency(current_depth, nearest_depth, current_cam, nearest_cam,
                                       consistency_threshold=0.15):
    """计算增强的深度一致性"""
    device = current_depth.device

    # 统一深度图维度
    try:
        if len(current_depth.shape) > 2:
            current_depth = current_depth.squeeze()
        if len(nearest_depth.shape) > 2:
            nearest_depth = nearest_depth.squeeze()

        H, W = current_depth.shape
    except:
        H, W = 64, 64
        default_mask = torch.ones(H, W, device=device, dtype=torch.bool)
        default_error = torch.zeros(H, W, device=device)
        return default_mask, default_error

    try:
        # 创建像素坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        # 相机内参
        fx = getattr(current_cam, 'Fx', 500.0)
        fy = getattr(current_cam, 'Fy', 500.0)
        cx = getattr(current_cam, 'Cx', W / 2)
        cy = getattr(current_cam, 'Cy', H / 2)

        # 反投影到3D空间
        z = current_depth
        x = (x_coords.float() - cx) * z / fx
        y = (y_coords.float() - cy) * z / fy

        points_3d = torch.stack([x, y, z], dim=-1).view(-1, 3)

        # 获取相机变换矩阵
        if hasattr(current_cam, 'world_view_transform') and hasattr(nearest_cam, 'world_view_transform'):
            curr_w2v = current_cam.world_view_transform[:3, :].cuda()
            near_w2v = nearest_cam.world_view_transform[:3, :].cuda()

            curr_R = curr_w2v[:3, :3].transpose(0, 1)
            curr_t = curr_w2v[:3, 3]

            near_R = near_w2v[:3, :3]
            near_t = near_w2v[:3, 3]

            R_rel = near_R @ curr_R
            t_rel = near_R @ (-curr_t) + near_t
        else:
            R_rel = torch.eye(3, device=device)
            t_rel = torch.zeros(3, device=device)

        # 变换3D点到邻近视角
        transformed_points = (R_rel @ points_3d.T).T + t_rel
        transformed_points = transformed_points.view(H, W, 3)

        # 投影到邻近视角像素坐标
        fx_near = getattr(nearest_cam, 'Fx', 500.0)
        fy_near = getattr(nearest_cam, 'Fy', 500.0)
        cx_near = getattr(nearest_cam, 'Cx', W / 2)
        cy_near = getattr(nearest_cam, 'Cy', H / 2)

        z_proj = transformed_points[:, :, 2]
        x_proj = transformed_points[:, :, 0] * fx_near / (z_proj + 1e-8) + cx_near
        y_proj = transformed_points[:, :, 1] * fy_near / (z_proj + 1e-8) + cy_near

        # 双线性插值采样邻近视角深度
        x_norm = 2.0 * x_proj / (W - 1) - 1.0
        y_norm = 2.0 * y_proj / (H - 1) - 1.0

        valid_mask = (
                (x_norm >= -1) & (x_norm <= 1) &
                (y_norm >= -1) & (y_norm <= 1) &
                (z_proj > 0)
        )

        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
        sampled_depth = F.grid_sample(
            nearest_depth.unsqueeze(0).unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze()

        # 计算深度重投影误差
        depth_diff = torch.abs(z_proj - sampled_depth)
        relative_error = depth_diff / (z_proj + 1e-8)

        relative_error[~valid_mask] = 1.0

        # 应用阈值过滤
        consistency_mask = (relative_error < consistency_threshold) & valid_mask

        return consistency_mask, relative_error

    except Exception as e:
        print(f"Warning: Depth consistency computation failed: {e}")
        default_mask = torch.ones(H, W, device=device, dtype=torch.bool)
        default_error = torch.zeros(H, W, device=device)
        return default_mask, default_error


def compute_enhanced_uncertainty_weights(current_depth, nearest_depth, current_cam, nearest_cam,
                                         normal_map, rendered_distance, scaling, visibility_filter,
                                         iteration, consistency_threshold=0.15):
    """基于深度先验方法的增强不确定性权重计算"""
    device = current_depth.device

    try:
        if len(current_depth.shape) > 2:
            current_depth = current_depth.squeeze()
        H, W = current_depth.shape
    except:
        H, W = 64, 64
        return torch.ones(H, W, device=device), torch.zeros(H, W, device=device)

    # 多视图深度一致性检查
    consistency_mask, reprojection_error = compute_enhanced_depth_consistency(
        current_depth, nearest_depth, current_cam, nearest_cam, consistency_threshold
    )

    # 几何约束不确定性
    geometric_uncertainty = torch.zeros(H, W, device=device)
    if normal_map is not None:
        try:
            combined_uncertainty, depth_uncertainty, normal_uncertainty = compute_depth_uncertainty(
                current_depth.unsqueeze(0), normal_map, scaling, visibility_filter
            )
            geometric_uncertainty = combined_uncertainty
        except:
            pass

    # 可见性不确定性
    visibility_uncertainty = torch.zeros(H, W, device=device)
    try:
        if visibility_filter.sum() > 0:
            visible_scaling = scaling[visibility_filter]
            scale_ratios = visible_scaling.max(dim=1).values / (visible_scaling.min(dim=1).values + 1e-8)
            high_ratio_count = (scale_ratios > 10.0).sum().float() / len(scale_ratios)
            visibility_uncertainty.fill_(high_ratio_count * 0.5)
    except:
        pass

    # 训练阶段自适应调整
    if iteration < 1000:
        stage_factor = 0.5
        min_weight, max_weight = 0.2, 1.5
    elif iteration < 5000:
        stage_factor = 1.0
        min_weight, max_weight = 0.01, 3.0
    else:
        stage_factor = 0.8
        min_weight, max_weight = 0.05, 2.0

    # 综合不确定性
    total_uncertainty = (
            0.6 * torch.sigmoid(reprojection_error * 5.0) * stage_factor +
            0.3 * geometric_uncertainty +
            0.1 * visibility_uncertainty
    )

    # 应用一致性mask
    confidence_base = 1.0 - total_uncertainty

    confidence_weights = torch.where(
        consistency_mask,
        min_weight + (max_weight - min_weight) * torch.pow(confidence_base, 0.5),
        torch.full_like(confidence_base, min_weight * 0.1)
    )

    return confidence_weights, total_uncertainty


def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """主训练函数"""
    torch.cuda.reset_peak_memory_stats()
    peak_memory_usage = 0.0
    scene_type = "outdoor"

    first_iter = 0
    prepare_output_and_logger(dataset)

    # 备份代码
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scripts {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 初始化EMA损失
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    ema_freq_loss_for_log = 0.0

    normal_loss, geo_loss, ncc_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    indices = None
    training_generator = DataLoader(scene.getTrainCameras(), num_workers=8, prefetch_factor=1,
                                    persistent_workers=True, collate_fn=direct_collate)
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    # 频率正则化相关目录
    freq_viz_dir = os.path.join(scene.model_path, "visualizations")
    os.makedirs(freq_viz_dir, exist_ok=True)

    iteration = first_iter
    while iteration < opt.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                background = torch.rand((3), dtype=torch.float32, device="cuda")
                # background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")  # 固定背景颜色

                # 移动相机参数到GPU
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                # 网络GUI处理
                if not args.disable_viewer:
                    if network_gui.conn == None:
                        network_gui.try_connect()
                    while network_gui.conn != None:
                        try:
                            net_image_bytes = None
                            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                            if custom_cam != None:
                                if keep_alive:
                                    net_image = \
                                    render(custom_cam, gaussians, pipe, background, scaling_modifer, indices=indices)[
                                        "render"]
                                else:
                                    net_image = \
                                    render(custom_cam, gaussians, pipe, background, scaling_modifer, indices=indices)[
                                        "depth"].repeat(3, 1, 1)
                                net_image_bytes = memoryview(
                                    (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                  0).contiguous().cpu().numpy())
                            network_gui.send(net_image_bytes, dataset.source_path)
                            if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                                break
                        except Exception as e:
                            network_gui.conn = None

                iter_start.record()

                gaussians.update_learning_rate(iteration)

                # 每1000次迭代增加SH度数
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                # 渲染
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                render_pkg = render(viewpoint_cam, gaussians, pipe, background, indices=indices,
                                    use_trained_exp=True,
                                    return_plane=iteration > opt.single_view_weight_from_iter,
                                    return_depth_normal=iteration > opt.single_view_weight_from_iter)

                image, invDepth, Plane_Depth, viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["render"], render_pkg["depth"], render_pkg["plane_depth"],
                    render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                )

                # 损失计算
                gt_image = viewpoint_cam.original_image.cuda()
                gt_image_gray = viewpoint_cam.gt_gray_image.cuda()

                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    image *= alpha_mask

                # 改进版本
                freq_loss, high_freq_mask, debug_info = frequency_regularization_pyramid_scale(
                    rendered_image=image,
                    gt_image=gt_image,
                    gaussians=gaussians,
                    scene=scene,
                    viewpoint_cam=viewpoint_cam,
                    visibility_filter=visibility_filter,
                    iteration=iteration,
                    lambda_freq=0.1,  # 适中权重，包含FFT损失
                    lambda_scale=10.0,  # 适中权重
                    num_levels=3,  # 3层金字塔
                    high_freq_thresh=0.2,  # 高频阈值
                    save_results=True,  # 保存FFT可视化
                    save_dir=freq_viz_dir,
                    debug=False  # 查看FFT计算状态
                )

                # 高频加权损失
                if iteration > 200 and high_freq_mask is not None and high_freq_mask.sum() > 0:
                    weight_mask = torch.ones_like(high_freq_mask)
                    weight_mask[high_freq_mask > 0.5] = 2.0
                    pixel_l1 = torch.abs(image - gt_image).mean(dim=0)
                    Ll1 = (pixel_l1 * weight_mask).mean()

                    if iteration % 500 == 0:
                        high_freq_ratio = (high_freq_mask > 0.5).float().mean()
                        print(f"[{iteration}] 高频加权: {high_freq_ratio:.1%}")
                else:
                    Ll1 = l1_loss(image, gt_image)

                # Ll1 = l1_loss(image, gt_image)  # 损失加权的时候注释掉
                Lssim = (1.0 - ssim(image, gt_image))
                photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
                loss = photo_loss.clone()

                loss += freq_loss  # 将正则化损失加入总损失

                # 深度损失
                Ll1depth_pure = 0.0
                if (depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable):
                    # if iteration > opt.do_Ll1depth_from_iter:
                    #     invDepth = torch.zeros_like(Plane_Depth)  # 用平面深度图监督
                    #     valid_mask = Plane_Depth > 0
                    #     invDepth[valid_mask] = 1 / Plane_Depth[valid_mask]
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    depth_mask = viewpoint_cam.depth_mask.cuda()
                    Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                    loss += Ll1depth
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0

                # 缩放损失（全局）
                if visibility_filter.sum() > 0:
                    scaling = gaussians.get_scaling[visibility_filter]

                    # 基础统计
                    min_scale = scaling.min(dim=1).values
                    max_scale = scaling.max(dim=1).values
                    mean_scale = scaling.mean(dim=1)

                    # 最小缩放比例损失 - 防止高斯过扁
                    min_scale_loss = torch.clamp(min_scale - 0.01, max=0).pow(2).mean()

                    # 最大缩放比例损失 - 防止高斯过大
                    scene_scale = 0.1 * scene.cameras_extent
                    max_scale_loss = torch.clamp(max_scale - scene_scale, min=0).pow(2).mean() if iteration > 5000 else 0.0

                    # 轴比例正则化 - 防止轴比例差异过大
                    axis_ratio = max_scale / torch.clamp(min_scale, min=1e-8)  # 防止除零
                    ratio_mask = axis_ratio > 20.0  # 轴比例阈值

                    # 计算需要正则化的高斯体的损失
                    axis_reg_loss = 0.0
                    if ratio_mask.any():
                        selected_scaling = scaling[ratio_mask]
                        axis_reg_loss = torch.abs(selected_scaling - selected_scaling.mean(dim=1, keepdim=True)).mean()

                # 合并所有缩放相关损失
                scale_loss = 100.0 * min_scale_loss + 10.0 * max_scale_loss + 0.001 * axis_reg_loss
                loss += scale_loss

                # 单视图损失
                if iteration > opt.single_view_weight_from_iter:
                    weight = opt.single_view_weight
                    normal = render_pkg["rendered_normal"]
                    depth_normal = render_pkg["depth_normal"]

                    image_weight = (1.0 - get_img_grad_weight(gt_image))
                    image_weight = (image_weight).clamp(0, 1).detach() ** 5
                    image_weight = erode(image_weight[None, None]).squeeze()

                    normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
                    loss += (normal_loss)

                # 多视图损失（深度和法线不确定性加权）
                if iteration > opt.multi_view_weight_from_iter:
                    nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras(multi_view=True)[random.sample(viewpoint_cam.nearest_id, 1)[0]]
                    use_virtul_cam = False

                    if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                        nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis,
                                                     deg_noise=dataset.multi_view_max_angle)
                        use_virtul_cam = True

                    if nearest_cam is not None:
                        patch_size = opt.multi_view_patch_size
                        sample_num = opt.multi_view_sample_num
                        pixel_noise_th = opt.multi_view_pixel_noise_th
                        total_patch_size = (patch_size * 2 + 1) ** 2
                        ncc_weight = opt.multi_view_ncc_weight
                        geo_weight = opt.multi_view_geo_weight

                        normal = render_pkg.get("rendered_normal", None)
                        rendered_distance = render_pkg.get("rendered_distance", None)

                        # 初始化不确定性相关变量
                        confidence_weight = torch.ones_like(render_pkg['plane_depth'].squeeze())
                        total_uncertainty = torch.zeros_like(confidence_weight)
                        depth_uncertainty = torch.zeros_like(confidence_weight)
                        normal_uncertainty = torch.zeros_like(confidence_weight)
                        combined_uncertainty = torch.zeros_like(confidence_weight)
                        normal_specific_uncertainty = torch.zeros_like(confidence_weight)
                        uncertainty_computed = False

                        # 计算深度和法线不确定性
                        if normal is not None and rendered_distance is not None:
                            try:
                                scaling = gaussians.get_scaling
                                combined_uncertainty, depth_uncertainty, normal_uncertainty = compute_depth_uncertainty(
                                    render_pkg['plane_depth'], normal, scaling, visibility_filter)

                                normal_specific_uncertainty = compute_normal_uncertainty(normal, rendered_distance)

                                total_uncertainty = 0.4 * combined_uncertainty + 0.3 * normal_specific_uncertainty + 0.3 * depth_uncertainty

                                confidence_weight = 1.0 - total_uncertainty
                                confidence_weight = torch.clamp(confidence_weight, min=0.1, max=1.0)

                                uncertainty_computed = True

                            except Exception as e:
                                print(f"Warning: Error computing uncertainty at iteration {iteration}: {e}")

                        # 计算几何一致性
                        H, W = render_pkg['plane_depth'].squeeze().shape
                        ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                        pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                        nearest_render_pkg = render(nearest_cam, gaussians, pipe, background, indices=indices,
                                                    use_trained_exp=True)

                        confidence_weights, total_uncertainty = compute_enhanced_uncertainty_weights(
                            render_pkg['plane_depth'],
                            nearest_render_pkg['plane_depth'],
                            viewpoint_cam,
                            nearest_cam,
                            normal,
                            rendered_distance,
                            gaussians.get_scaling,
                            visibility_filter,
                            iteration,
                            consistency_threshold=0.15
                        )

                        uncertainty_computed = True

                        pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                        pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3].cuda() + nearest_cam.world_view_transform[3, :3].cuda()
                        map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)

                        pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
                        pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
                        R = torch.tensor(nearest_cam.R).float().cuda()
                        T = torch.tensor(nearest_cam.T).float().cuda()
                        pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
                        pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,
                                                 :3] + viewpoint_cam.world_view_transform[3, :3]

                        pts_projections = torch.stack([
                            pts_in_view_cam[:, 0] * viewpoint_cam.Fx / pts_in_view_cam[:, 2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:, 1] * viewpoint_cam.Fy / pts_in_view_cam[:, 2] + viewpoint_cam.Cy
                        ], -1).float()

                        pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                        d_mask = d_mask & (pixel_noise < pixel_noise_th)
                        weights = (1.0 / torch.exp(pixel_noise)).detach()
                        weights[~d_mask] = 0

                        # 应用不确定性权重
                        confidence_weight_flat = confidence_weights.reshape(-1)
                        uncertainty_weighted_weights = weights * confidence_weight_flat

                        # 可视化调试信息
                        if iteration % 200 == 0:
                            gt_img_show = ((gt_image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                           [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)

                            if 'app_image' in render_pkg:
                                img_show = ((render_pkg['app_image']).permute(1, 2, 0).clamp(0, 1)[:, :,
                                            [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                            else:
                                img_show = ((image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                            [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)

                            uncertainty_vis = (total_uncertainty.detach().cpu().numpy() * 255).astype(np.uint8)
                            confidence_vis = (confidence_weights.detach().cpu().numpy() * 255).astype(np.uint8)

                            black_regions = (confidence_weights < 0.1).float().detach().cpu().numpy() * 255

                            uncertainty_color = cv2.applyColorMap(uncertainty_vis, cv2.COLORMAP_HOT)
                            confidence_color = cv2.applyColorMap(confidence_vis, cv2.COLORMAP_VIRIDIS)
                            black_regions_vis = cv2.applyColorMap(black_regions.astype(np.uint8), cv2.COLORMAP_BONE)

                            if normal is not None:
                                normal_show = (((normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0,
                                                                                             1) * 255).detach().cpu().numpy().astype(
                                    np.uint8)
                            else:
                                normal_show = np.zeros_like(gt_img_show)

                            if "depth_normal" in render_pkg:
                                depth_normal = render_pkg["depth_normal"]
                                depth_normal_show = (((depth_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0,
                                                                                                         1) * 255).detach().cpu().numpy().astype(
                                    np.uint8)
                            else:
                                depth_normal_show = np.zeros_like(gt_img_show)

                            d_mask_show = (uncertainty_weighted_weights.float() * 255).detach().cpu().numpy().astype(
                                np.uint8).reshape(H, W)
                            d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)

                            depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

                            row0 = np.concatenate([gt_img_show, img_show, uncertainty_color, black_regions_vis], axis=1)
                            row1 = np.concatenate([confidence_color, depth_color, normal_show, d_mask_show_color],
                                                  axis=1)
                            image_to_show = np.concatenate([row0, row1], axis=0)

                            cv2.imwrite(os.path.join(debug_path, "%05d" % iteration + "_uncertainty_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                        # 几何损失计算
                        if d_mask.sum() > 0:
                            geo_loss = geo_weight * ((uncertainty_weighted_weights * pixel_noise)[d_mask]).mean()
                            loss += geo_loss

                            # 光度一致性损失计算
                            if use_virtul_cam is False:
                                with torch.no_grad():
                                    d_mask = d_mask.reshape(-1)
                                    valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                                    if d_mask.sum() > sample_num:
                                        index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                                        valid_indices = valid_indices[index]

                                    weights_selected = uncertainty_weighted_weights.reshape(-1)[valid_indices]
                                    pixels = pixels.reshape(-1, 2)[valid_indices]
                                    offsets = patch_offsets(patch_size, pixels.device)
                                    ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                                    H_gray, W_gray = gt_image_gray.squeeze().shape
                                    pixels_patch = ori_pixels_patch.clone()
                                    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W_gray - 1) - 1.0
                                    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H_gray - 1) - 1.0
                                    ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                                    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                                    ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2).cuda() @ viewpoint_cam.world_view_transform[:3, :3].cuda()
                                    ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3, :3].cuda() + nearest_cam.world_view_transform[3, :3].cuda()

                                # 计算Homography
                                if normal is not None and rendered_distance is not None:
                                    ref_local_n = normal.permute(1, 2, 0)
                                    ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

                                    ref_local_d = rendered_distance.squeeze()
                                    ref_local_d = ref_local_d.reshape(-1)[valid_indices]

                                    H_ref_to_neareast = ref_to_neareast_r[None] - torch.matmul(
                                        ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                        ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2, 1)
                                    ) / ref_local_d[..., None, None]

                                    H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                                    H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                                    grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
                                    grid[:, :, 0] = 2 * grid[:, :, 0] / (W_gray - 1) - 1.0
                                    grid[:, :, 1] = 2 * grid[:, :, 1] / (H_gray - 1) - 1.0
                                    nearest_image_gray = nearest_cam.gt_gray_image.cuda()
                                    sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                                    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                                    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                                    mask = ncc_mask.reshape(-1)
                                    ncc = ncc.reshape(-1) * weights_selected
                                    ncc = ncc[mask].squeeze()

                                    if mask.sum() > 0:
                                        ncc_loss = ncc_weight * ncc.mean()
                                        loss += ncc_loss

                        # 统计信息
                        if iteration % 1000 == 0:
                            avg_uncertainty = total_uncertainty.mean().item()
                            avg_confidence = confidence_weights.mean().item()
                            black_region_ratio = (confidence_weights < 0.1).float().mean().item()
                            high_confidence_ratio = (confidence_weights > 1.5).float().mean().item()

                            print(f"[{iteration}] Enhanced Uncertainty Stats:")
                            print(f"  平均不确定性: {avg_uncertainty:.3f}")
                            print(f"  平均置信度: {avg_confidence:.3f}")
                            print(f"  黑色区域比例: {black_region_ratio:.1%}")
                            print(f"  高置信度区域比例: {high_confidence_ratio:.1%}")

                # loss.backward()
                loss.backward(retain_graph=True)
                iter_end.record()

                with torch.no_grad():
                    # 更新EMA损失
                    ema_loss_for_log = 0.4 * photo_loss.item() + 0.6 * ema_loss_for_log
                    ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
                    ema_freq_loss_for_log = 0.4 * freq_loss.item() + 0.6 * ema_freq_loss_for_log
                    ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
                    ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
                    ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log

                    if iteration % 10 == 0:
                        progress_bar.set_postfix({
                            "Loss": f"{ema_loss_for_log:.{7}f}",
                            "Depth Loss": f"{ema_Ll1depth_for_log:.{5}f}",
                            "Freq": f"{ema_freq_loss_for_log:.{5}f}",
                            "Single": f"{ema_single_view_for_log:.{5}f}",
                            "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                            "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                            "Size": f"{gaussians._xyz.size(0)}"
                        })
                        progress_bar.update(10)

                    # 保存检查点
                    if (iteration in saving_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                        print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))

                    if iteration == opt.iterations:
                        progress_bar.close()
                        return

                    # 稠密化
                    if iteration < opt.densify_until_iter:
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                             radii)
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_path = os.path.join(args.model_path, "iteration_size.txt")
                            with open(size_path, 'a') as f:
                                f.write(f"{iteration}\t{gaussians._xyz.size(0)}\n")

                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                                                        0.005, scene.cameras_extent, size_threshold)

                        if iteration % opt.opacity_reset_interval == 0 or (
                                dataset.white_background and iteration == opt.densify_from_iter):
                            print("-----------------RESET OPACITY!-------------")
                            gaussians.reset_opacity()

                    # 多视图观察修剪
                    if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                        observe_the = 2
                        observe_cnt = torch.zeros_like(gaussians.get_opacity)
                        for view in scene.getTrainCameras(True):
                            render_pkg_tmp = render(view, gaussians, pipe, background, return_plane=False, return_depth_normal=False)
                            out_observe = render_pkg_tmp["out_observe"]
                            observe_cnt[out_observe > 0] += 1
                        prune_mask = (observe_cnt < observe_the).squeeze()
                        if prune_mask.sum() > 0:
                            gaussians.prune_points(prune_mask)

                    # 优化器步骤
                    if iteration < opt.iterations:
                        gaussians.exposure_optimizer.step()
                        gaussians.exposure_optimizer.zero_grad(set_to_none=True)

                        if gaussians._xyz.grad != None and gaussians.skybox_locked:
                            gaussians._xyz.grad[:gaussians.skybox_points, :] = 0
                            gaussians._rotation.grad[:gaussians.skybox_points, :] = 0
                            gaussians._features_dc.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._features_rest.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._opacity.grad[:gaussians.skybox_points, :] = 0
                            gaussians._scaling.grad[:gaussians.skybox_points, :] = 0

                        if gaussians._opacity.grad != None:
                            relevant = (gaussians._opacity.grad.flatten() != 0).nonzero()
                            relevant = relevant.flatten().long()
                            if (relevant.size(0) > 0):
                                gaussians.optimizer.step(relevant)
                            else:
                                gaussians.optimizer.step(relevant)
                                print("No grads!")
                            gaussians.optimizer.zero_grad(set_to_none=True)

                    # 缩放控制
                    if not args.skip_scale_big_gauss:
                        with torch.no_grad():
                            vals, _ = gaussians.get_scaling.max(dim=1)
                            violators = vals > scene.cameras_extent * 0.02
                            if gaussians.scaffold_points is not None:
                                violators[:gaussians.scaffold_points] = False
                            gaussians._scaling[violators] = gaussians.scaling_inverse_activation(
                                gaussians.get_scaling[violators] * 0.8)

                    if (iteration in checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration),
                                   scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    iteration += 1


def prepare_output_and_logger(args):
    """准备输出目录和日志"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    # 命令行参数解析
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    if args.eval and args.exposure_lr_init > 0 and not args.train_test_exp:
        print(
            "Reconstructing for evaluation (--eval) with exposure optimization on the train set but not for the test set.")
        print(
            "This will lead to high error when computing metrics. To optimize exposure on the left half of the test images, use --train_test_exp")

    # 初始化系统状态
    safe_state(args.quiet)

    # 启动GUI服务器并运行训练
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")