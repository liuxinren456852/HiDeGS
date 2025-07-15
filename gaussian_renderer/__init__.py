#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
from diff_gaussian_rasterization import _C
import numpy as np

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale],
                                            intrinsic_matrix.to(depth.device),
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref


def render(
        viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, indices = None, use_trained_exp=False, return_plane = True, return_depth_normal = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # 创建一个与 pc.get_xyz 形状相同的零张量，用于记录 2D（屏幕空间）均值的梯度
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()  # 保留该张量的梯度，以便后续使用
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # 初始化渲染索引、父索引、插值权重和兄弟节点数量的张量
    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_siblings = torch.empty(0).int().cuda()
    # 创建高斯光栅化设置对象，包含图像尺寸、视场角、背景颜色、变换矩阵等信息
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        render_geo=return_plane,
        debug=pipe.debug,
        do_depth=True,
        # do_depth=False,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_siblings
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # 获取高斯模型的 3D 中心点、2D 屏幕空间点和不透明度
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    # 初始化缩放、旋转和 3D 协方差矩阵
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    # 初始化球谐函数系数和预计算颜色
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    
    if indices is not None:
        means3D = means3D[indices].contiguous()  # 调用 contiguous() 方法，保证新张量在内存中是连续存储的。
        means2D = means2D[indices].contiguous()
        shs = shs[indices].contiguous()
        opacity = opacity[indices].contiguous()
        scales = scales[indices].contiguous()
        rotations = rotations[indices].contiguous()


    # 使用光栅化器对可见的高斯点进行渲染，得到渲染图像、半径和深度图像
    if not return_plane:
        rendered_image, radii, out_observe, out_all_map, plane_depth, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        if use_trained_exp:
            exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)  # 从高斯模型 pc 中获取与当前相机图像名称对应的曝光矩阵
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0,
                                                                                                     1) + exposure[:3, 3, None, None]  # 对渲染图像进行曝光调整
        rendered_image = rendered_image.clamp(0, 1)  # 对渲染图像进行裁剪，确保像素值在 0 到 1 之间
        # 创建一个布尔掩码，用于过滤半径大于 0 的高斯点
        subfilter = radii > 0
        if indices is not None:
            vis_filter = torch.zeros(pc._xyz.size(0), dtype=bool, device="cuda")
            w = vis_filter[indices]
            w[subfilter] = True
            vis_filter[indices] = w
        else:
            vis_filter = subfilter
        # 那些被视锥体剔除或者半径为 0 的高斯点是不可见的    # 它们将被排除在用于分裂标准的值更新之外    # 返回一个字典，包含渲染结果的各个部分
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "plane_depth": plane_depth,
                "depth": depth_image,
                "out_observe": out_observe,
                "viewspace_points": screenspace_points,  # 屏幕空间点
                "visibility_filter": vis_filter.nonzero().flatten().long(),  # 可见性过滤器，返回可见高斯点的索引
                "radii": radii[subfilter]}

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3, :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance
    rendered_image, radii, out_observe, out_all_map, plane_depth, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map=input_all_map,
        cov3D_precomp = cov3D_precomp)
    
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name) # 从高斯模型 pc 中获取与当前相机图像名称对应的曝光矩阵
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]   # 对渲染图像进行曝光调整
    rendered_image = rendered_image.clamp(0, 1)  # 对渲染图像进行裁剪，确保像素值在 0 到 1 之间
    # 创建一个布尔掩码，用于过滤半径大于 0 的高斯点
    subfilter = radii > 0
    if indices is not None:
        vis_filter = torch.zeros(pc._xyz.size(0), dtype=bool, device="cuda")
        w = vis_filter[indices]
        w[subfilter] = True
        vis_filter[indices] = w
    else:
        vis_filter = subfilter
    # 那些被视锥体剔除或者半径为 0 的高斯点是不可见的    # 它们将被排除在用于分裂标准的值更新之外    # 返回一个字典，包含渲染结果的各个部分
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]

    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()

    return {"render": rendered_image,
            "depth" : depth_image,
            "viewspace_points": screenspace_points,  #  屏幕空间点
            "visibility_filter" : vis_filter.nonzero().flatten().long(),  # 可见性过滤器，返回可见高斯点的索引
            "radii": radii[subfilter],
            "out_observe": out_observe,
            "rendered_normal": rendered_normal,
            "plane_depth": plane_depth,
            "rendered_distance": rendered_distance,
            "depth_normal": depth_normal
            }


def render_post(
        viewpoint_camera, 
        pc : GaussianModel, 
        pipe, 
        bg_color : torch.Tensor, 
        scaling_modifier = 1.0, 
        override_color = None, 
        render_indices = torch.Tensor([]).int(),
        parent_indices = torch.Tensor([]).int(),
        interpolation_weights = torch.Tensor([]).float(),
        num_node_kids = torch.Tensor([]).int(),
        interp_python = True,
        use_trained_exp = False):
    """
    Render the scene from a hierarchy.  
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
        
    if render_indices.size(0) != 0:
        render_inds = render_indices.long()
        if interp_python:
            num_entries = render_indices.size(0)

            interps = interpolation_weights[:num_entries].unsqueeze(1)
            interps_inv = (1 - interpolation_weights[:num_entries]).unsqueeze(1)
            parent_inds = parent_indices[:num_entries].long()

            means3D_base = (interps * means3D[render_inds] + interps_inv * means3D[parent_inds]).contiguous()
            scales_base = (interps * scales[render_inds] + interps_inv * scales[parent_inds]).contiguous()
            shs_base = (interps.unsqueeze(2) * shs[render_inds] + interps_inv.unsqueeze(2) * shs[parent_inds]).contiguous()
            
            parents = rotations[parent_inds]
            rots = rotations[render_inds]
            dots = torch.bmm(rots.unsqueeze(1), parents.unsqueeze(2)).flatten()
            parents[dots < 0] *= -1
            rotations_base = ((interps * rots) + interps_inv * parents).contiguous()
            
            opacity_base = (interps * opacity[render_inds] + interps_inv * opacity[parent_inds]).contiguous()

            if pc.skybox_points == 0:
                skybox_inds = torch.Tensor([]).long()
            else:
                skybox_inds = torch.range(pc._xyz.size(0) - pc.skybox_points, pc._xyz.size(0)-1, device="cuda").long()

            means3D = torch.cat((means3D_base, means3D[skybox_inds])).contiguous()  
            shs = torch.cat((shs_base, shs[skybox_inds])).contiguous() 
            opacity = torch.cat((opacity_base, opacity[skybox_inds])).contiguous()  
            rotations = torch.cat((rotations_base, rotations[skybox_inds])).contiguous()    
            means2D = means2D[:(num_entries + pc.skybox_points)].contiguous()     
            scales = torch.cat((scales_base, scales[skybox_inds])).contiguous()  

            interpolation_weights = interpolation_weights.clone().detach()
            interpolation_weights[num_entries:num_entries+pc.skybox_points] = 1.0 
            num_node_kids[num_entries:num_entries+pc.skybox_points] = 1 
        
        else:
            means3D = means3D[render_inds].contiguous()
            means2D = means2D[render_inds].contiguous()
            shs = shs[render_inds].contiguous()
            opacity = opacity[render_inds].contiguous()
            scales = scales[render_inds].contiguous()
            rotations = rotations[render_inds].contiguous() 

        render_indices = torch.Tensor([]).int()
        parent_indices = torch.Tensor([]).int()
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_node_kids,
        do_depth=False,
        render_geo=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, _, _, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    if use_trained_exp and pc.pretrained_exposures:
        try:
            exposure = pc.pretrained_exposures[viewpoint_camera.image_name]
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
        except Exception as e:
            print(f"Exposures should be optimized in single. Missing exposure for image {viewpoint_camera.image_name}")
    rendered_image = rendered_image.clamp(0, 1)
    
    vis_filter = radii > 0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter,
            "radii": radii[vis_filter]}

def render_coarse(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, zfar=0.0, override_color = None, indices = None):
    """
    Render the scene for the coarse optimization. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    render_indices = torch.empty(0).int().cuda()
    parent_indices = torch.empty(0).int().cuda()
    interpolation_weights = torch.empty(0).float().cuda()
    num_siblings = torch.empty(0).int().cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=True,
        do_depth=False,
        render_indices=render_indices,
        parent_indices=parent_indices,
        interpolation_weights=interpolation_weights,
        num_node_kids=num_siblings,
        render_geo=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if indices is not None:
        means3D = means3D[indices].contiguous()
        means2D = means2D[indices].contiguous()
        shs = shs[indices].contiguous()
        opacity = opacity[indices].contiguous()
        scales = scales[indices].contiguous()
        rotations = rotations[indices].contiguous() 

    rendered_image, radii, _, _, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    

    
    subfilter = radii > 0
    if indices is not None:
        vis_filter = torch.zeros(pc._xyz.size(0), dtype=bool, device="cuda")
        w = vis_filter[indices]
        w[subfilter] = True
        vis_filter[indices] = w
    else:
        vis_filter = subfilter

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : vis_filter,
            "radii": radii[subfilter]}
