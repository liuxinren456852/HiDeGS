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

import os
import torch
import numpy as np
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import camera_to_JSON, CameraDataset
from utils.camera_utils import *
from utils.system_utils import mkdir_p
import time
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], create_from_hier=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.train_cameras_nearest = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.alpha_masks, args.depths, args.eval, args.train_test_exp)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:   # 打开场景信息中的点云文件（以二进制只读模式）  # 打开目标文件，用于将点云文件复制到模型路径下，命名为 input.ply（以二进制写入模式)
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())  # 将源点云文件的内容读取并写入到目标文件中
            json_cams = []  # 初始化一个空列表，用于存储相机的 JSON 数据
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))  # 调用 camera_to_JSON 函数，将相机信息转换为 JSON 格式，并添加到 json_cams 列表中
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)  # 使用 json.dump 函数将 json_cams 列表中的数据以 JSON 格式写入到文件中

        if shuffle:  # 这样做可以保证多分辨率处理时的随机性是一致的
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        # 从场景信息里获取 NeRF 归一化的半径，将其设为相机的范围
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # 遍历不同的分辨率缩放比例
        self.multi_view_num = args.multi_view_num
        for resolution_scale in resolution_scales:
            print("Making Training Dataset")
            self.train_cameras[resolution_scale] = CameraDataset(scene_info.train_cameras, args, resolution_scale, False)
            self.train_cameras_nearest[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, is_test_dataset=False)

            print("Making Test Dataset")
            self.test_cameras[resolution_scale] = CameraDataset(scene_info.test_cameras, args, resolution_scale, True)

            print("computing nearest_id")  # 计算每个相机的最近视图
            self.world_view_transforms = []
            camera_centers = []
            center_rays = []
            for id, cur_cam in enumerate(self.train_cameras[resolution_scale].list_cam_infos):
                self.world_view_transforms.append(torch.tensor(cur_cam.R).float().cuda())  # 提取变换矩阵
                camera_centers.append(torch.tensor(cur_cam.T).float().cuda())  # 提取相机中心
                R = torch.tensor(cur_cam.R).float().cuda()
                T = torch.tensor(cur_cam.T).float().cuda()
                center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
                center_ray = center_ray @ R.transpose(-1, -2)  # 计算相机的中心射线方向
                center_rays.append(center_ray)
            # self.world_view_transforms = [torch.tensor(R).float().cuda() for R in self.world_view_transforms]
            self.world_view_transforms = [R.clone().detach().float().cuda() for R in self.world_view_transforms]
            camera_centers = torch.stack(camera_centers, dim=0)
            center_rays = torch.stack(center_rays, dim=0)
            center_rays = torch.nn.functional.normalize(center_rays, dim=-1)  # 对中心射线进行归一化
            diss = torch.norm(camera_centers[:, None] - camera_centers[None],
                              dim=-1).detach().cpu().numpy()  # 计算所有相机中心之间的欧几里得距离[N,N]
            tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1)  # 通过中心射线的点积计算夹角
            angles = torch.arccos(tmp) * 180 / 3.14159
            angles = angles.detach().cpu().numpy()
            # 计算最近相机的索引：
            # 按角度（angles）和距离（diss）进行排序。
            # 使用过滤条件：
            # 角度小于 args.multi_view_max_angle。
            # 距离在 args.multi_view_min_dis 和 args.multi_view_max_dis 之间。
            # 保存最近视图信息：
            # 对每个相机，存储与其最近的多视图相机的图像名称。
            # 将结果写入 JSON 文件
            with open(os.path.join(args.model_path, "multi_view.json"), 'w') as file:
                for id, cur_cam in enumerate(self.train_cameras[resolution_scale].list_cam_infos):
                    cur_cam.nearest_id.clear()  # 清空列表
                    cur_cam.nearest_names.clear()  # 清空列表
                    sorted_indices = np.lexsort((angles[id], diss[id]))
                    mask = (angles[id][sorted_indices] < args.multi_view_max_angle) & \
                           (diss[id][sorted_indices] > args.multi_view_min_dis) & \
                           (diss[id][sorted_indices] < args.multi_view_max_dis)
                    sorted_indices = sorted_indices[mask]
                    multi_view_num = min(args.multi_view_num, len(sorted_indices))
                    json_d = {'ref_name': cur_cam.image_name, 'nearest_name': []}
                    for index in sorted_indices[:multi_view_num]:
                        cur_cam.nearest_id.append(index)
                        cur_cam.nearest_names.append(self.train_cameras[resolution_scale].list_cam_infos[index].image_name)
                        json_d["nearest_name"].append(self.train_cameras[resolution_scale].list_cam_infos[index].image_name)

                    json_str = json.dumps(json_d, separators=(',', ':'))
                    file.write(json_str)
                    file.write('\n')

        if self.loaded_iter:  # 检查是否已经加载了特定的迭代信息
            self.gaussians.load_ply(os.path.join(self.model_path,  # 路径结构为模型路径 -> point_cloud 文件夹 -> 对应迭代次数的文件夹 -> point_cloud.ply 文件
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif args.pretrained:  # 若没有加载迭代信息，但指定了预训练模型文件
            self.gaussians.create_from_pt(args.pretrained, self.cameras_extent)  # self.cameras_extent 是相机的范围，用于归一化等操作  # 从层次结构文件创建高斯点云   # args.hierarchy 是层次结构文件的路径
        elif create_from_hier:  # 若既没有加载迭代信息，也没有指定预训练模型文件，但设置了从层次结构创建
            self.gaussians.create_from_hier(args.hierarchy, self.cameras_extent, args.scaffold_file)   # self.cameras_extent 是相机的范围   # args.scaffold_file 是脚手架文件的路径，可能用于辅助构建点云
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, 
                                           scene_info.train_cameras,
                                           self.cameras_extent, 
                                           args.skybox_num,
                                           args.scaffold_file,
                                           args.bounds_file,
                                           args.skybox_locked)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        mkdir_p(point_cloud_path)
        if self.gaussians.nodes is not None:
            self.gaussians.save_hier()
        else:
            with open(os.path.join(point_cloud_path, "pc_info.txt"), "w") as f:
                f.write(str(self.gaussians.skybox_points))
            if self.gaussians._xyz.size(0) > 8_000_000:
                self.gaussians.save_pt(point_cloud_path)
            else:
                self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

            exposure_dict = {
                image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                for image_name in self.gaussians.exposure_mapping
            }

            with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
                json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0, multi_view=False):
        if multi_view == False:
            return self.train_cameras[scale]
        else:
            return self.train_cameras_nearest[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
