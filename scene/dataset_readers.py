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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
from dataclasses import dataclass, field
from typing import List, Dict

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    primx:float
    primy:float
    depth_params: dict
    image_path: str
    mask_path: str
    depth_path: str
    image_name: str
    width: int
    height: int
    is_test: bool
    nearest_id: list
    nearest_names: list

    # def __post_init__(self):
    #     if self.nearest_id is None:
    #         self.nearest_id = []
    #     if self.nearest_names is None:
    #         self.nearest_names = []

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        #  计算相机中心的平均位置以及一个合适的对角线长度。
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.quantile(dist, 0.9)
        return center.flatten(), diagonal

    cam_centers = []
    # 遍历输入的相机信息列表，计算每个相机的中心位置并存储在 cam_centers 列表中。
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    # 调用 get_center_and_diag 函数计算平均相机中心位置和对角线长度，然后根据这些结果计算归一化所需的平移向量和半径
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            primx = float(intr.params[1]) / width
            primy = float(intr.params[2]) / height
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            primx = float(intr.params[2]) / width
            primy = float(intr.params[3]) / height
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{extr.name[:-n_remove]}.jpg")
            image_name = f"{extr.name[:-n_remove]}.jpg"
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{extr.name[:-n_remove]}.png")
            image_name = f"{extr.name[:-n_remove]}.png"

        mask_path = os.path.join(masks_folder, f"{extr.name[:-n_remove]}.png") if masks_folder != "" else ""
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        # Initialize nearest_id and nearest_names as empty lists
        nearest_id = []
        nearest_names = []

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, primx=primx, primy=primy, depth_params=depth_params,
                              image_path=image_path, mask_path=mask_path, depth_path=depth_path, image_name=image_name, 
                              width=width, height=height, is_test=image_name in test_cam_names_list, nearest_id=nearest_id, nearest_names=nearest_names)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if('red' in vertices):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.ones_like(positions) * 0.5
    if('nx' in vertices):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def fetchPly_las(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']

    # 直接将顶点颜色设置为白色
    vertices['red'] = 255
    vertices['green'] = 255
    vertices['blue'] = 255

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T ## lcc
    normals = np.vstack([0,0,0]).T ## 通常自采集数据集不会有点云法向量 normals，因此我们根据其 colmap 部分的代码，也将其设置为 0
    # isSeed = np.ones((positions.shape[0], 1), dtype=np.float32) # 创建一个形状为 [N, 1] 的数组
    # seededFrom = -np.ones((positions.shape[0], 1), dtype=np.float32) # 创建一个形状为 [N, 1] 的数组

    return BasicPointCloud(points=positions, colors=colors, normals=normals)
    # return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPt(xyz_path, rgb_path):
    positions_tensor = torch.jit.load(xyz_path).state_dict()['0']

    positions = positions_tensor.numpy()

    colors_tensor = torch.jit.load(rgb_path).state_dict()['0']
    if colors_tensor.size(0) == 0:
        colors_tensor = 255 * (torch.ones_like(positions_tensor) * 0.5)
    colors = (colors_tensor.float().numpy()) / 255.0
    normals = torch.Tensor([]).numpy()

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, masks, depths, eval, train_test_exp, llffhold=True):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])   # 从深度参数字典中提取所有的缩放因子，并转换为 numpy 数组
            if (all_scales > 0).sum():  # 如果存在大于 0 的缩放因子，则计算这些正缩放因子的中位数
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:  # 为每个图像的深度参数添加一个新的键值对，存储中位数缩放因子
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    
    try:
        xyz_path = os.path.join(path, "sparse/0/xyz.pt")
        rgb_path = os.path.join(path, "sparse/0/rgb.pt")
        pcd = fetchPt(xyz_path, rgb_path)
    except:
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        pcd = fetchPly(ply_path)
        # pcd = fetchPly_las(ply_path)  # 读取激光电云数据进行初始化

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            llffhold = 8
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    masks_reading_dir = masks if masks == "" else os.path.join(path, masks)
    # 对相机信息进行排序，排序依据是相机对应的图像名称
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params, 
        images_folder=os.path.join(path, reading_dir), masks_folder=masks_reading_dir,
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # 如果 train_test_exp 为 True 或者相机不是测试相机，则将其作为训练相机信息
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]  # 仅选择那些标记为测试相机的信息
    print(len(test_cam_infos), "test images")
    print(len(train_cam_infos), "train images")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
}