#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from read_write_model import *

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]
    # 获取当前图像中所有3D点的索引
    pts_idx = images_metas[key].point3D_ids
    # 创建一个掩码，用于筛选有效的3D点索引
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)
    # 应用掩码，筛选出有效的3D点索引
    pts_idx = pts_idx[mask]
    # 获取有效3D点在图像上的投影坐标
    valid_xys = image_meta.xys[mask]
    # 如果存在有效的3D点，则获取这些点的坐标
    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        # 若不存在有效3D点，则初始化为零向量
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    # 将3D点从世界坐标系转换到相机坐标系
    pts = np.dot(pts, R.T) + image_meta.tvec
    # 计算相机坐标系下3D点的逆深度
    invcolmapdepth = 1. / pts[..., 2]
    # 计算文件名中需要去除的扩展名长度
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    # 读取对应的单目深度图
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        return None
    # 如果单目深度图不是二维的，则取第一个通道
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]
    # 将单目深度图转换为浮点数类型，并进行归一化
    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height  # 计算单目深度图与相机内参中图像高度的缩放比例
    # 对有效3D点的投影坐标进行缩放
    maps = (valid_xys * s).astype(np.float32)
    valid = (  # 创建一个掩码，用于筛选出在单目深度图范围内且逆深度大于零的点
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    # 如果有效点的数量大于10且逆深度的变化范围大于阈值，则进行缩放参数计算
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :] # 应用掩码，筛选出有效的投影坐标和逆深度
        invcolmapdepth = invcolmapdepth[valid]
        # 使用双线性插值方法从单目深度图中获取对应点的逆深度
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev # 计算相机坐标系下逆深度的中位数
        t_colmap = np.median(invcolmapdepth)
        # 计算相机坐标系下逆深度相对于中位数的平均绝对偏差
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        # 计算缩放因子
        scale = s_colmap / s_mono
        # 计算偏移量
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--depths_dir', required=True)
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()

    # 读取相机内参、图像元数据和3D点数据
    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")
    # 获取所有3D点的索引
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    # 使用并行计算的方式调用 get_scales 函数，计算每个图像的深度缩放参数
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(0)
