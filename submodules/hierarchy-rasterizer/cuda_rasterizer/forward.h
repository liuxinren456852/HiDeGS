/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,  // p高斯点的数量； D SH（球谐函数）的阶数。；M 每个高斯点的最大子节点数量
		const int* indices,  // indices 要处理的高斯点的索引数组
		const int* parent_indices,  // parent_indices 每个高斯点的父节点索引数组
		const float* ts, // ts 插值权重数组，用于混合高斯点的属性。
		const float* means3D, // means3D 三维空间中高斯点的中心位置数组
		const glm::vec3* scales, // 每个高斯点在三个轴上的缩放因子数组，使用glm::vec3类型表示
		const float scale_modifier,  // 缩放因子的修正值，用于调整高斯点的整体缩放
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,  // 布尔数组，指示每个高斯点是否被裁剪
		bool* p_clamped,  // 布尔数组，指示每个父节点高斯点是否被裁剪
		const float* cov3D_precomp,  // 预计算的三维协方差矩阵数组
		const float* colors_precomp,  // 预计算的颜色数组
		const float* viewmatrix,  // 视图矩阵，用于将世界坐标转换为相机坐标
		const float* projmatrix,  // 投影矩阵，用于将相机坐标转换为裁剪坐标
		const glm::vec3* cam_pos,  // 相机在三维空间中的位置
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* means2D,  // 二维屏幕空间中高斯点的中心位置数组
		float* depths,  // 每个高斯点在相机空间中的深度值数组
		float* cov3Ds,
		float* rgb,  // 每个高斯点的颜色数组
		float4* conic_opacity,  // 每个高斯点的圆锥不透明度数组
		const dim3 grid,  // CUDA网格的维度，用于并行计算
		uint32_t* tiles_touched,  // 每个高斯点接触的tile数量数组
		bool prefiltered,  // 是否进行预过滤的标志
		int2* rects,  // 每个高斯点的边界矩形数组
		float3 boxmin,  // 场景的最小边界框
		float3 boxmax,  // 场景的最大边界框
		int skyboxnum,  // 天空盒的高斯点数量
		cudaStream_t stream,  // 用于异步操作
		float biglimit,  // 一个大的限制值，用于某些计算的边界检查
		bool on_cpu);  // 是否在CPU上执行的标志

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,  // grid CUDA网格的维度，用于并行计算;block CUDA块的维度，用于并行计算;
		const uint2* ranges,  // ranges 每个瓦片内高斯点的范围数组，使用uint2类型表示。
		const uint32_t* point_list, // point_list 高斯点的索引列表，指示每个瓦片内包含哪些高斯点
		int W, int H,
		const float focal_x, const float focal_y,
		const float cx, const float cy,
		const float* ts,  // ts 插值权重数组，用于混合高斯点的属性
		const int* kids,  // kids 每个高斯点的子节点数量数组
		const float2* points_xy_image,  // points_xy_image 二维屏幕空间中高斯点的中心位置数组，使用float2类型表示
		const float* features,  // features 每个高斯点的特征数组，例如颜色、不透明度等
		const float* all_map,
		const float4* conic_opacity,  // conic_opacity 每个高斯点的圆锥不透明度数组，使用float4类型表示
		float* final_T,  //final_T 最终的透明度累积数组
		uint32_t* n_contrib,  // 每个像素上贡献的高斯点数量数组
		const float* bg_color,
		float* out_color,  //输出的图像颜色数组
		int P,  // 高斯点的数量
		int skyboxnum,
		cudaStream_t stream,
		float* depths,  // depths 每个高斯点在视图空间中的深度值数组
		float* depth,
		int* out_observe,
		float* out_all_map,
		float* out_plane_depth,
		const bool render_geo);  // depth 输出的深度图像数组
}


#endif