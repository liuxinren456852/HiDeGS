# import torch
# import torch.nn.functional as F
# import cv2
# import numpy as np
# import os
# from typing import Tuple, Optional, Dict
#
# class FrequencyPyramidRegularizer:
#     """频率金字塔正则化器"""
#
#     def __init__(self, num_levels: int = 3, high_freq_thresh: float = 0.15):
#         """
#         初始化频率金字塔正则化器
#
#         Args:
#             num_levels: 金字塔层数
#             high_freq_thresh: 高频区域阈值
#         """
#         self.num_levels = num_levels
#         self.high_freq_thresh = high_freq_thresh
#
#         # 预定义滤波器
#         self.register_filters()
#
#     def register_filters(self):
#         """注册常用的滤波器"""
#         # Sobel算子 - 用于边缘检测
#         self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                                     dtype=torch.float32).view(1, 1, 3, 3)
#         self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                                     dtype=torch.float32).view(1, 1, 3, 3)
#
#         # 拉普拉斯算子 - 用于检测高频
#         self.laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
#                                       dtype=torch.float32).view(1, 1, 3, 3)
#
#         # 高斯核 - 用于低通滤波
#         self.gaussian_kernel = self._create_gaussian_kernel(5, 1.0)
#
#     def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
#         """创建高斯核"""
#         coords = torch.arange(size, dtype=torch.float32) - size // 2
#         grid = torch.meshgrid(coords, coords, indexing='ij')
#         kernel = torch.exp(-(grid[0] ** 2 + grid[1] ** 2) / (2 * sigma ** 2))
#         kernel = kernel / kernel.sum()
#         return kernel.view(1, 1, size, size)
#
#     def build_pyramid(self, image: torch.Tensor) -> list:
#         """构建图像金字塔"""
#         pyramid = [image]
#         current = image
#
#         for i in range(1, self.num_levels):
#             # 下采样
#             current = F.avg_pool2d(current, kernel_size=2, stride=2)
#             pyramid.append(current)
#
#         return pyramid
#
#     def detect_high_frequency_regions(self, image: torch.Tensor) -> torch.Tensor:
#         """检测高频区域"""
#         device = image.device
#
#         # 确保滤波器在正确设备上
#         sobel_x = self.sobel_x.to(device)
#         sobel_y = self.sobel_y.to(device)
#         laplacian = self.laplacian.to(device)
#
#         # 转换为灰度图
#         if image.dim() == 4:  # [B, C, H, W]
#             gray = image.mean(dim=1, keepdim=True)
#         else:  # [C, H, W]
#             gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
#
#         # 计算梯度幅度
#         grad_x = F.conv2d(gray, sobel_x, padding=1)
#         grad_y = F.conv2d(gray, sobel_y, padding=1)
#         gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
#
#         # 计算拉普拉斯响应
#         laplacian_response = torch.abs(F.conv2d(gray, laplacian, padding=1))
#
#         # 结合梯度和拉普拉斯信息
#         high_freq_score = 0.7 * gradient_magnitude + 0.3 * laplacian_response
#
#         # 归一化到[0,1]
#         high_freq_score = (high_freq_score - high_freq_score.min()) / (
#                 high_freq_score.max() - high_freq_score.min() + 1e-8)
#
#         # 生成高频掩码
#         high_freq_mask = (high_freq_score > self.high_freq_thresh).float()
#
#         return high_freq_mask.squeeze()  # 移除batch维度
#
#     def compute_frequency_loss(self, rendered_pyramid: list, gt_pyramid: list) -> torch.Tensor:
#         """计算频率域损失"""
#         total_loss = 0.0
#         weights = [1.0, 0.5, 0.25]  # 不同层的权重
#
#         for level in range(self.num_levels):
#             rendered = rendered_pyramid[level]
#             gt = gt_pyramid[level]
#
#             # 计算该层的频率损失
#             level_loss = self._compute_level_frequency_loss(rendered, gt)
#             total_loss += weights[level] * level_loss
#
#         return total_loss
#
#     def _compute_level_frequency_loss(self, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#         """计算单层频率损失"""
#         device = rendered.device
#         sobel_x = self.sobel_x.to(device)
#         sobel_y = self.sobel_y.to(device)
#
#         # 转为灰度
#         if rendered.dim() == 4:
#             rendered_gray = rendered.mean(dim=1, keepdim=True)
#             gt_gray = gt.mean(dim=1, keepdim=True)
#         else:
#             rendered_gray = rendered.mean(dim=0, keepdim=True).unsqueeze(0)
#             gt_gray = gt.mean(dim=0, keepdim=True).unsqueeze(0)
#
#         # 计算梯度
#         rendered_grad_x = F.conv2d(rendered_gray, sobel_x, padding=1)
#         rendered_grad_y = F.conv2d(rendered_gray, sobel_y, padding=1)
#         gt_grad_x = F.conv2d(gt_gray, sobel_x, padding=1)
#         gt_grad_y = F.conv2d(gt_gray, sobel_y, padding=1)
#
#         # 梯度幅度
#         rendered_grad_mag = torch.sqrt(rendered_grad_x ** 2 + rendered_grad_y ** 2 + 1e-8)
#         gt_grad_mag = torch.sqrt(gt_grad_x ** 2 + gt_grad_y ** 2 + 1e-8)
#
#         # 计算损失
#         loss = F.mse_loss(rendered_grad_mag, gt_grad_mag)
#         return loss
#
#     def compute_scale_regularization(self, gaussians, visibility_filter: torch.Tensor,
#                                      high_freq_mask: torch.Tensor, viewpoint_cam) -> torch.Tensor:
#         """计算高频区域的尺度正则化"""
#         try:
#             # 获取高斯点的缩放
#             if hasattr(gaussians, 'get_scaling'):
#                 scaling = gaussians.get_scaling
#             elif hasattr(gaussians, '_scaling'):
#                 scaling = gaussians._scaling
#             else:
#                 return torch.tensor(0.0, device='cuda')
#
#             # 处理可见性过滤器
#             if visibility_filter.dtype == torch.bool:
#                 visible_scaling = scaling[visibility_filter]
#             else:
#                 visible_indices = visibility_filter.long()
#                 valid_indices = visible_indices[visible_indices < scaling.shape[0]]
#                 if len(valid_indices) == 0:
#                     return torch.tensor(0.0, device=scaling.device)
#                 visible_scaling = scaling[valid_indices]
#
#             if visible_scaling.numel() == 0:
#                 return torch.tensor(0.0, device=scaling.device)
#
#             # 获取高斯点的2D投影位置
#             try:
#                 # 获取可见高斯点的世界坐标
#                 if hasattr(gaussians, 'get_xyz'):
#                     xyz = gaussians.get_xyz
#                 elif hasattr(gaussians, '_xyz'):
#                     xyz = gaussians._xyz
#                 else:
#                     return torch.tensor(0.0, device=scaling.device)
#
#                 if visibility_filter.dtype == torch.bool:
#                     visible_xyz = xyz[visibility_filter]
#                 else:
#                     visible_xyz = xyz[valid_indices]
#
#                 # 投影到屏幕空间
#                 screen_coords = self._project_to_screen(visible_xyz, viewpoint_cam)
#
#                 # 检查哪些高斯点在高频区域
#                 h, w = high_freq_mask.shape
#                 valid_coords = ((screen_coords[:, 0] >= 0) & (screen_coords[:, 0] < w) &
#                                 (screen_coords[:, 1] >= 0) & (screen_coords[:, 1] < h))
#
#                 if not valid_coords.any():
#                     return torch.tensor(0.0, device=scaling.device)
#
#                 # 获取高频区域内的高斯点
#                 valid_screen_coords = screen_coords[valid_coords].long()
#                 in_high_freq = high_freq_mask[valid_screen_coords[:, 1], valid_screen_coords[:, 0]] > 0.5
#
#                 if not in_high_freq.any():
#                     return torch.tensor(0.0, device=scaling.device)
#
#                 # 对高频区域内的高斯点进行尺度正则化
#                 high_freq_scaling = visible_scaling[valid_coords][in_high_freq]
#
#                 # 计算尺度损失
#                 max_scale = high_freq_scaling.max(dim=1).values
#
#                 # 自适应阈值
#                 scale_threshold = 0.005  # 更严格的阈值用于高频区域
#                 oversized_mask = max_scale > scale_threshold
#
#                 if oversized_mask.any():
#                     oversized_values = max_scale[oversized_mask]
#                     loss = ((oversized_values - scale_threshold) ** 2).mean()
#                 else:
#                     loss = torch.tensor(0.0, device=scaling.device)
#
#                 return loss
#
#             except Exception as e:
#                 # 简化版本：只对可见高斯点进行基本的尺度正则化
#                 max_scale = visible_scaling.max(dim=1).values
#                 scale_threshold = 0.01
#                 oversized_mask = max_scale > scale_threshold
#
#                 if oversized_mask.any():
#                     oversized_values = max_scale[oversized_mask]
#                     loss = ((oversized_values - scale_threshold) ** 2).mean()
#                 else:
#                     loss = torch.tensor(0.0, device=scaling.device)
#
#                 return loss
#
#         except Exception as e:
#             print(f"尺度正则化计算失败: {e}")
#             return torch.tensor(0.0, device='cuda')
#
#     def _project_to_screen(self, xyz: torch.Tensor, viewpoint_cam) -> torch.Tensor:
#         """将3D点投影到屏幕空间"""
#         # 转换到相机坐标系
#         xyz_cam = xyz @ viewpoint_cam.world_view_transform[:3, :3].T + viewpoint_cam.world_view_transform[3, :3]
#
#         # 透视投影
#         if hasattr(viewpoint_cam, 'Fx') and hasattr(viewpoint_cam, 'Fy'):
#             # 使用相机内参
#             x_screen = xyz_cam[:, 0] * viewpoint_cam.Fx / (xyz_cam[:, 2] + 1e-8) + viewpoint_cam.Cx
#             y_screen = xyz_cam[:, 1] * viewpoint_cam.Fy / (xyz_cam[:, 2] + 1e-8) + viewpoint_cam.Cy
#         else:
#             # 使用默认投影
#             fov_x = viewpoint_cam.FoVx if hasattr(viewpoint_cam, 'FoVx') else 1.0
#             fov_y = viewpoint_cam.FoVy if hasattr(viewpoint_cam, 'FoVy') else 1.0
#
#             focal_x = viewpoint_cam.image_width / (2.0 * np.tan(fov_x * 0.5))
#             focal_y = viewpoint_cam.image_height / (2.0 * np.tan(fov_y * 0.5))
#
#             x_screen = xyz_cam[:, 0] * focal_x / (xyz_cam[:, 2] + 1e-8) + viewpoint_cam.image_width * 0.5
#             y_screen = xyz_cam[:, 1] * focal_y / (xyz_cam[:, 2] + 1e-8) + viewpoint_cam.image_height * 0.5
#
#         return torch.stack([x_screen, y_screen], dim=1)
#
#     def save_visualizations(self, rendered_image: torch.Tensor, gt_image: torch.Tensor,
#                             high_freq_mask: torch.Tensor, save_dir: str, iteration: int,
#                             camera_name: str = ""):
#         """保存可视化结果 - 四张图像组合为2x2布局"""
#         os.makedirs(save_dir, exist_ok=True)
#
#         # 转换为numpy格式
#         if rendered_image.dim() == 4:
#             rendered_np = rendered_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#             gt_np = gt_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#         else:
#             rendered_np = rendered_image.permute(1, 2, 0).detach().cpu().numpy()
#             gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
#
#         # 确保值在[0,1]范围内
#         rendered_np = np.clip(rendered_np, 0, 1)
#         gt_np = np.clip(gt_np, 0, 1)
#
#         # 转换为BGR格式（OpenCV格式）
#         rendered_bgr = (rendered_np[:, :, [2, 1, 0]] * 255).astype(np.uint8)
#         gt_bgr = (gt_np[:, :, [2, 1, 0]] * 255).astype(np.uint8)
#
#         # 获取图像尺寸
#         h, w = rendered_bgr.shape[:2]
#
#         # 高频掩码处理
#         mask_np = high_freq_mask.detach().cpu().numpy()
#
#         # 如果掩码尺寸与原图不同，进行调整
#         if mask_np.shape != (h, w):
#             mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
#
#         mask_np = (mask_np * 255).astype(np.uint8)
#
#         # 将掩码转换为3通道图像（黑白图）
#         mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
#
#         # 创建高频区域叠加图
#         overlay = gt_bgr.copy()
#         red_overlay = np.zeros_like(overlay)
#         red_overlay[:, :, 2] = 255  # 红色通道
#
#         # 在高频区域添加红色叠加
#         mask_3d = np.stack([mask_np, mask_np, mask_np], axis=2) / 255.0
#         overlay = (overlay * (1 - mask_3d * 0.5) + red_overlay * mask_3d * 0.5).astype(np.uint8)
#
#         # 添加文字标题
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.8
#         font_color = (255, 255, 255)  # 白色
#         thickness = 2
#
#         def add_title(img, title):
#             img_with_title = img.copy()
#             # 计算文字位置
#             text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
#             text_x = (img.shape[1] - text_size[0]) // 2
#             text_y = 30
#
#             # 添加黑色背景
#             cv2.rectangle(img_with_title, (text_x - 5, text_y - 20),
#                           (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
#             # 添加白色文字
#             cv2.putText(img_with_title, title, (text_x, text_y),
#                         font, font_scale, font_color, thickness)
#             return img_with_title
#
#         # 为每张图像添加标题
#         rendered_titled = add_title(rendered_bgr, "Rendered Image")
#         gt_titled = add_title(gt_bgr, "Ground Truth")
#         mask_titled = add_title(mask_bgr, "High Frequency Mask")
#         overlay_titled = add_title(overlay, "High Freq Regions Overlay")
#
#         # 创建2x2布局
#         # 上排：渲染图像 | 真实图像
#         top_row = np.hstack([rendered_titled, gt_titled])
#
#         # 下排：高频掩码 | 高频叠加图
#         bottom_row = np.hstack([mask_titled, overlay_titled])
#
#         # 合并上下两排
#         combined_image = np.vstack([top_row, bottom_row])
#
#         # 保存组合图像
#         filename_prefix = f"{iteration:05d}"
#         if camera_name:
#             filename_prefix += f"_{camera_name}"
#
#         combined_filename = os.path.join(save_dir, f"{filename_prefix}_frequency_analysis.jpg")
#         cv2.imwrite(combined_filename, combined_image)
#
#
# def frequency_regularization_pyramid_scale(
#         rendered_image: torch.Tensor,
#         gt_image: torch.Tensor,
#         gaussians,
#         scene,
#         viewpoint_cam,
#         visibility_filter: torch.Tensor,
#         iteration: int,
#         lambda_freq: float = 0.01,
#         lambda_scale: float = 0.1,
#         num_levels: int = 3,
#         high_freq_thresh: float = 0.15,
#         save_results: bool = False,
#         save_dir: Optional[str] = None,
#         warmup_iterations: int = 1000,
#         debug: bool = True
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#     """
#     主要的频率金字塔正则化函数
#
#     Args:
#         rendered_image: 渲染图像 [C, H, W]
#         gt_image: 真实图像 [C, H, W]
#         gaussians: 高斯对象
#         scene: 场景对象
#         viewpoint_cam: 视角相机
#         visibility_filter: 可见性过滤器
#         iteration: 当前迭代次数
#         lambda_freq: 频率损失权重
#         lambda_scale: 尺度损失权重
#         num_levels: 金字塔层数
#         high_freq_thresh: 高频阈值
#         save_results: 是否保存结果
#         save_dir: 保存目录
#         warmup_iterations: 预热迭代次数
#         debug: 调试模式
#
#     Returns:
#         总的正则化损失
#     """
#
#     # 预热期间跳过正则化
#     if iteration < warmup_iterations:
#         return torch.tensor(0.0, device=rendered_image.device), None
#
#     # 创建正则化器
#     regularizer = FrequencyPyramidRegularizer(num_levels, high_freq_thresh)
#
#     device = rendered_image.device
#     total_loss = torch.tensor(0.0, device=device)
#
#     # 确保输入是4D张量 [B, C, H, W]
#     if rendered_image.dim() == 3:
#         rendered_image = rendered_image.unsqueeze(0)
#     if gt_image.dim() == 3:
#         gt_image = gt_image.unsqueeze(0)
#
#     try:
#         # 1. 构建金字塔
#         rendered_pyramid = regularizer.build_pyramid(rendered_image)
#         gt_pyramid = regularizer.build_pyramid(gt_image)
#
#         # 2. 计算频率损失
#         if lambda_freq > 0:
#             freq_loss = regularizer.compute_frequency_loss(rendered_pyramid, gt_pyramid)
#             total_loss += lambda_freq * freq_loss
#
#             if debug:
#                 print(f"频率损失: {freq_loss:.6f}")
#
#         # 3. 检测高频区域（仅第一层）
#         high_freq_mask = regularizer.detect_high_frequency_regions(gt_pyramid[0])
#
#         # 4. 计算尺度正则化（仅对高频区域）
#         if lambda_scale > 0 and high_freq_mask.sum() > 0:
#             scale_loss = regularizer.compute_scale_regularization(
#                 gaussians, visibility_filter, high_freq_mask, viewpoint_cam)
#             total_loss += lambda_scale * scale_loss
#
#             if debug:
#                 print(f"尺度损失: {scale_loss:.6f}")
#
#         # 5. 保存可视化结果
#         if save_results and save_dir and iteration % 1000 == 0:
#             camera_name = getattr(viewpoint_cam, 'image_name', '').replace('.jpg', '').replace('.png', '')
#             regularizer.save_visualizations(
#                 rendered_image, gt_image, high_freq_mask,
#                 save_dir, iteration, camera_name)
#
#         if debug:
#             print(f"总正则化损失: {total_loss:.6f}")
#             print(f"高频区域像素数: {high_freq_mask.sum().item()}")
#
#         return total_loss, high_freq_mask
#
#     except Exception as e:
#         print(f"频率正则化失败: {e}")
#         return torch.tensor(0.0, device=device), None
#
# import torch
# import torch.nn.functional as F
# import cv2
# import numpy as np
# import os
# from typing import Tuple, Optional, Dict, List
# import math
#
#
# class TrueFrequencyPyramidRegularizer:
#     """真正的频率金字塔正则化器 - 包含FFT计算的稳定版本"""
#
#     def __init__(self,
#                  num_levels: int = 3,
#                  high_freq_thresh: float = 0.2,
#                  freq_bands: int = 4,
#                  use_fft: bool = True):
#         """
#         初始化真正的频率金字塔正则化器
#
#         Args:
#             num_levels: 金字塔层数
#             high_freq_thresh: 高频区域阈值
#             freq_bands: 频率带数量
#             use_fft: 是否使用FFT频率分析
#         """
#         self.num_levels = num_levels
#         self.high_freq_thresh = high_freq_thresh
#         self.freq_bands = freq_bands
#         self.use_fft = use_fft
#
#     def get_sobel_filters(self, device: torch.device):
#         """动态创建Sobel滤波器"""
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
#                                dtype=torch.float32, device=device).view(1, 1, 3, 3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
#                                dtype=torch.float32, device=device).view(1, 1, 3, 3)
#         return sobel_x, sobel_y
#
#     def get_laplacian_filter(self, device: torch.device):
#         """动态创建拉普拉斯滤波器"""
#         return torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
#                             dtype=torch.float32, device=device).view(1, 1, 3, 3)
#
#     def build_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
#         """构建图像金字塔"""
#         pyramid = [image]
#         current = image
#
#         for i in range(1, self.num_levels):
#             current = F.avg_pool2d(current, kernel_size=2, stride=2)
#             pyramid.append(current)
#
#         return pyramid
#
#     def compute_fft_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """计算FFT频率特征 - 稳定版本"""
#         if not self.use_fft:
#             return {}
#
#         device = image.device
#
#         try:
#             # 转换为灰度图
#             if image.dim() == 4:  # [B, C, H, W]
#                 gray = image.mean(dim=1, keepdim=True).squeeze(0).squeeze(0)
#             else:  # [C, H, W]
#                 gray = image.mean(dim=0)
#
#             # 确保尺寸是偶数，FFT计算更稳定
#             h, w = gray.shape
#             if h % 2 != 0:
#                 gray = gray[:-1, :]
#                 h -= 1
#             if w % 2 != 0:
#                 gray = gray[:, :-1]
#                 w -= 1
#
#             # FFT变换
#             fft = torch.fft.fft2(gray)
#             fft_shifted = torch.fft.fftshift(fft)
#
#             # 幅度谱和相位谱
#             magnitude = torch.abs(fft_shifted)
#             phase = torch.angle(fft_shifted)
#
#             # 对数幅度谱 - 更稳定的表示
#             log_magnitude = torch.log(magnitude + 1e-8)
#
#             # 频率分析
#             center_h, center_w = h // 2, w // 2
#
#             # 创建频率掩码
#             y, x = torch.meshgrid(
#                 torch.arange(h, device=device, dtype=torch.float32),
#                 torch.arange(w, device=device, dtype=torch.float32),
#                 indexing='ij'
#             )
#
#             # 计算到中心的距离
#             distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
#             max_distance = min(center_h, center_w) * 0.9  # 稍微保守一点
#
#             # 定义频率带
#             freq_bands = []
#             band_energies = []
#
#             for i in range(self.freq_bands):
#                 inner_radius = i * max_distance / self.freq_bands
#                 outer_radius = (i + 1) * max_distance / self.freq_bands
#
#                 band_mask = (distance >= inner_radius) & (distance < outer_radius)
#                 freq_bands.append(band_mask)
#
#                 # 计算频率带能量 - 使用对数幅度
#                 if band_mask.sum() > 0:
#                     energy = (log_magnitude * band_mask.float()).sum() / (band_mask.sum() + 1e-8)
#                 else:
#                     energy = torch.tensor(0.0, device=device)
#                 band_energies.append(energy)
#
#             return {
#                 'magnitude': magnitude,
#                 'log_magnitude': log_magnitude,
#                 'phase': phase,
#                 'band_energies': torch.stack(band_energies),
#                 'freq_bands': freq_bands,
#                 'valid': True
#             }
#
#         except Exception as e:
#             print(f"FFT特征计算失败: {e}")
#             return {'valid': False}
#
#     def detect_true_high_frequency_regions(self, image: torch.Tensor) -> torch.Tensor:
#         """基于真正FFT的高频区域检测"""
#         device = image.device
#
#         try:
#             # 1. 空间域特征
#             sobel_x, sobel_y = self.get_sobel_filters(device)
#             laplacian = self.get_laplacian_filter(device)
#
#             # 转换为灰度图
#             if image.dim() == 4:
#                 gray = image.mean(dim=1, keepdim=True)
#             else:
#                 gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
#
#             # 梯度特征
#             grad_x = F.conv2d(gray, sobel_x, padding=1)
#             grad_y = F.conv2d(gray, sobel_y, padding=1)
#             gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
#
#             # 拉普拉斯特征
#             laplacian_response = torch.abs(F.conv2d(gray, laplacian, padding=1))
#
#             # 2. 频率域特征
#             fft_features = self.compute_fft_features(image)
#
#             # 基础高频分数
#             spatial_score = 0.6 * gradient_magnitude + 0.4 * laplacian_response
#
#             # 如果FFT计算成功，添加频率域信息
#             if fft_features.get('valid', False):
#                 try:
#                     magnitude = fft_features['magnitude']
#                     h_orig, w_orig = spatial_score.shape[-2:]
#                     h_fft, w_fft = magnitude.shape
#
#                     # 创建高频掩码（频率域）
#                     center_h, center_w = h_fft // 2, w_fft // 2
#                     y, x = torch.meshgrid(
#                         torch.arange(h_fft, device=device),
#                         torch.arange(w_fft, device=device),
#                         indexing='ij'
#                     )
#                     distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
#                     max_distance = min(center_h, center_w)
#
#                     # 高频区域定义（距离中心较远）
#                     high_freq_radius = max_distance * 0.3  # 外30%区域为高频
#                     freq_high_mask = distance > high_freq_radius
#
#                     # 高频能量
#                     high_freq_energy = magnitude * freq_high_mask.float()
#
#                     # 反FFT回到空间域
#                     high_freq_fft = torch.zeros_like(fft_features['magnitude'], dtype=torch.complex64)
#                     high_freq_fft.real = high_freq_energy
#
#                     # 反FFT变换
#                     high_freq_fft_shifted = torch.fft.ifftshift(high_freq_fft)
#                     high_freq_spatial = torch.abs(torch.fft.ifft2(high_freq_fft_shifted))
#
#                     # 调整尺寸匹配
#                     if high_freq_spatial.shape != (h_orig, w_orig):
#                         high_freq_spatial = F.interpolate(
#                             high_freq_spatial.unsqueeze(0).unsqueeze(0),
#                             size=(h_orig, w_orig),
#                             mode='bilinear',
#                             align_corners=False
#                         ).squeeze()
#
#                     # 归一化频率域特征
#                     if high_freq_spatial.max() > 1e-8:
#                         high_freq_spatial = high_freq_spatial / high_freq_spatial.max()
#
#                     # 结合空间域和频率域特征
#                     combined_score = 0.7 * spatial_score.squeeze() + 0.3 * high_freq_spatial
#
#                 except Exception as freq_e:
#                     print(f"频率域特征结合失败，仅使用空间域: {freq_e}")
#                     combined_score = spatial_score.squeeze()
#             else:
#                 combined_score = spatial_score.squeeze()
#
#             # 数值稳定化
#             combined_score = torch.clamp(combined_score, 0, 5.0)
#
#             # 安全归一化
#             score_min = combined_score.min()
#             score_max = combined_score.max()
#             score_range = score_max - score_min
#
#             if score_range > 1e-6:
#                 combined_score = (combined_score - score_min) / score_range
#             else:
#                 combined_score = torch.zeros_like(combined_score)
#
#             # 生成高频掩码
#             high_freq_mask = (combined_score > self.high_freq_thresh).float()
#
#             return high_freq_mask
#
#         except Exception as e:
#             print(f"高频检测失败，使用简化版本: {e}")
#             # 降级到简化版本
#             return self._fallback_high_freq_detection(image)
#
#     def _fallback_high_freq_detection(self, image: torch.Tensor) -> torch.Tensor:
#         """降级的高频检测方法"""
#         device = image.device
#         sobel_x, sobel_y = self.get_sobel_filters(device)
#
#         if image.dim() == 4:
#             gray = image.mean(dim=1, keepdim=True)
#         else:
#             gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
#
#         grad_x = F.conv2d(gray, sobel_x, padding=1)
#         grad_y = F.conv2d(gray, sobel_y, padding=1)
#         gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
#
#         # 简单归一化
#         if gradient_magnitude.max() > 1e-8:
#             gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
#
#         return (gradient_magnitude.squeeze() > self.high_freq_thresh).float()
#
#     def compute_true_frequency_loss(self, rendered_pyramid: List[torch.Tensor],
#                                     gt_pyramid: List[torch.Tensor]) -> torch.Tensor:
#         """计算真正的频率域损失"""
#         device = rendered_pyramid[0].device
#         total_loss = torch.tensor(0.0, device=device)
#
#         # 金字塔层权重
#         pyramid_weights = [0.1, 0.05, 0.025][:len(rendered_pyramid)]
#
#         try:
#             for level in range(len(rendered_pyramid)):
#                 rendered = rendered_pyramid[level]
#                 gt = gt_pyramid[level]
#
#                 # 1. 空间域频率损失
#                 spatial_loss = self._compute_spatial_frequency_loss(rendered, gt)
#
#                 # 2. FFT频率域损失
#                 fft_loss = self._compute_fft_frequency_loss(rendered, gt)
#
#                 # 组合损失
#                 level_loss = 0.7 * spatial_loss + 0.3 * fft_loss
#
#                 # 数值保护
#                 level_loss = torch.clamp(level_loss, 0, 0.1)
#
#                 total_loss += pyramid_weights[level] * level_loss
#
#             return torch.clamp(total_loss, 0, 0.1)
#
#         except Exception as e:
#             print(f"频率损失计算失败: {e}")
#             return torch.tensor(0.0, device=device)
#
#     def _compute_spatial_frequency_loss(self, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#         """计算空间域频率损失"""
#         device = rendered.device
#
#         try:
#             sobel_x, sobel_y = self.get_sobel_filters(device)
#             laplacian = self.get_laplacian_filter(device)
#
#             # 转为灰度
#             if rendered.dim() == 4:
#                 rendered_gray = rendered.mean(dim=1, keepdim=True)
#                 gt_gray = gt.mean(dim=1, keepdim=True)
#             else:
#                 rendered_gray = rendered.mean(dim=0, keepdim=True).unsqueeze(0)
#                 gt_gray = gt.mean(dim=0, keepdim=True).unsqueeze(0)
#
#             # 梯度损失
#             rendered_grad_x = F.conv2d(rendered_gray, sobel_x, padding=1)
#             rendered_grad_y = F.conv2d(rendered_gray, sobel_y, padding=1)
#             gt_grad_x = F.conv2d(gt_gray, sobel_x, padding=1)
#             gt_grad_y = F.conv2d(gt_gray, sobel_y, padding=1)
#
#             grad_loss = F.mse_loss(rendered_grad_x, gt_grad_x) + F.mse_loss(rendered_grad_y, gt_grad_y)
#
#             # 拉普拉斯损失
#             rendered_laplacian = F.conv2d(rendered_gray, laplacian, padding=1)
#             gt_laplacian = F.conv2d(gt_gray, laplacian, padding=1)
#             laplacian_loss = F.mse_loss(rendered_laplacian, gt_laplacian)
#
#             return torch.clamp(0.7 * grad_loss + 0.3 * laplacian_loss, 0, 1.0)
#
#         except Exception as e:
#             print(f"空间频率损失计算失败: {e}")
#             return torch.tensor(0.0, device=device)
#
#     def _compute_fft_frequency_loss(self, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#         """计算FFT频率域损失"""
#         if not self.use_fft:
#             return torch.tensor(0.0, device=rendered.device)
#
#         try:
#             # 计算FFT特征
#             rendered_fft = self.compute_fft_features(rendered)
#             gt_fft = self.compute_fft_features(gt)
#
#             if not (rendered_fft.get('valid', False) and gt_fft.get('valid', False)):
#                 return torch.tensor(0.0, device=rendered.device)
#
#             total_fft_loss = torch.tensor(0.0, device=rendered.device)
#
#             # 1. 对数幅度谱损失 - 更稳定
#             if 'log_magnitude' in rendered_fft and 'log_magnitude' in gt_fft:
#                 mag_loss = F.mse_loss(rendered_fft['log_magnitude'], gt_fft['log_magnitude'])
#                 mag_loss = torch.clamp(mag_loss, 0, 10.0)  # 对数域的合理范围
#                 total_fft_loss += 0.6 * mag_loss
#
#             # 2. 相位损失 - 处理周期性
#             if 'phase' in rendered_fft and 'phase' in gt_fft:
#                 phase_diff = torch.abs(rendered_fft['phase'] - gt_fft['phase'])
#                 # 处理相位的2π周期性
#                 phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)
#                 phase_loss = phase_diff.mean()
#                 phase_loss = torch.clamp(phase_loss, 0, math.pi)
#                 total_fft_loss += 0.2 * phase_loss
#
#             # 3. 频率带能量损失
#             if 'band_energies' in rendered_fft and 'band_energies' in gt_fft:
#                 band_loss = F.mse_loss(rendered_fft['band_energies'], gt_fft['band_energies'])
#                 band_loss = torch.clamp(band_loss, 0, 100.0)  # 能量域的合理范围
#                 total_fft_loss += 0.2 * band_loss
#
#             return torch.clamp(total_fft_loss, 0, 10.0)
#
#         except Exception as e:
#             print(f"FFT频率损失计算失败: {e}")
#             return torch.tensor(0.0, device=rendered.device)
#
#     def compute_scale_regularization(self, gaussians, visibility_filter: torch.Tensor,
#                                      high_freq_mask: torch.Tensor, viewpoint_cam) -> torch.Tensor:
#         """计算尺度正则化"""
#         try:
#             if hasattr(gaussians, 'get_scaling'):
#                 scaling = gaussians.get_scaling
#             elif hasattr(gaussians, '_scaling'):
#                 scaling = gaussians._scaling
#             else:
#                 return torch.tensor(0.0, device=high_freq_mask.device)
#
#             # 处理可见性过滤
#             if visibility_filter.dtype == torch.bool:
#                 if visibility_filter.sum() == 0:
#                     return torch.tensor(0.0, device=scaling.device)
#                 visible_scaling = scaling[visibility_filter]
#             else:
#                 visible_indices = visibility_filter.long()
#                 valid_mask = (visible_indices >= 0) & (visible_indices < scaling.shape[0])
#                 if not valid_mask.any():
#                     return torch.tensor(0.0, device=scaling.device)
#                 valid_indices = visible_indices[valid_mask]
#                 visible_scaling = scaling[valid_indices]
#
#             if visible_scaling.numel() == 0:
#                 return torch.tensor(0.0, device=scaling.device)
#
#             # 尺度正则化
#             max_scale = visible_scaling.max(dim=1).values
#             scale_threshold = 0.01
#
#             oversized_mask = max_scale > scale_threshold
#             if oversized_mask.any():
#                 oversized_values = max_scale[oversized_mask]
#                 loss = ((oversized_values - scale_threshold) ** 2).mean()
#                 return torch.clamp(loss, 0, 0.01)
#             else:
#                 return torch.tensor(0.0, device=scaling.device)
#
#         except Exception as e:
#             print(f"尺度正则化失败: {e}")
#             return torch.tensor(0.0, device=high_freq_mask.device)
#
#     def save_visualizations(self, rendered_image: torch.Tensor, gt_image: torch.Tensor,
#                             high_freq_mask: torch.Tensor, fft_features: Dict,
#                             save_dir: str, iteration: int, camera_name: str = ""):
#         """保存包含FFT特征的可视化结果"""
#         try:
#             os.makedirs(save_dir, exist_ok=True)
#
#             # 基本图像处理
#             if rendered_image.dim() == 4:
#                 rendered_np = rendered_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#                 gt_np = gt_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#             else:
#                 rendered_np = rendered_image.permute(1, 2, 0).detach().cpu().numpy()
#                 gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
#
#             rendered_np = np.clip(rendered_np, 0, 1)
#             gt_np = np.clip(gt_np, 0, 1)
#
#             rendered_bgr = (rendered_np[:, :, [2, 1, 0]] * 255).astype(np.uint8)
#             gt_bgr = (gt_np[:, :, [2, 1, 0]] * 255).astype(np.uint8)
#
#             # 高频掩码
#             h, w = rendered_bgr.shape[:2]
#             mask_np = high_freq_mask.detach().cpu().numpy()
#             if mask_np.shape != (h, w):
#                 mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
#             mask_np = (mask_np * 255).astype(np.uint8)
#             mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
#
#             # FFT幅度谱可视化
#             fft_vis = None
#             if fft_features.get('valid', False) and 'log_magnitude' in fft_features:
#                 magnitude = fft_features['log_magnitude'].detach().cpu().numpy()
#                 # 归一化到0-255
#                 magnitude_norm = ((magnitude - magnitude.min()) /
#                                   (magnitude.max() - magnitude.min() + 1e-8) * 255).astype(np.uint8)
#                 fft_vis = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)
#                 fft_vis = cv2.resize(fft_vis, (w, h))
#             else:
#                 fft_vis = np.zeros_like(mask_bgr)
#
#             # 创建2x2布局
#             top_row = np.hstack([rendered_bgr, gt_bgr])
#             bottom_row = np.hstack([mask_bgr, fft_vis])
#             combined_image = np.vstack([top_row, bottom_row])
#
#             # 保存
#             filename = f"{iteration:05d}_{camera_name}_freq_fft_analysis.jpg"
#             cv2.imwrite(os.path.join(save_dir, filename), combined_image)
#
#         except Exception as e:
#             print(f"可视化保存失败: {e}")
#
#
# def true_frequency_regularization_pyramid_scale(
#         rendered_image: torch.Tensor,
#         gt_image: torch.Tensor,
#         gaussians,
#         scene,
#         viewpoint_cam,
#         visibility_filter: torch.Tensor,
#         iteration: int,
#         lambda_freq: float = 0.001,  # 适中的权重
#         lambda_scale: float = 0.005,  # 适中的权重
#         num_levels: int = 3,
#         high_freq_thresh: float = 0.2,
#         save_results: bool = False,
#         save_dir: Optional[str] = None,
#         warmup_iterations: int = 1000,
#         debug: bool = False
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
#     """
#     真正的频率金字塔正则化函数 - 包含FFT计算
#
#     Args:
#         rendered_image: 渲染图像 [C, H, W]
#         gt_image: 真实图像 [C, H, W]
#         gaussians: 高斯对象
#         scene: 场景对象
#         viewpoint_cam: 视角相机
#         visibility_filter: 可见性过滤器
#         iteration: 当前迭代次数
#         lambda_freq: 频率损失权重
#         lambda_scale: 尺度损失权重
#         num_levels: 金字塔层数
#         high_freq_thresh: 高频阈值
#         save_results: 是否保存结果
#         save_dir: 保存目录
#         warmup_iterations: 预热迭代次数
#         debug: 调试模式
#
#     Returns:
#         总的正则化损失, 高频掩码, 调试信息字典
#     """
#
#     # 预热期间跳过
#     if iteration < warmup_iterations:
#         return torch.tensor(0.0, device=rendered_image.device), None, {'warmup': True}
#
#     device = rendered_image.device
#     debug_info = {}
#
#     try:
#         # 创建真正的频率正则化器
#         regularizer = TrueFrequencyPyramidRegularizer(
#             num_levels=num_levels,
#             high_freq_thresh=high_freq_thresh,
#             use_fft=True
#         )
#
#         # 确保输入格式
#         if rendered_image.dim() == 3:
#             rendered_image = rendered_image.unsqueeze(0)
#         if gt_image.dim() == 3:
#             gt_image = gt_image.unsqueeze(0)
#
#         total_loss = torch.tensor(0.0, device=device)
#
#         # 1. 构建金字塔
#         rendered_pyramid = regularizer.build_pyramid(rendered_image)
#         gt_pyramid = regularizer.build_pyramid(gt_image)
#
#         debug_info['pyramid_levels'] = len(rendered_pyramid)
#
#         # 2. 计算真正的频率损失（包含FFT）
#         if lambda_freq > 0:
#             freq_loss = regularizer.compute_true_frequency_loss(rendered_pyramid, gt_pyramid)
#             total_loss += lambda_freq * freq_loss
#             debug_info['freq_loss'] = freq_loss.item()
#
#             if debug:
#                 print(f"真正频率损失: {freq_loss:.6f}")
#
#         # 3. 基于FFT的高频区域检测
#         high_freq_mask = regularizer.detect_true_high_frequency_regions(gt_pyramid[0])
#         debug_info['high_freq_pixels'] = high_freq_mask.sum().item()
#         debug_info['high_freq_ratio'] = (high_freq_mask.sum() / high_freq_mask.numel()).item()
#
#         # 4. 尺度正则化
#         if lambda_scale > 0 and high_freq_mask.sum() > 0:
#             scale_loss = regularizer.compute_scale_regularization(
#                 gaussians, visibility_filter, high_freq_mask, viewpoint_cam)
#             total_loss += lambda_scale * scale_loss
#             debug_info['scale_loss'] = scale_loss.item()
#
#             if debug:
#                 print(f"尺度损失: {scale_loss:.6f}")
#
#         # 5. FFT特征分析
#         fft_features = regularizer.compute_fft_features(gt_pyramid[0])
#         debug_info['fft_valid'] = fft_features.get('valid', False)
#
#         if fft_features.get('valid', False):
#             debug_info['freq_band_energies'] = fft_features['band_energies'].detach().cpu().numpy().tolist()
#
#         # 6. 保存可视化
#         if save_results and save_dir and iteration % 1000 == 0:
#             camera_name = getattr(viewpoint_cam, 'image_name', '').replace('.jpg', '').replace('.png', '')
#             regularizer.save_visualizations(
#                 rendered_image, gt_image, high_freq_mask, fft_features,
#                 save_dir, iteration, camera_name)
#
#         # 最终保护
#         total_loss = torch.clamp(total_loss, 0, 1.0)
#         debug_info['total_loss'] = total_loss.item()
#
#         if debug:
#             print(f"总频率正则化损失: {total_loss:.6f}")
#             print(f"FFT计算状态: {'成功' if debug_info['fft_valid'] else '失败'}")
#
#         return total_loss, high_freq_mask, debug_info
#
#     except Exception as e:
#         print(f"频率正则化失败: {e}")
#         debug_info['error'] = str(e)
#         return torch.tensor(0.0, device=device), None, debug_info


import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, List
import math


class TrueFrequencyPyramidRegularizer:
    """真正的频率金字塔正则化器 - 修复shape匹配问题"""

    def __init__(self,
                 num_levels: int = 3,
                 high_freq_thresh: float = 0.2,
                 freq_bands: int = 4,
                 use_fft: bool = True):
        self.num_levels = num_levels
        self.high_freq_thresh = high_freq_thresh
        self.freq_bands = freq_bands
        self.use_fft = use_fft

    def get_sobel_filters(self, device: torch.device):
        """动态创建Sobel滤波器"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        return sobel_x, sobel_y

    def get_laplacian_filter(self, device: torch.device):
        """动态创建拉普拉斯滤波器"""
        return torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                            dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def build_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        """构建图像金字塔"""
        pyramid = [image]
        current = image

        for i in range(1, self.num_levels):
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            pyramid.append(current)

        return pyramid

    def compute_fft_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算FFT频率特征 - 修复版本"""
        if not self.use_fft:
            return {'valid': False}

        device = image.device

        try:
            # 正确处理输入维度
            if image.dim() == 4:  # [B, C, H, W]
                gray = image.mean(dim=1).squeeze(0)  # 先平均通道，再去掉batch维度
            elif image.dim() == 3:  # [C, H, W]
                gray = image.mean(dim=0)  # 平均通道维度
            else:  # [H, W]
                gray = image

            # 确保是2D tensor
            if gray.dim() != 2:
                raise ValueError(f"处理后的gray图像维度不正确: {gray.shape}")

            h_orig, w_orig = gray.shape

            # 数值范围检查和预处理
            gray = torch.clamp(gray, 0, 1)

            # FFT变换
            fft = torch.fft.fft2(gray)
            fft_shifted = torch.fft.fftshift(fft)

            # 幅度谱和相位谱
            magnitude = torch.abs(fft_shifted)
            phase = torch.angle(fft_shifted)

            # 改进的对数幅度谱处理
            log_magnitude = torch.log(magnitude + 1e-6)

            # 频率分析
            center_h, center_w = h_orig // 2, w_orig // 2

            # 创建频率掩码
            y, x = torch.meshgrid(
                torch.arange(h_orig, device=device, dtype=torch.float32),
                torch.arange(w_orig, device=device, dtype=torch.float32),
                indexing='ij'
            )

            # 计算到中心的距离
            distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
            max_distance = min(center_h, center_w)

            # 定义频率带
            freq_bands = []
            band_energies = []

            for i in range(self.freq_bands):
                inner_radius = i * max_distance / self.freq_bands
                outer_radius = (i + 1) * max_distance / self.freq_bands

                band_mask = (distance >= inner_radius) & (distance < outer_radius)
                freq_bands.append(band_mask)

                # 计算频率带能量
                if band_mask.sum() > 0:
                    energy = (magnitude * band_mask.float()).sum() / (band_mask.sum() + 1e-8)
                else:
                    energy = torch.tensor(0.0, device=device)
                band_energies.append(energy)

            return {
                'magnitude': magnitude,
                'log_magnitude': log_magnitude,
                'phase': phase,
                'band_energies': torch.stack(band_energies),
                'freq_bands': freq_bands,
                'original_shape': (h_orig, w_orig),
                'valid': True
            }

        except Exception as e:
            print(f"FFT特征计算失败: {e}")
            return {'valid': False, 'error': str(e)}

    def detect_true_high_frequency_regions(self, image: torch.Tensor) -> torch.Tensor:
        """基于真正FFT的高频区域检测 - 修复shape问题"""
        device = image.device

        try:
            # 1. 空间域特征
            sobel_x, sobel_y = self.get_sobel_filters(device)
            laplacian = self.get_laplacian_filter(device)

            # 正确转换为灰度图并保持正确维度
            if image.dim() == 4:  # [B, C, H, W]
                gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                gray_2d = image.mean(dim=1).squeeze(0)  # [H, W] for FFT
            elif image.dim() == 3:  # [C, H, W]
                gray = image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
                gray_2d = image.mean(dim=0)  # [H, W] for FFT
            else:
                raise ValueError(f"不支持的图像维度: {image.shape}")

            # 梯度特征
            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

            # 拉普拉斯特征
            laplacian_response = torch.abs(F.conv2d(gray, laplacian, padding=1))

            # 基础高频分数 - 转换为2D
            spatial_score = 0.6 * gradient_magnitude.squeeze() + 0.4 * laplacian_response.squeeze()

            # 2. 频率域特征
            fft_features = self.compute_fft_features(gray_2d)  # 使用2D灰度图

            # 如果FFT计算成功，添加频率域信息
            if fft_features.get('valid', False):
                try:
                    magnitude = fft_features['magnitude']
                    h_spatial, w_spatial = spatial_score.shape
                    h_fft, w_fft = magnitude.shape

                    # 创建高频掩码（频率域）
                    center_h, center_w = h_fft // 2, w_fft // 2
                    y, x = torch.meshgrid(
                        torch.arange(h_fft, device=device),
                        torch.arange(w_fft, device=device),
                        indexing='ij'
                    )
                    distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
                    max_distance = min(center_h, center_w)

                    # 高频区域定义（距离中心较远）
                    high_freq_radius = max_distance * 0.3
                    freq_high_mask = distance > high_freq_radius

                    # 创建高频FFT
                    fft_shifted = torch.fft.fftshift(torch.fft.fft2(gray_2d))
                    high_freq_fft = torch.zeros_like(fft_shifted)
                    high_freq_fft[freq_high_mask] = fft_shifted[freq_high_mask]

                    # 反FFT变换
                    high_freq_fft_ishifted = torch.fft.ifftshift(high_freq_fft)
                    high_freq_spatial = torch.abs(torch.fft.ifft2(high_freq_fft_ishifted))

                    # 调整尺寸匹配
                    if high_freq_spatial.shape != (h_spatial, w_spatial):
                        high_freq_spatial = F.interpolate(
                            high_freq_spatial.unsqueeze(0).unsqueeze(0),
                            size=(h_spatial, w_spatial),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()

                    # 归一化频率域特征
                    if high_freq_spatial.max() > 1e-8:
                        high_freq_spatial = high_freq_spatial / high_freq_spatial.max()

                    # 结合空间域和频率域特征
                    combined_score = 0.7 * spatial_score + 0.3 * high_freq_spatial

                except Exception as freq_e:
                    print(f"频率域特征结合失败，仅使用空间域: {freq_e}")
                    combined_score = spatial_score
            else:
                combined_score = spatial_score

            # 数值稳定化
            combined_score = torch.clamp(combined_score, 0, 5.0)

            # 安全归一化
            score_min = combined_score.min()
            score_max = combined_score.max()
            score_range = score_max - score_min

            if score_range > 1e-6:
                combined_score = (combined_score - score_min) / score_range
            else:
                combined_score = torch.zeros_like(combined_score)

            # 生成高频掩码
            high_freq_mask = (combined_score > self.high_freq_thresh).float()

            return high_freq_mask

        except Exception as e:
            print(f"高频检测失败，使用简化版本: {e}")
            return self._fallback_high_freq_detection(image)

    def _fallback_high_freq_detection(self, image: torch.Tensor) -> torch.Tensor:
        """降级的高频检测方法"""
        device = image.device
        sobel_x, sobel_y = self.get_sobel_filters(device)

        if image.dim() == 4:
            gray = image.mean(dim=1, keepdim=True)
        else:
            gray = image.mean(dim=0, keepdim=True).unsqueeze(0)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 简单归一化
        if gradient_magnitude.max() > 1e-8:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()

        return (gradient_magnitude.squeeze() > self.high_freq_thresh).float()

    def compute_true_frequency_loss(self, rendered_pyramid: List[torch.Tensor],
                                    gt_pyramid: List[torch.Tensor]) -> torch.Tensor:
        """计算真正的频率域损失"""
        device = rendered_pyramid[0].device
        total_loss = torch.tensor(0.0, device=device)

        # 金字塔层权重
        pyramid_weights = [0.1, 0.05, 0.025][:len(rendered_pyramid)]

        try:
            for level in range(len(rendered_pyramid)):
                rendered = rendered_pyramid[level]
                gt = gt_pyramid[level]

                # 1. 空间域频率损失
                spatial_loss = self._compute_spatial_frequency_loss(rendered, gt)

                # 2. FFT频率域损失
                fft_loss = self._compute_fft_frequency_loss(rendered, gt)

                # 组合损失
                level_loss = 0.7 * spatial_loss + 0.3 * fft_loss

                # 数值保护
                level_loss = torch.clamp(level_loss, 0, 0.1)

                total_loss += pyramid_weights[level] * level_loss

            return torch.clamp(total_loss, 0, 0.1)

        except Exception as e:
            print(f"频率损失计算失败: {e}")
            return torch.tensor(0.0, device=device)

    def _compute_spatial_frequency_loss(self, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """计算空间域频率损失"""
        device = rendered.device

        try:
            sobel_x, sobel_y = self.get_sobel_filters(device)
            laplacian = self.get_laplacian_filter(device)

            # 转为灰度
            if rendered.dim() == 4:
                rendered_gray = rendered.mean(dim=1, keepdim=True)
                gt_gray = gt.mean(dim=1, keepdim=True)
            else:
                rendered_gray = rendered.mean(dim=0, keepdim=True).unsqueeze(0)
                gt_gray = gt.mean(dim=0, keepdim=True).unsqueeze(0)

            # 梯度损失
            rendered_grad_x = F.conv2d(rendered_gray, sobel_x, padding=1)
            rendered_grad_y = F.conv2d(rendered_gray, sobel_y, padding=1)
            gt_grad_x = F.conv2d(gt_gray, sobel_x, padding=1)
            gt_grad_y = F.conv2d(gt_gray, sobel_y, padding=1)

            grad_loss = F.mse_loss(rendered_grad_x, gt_grad_x) + F.mse_loss(rendered_grad_y, gt_grad_y)

            # 拉普拉斯损失
            rendered_laplacian = F.conv2d(rendered_gray, laplacian, padding=1)
            gt_laplacian = F.conv2d(gt_gray, laplacian, padding=1)
            laplacian_loss = F.mse_loss(rendered_laplacian, gt_laplacian)

            return torch.clamp(0.7 * grad_loss + 0.3 * laplacian_loss, 0, 1.0)

        except Exception as e:
            print(f"空间频率损失计算失败: {e}")
            return torch.tensor(0.0, device=device)

    def _compute_fft_frequency_loss(self, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """计算FFT频率域损失"""
        if not self.use_fft:
            return torch.tensor(0.0, device=rendered.device)

        try:
            # 计算FFT特征
            rendered_fft = self.compute_fft_features(rendered)
            gt_fft = self.compute_fft_features(gt)

            if not (rendered_fft.get('valid', False) and gt_fft.get('valid', False)):
                return torch.tensor(0.0, device=rendered.device)

            total_fft_loss = torch.tensor(0.0, device=rendered.device)

            # 1. 对数幅度谱损失
            if 'log_magnitude' in rendered_fft and 'log_magnitude' in gt_fft:
                mag_loss = F.mse_loss(rendered_fft['log_magnitude'], gt_fft['log_magnitude'])
                mag_loss = torch.clamp(mag_loss, 0, 10.0)
                total_fft_loss += 0.6 * mag_loss

            # 2. 相位损失
            if 'phase' in rendered_fft and 'phase' in gt_fft:
                phase_diff = torch.abs(rendered_fft['phase'] - gt_fft['phase'])
                phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)
                phase_loss = phase_diff.mean()
                phase_loss = torch.clamp(phase_loss, 0, math.pi)
                total_fft_loss += 0.2 * phase_loss

            # 3. 频率带能量损失
            if 'band_energies' in rendered_fft and 'band_energies' in gt_fft:
                band_loss = F.mse_loss(rendered_fft['band_energies'], gt_fft['band_energies'])
                band_loss = torch.clamp(band_loss, 0, 100.0)
                total_fft_loss += 0.2 * band_loss

            return torch.clamp(total_fft_loss, 0, 10.0)

        except Exception as e:
            print(f"FFT频率损失计算失败: {e}")
            return torch.tensor(0.0, device=rendered.device)

    def compute_scale_regularization(self, gaussians, visibility_filter: torch.Tensor,
                                     high_freq_mask: torch.Tensor, viewpoint_cam) -> torch.Tensor:
        """计算尺度正则化"""
        try:
            if hasattr(gaussians, 'get_scaling'):
                scaling = gaussians.get_scaling
            elif hasattr(gaussians, '_scaling'):
                scaling = gaussians._scaling
            else:
                return torch.tensor(0.0, device=high_freq_mask.device)

            # 处理可见性过滤
            if visibility_filter.dtype == torch.bool:
                if visibility_filter.sum() == 0:
                    return torch.tensor(0.0, device=scaling.device)
                visible_scaling = scaling[visibility_filter]
            else:
                visible_indices = visibility_filter.long()
                valid_mask = (visible_indices >= 0) & (visible_indices < scaling.shape[0])
                if not valid_mask.any():
                    return torch.tensor(0.0, device=scaling.device)
                valid_indices = visible_indices[valid_mask]
                visible_scaling = scaling[valid_indices]

            if visible_scaling.numel() == 0:
                return torch.tensor(0.0, device=scaling.device)

            # 尺度正则化
            max_scale = visible_scaling.max(dim=1).values
            scale_threshold = 0.01

            oversized_mask = max_scale > scale_threshold
            if oversized_mask.any():
                oversized_values = max_scale[oversized_mask]
                loss = ((oversized_values - scale_threshold) ** 2).mean()
                return torch.clamp(loss, 0, 0.01)
            else:
                return torch.tensor(0.0, device=scaling.device)

        except Exception as e:
            print(f"尺度正则化失败: {e}")
            return torch.tensor(0.0, device=high_freq_mask.device)

    def create_better_fft_visualization(self, fft_features: Dict, target_size: Tuple[int, int]) -> np.ndarray:
        """创建更好的FFT可视化，减少人工伪影"""
        h, w = target_size

        if not fft_features.get('valid', False):
            no_data_img = np.full((h, w, 3), 128, dtype=np.uint8)
            cv2.putText(no_data_img, 'FFT Failed', (w // 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return no_data_img

        try:
            magnitude = fft_features['magnitude'].detach().cpu().numpy()

            # 使用更自然的可视化策略
            # 1. 应用窗口函数减少频谱泄漏
            h_fft, w_fft = magnitude.shape
            window_h = np.hanning(h_fft).reshape(-1, 1)
            window_w = np.hanning(w_fft).reshape(1, -1)
            window_2d = window_h @ window_w
            windowed_magnitude = magnitude * window_2d

            # 2. 对数变换，但使用更温和的变换
            log_magnitude = np.log(windowed_magnitude + 1e-8)

            # 3. 使用更鲁棒的归一化 - 去除极值影响
            p1, p99 = np.percentile(log_magnitude, [1, 99])
            log_magnitude_clipped = np.clip(log_magnitude, p1, p99)

            # 4. 平滑归一化到[0, 1]
            range_val = p99 - p1 + 1e-8
            normalized = (log_magnitude_clipped - p1) / range_val

            # 5. 应用轻微的高斯平滑以减少噪声
            from scipy import ndimage
            normalized_smooth = ndimage.gaussian_filter(normalized, sigma=0.5)

            # 6. 转换到[0, 255]
            magnitude_uint8 = (normalized_smooth * 255).astype(np.uint8)

            # 7. 调整尺寸
            if magnitude_uint8.shape != (h, w):
                magnitude_uint8 = cv2.resize(magnitude_uint8, (w, h), interpolation=cv2.INTER_LINEAR)

            # 8. 使用更自然的颜色映射
            fft_vis = cv2.applyColorMap(magnitude_uint8, cv2.COLORMAP_VIRIDIS)

            return fft_vis

        except Exception as e:
            print(f"FFT可视化创建失败: {e}")
            error_img = np.full((h, w, 3), 64, dtype=np.uint8)
            cv2.putText(error_img, 'Vis Error', (w // 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_img

    def save_visualizations(self, rendered_image: torch.Tensor, gt_image: torch.Tensor,
                            high_freq_mask: torch.Tensor, fft_features: Dict,
                            save_dir: str, iteration: int, camera_name: str = ""):
        """保存可视化结果"""
        try:
            os.makedirs(save_dir, exist_ok=True)

            # 基本图像处理
            if rendered_image.dim() == 4:
                rendered_np = rendered_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                gt_np = gt_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            else:
                rendered_np = rendered_image.permute(1, 2, 0).detach().cpu().numpy()
                gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()

            rendered_np = np.clip(rendered_np, 0, 1)
            gt_np = np.clip(gt_np, 0, 1)

            rendered_bgr = (rendered_np[:, :, [2, 1, 0]] * 255).astype(np.uint8)
            gt_bgr = (gt_np[:, :, [2, 1, 0]] * 255).astype(np.uint8)

            # 高频掩码处理
            h, w = rendered_bgr.shape[:2]
            mask_np = high_freq_mask.detach().cpu().numpy()
            if mask_np.shape != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_np = (mask_np * 255).astype(np.uint8)
            mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)

            # 使用改进的FFT可视化
            fft_vis = self.create_better_fft_visualization(fft_features, (h, w))

            # 添加标题函数
            def add_title(img, title, font_scale=0.8):
                img_with_title = img.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_color = (255, 255, 255)
                thickness = 2

                text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = 25

                cv2.rectangle(img_with_title, (text_x - 5, text_y - 20),
                              (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(img_with_title, title, (text_x, text_y),
                            font, font_scale, font_color, thickness)
                return img_with_title

            # 为每张图像添加标题
            rendered_titled = add_title(rendered_bgr, "Rendered Image")
            gt_titled = add_title(gt_bgr, "Ground Truth")
            mask_titled = add_title(mask_bgr, "High Freq Mask")

            fft_title = "FFT Spectrum"
            if not fft_features.get('valid', False):
                fft_title += " (Failed)"
            fft_titled = add_title(fft_vis, fft_title)

            # 创建2x2布局
            top_row = np.hstack([rendered_titled, gt_titled])
            bottom_row = np.hstack([mask_titled, fft_titled])
            combined_image = np.vstack([top_row, bottom_row])

            # 保存组合图像
            filename = f"{iteration:05d}_{camera_name}_freq_analysis_fixed.jpg"
            save_path = os.path.join(save_dir, filename)
            success = cv2.imwrite(save_path, combined_image)

            if success:
                print(f"可视化结果已保存到: {save_path}")
            else:
                print(f"保存失败: {save_path}")

        except Exception as e:
            print(f"可视化保存失败: {e}")


def frequency_regularization_pyramid_scale(
        rendered_image: torch.Tensor,
        gt_image: torch.Tensor,
        gaussians,
        scene,
        viewpoint_cam,
        visibility_filter: torch.Tensor,
        iteration: int,
        lambda_freq: float = 0.001,
        lambda_scale: float = 0.005,
        num_levels: int = 3,
        high_freq_thresh: float = 0.2,
        save_results: bool = False,
        save_dir: Optional[str] = None,
        warmup_iterations: int = 1000,
        debug: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
    """
    修复后的频率金字塔正则化函数

    主要修复：
    1. 维度匹配问题
    2. FFT可视化中的人工伪影
    3. 更健壮的错误处理
    """

    if iteration < warmup_iterations:
        return torch.tensor(0.0, device=rendered_image.device), None, {'warmup': True}

    device = rendered_image.device
    debug_info = {}

    try:
        regularizer = TrueFrequencyPyramidRegularizer(
            num_levels=num_levels,
            high_freq_thresh=high_freq_thresh,
            use_fft=True
        )

        # 确保输入格式
        if rendered_image.dim() == 3:
            rendered_image = rendered_image.unsqueeze(0)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)

        total_loss = torch.tensor(0.0, device=device)

        # 1. 构建金字塔
        rendered_pyramid = regularizer.build_pyramid(rendered_image)
        gt_pyramid = regularizer.build_pyramid(gt_image)

        debug_info['pyramid_levels'] = len(rendered_pyramid)

        # 2. 计算频率损失
        if lambda_freq > 0:
            freq_loss = regularizer.compute_true_frequency_loss(rendered_pyramid, gt_pyramid)
            total_loss += lambda_freq * freq_loss
            debug_info['freq_loss'] = freq_loss.item()

        # 3. 高频区域检测（修复shape问题）
        high_freq_mask = regularizer.detect_true_high_frequency_regions(gt_pyramid[0])
        debug_info['high_freq_pixels'] = high_freq_mask.sum().item()
        debug_info['high_freq_ratio'] = (high_freq_mask.sum() / high_freq_mask.numel()).item()

        # 4. 尺度正则化
        if lambda_scale > 0 and high_freq_mask.sum() > 0:
            scale_loss = regularizer.compute_scale_regularization(
                gaussians, visibility_filter, high_freq_mask, viewpoint_cam)
            total_loss += lambda_scale * scale_loss
            debug_info['scale_loss'] = scale_loss.item()

        # 5. FFT特征分析
        fft_features = regularizer.compute_fft_features(gt_pyramid[0])
        debug_info['fft_valid'] = fft_features.get('valid', False)

        if fft_features.get('valid', False):
            debug_info['freq_band_energies'] = fft_features['band_energies'].detach().cpu().numpy().tolist()

        # 6. 保存可视化
        if save_results and save_dir and iteration % 1000 == 0:
            camera_name = getattr(viewpoint_cam, 'image_name', '').replace('.jpg', '').replace('.png', '')
            regularizer.save_visualizations(
                rendered_image, gt_image, high_freq_mask, fft_features,
                save_dir, iteration, camera_name)

        total_loss = torch.clamp(total_loss, 0, 1.0)
        debug_info['total_loss'] = total_loss.item()

        if debug:
            print(f"修复后的频率正则化损失: {total_loss:.6f}")
            print(f"FFT计算状态: {'成功' if debug_info['fft_valid'] else '失败'}")

        return total_loss, high_freq_mask, debug_info

    except Exception as e:
        print(f"频率正则化失败: {e}")
        debug_info['error'] = str(e)
        return torch.tensor(0.0, device=device), None, debug_info