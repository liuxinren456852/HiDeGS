# # #
# # # Copyright (C) 2023 - 2024, Inria
# # # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # # All rights reserved.
# # #
# # # This software is free for non-commercial, research and evaluation use
# # # under the terms of the LICENSE.md file.
# # #
# # # For inquiries contact  george.drettakis@inria.fr
# # #
# #
# # import math
# # import os
# # import torch
# # from random import randint
# # from utils.loss_utils import ssim
# # from gaussian_renderer import render  # 改用标准render函数
# # import sys
# # from scene import Scene, GaussianModel
# # from tqdm import tqdm
# # from utils.image_utils import psnr
# # from argparse import ArgumentParser
# # from arguments import ModelParams, PipelineParams, OptimizationParams
# # import torchvision
# # from lpipsPyTorch import lpips
# #
# #
# # def direct_collate(x):
# #     return x
# #
# #
# # @torch.no_grad()
# # def render_set(args, scene, pipe, out_dir, eval):
# #     render_path = out_dir
# #
# #     psnr_test = 0.0
# #     ssims = 0.0
# #     lpipss = 0.0
# #
# #     cameras = scene.getTestCameras() if eval else scene.getTrainCameras()
# #
# #     for viewpoint in tqdm(cameras):
# #         viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
# #         viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
# #         viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
# #         viewpoint.camera_center = viewpoint.camera_center.cuda()
# #
# #         # 直接渲染所有gaussians，不进行分层处理
# #         image = torch.clamp(render(
# #             viewpoint,
# #             scene.gaussians,
# #             pipe,
# #             torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
# #         )["render"], 0.0, 1.0)
# #
# #         gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
# #         alpha_mask = viewpoint.alpha_mask.cuda()
# #
# #         if args.train_test_exp:
# #             image = image[..., image.shape[-1] // 2:]
# #             gt_image = gt_image[..., gt_image.shape[-1] // 2:]
# #             alpha_mask = alpha_mask[..., alpha_mask.shape[-1] // 2:]
# #
# #         try:
# #             torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))
# #         except:
# #             os.makedirs(os.path.dirname(os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png")),
# #                         exist_ok=True)
# #             torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))
# #
# #         if eval:
# #             image *= alpha_mask
# #             gt_image *= alpha_mask
# #             psnr_test += psnr(image, gt_image).mean().double()
# #             ssims += ssim(image, gt_image).mean().double()
# #             lpipss += lpips(image, gt_image, net_type='vgg').mean().double()
# #
# #         torch.cuda.empty_cache()
# #
# #     if eval and len(scene.getTestCameras()) > 0:
# #         psnr_test /= len(scene.getTestCameras())
# #         ssims /= len(scene.getTestCameras())
# #         lpipss /= len(scene.getTestCameras())
# #         print(f"PSNR: {psnr_test:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")
# #
# #
# # if __name__ == "__main__":
# #     # Set up command line argument parser
# #     parser = ArgumentParser(description="Rendering script parameters")
# #     lp = ModelParams(parser)
# #     op = OptimizationParams(parser)
# #     pp = PipelineParams(parser)
# #     parser.add_argument('--out_dir', type=str, default="")
# #     args = parser.parse_args(sys.argv[1:])
# #
# #     print("Rendering " + args.model_path)
# #
# #     dataset, pipe = lp.extract(args), pp.extract(args)
# #     gaussians = GaussianModel(dataset.sh_degree)
# #     gaussians.active_sh_degree = dataset.sh_degree
# #
# #     # 直接从ply文件加载，不创建分层结构
# #     scene = Scene(dataset, gaussians, resolution_scales=[1], create_from_hier=False)
# #     gaussians.load_ply(os.path.join(args.model_path, "point_cloud.ply"))
# #
# #     render_set(args, scene, pipe, args.out_dir, args.eval)
#
# #
# # Copyright (C) 2023 - 2024, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #
#
# import torch
# from scene import Scene
# import os
# from tqdm import tqdm
# from os import makedirs
# from gaussian_renderer import render
# import torchvision
# from utils.general_utils import safe_state
# from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
# from utils.loss_utils import ssim
# from utils.image_utils import psnr
#
#
# @torch.no_grad()
# def evaluate_set(model_path, name, iteration, views, scene, gaussians, pipeline, background):
#     """评估渲染质量"""
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#
#     makedirs(gts_path, exist_ok=True)
#     makedirs(render_path, exist_ok=True)
#
#     psnr_test = 0.0
#     ssims = 0.0
#
#     for idx, view in enumerate(tqdm(views, desc=f"Evaluating {name}")):
#         # 获取真实图像并移动到GPU
#         gt = view.original_image
#         gt = gt.cuda()
#
#         # 渲染图像
#         out = render(view, gaussians, pipeline, background)
#         rendering = out["render"].clamp(0.0, 1.0)
#
#         # 保存图像
#         torchvision.utils.save_image(gt.clamp(0.0, 1.0),
#                                      os.path.join(gts_path, view.image_name + ".png"))
#         torchvision.utils.save_image(rendering,
#                                      os.path.join(render_path, view.image_name + ".png"))
#
#         # 计算评估指标
#         if hasattr(view, 'alpha_mask') and view.alpha_mask is not None:
#             alpha_mask = view.alpha_mask.cuda()
#             rendering = rendering * alpha_mask
#             gt = gt * alpha_mask
#
#         psnr_test += psnr(rendering, gt).mean().double()
#         ssims += ssim(rendering, gt).mean().double()
#
#     if len(views) > 0:
#         psnr_test /= len(views)
#         ssims /= len(views)
#         print(f"{name.upper()} - PSNR: {psnr_test:.5f} SSIM: {ssims:.5f}")
#
#     return psnr_test, ssims
#
#
# @torch.no_grad()
# def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background):
#     """仅渲染图像，不计算指标"""
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     makedirs(render_path, exist_ok=True)
#
#     for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
#         out = render(view, gaussians, pipeline, background)
#         rendering = out["render"].clamp(0.0, 1.0)
#         torchvision.utils.save_image(rendering,
#                                      os.path.join(render_path, view.image_name + ".png"))
#
#
# def main(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
#          skip_train: bool, skip_test: bool, eval_mode: bool, out_dir: str):
#     with torch.no_grad():
#         # 初始化高斯模型
#         gaussians = GaussianModel(dataset.sh_degree)
#
#         # 加载场景和模型
#         scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
#
#         # 设置背景色
#         bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#
#         # 设置输出路径
#         output_path = out_dir if out_dir else dataset.model_path
#
#         total_psnr = 0.0
#         total_ssim = 0.0
#         eval_count = 0
#
#         if not skip_train:
#             if eval_mode:
#                 psnr_train, ssim_train = evaluate_set(output_path, "train", scene.loaded_iter,
#                                                       scene.getTrainCameras(), scene, gaussians,
#                                                       pipeline, background)
#                 total_psnr += psnr_train
#                 total_ssim += ssim_train
#                 eval_count += 1
#             else:
#                 render_set(output_path, "train", scene.loaded_iter,
#                            scene.getTrainCameras(), scene, gaussians, pipeline, background)
#
#         if not skip_test:
#             if eval_mode:
#                 psnr_test, ssim_test = evaluate_set(output_path, "test", scene.loaded_iter,
#                                                     scene.getTestCameras(), scene, gaussians,
#                                                     pipeline, background)
#                 total_psnr += psnr_test
#                 total_ssim += ssim_test
#                 eval_count += 1
#             else:
#                 render_set(output_path, "test", scene.loaded_iter,
#                            scene.getTestCameras(), scene, gaussians, pipeline, background)
#
#         if eval_mode and eval_count > 0:
#             avg_psnr = total_psnr / eval_count
#             avg_ssim = total_ssim / eval_count
#             print(f"\nOVERALL AVERAGE - PSNR: {avg_psnr:.5f} SSIM: {avg_ssim:.5f}")
#
#
# if __name__ == "__main__":
#     # 设置命令行参数解析器
#     parser = ArgumentParser(description="Evaluation script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train", action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument("--out_dir", type=str, default="")
#     parser.add_argument("--compute_metrics", action="store_true", help="Evaluate metrics (PSNR, SSIM)")
#
#     args = get_combined_args(parser)
#     print("Rendering " + args.model_path)
#
#     # 初始化系统状态
#     safe_state(args.quiet)
#
#     # 运行评估
#     main(model.extract(args), args.iteration, pipeline.extract(args),
#          args.skip_train, args.skip_test, args.compute_metrics, args.out_dir)

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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips  # 添加LPIPS导入


@torch.no_grad()
def evaluate_set(model_path, name, iteration, views, scene, gaussians, pipeline, background):
    """评估渲染质量"""
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)

    psnr_test = 0.0
    ssims = 0.0
    lpipss = 0.0  # 添加LPIPS累积变量

    for idx, view in enumerate(tqdm(views, desc=f"Evaluating {name}")):
        # 获取真实图像并移动到GPU
        gt = view.original_image
        gt = gt.cuda()

        # 渲染图像
        out = render(view, gaussians, pipeline, background)
        rendering = out["render"].clamp(0.0, 1.0)

        # 保存图像
        torchvision.utils.save_image(gt.clamp(0.0, 1.0),
                                     os.path.join(gts_path, view.image_name + ".png"))
        torchvision.utils.save_image(rendering,
                                     os.path.join(render_path, view.image_name + ".png"))

        # 应用alpha mask（如果存在）
        if hasattr(view, 'alpha_mask') and view.alpha_mask is not None:
            alpha_mask = view.alpha_mask.cuda()
            rendering = rendering * alpha_mask
            gt = gt * alpha_mask

        # 计算评估指标
        psnr_test += psnr(rendering, gt).mean().double()
        ssims += ssim(rendering, gt).mean().double()
        lpipss += lpips(rendering, gt, net_type='vgg').mean().double()  # 添加LPIPS计算

    if len(views) > 0:
        psnr_test /= len(views)
        ssims /= len(views)
        lpipss /= len(views)  # 计算LPIPS平均值
        print(f"{name.upper()} - PSNR: {psnr_test:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")

    return psnr_test, ssims, lpipss  # 返回LPIPS值


@torch.no_grad()
def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background):
    """仅渲染图像，不计算指标"""
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        out = render(view, gaussians, pipeline, background)
        rendering = out["render"].clamp(0.0, 1.0)
        torchvision.utils.save_image(rendering,
                                     os.path.join(render_path, view.image_name + ".png"))


def main(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
         skip_train: bool, skip_test: bool, eval_mode: bool, out_dir: str):
    with torch.no_grad():
        # 初始化高斯模型
        gaussians = GaussianModel(dataset.sh_degree)

        # 加载场景和模型
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 设置背景色
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 设置输出路径
        output_path = out_dir if out_dir else dataset.model_path

        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0  # 添加LPIPS总和变量
        eval_count = 0

        if not skip_train:
            if eval_mode:
                psnr_train, ssim_train, lpips_train = evaluate_set(output_path, "train", scene.loaded_iter,
                                                      scene.getTrainCameras(), scene, gaussians,
                                                      pipeline, background)
                total_psnr += psnr_train
                total_ssim += ssim_train
                total_lpips += lpips_train  # 累积LPIPS
                eval_count += 1
            else:
                render_set(output_path, "train", scene.loaded_iter,
                           scene.getTrainCameras(), scene, gaussians, pipeline, background)

        if not skip_test:
            if eval_mode:
                psnr_test, ssim_test, lpips_test = evaluate_set(output_path, "test", scene.loaded_iter,
                                                    scene.getTestCameras(), scene, gaussians,
                                                    pipeline, background)
                total_psnr += psnr_test
                total_ssim += ssim_test
                total_lpips += lpips_test  # 累积LPIPS
                eval_count += 1
            else:
                render_set(output_path, "test", scene.loaded_iter,
                           scene.getTestCameras(), scene, gaussians, pipeline, background)

        if eval_mode and eval_count > 0:
            avg_psnr = total_psnr / eval_count
            avg_ssim = total_ssim / eval_count
            avg_lpips = total_lpips / eval_count  # 计算LPIPS平均值
            print(f"\nOVERALL AVERAGE - PSNR: {avg_psnr:.5f} SSIM: {avg_ssim:.5f} LPIPS: {avg_lpips:.5f}")


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Evaluation script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--compute_metrics", action="store_true", help="Evaluate metrics (PSNR, SSIM, LPIPS)")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # 初始化系统状态
    safe_state(args.quiet)

    # 运行评估
    main(model.extract(args), args.iteration, pipeline.extract(args),
         args.skip_train, args.skip_test, args.compute_metrics, args.out_dir)