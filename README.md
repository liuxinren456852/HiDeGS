# HiDeGS: High-Frequency Detail-Enhanced Gaussian Splatting for UAV Multi-View Reconstruction

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://github.com/SongJiang-WHU/HiDeGS)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://github.com/SongJiang-WHU/HiDeGS)
[![Dataset](https://img.shields.io/badge/Dataset-WHU__Dataset-green)](https://github.com/SongJiang-WHU/HiDeGS)

</div>

[**Jiang Song**]()*<sup>1</sup> · [**Changjun Chen***]()<sup>1</sup> · [**Hong Xie**]()<sup>1,2</sup> · [**Zongqian Zhan**]()<sup>1</sup> · [**Ronglin Zhang**]()<sup>3</sup> · [**Chengcheng Luo**]()<sup>1</sup> · [**Yuquan Zhou**]()<sup>1</sup> · [**Linxia Ji**]()<sup>1</sup> · [**Li Yan**]()<sup>*,1,2</sup>

<sup>1</sup>School of Geodesy and Geomatics, Wuhan University, 129 Luoyu Rd, Wuhan, 430079, China  
<sup>2</sup>Hubei Luojia Laboratory, 129 Luoyu Rd, Wuhan, 430079, China  
<sup>3</sup>Department of Land Surveying & Geo-Informatics, The Hong Kong Polytechnic University, Hung Hom, Kowloon, Hong Kong, China

*Corresponding author

---

## 🏗️ Method Overview

<div align="center">
<img src="assets/pipeline.png" width="90%">
<p><i>Figure 1: Overall architecture of HiDeGS framework</i></p>
</div>

Our method consists of three innovative modules working in coordination:

- **(a) Multi-scale Frequency-Spatial Joint Regularization**: Dual-domain optimization for high-frequency detail preservation
- **(b) Scale Regularization and Loss Weighting**: Adaptive control for Gaussian primitives in high-frequency regions
- **(c) Multi-dimensional Uncertainty Fusion**: Enhanced geometric consistency through dynamic constraint adjustment

---


## 📋 Abstract

HiDeGS is a novel 3D reconstruction method specifically designed for UAV multi-view scenarios. It addresses the critical challenges of **high-frequency detail loss** and **insufficient geometric consistency** in 3D Gaussian Splatting (3DGS) through multi-dimensional joint optimization strategies.

**Key Improvements:**
- 🎯 **+1.34 dB** PSNR and **+0.085** SSIM on ISPRS Dataset
- 🎯 **+1.95 dB** PSNR and **+0.1** SSIM on WHU Dataset
- ✨ Accurate recovery of fine geometric structures (building contours, road markings)
- 🚫 Effective reduction of "floaters" artifacts and geometric drift

---

## 🌟 Key Features

### 1. Multi-Scale Frequency-Spatial Joint Regularization
- Compensates high-frequency information loss via frequency-domain constraints
- Integrates spatial gradient enhancement and edge-preserving constraints
- Overcomes inadequate sensitivity of pixel-level loss to high-frequency components

### 2. High-Frequency-Aware Scale Regularization
- Imposes strict scale restrictions on Gaussian primitives in high-frequency regions
- Prevents geometric distortion from excessive primitive expansion
- Prioritizes optimization of tiny geometric structures through adaptive loss weighting

### 3. Enhanced Depth Geometric Consistency
- Quantifies uncertainty through depth gradient mutations and normal consistency
- Dynamically adjusts multi-view constraint strength
- Improves robustness in low-texture regions and edge areas


## 📊 Experimental Results

### Novel View Synthesis Comparison

<div align="center">

| Method | ISPRS Dataset | | | WHU Dataset | | |
|--------|--------|------|-------|--------|------|-------|
| | PSNR↑ | SSIM↑ | LPIPS↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
| 3DGS | 25.35 | 0.802 | 0.200 | 23.77 | 0.708 | 0.348 |
| PGSR | 25.41 | 0.850 | 0.235 | 23.96 | 0.722 | 0.335 |
| GaussianPro | 26.47 | 0.873 | 0.196 | 24.94 | 0.755 | 0.288 |
| RaDeGS | 26.37 | 0.879 | **0.144** | 25.17 | 0.781 | **0.254** |
| **HiDeGS (Ours)** | **26.69** | **0.887** | 0.145 | **25.72** | **0.808** | 0.261 |

</div>

### Novel View Synthesis Comparisons

---

## 🎬 Demo Videos

### Scene Reconstruction Demos

<div align="center">
  <img src="https://github.com/SongJiang-WHU/HiDeGS/blob/master/demos/zhl-h-3dgs11.gif?raw=true" width="100%" alt="3D Gaussian Splatting Demo">
  <p><i>ZHL</i></p>
</div>

<div align="center">
  <img src="https://github.com/SongJiang-WHU/HiDeGS/blob/master/demos/xh-h-pgsr.gif" width="100%" alt="3D Gaussian Splatting Demo">
  <p><i>XH</i></p>
</div>

---
## 🗄️ WHU Dataset

We introduce the **Wuhan University UAV Dataset (WHU_Dataset)** covering 8 typical scenarios including building complexes, vegetation, plazas, and more.

### Dataset Structure
```
WHU_Dataset/
├── PF/          # Plaza with Fountain
├── SQ/          # Square Scene
├── XH/          # XiangHu Lake
├── XZB/         # XinZhaiBu
├── YD/          # YunDi
├── ZE/          # ZongEr
├── ZHB/         # ZhaoHuBao
└── GXB/         # GuanXueBu
```

### Dataset Characteristics
- **Images per scene:** 78-367
- **Flight altitude:** 40-65 meters (adaptively adjusted)
- **Forward overlap:** >80%
- **Side overlap:** >60%
- **Camera tilt angle:** >65°
- **Acquisition device:** DJI Mini 3 Pro, DJI Mavic 2 Pro

### Download
🔗 [Download WHU_Dataset](#) (Link to be updated)

---

## 🚀 Installation

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/SongJiang-WHU/HiDeGS.git
cd HiDeGS

# Create conda environment
conda create -n hidegs python=3.9
conda activate hidegs

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Dependencies
- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8
- NumPy
- OpenCV
- [Add other specific dependencies]

---

## 💻 Usage

### Training
```bash
python train.py \
    --source_path /path/to/your/data \
    --model_path /path/to/output \
    --iterations 30000
```

### Evaluation
```bash
python evaluate.py \
    --model_path /path/to/trained/model \
    --source_path /path/to/test/data
```

### Rendering
```bash
python render.py \
    --model_path /path/to/trained/model \
    --source_path /path/to/test/data \
    --output_path /path/to/output
```

---

## 📝 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{song2025hidegs,
  title={HiDeGS: High-Frequency Detail-Enhanced Gaussian Splatting for UAV Multi-View Reconstruction},
  author={Song, Jiang and Chen, Changjun and Xie, Hong and Zhan, Zongqian and Zhang, Rongling and Luo, Chengcheng and Zhou, Yuquan and Ji, Linxia and Yan, Li},
  journal={[Journal Name]},
  year={2025}
}
```

---

## 🙏 Acknowledgements

This work was partially supported by:
- National Key Research and Development Program of China (Grant 2024YFC3013301)
- National Natural Science Foundation of China (Grants 42371451 and 42394061)
- Natural Science Foundation of Wuhan (No.2024040701010028)

We thank the following projects for their excellent work:
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [ISPRS Benchmark](http://www2.isprs.org/commissions/comm3/wg4/detection-and-reconstruction.html)

---

## 📧 Contact

For questions and discussions, please contact:
- **Jiang Song**: [songjiang@whu.edu.cn](mailto:songjiang@whu.edu.cn)
- **Li Yan** (Corresponding author): [liyan@whu.edu.cn](mailto:liyan@whu.edu.cn)

School of Geodesy and Geomatics, Wuhan University

---

<div align="center">
<p>⭐ If you find this project helpful, please consider giving it a star! ⭐</p>
</div>
