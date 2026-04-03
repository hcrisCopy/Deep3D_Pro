# Deep3D: Automatic 2D-to-3D Video Conversion with CNNs (PyTorch)

本项目是 Deep3D 的 **PyTorch 重写版本**，原版基于 MXNet 实现。Deep3D 能够自动将 2D 图像/视频转换为 3D 立体效果。

![teaser](https://raw.githubusercontent.com/piiswrong/deep3d/master/img/teaser.png)

## 项目结构

```
deep3d_pytorch/
├── environment.yml       # Conda 环境配置
├── README.md             # 本文档
├── model.py              # Deep3D 网络模型定义
├── depth_dot.py          # DepthDot 算子（纯 PyTorch 实现）
├── dataset.py            # 数据加载：图片目录 / 视频文件
├── preprocess.py         # 数据预处理：从立体视频提取帧对
├── train.py              # 训练脚本
├── predict.py            # 单图推理 & 深度可视化
└── convert_movie.py      # 2D 视频转 3D 视频
```

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n deep3d python=3.10
conda activate deep3d
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 数据准备

### 方式一：从立体 3D 视频提取帧对

如果你有 Side-by-Side (SBS) 3D 视频文件：

```bash
python preprocess.py movie_3d.mkv data/train_set/1 --sbs3d --auto_clip
```

这将从视频中提取左右视图帧对，保存到 `data/train_set/1/left/` 和 `data/train_set/1/right/` 目录。

可选参数：
- `--reshape 384 160`：输出帧尺寸（宽 高）
- `--max_frames 10000`：最多提取帧数
- `--frame_interval 3`：每 3 帧取 1 帧
- `--skip_frames 2880`：跳过开头帧（如 logo）
- `--auto_clip`：自动检测并裁剪黑边

### 方式二：图片目录

仅支持以下目录结构（多片段）：

```
data/train_set/
├── 1/
│   ├── left/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── right/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
├── 2/
│   ├── left/
│   └── right/
└── ...
```

每个片段内 left 和 right 按同名文件自动配对。

## 训练

### 从图片目录训练

```bash
python train.py \
    --data_root ../data/train_set \
    --batch_size 16 \
    --epochs 2 \
    --lr 0.002 \
    --lr_step 20 \
    --lr_factor 0.1 \
    --gpu 0 \
    --exp_dir ../data/exp \
    --prefix deep3d
```

说明：
- 在 Deep3D_Pro 目录下运行时，默认 data_root 已设置为 ../data/train_set。
- 即使不显式传 --data_root，也会读取同级 data/train_set。

### 从视频文件直接训练

```bash
python train.py \
    --video_path data/raw/movie.mkv \
    --batch_size 16 \
    --epochs 100 \
    --gpu 0
```

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | ../data/train_set | 数据根目录，仅支持 clip_id/left,right 结构 |
| `--video_path` | - | SBS 3D 视频路径 |
| `--batch_size` | 16 | 批大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 0.002 | 初始学习率 |
| `--lr_step` | 20 | 每 N 轮学习率衰减 |
| `--lr_factor` | 0.1 | 学习率衰减因子 |
| `--data_shape` | 384 160 | 输入图像尺寸（宽 高） |
| `--gpu` | 0 | GPU 编号（-1 为 CPU） |
| `--no_pretrained` | False | 不使用预训练 VGG16 |
| `--resume` | - | 从检查点恢复训练 |

### 恢复训练

```bash
python train.py --data_root ../data/train_set --resume exp/deep3d-0050.pth
```

### TensorBoard 监控

```bash
tensorboard --logdir exp/tb_logs
```

## 推理

### 单图推理

```bash
python predict.py exp/deep3d-best.pth input.jpg --output result
```

输出文件：
- `result_right.png`：预测的右视图
- `result_anaglyph.png`：红青 3D 图（需 3D 眼镜观看）
- `result_sbs.png`：左右并排 3D 图
- `result_depth.png`：深度概率可视化（需 `--save_depth`）

### 批量图片推理

```bash
python predict.py exp/deep3d-best.pth input_dir/ --output output_dir/
```

### 2D 视频转 3D

```bash
python convert_movie.py exp/deep3d-best.pth input_2d.mp4 \
    --output output_3d \
    --gpu 0 \
    --batch_size 10
```

输出文件：
- `output_3d_anaglyph.mp4`：红青 3D 视频
- `output_3d_sbs.mp4`：左右并排 3D 视频
- `output_3d_left.mp4`：原始左视图
- `output_3d_pred_right.mp4`：预测右视图

## 技术细节

### 网络架构

- **骨干网络**：简化的 VGG16（conv1-conv5，每组 1-2 层卷积 + MaxPool）
- **多尺度深度预测**：从 pool1-pool5 各层提取特征，经 BatchNorm + 1x1/3x3 卷积预测 33 通道深度图
- **特征融合**：通过反卷积将各尺度预测上采样到 pool1 分辨率后逐元素相加
- **视图重建**：Softmax 归一化深度概率 + DepthDot 算子根据深度概率对左视图做水平位移加权求和，生成右视图

### DepthDot 算子

原版使用 CUDA C++ 自定义算子，本版本使用纯 PyTorch 实现，支持 CPU 和 GPU，无需编译。

### 损失函数

L1 Loss（MAE），与原版一致。

### 预训练权重

默认使用 torchvision 提供的 VGG16 ImageNet 预训练权重初始化骨干网络，反卷积层使用双线性插值核初始化。

## 与原版的主要区别

| 特性 | 原版 (MXNet) | PyTorch 版 |
|------|-------------|-----------|
| 框架 | MXNet | PyTorch |
| 数据格式 | LMDB + RecordIO | 图片目录 / 视频 |
| 自定义算子 | CUDA C++ 编译 | 纯 PyTorch |
| 预训练权重 | VGG16 .params 文件 | torchvision VGG16 |
| 训练监控 | 日志文件 | TensorBoard + 日志 |
| Python | 2.7 | 3.10+ |
