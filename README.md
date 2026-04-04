# Deep3D v1.0 — 实时端到端 2D-to-3D 视频转换

基于深度学习的实时 2D-to-3D 视频转换系统。受 [piiswrong/deep3d](https://github.com/piiswrong/deep3d) 启发，使用 PyTorch 重新构建网络，并在时域和推理速度方面进行了大幅优化。

<div align="center">
  <img src="./medias/wood_result_360p.gif"><br>
  <em>左：输入 2D 视频 | 右：输出具有<a href="https://en.wikipedia.org/wiki/Parallax">视差</a>效果的 3D 视频</em>
</div>

## 推理性能

|           平台           | 360p (FPS) | 720p (FPS) | 1080p (FPS) | 4K (FPS) |
| :----------------------: | :--------: | :--------: | :---------: | :------: |
|       GPU (2080Ti)       |     84     |     87     |     77      |    26    |
| CPU (Xeon Platinum 8260) |    27.7    |    14.1    |     7.2     |   2.0    |

## 环境要求

- Linux / Mac OS / Windows
- Python 3.7+
- [FFmpeg 3.4.6+](http://ffmpeg.org/)
- [PyTorch 1.7.1+](https://pytorch.org/)
- CPU 或 NVIDIA GPU

```bash
pip install opencv-python
```

## 预训练模型

[[Google Drive]](https://drive.google.com/drive/folders/1o-JRU9A38rHwoozHZNJDxKKAydgK_z04?usp=sharing) [[百度云 (提取码 xxo0)]](https://pan.baidu.com/s/1Qml48TBI7_AC_d5oiZEAyQ)

## 项目结构

```
Deep3D/
├── model.py            # 网络定义: Deep3DNet, RefinementBlock, backwarp
├── train.py            # 训练脚本: 训练循环, 验证, checkpoint, 可视化
├── inference.py        # 推理脚本: 视频逐帧处理, 时序递归
├── data/
│   ├── dataset.py      # TemporalStereoDataset
│   ├── transform.py    # 图像预处理 (PreProcess)
│   ├── impro.py        # 图像 I/O 工具
│   └── degradater.py   # 数据退化增强
└── utils/
    ├── util.py         # 通用工具
    └── ffmpeg.py       # 视频编解码封装
```

---

## 快速推理

```bash
python inference.py \
    --model ./export/deep3d_v1.0_640x360_cuda.pt \
    --video ./medias/wood.mp4 \
    --out ./result/wood.mp4 \
    --inv   # 部分视频需要翻转左右视图
```

---

## 训练

### 数据准备

训练数据需要按视频片段组织的左右立体帧序列：

```
data/train_set/
├── 1/
│   ├── left/       # 左视图帧（按文件名排序）
│   │   ├── 0001.jpg
│   │   └── ...
│   └── right/      # 右视图帧（文件名与 left/ 一一对应）
│       ├── 0001.jpg
│       └── ...
├── 2/
│   ├── left/
│   └── right/
└── ...
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp` 格式，每个片段建议至少 11 帧。

### 训练命令

```bash
# 从零训练
python train.py --data_root ../data/train_set --batch_size 4 --epochs 100

# 加载预训练权重微调
python train.py \
    --data_root ../data/train_set \
    --pretrained ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
    --lr 0.0001 --epochs 50

# 从 checkpoint 恢复
python train.py \
    --data_root ../data/train_set \
    --resume ../data/exp/20260404/deep3d_031017/deep3d-best.pth
```

### 关键训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `../data/train_set` | 训练数据根目录 |
| `--data_shape` | `640 360` | 输入分辨率 (宽 高) |
| `--alpha` | `5` | 远前/远后帧的时序偏移 |
| `--prev_mode` | `right_gt` | x0 来源: Teacher Forcing |
| `--batch_size` | `4` | 批大小 |
| `--lr` | `1e-3` | 初始学习率（Adam） |
| `--lr_step` | `30` | LR 衰减周期 |
| `--lr_factor` | `0.5` | LR 衰减因子 |
| `--pretrained` | `None` | 预训练 JIT 模型路径 |
| `--gpu` | `0` | GPU 编号（-1 为 CPU） |

---

## 致谢

This code borrows from [[deep3d]](https://github.com/piiswrong/deep3d) [[DeepMosaics]](https://github.com/HypoX64/DeepMosaics)

---

## 方法详解：从输入到输出的完整流程

### 一、核心思想

Deep3D v1.0 是一个**基于多帧时序信息的光流预测网络**。核心思路是：给定一段连续的左眼视频帧，通过预测**光流场**（optical flow）和**混合掩码**（blending mask），将多个左眼参考帧**反向变形**（backward warp）后加权混合，合成右眼视图。

关键创新：
1. **多帧时序推理**：利用当前帧前后各数帧的信息，从多个视角综合重建
2. **4 级由粗到精（Coarse-to-Fine）**：光流从 1/8 分辨率逐步精化到全分辨率
3. **自回归递归**：前一帧的预测输出作为当前帧的输入之一，保证时序连贯性
4. **全卷积架构**：仅 ~5.2M 参数，支持任意分辨率输入，实现实时推理

### 二、输入格式

模型输入为 **18 通道张量** `(N, 18, H, W)`，由 6 个 RGB 帧拼接而成：

| 通道 | 符号 | 含义 |
|:---:|:---:|:---|
| `[0:3]` | $x_1 = L_{t-\alpha}$ | **远前帧**（默认 $\alpha=5$，即 5 帧前的左视图） |
| `[3:6]` | $x_2 = L_{t-1}$ | **近前帧**（前一帧左视图） |
| `[6:9]` | $x_0 = \hat{R}_{t-1}$ | **前一帧右视图预测**（训练时为 GT，推理时为模型输出） |
| `[9:12]` | $x_3 = L_t$ | **当前帧**（要处理的目标左视图） |
| `[12:15]` | $x_4 = L_{t+1}$ | **近后帧** |
| `[15:18]` | $x_5 = L_{t+\alpha}$ | **远后帧** |

边界帧通过 clamp 处理（例如第 0 帧时，远前帧也用第 0 帧替代）。

### 三、网络架构

#### 3.1 整体流水线

```
输入 (N, 18, H, W) + onehot 指示通道 → (N, 19, H, W)
      │
      ├── [Block 0] 1/8 分辨率粗估计  → flow₀(8ch), mask₀(3ch)
      │        ↓ backward warp × 4 个参考帧
      ├── [Block 1] 1/4 分辨率精化    → flow₁, mask₁ (残差累加)
      │        ↓ backward warp × 4
      ├── [Block 2] 1/2 分辨率精化    → flow₂, mask₂ (残差累加)
      │        ↓ backward warp × 4
      ├── [Block 3] 全分辨率精化      → flow₃, mask₃ (残差累加)
      │        ↓ backward warp × 4
      │
      └── Sigmoid Mask 层级混合 → 输出 (N, 3, H, W) — 预测右眼视图
```

**参数量**：5,237,564（约 5.2M），全卷积无全连接层。

#### 3.2 RefinementBlock 结构

每个 Block 的内部结构如下：

```
输入 (N, C_in, H, W), 可选 flow_in (N, 8, H, W)
  │
  ├─ F.interpolate 空间下采样到目标 spatial_scale
  ├─ 若有 flow_in: 同步下采样 + 缩放后拼接
  ├─ conv0: 两层 Conv-BN-PReLU (stride=2 × 2, 共降 4×)
  ├─ convblock: 8 层 Conv-BN-PReLU (stride=1) + 残差连接
  ├─ lastconv: ConvTranspose2d 上采样 2× → 11 通道
  └─ F.interpolate 回到原始尺寸
      → flow_residual (8ch) × scale,  mask_residual (3ch)
```

各 Block 配置：

| Block | 处理尺度 | 输入通道 | 中间通道 | 输出通道 | flow_out_scale |
|:-----:|:-------:|:------:|:------:|:------:|:------------:|
| block0 | 1/8 | 19 | 96 | 192 | 16.0 |
| block1 | 1/4 | 24 | 64 | 128 | 8.0 |
| block2 | 1/2 | 24 | 48 | 96 | 4.0 |
| block3 | 1.0 | 24 | 32 | 64 | 2.0 |

#### 3.3 光流 (Flow) — 8 通道

flow 共 8 通道 = 4 组 × 2 通道 $(dx, dy)$，分别控制对 4 个参考帧的反向变形：

| flow 通道 | 变形对象 | 来源帧 |
|:---------:|:-------:|:-----:|
| `[0:2]` | 远前帧 | $L_{t-\alpha}$ |
| `[2:4]` | 前一帧预测 | $\hat{R}_{t-1}$ |
| `[4:6]` | 当前帧 | $L_t$ |
| `[6:8]` | 远后帧 | $L_{t+\alpha}$ |

注意：$x_2$ ($L_{t-1}$) 和 $x_4$ ($L_{t+1}$) **不直接参与 warp**，仅作为网络的上下文信息。

#### 3.4 Backward Warp（反向变形）

标准的可微分反向变形操作：

$$\text{warped}(x, y) = \text{bilinear\_sample}\bigl(\text{ref},\; [x + \text{flow}_x,\; y + \text{flow}_y]\bigr)$$

实现步骤：
1. 在 $[-1, 1]$ 范围构建基础采样网格
2. 将像素单位的 flow 归一化：$\text{flow\_norm}_x = \frac{\text{flow}_x}{(W-1)/2}$
3. 新采样坐标 = base_grid + flow_norm
4. `F.grid_sample` 双线性插值采样，边缘模式 `border`

全程可微，支持梯度反传。

### 四、逐级精化过程

#### Stage 0 — 粗估计（1/8 分辨率）

```
x = concat([18ch 图像, 1ch onehot]) → (N, 19, H, W)
flow₀, mask₀ = Block0(x)
w_bef  = backwarp(L_{t-α},    flow₀[0:2])
w_pred = backwarp(R̂_{t-1},    flow₀[2:4])
w_cur  = backwarp(L_t,        flow₀[4:6])
w_aft  = backwarp(L_{t+α},    flow₀[6:8])
```

#### Stage 1–3 — 残差精化

每一级重复相同模式：

```
x_in = concat([w_bef, w_pred, w_cur, w_aft, mask, onehot])  → 16ch
flow_resid, mask_resid = Block_k(x_in, flow)                 → 24ch 输入

flow = flow + flow_resid    ← 残差累加
mask = mask + mask_resid

重新进行 backward warp × 4
```

### 五、最终混合（Sigmoid Mask Blending）

经过 4 级精化后，3 通道 mask 经 sigmoid 激活控制层级混合：

```python
mask = sigmoid(mask)             # 每通道 → [0, 1]
mask_bef, mask_aft, mask_pred = mask 的 3 个通道

# 三级层式 alpha 混合
output = w_bef  × mask_bef  + w_cur × (1 - mask_bef)    # ① 当前帧 vs 远前帧
output = w_aft  × mask_aft  + output × (1 - mask_aft)    # ② 混入远后帧
output = w_pred × mask_pred + output × (1 - mask_pred)   # ③ 混入前一帧预测
```

**最终输出**：`(N, 3, H, W)` 预测右眼视图，值域 $[0, 1]$。

### 六、推理时的时序递归

推理时逐帧处理视频，$x_0$ 通道形成 autoregressive 循环：

$$\hat{R}_t = f\bigl(L_{t-\alpha},\; L_{t-1},\; \hat{R}_{t-1},\; L_t,\; L_{t+1},\; L_{t+\alpha}\bigr)$$

```
x0 = process(第一帧左视图)         ← 初始化

对每一帧 t:
    input = concat([x1, x2, x0, x3, x4, x5])   ← 18 通道
    out = model(input)                            ← 前向推理
    x0 = out                                      ← 输出用作下一帧输入
```

训练时通过 Teacher Forcing（$x_0 = R_{t-1}$ 真值）避免误差累积。

### 七、训练流程

1. **数据加载**：`TemporalStereoDataset` 从左右立体帧序列中按时序采样 6 帧 → 18ch 输入 + 3ch 目标
2. **损失函数**：L1 Loss — $\mathcal{L} = \|\hat{R}_t - R_t\|_1$
3. **优化器**：Adam（lr=1e-3, weight_decay=1e-5）
4. **学习率调度**：StepLR 每 30 epoch × 0.5
5. **验证划分**：训练集 90%，验证集 10%（seed=42）
6. **可视化输出**：每 5 epoch 保存 left | right_gt | right_pred | anaglyph
7. **Checkpoint**：自动保存最佳 val_loss 模型 (`deep3d-best.pth`)

---

## 方法对比：Deep3D v1.0 vs Deep3D_Pro

Deep3D_Pro 是原始 [piiswrong/deep3d](https://github.com/piiswrong/deep3d)（MXNet）的 PyTorch 重写版本，采用了完全不同的技术路线。以下是两种方法的详细对比。

### Deep3D_Pro 方法简述

Deep3D_Pro 使用 **VGG16 骨干网络** 从单张左视图中预测**视差概率分布**（33 个离散视差级别），然后通过 **DepthDot 算子**将左图按概率加权水平位移重建右视图：

$$R(x, y) = \sum_{d=d_{\min}}^{d_{\max}} P(d \mid x, y) \cdot L(x - d, y)$$

- 输入：单帧 RGB 图像 `(N, 3, H, W)`，默认 384×160
- 骨干：VGG16 (Conv1→Pool1→...→Pool5→FC6→FC7→FC8)
- 5 个尺度分支各输出 33 通道视差概率，上采样后逐元素求和
- Softmax → DepthDot（水平位移 + 加权求和）→ 预测右视图

### 核心差异对比

| 维度 | **Deep3D v1.0（本项目）** | **Deep3D_Pro** |
|:---|:---|:---|
| **核心方法** | 多帧光流预测 + 反向变形 + 掩码混合 | 单帧视差概率预测 + DepthDot 加权位移 |
| **输入** | 18 通道（6 帧时序 RGB） | 3 通道（单帧 RGB） |
| **时序信息** | 利用前后各 $\alpha$ 帧 + 自回归递归 | 无时序信息，逐帧独立处理 |
| **右视图生成方式** | 光流变形 4 个参考帧后 sigmoid 掩码混合 | 对左图做 33 级水平位移后 softmax 概率加权求和 |
| **深度表达** | 隐式（嵌入在光流中） | 显式（33 级视差概率分布） |
| **骨干网络** | 轻量全卷积（Conv-BN-PReLU × 4 级） | VGG16（含 FC 层） |
| **参数量** | ~5.2M | ~138M（VGG16）|
| **默认分辨率** | 640×360 | 384×160 |
| **分辨率灵活性** | 全卷积，支持任意分辨率 | FC 层限制，需固定空间尺寸 |
| **推理速度** | 84 FPS @ 360p (2080Ti) | 较慢（VGG16 + 33 次位移操作） |
| **视频处理** | 帧间自回归，时序连贯 | 逐帧独立，可能闪烁 |
| **损失函数** | L1 Loss | L1 Loss |
| **优化器** | Adam (lr=1e-3) | SGD (lr=0.002, momentum=0.9) |
| **数据增强** | 退化增强（模糊、降采样、噪声、JPEG 压缩） | 无 |
| **预训练骨干** | 无（从头训练或加载 JIT 权重） | ImageNet 预训练 VGG16 |
| **输出格式** | 左右拼接立体视频 (+FFmpeg 合成音频) | 右视图 / 红青立体图 / 左右拼接 |

### 方法流程对比图

```
                  Deep3D v1.0                          Deep3D_Pro
                  ─────────                            ──────────
输入：6 帧 (18ch)                           输入：单帧 (3ch)
  │                                           │
  ├─ Block0 (1/8)→flow₀+mask₀               ├─ VGG16 骨干特征提取
  │    ↓ warp×4                               │    ↓ 5 个池化层
  ├─ Block1 (1/4)→flow₁+mask₁               ├─ 5 个分支 → 各输出 33ch 视差概率
  │    ↓ warp×4                               │    ↓ 反卷积上采样到统一尺寸
  ├─ Block2 (1/2)→flow₂+mask₂               ├─ 逐元素求和 → ReLU
  │    ↓ warp×4                               │    ↓ 最终上采样 + Softmax
  ├─ Block3 (1/1)→flow₃+mask₃               ├─ DepthDot: 33 次水平位移加权求和
  │    ↓ warp×4                               │
  └─ sigmoid mask 层级混合                    └─ 输出右视图
  │                                           │
输出：预测右眼视图                           输出：预测右眼视图
```

### 各自的优势

**Deep3D v1.0**：
- 速度极快，适合实时视频处理
- 多帧时序推理提供更丰富的运动视差信息
- 自回归机制保证帧间一致性
- 轻量级参数量，训练和部署成本低
- 任意分辨率输入

**Deep3D_Pro**：
- 单帧即可工作，无需视频上下文
- 显式视差概率可直接可视化深度分布
- ImageNet 预训练的 VGG16 骨干提供强大的语义理解
- 实现更简单，完全基于纯 PyTorch（无 grid_sample）
- 适合静态图像的 2D→3D 转换