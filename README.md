# Deep3D_Pro

基于 PyTorch 的多帧时序 2D-to-3D 视频转换项目。仓库目前已经按职责整理为训练、推理、模型、数据和工具几个子目录；推理链路不包含音频抽取、音频合成或音视频重新封装逻辑，默认输出无音轨的左右拼接立体视频。

<div align="center">
  <img src="./medias/wood_result_360p.gif"><br>
  <em>左：输入 2D 视频；右：输出的左右拼接 3D 结果示意</em>
</div>

## 项目结构

```text
Deep3D_Pro/
├── training/
│   └── train.py              # 训练入口
├── inference/
│   └── run_inference.py      # 推理入口
├── models/
│   └── deep3d_network.py     # 网络结构与预训练权重加载
├── data/
│   ├── dataset.py            # 时序立体数据集
│   └── transforms.py         # 预处理与历史增强函数
├── tools/
│   ├── file_utils.py         # 文件与路径工具
│   ├── image_io.py           # 图像读写与传统图像处理工具
│   ├── image_degradation.py  # 数据退化增强
│   └── video_utils.py        # 视频元数据与写出工具
└── medias/
    ├── wood.mp4
    └── wood_result_360p.gif
```

推荐的目录协作方式如下：

```text
/root/autodl-tmp/deep3d/
├── Deep3D_Pro/              # 代码目录
└── data/                    # 数据、权重、训练输出统一放这里
    ├── train_set/
    ├── pretrained/
    ├── exp/
    └── results/
```

这样做的好处是代码目录保持干净，训练集、预训练权重、checkpoint、可视化结果和推理输出都集中在同级 `data/` 目录中；README 示例也建议统一使用相对路径 `../data/...`，方便后续脚本、配置和批处理对接。

## 环境配置

推荐使用 `conda + pip` 搭建与当前项目一致的运行环境：

```bash
conda create -n deep3d python=3.10
conda activate deep3d
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

说明：

- 项目训练与推理代码都基于 PyTorch。
- 如果机器没有可用的 NVIDIA GPU，也可以在 CPU 上运行，但训练和推理速度会明显下降。
- `requirements.txt` 中已经包含 OpenCV、tqdm、tensorboard 等依赖，无需再逐个手动安装。

## 预训练模型

[[Google Drive]](https://drive.google.com/drive/folders/1o-JRU9A38rHwoozHZNJDxKKAydgK_z04?usp=sharing)
[[百度云（提取码 xxo0）]](https://pan.baidu.com/s/1Qml48TBI7_AC_d5oiZEAyQ)

## 快速推理

```bash
python inference/run_inference.py \
  --model ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --video ./medias/wood.mp4 \
  --out ../data/results/wood.mp4
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--gpu_id` | GPU 编号，设为 `-1` 时走 CPU |
| `--alpha` | 时序窗口偏移，默认 `5` |
| `--resize WIDTH HEIGHT` | 指定推理输出尺寸 |
| `--inv` | 输出时交换左右视图 |

说明：

- 当前推理输出为静音视频，不会保留源视频音轨。
- 推理脚本不依赖 FFmpeg。

## 测试评测

针对你当前指定的数据集 `../data/test_set/mono2stereo_test`，项目已经补充了评测入口：

```bash
python inference/evaluate_mono2stereo.py \
  --gpu_id 0 \
  --model ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --data_root ../data/test_set/mono2stereo_test \
  --out_root ../data/test_on_mono
```

评测流程与 `Mono2Stereo` 参考逻辑对齐为：

- `mono2stereo_test` 被视作单图评测集，不再把不同图片拼成伪时序窗口。
- 每张左图会被重复填入网络要求的 6 个输入槽位，独立生成对应的右图预测。
- 先把测试图片统一调整到 `1280x800`。
- 送入当前 TorchScript 权重前，再调整到权重固定要求的 `1280x720`。
- 模型输出的右视图再放回 `1280x800` 后计算 `SSIM / PSNR / SIOU`。
- 同时统计整套评测吞吐、纯模型前向速度，以及进程内存和 CUDA 显存占用。

输出目录固定为：

```text
../data/test_on_mono/
├── summary.json              # 总体指标、速度、内存/显存汇总
├── per_clip_metrics.csv      # 每个子集的平均指标
├── per_frame_metrics.csv     # 每一帧的详细指标和耗时
├── predictions/              # 全量预测右视图
└── visualizations/           # 每个子集抽样可视化拼图
```

可视化拼图会同时展示 `Left / Pred Right / GT Right / Pred-GT Diff / Anaglyph / Pred-Left Diff`，方便快速判断模型是否正常运行。当前这次实跑生成的样例包括：

- `../data/test_on_mono/visualizations/animation/000000251_viz.jpg`
- `../data/test_on_mono/visualizations/complex/000000230_viz.jpg`
- `../data/test_on_mono/visualizations/indoor/000000440_viz.jpg`
- `../data/test_on_mono/visualizations/outdoor/000002383_viz.jpg`
- `../data/test_on_mono/visualizations/simple/00000001_viz.jpg`

说明：

- `FPS` 是整套评测循环的处理速度，包含数据搬运、resize、保存预测和指标计算。
- `Model FPS` 只统计 `net(input_data)` 的纯前向时间。
- 当前脚本会按子集预先载入并缓存图片，因此进程峰值内存会高于纯模型推理场景。

## 训练

### 训练数据集路径与目录结构

默认训练集根目录为：

```text
../data/train_set
```

`TemporalStereoDataset` 会把 `train_set` 下的每个一级子目录视作一个独立视频片段。结合你当前本地数据，推荐按下面这种“纯数字编号目录”组织：

```text
../data/train_set/
├── 1/
│   ├── left/
│   └── right/
├── 2/
│   ├── left/
│   └── right/
...
├── 9/
│   ├── left/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── 3.png
│   │   └── ...
│   └── right/
│       ├── 1.png
│       ├── 2.png
│       ├── 3.png
│       └── ...

```

数据组织规则：

- 每个编号目录表示一个独立片段，例如 `1/`、`2/`、`9/`。
- 每个片段下必须同时存在 `left/` 和 `right/` 两个子目录。
- `left/` 与 `right/` 中的文件名必须严格对应；数据集代码会取左右目录文件名交集作为可用样本。
- 当前代码支持 `.jpg`、`.jpeg`、`.png`、`.bmp`。
- 你本地样例中帧名使用 `1.png、2.png、3.png ...` 这种自然编号方式，README 建议继续保持这种命名，避免不同片段间规则不一致。
- 如果某个编号目录只有空文件夹、缺失 `left/`/`right/` 之一，或左右文件名对不上，该片段会被自动跳过。
- 数据集内部会对每一帧做统一 resize，默认目标尺寸为 `640x360`，由 `--data_shape WIDTH HEIGHT` 控制。
- 当 `alpha=5` 时，模型每个样本会访问 `t-5、t-1、t、t+1、t+5` 等位置，因此建议每个片段至少准备 `2 * alpha + 1 = 11` 帧；帧数越长，时序上下文越稳定。

补充说明：

- 边界帧不会被丢弃。代码在 `t=0`、`t=n-1` 等位置会采用首尾帧夹取补齐的方式构造时序窗口。
- 当前 `train.py` 的参数默认值仍写的是项目内路径，但按当前目录约定，实际使用时建议始终显式传入 `--data_root ../data/train_set`。

### 训练命令

```bash
python training/train.py \
  --data_root ../data/train_set \
  --exp_dir ../data/exp \
  --batch_size 4 \
  --epochs 2
```

加载 TorchScript 预训练权重微调：

```bash
python training/train.py \
  --data_root ../data/train_set \
  --pretrained ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --exp_dir ../data/exp \
  --lr 1e-4 \
  --epochs 2
```

从 checkpoint 恢复：

```bash
python training/train.py \
  --data_root ../data/train_set \
  --resume ../data/exp/20260404/deep3d_031017/deep3d-best.pth
```

### 关键训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data_root` | `./data/train_set` | 参数默认值仍指向项目内目录；实际使用建议显式传入 `../data/train_set` |
| `--val_ratio` | `0.1` | 验证集比例 |
| `--data_shape` | `640 360` | 输入分辨率，顺序为宽、高 |
| `--alpha` | `5` | 远前帧和远后帧的偏移 |
| `--prev_mode` | `right_gt` | `x0` 来源，默认使用 teacher forcing |
| `--batch_size` | `4` | 批大小 |
| `--epochs` | `100` | 训练轮数 |
| `--lr` | `1e-3` | 初始学习率 |
| `--lr_step` | `30` | 学习率衰减周期 |
| `--lr_factor` | `0.5` | 学习率衰减倍率 |
| `--pretrained` | `None` | TorchScript 预训练权重路径 |
| `--resume` | `None` | checkpoint 路径 |
| `--gpu` | `0` | GPU 编号，`-1` 为 CPU |

如果显式传入 `--exp_dir ../data/exp`，训练输出会统一保存在 `../data/exp/<日期>/<实验名>/`，其中包含日志、TensorBoard 文件、可视化结果、checkpoint，以及一份可直接被推理脚本加载的最佳 TorchScript 权重。

训练目录中的关键产物如下：

```text
../data/exp/<日期>/<实验名>/
├── deep3d-best.pth                     # 最佳 checkpoint，便于继续训练或恢复
├── deep3d_v1.0_<W>x<H>_<device>.pt    # 最佳 TorchScript 权重，命名风格与官方权重一致
├── deep3d-0005.pth                    # 按 save_interval 周期保存的 checkpoint
├── tb_logs/                           # TensorBoard 日志
└── vis/                               # 可视化结果
```

其中 `deep3d_v1.0_<W>x<H>_<device>.pt` 会根据 `--data_shape` 和当前训练设备自动生成，例如 `deep3d_v1.0_1280x720_cuda.pt` 或 `deep3d_v1.0_640x360_cpu.pt`。`vis/` 中不再固定保存前几张样本，而是每次从验证集中挑选当前 L1 Loss 最低的几张，文件名里会带上 rank、样本索引和 loss，方便快速筛查表现最稳定的样本。

## 方法流程

这一版 `Deep3D_Pro` 的核心思路不是“从单张左图直接回归右图”，而是把视频时序信息显式引入模型：当前时刻右眼视图由前后多帧左眼图像、上一时刻右眼状态，以及多尺度光流对齐与融合共同生成。

### 从输入到输出的完整流程

#### 1. 输入数据准备

训练阶段，每个样本来自某个编号片段中的第 `t` 帧。数据集会读取：

- `L(t-alpha)`：远前左视图
- `L(t-1)`：近前左视图
- `R(t-1)` 或 `L(t)`：历史输入 `x0`
- `L(t)`：当前左视图
- `L(t+1)`：近后左视图
- `L(t+alpha)`：远后左视图
- `R(t)`：当前时刻真实右视图标签

推理阶段，视频按时间顺序读取，每次维护一个长度为 `2 * alpha + 1` 的滑动窗口；窗口中心对应当前帧，窗口两端提供远前与远后上下文。

#### 2. 预处理

读入的图像或视频帧会经历如下处理：

1. 使用 OpenCV 读取 BGR 图像。
2. 按目标分辨率统一 resize。
3. 转为 RGB。
4. 归一化到 `[0, 1]`。
5. 调整为 PyTorch 张量格式 `(C, H, W)`。

在训练集中，6 张参考帧会拼接成一个 `18` 通道输入张量 `(18, H, W)`；当前右视图 `R(t)` 作为监督目标 `(3, H, W)`。

### 输入张量定义

模型输入固定是 18 通道，拼接顺序如下：

| 通道 | 记号 | 含义 |
|---|---|---|
| `[0:3]` | `x1 = L(t - alpha)` | 远前帧 |
| `[3:6]` | `x2 = L(t - 1)` | 近前帧 |
| `[6:9]` | `x0 = R(t - 1)` 或上一帧预测 | 历史右视图 / 时序状态 |
| `[9:12]` | `x3 = L(t)` | 当前左视图 |
| `[12:15]` | `x4 = L(t + 1)` | 近后帧 |
| `[15:18]` | `x5 = L(t + alpha)` | 远后帧 |

边界位置采用首尾帧夹取补齐，因此即使是片段的第一帧和最后一帧，也能被纳入训练或推理。

#### 3. 网络前向过程

`models/deep3d_network.py` 中的 `Deep3DNet` 是一个 4 级 coarse-to-fine 光流精化网络。它并不直接像普通 UNet 那样一次性输出右图，而是分阶段完成“估计位移 -> 对齐参考帧 -> 融合结果”的过程。

具体来说：

1. `block0` 在 `1/8` 尺度上先给出粗粒度预测，输出 `8` 通道 flow 和 `3` 通道 mask。
2. `8` 通道 flow 实际对应 4 组二维位移场，分别用于 warp：
   - 远前帧 `L(t-alpha)`
   - 历史右视图/上一帧预测 `x0`
   - 当前左视图 `L(t)`
   - 远后帧 `L(t+alpha)`
3. 网络使用 `backwarp` 对上述 4 个参考图像做 backward warping，把它们尽量对齐到当前时刻右视图坐标系。
4. `block1`、`block2`、`block3` 在 `1/4`、`1/2`、全分辨率尺度上继续预测 flow 残差和 mask 残差，不断细化几何对齐结果。
5. 最终的 3 通道 mask 经过 `sigmoid` 后，作为逐像素融合权重，按顺序混合多个 warped 结果，得到最终右视图。

最终输出张量大小为 `(N, 3, H, W)`，值域为 `[0, 1]`。

#### 4. 训练流程

训练入口在 `training/train.py`，主要步骤如下：

1. 读取数据集并构建 `TemporalStereoDataset`。
2. 按 `val_ratio` 随机划分训练集和验证集。
3. 初始化 `Deep3DNet`，可选加载 TorchScript 预训练权重。
4. 使用 `L1Loss` 作为重建损失，直接约束预测右图与真实右图之间的像素差异。
5. 每个 epoch 依次完成训练、验证、学习率更新。
6. 定期保存：
   - checkpoint：`deep3d-xxxx.pth`
   - 最优 checkpoint：`deep3d-best.pth`
   - 最优 TorchScript 权重：`deep3d_v1.0_<W>x<H>_<device>.pt`
   - 可视化图：从验证集中挑选当前 L1 Loss 最低的几张，展示左图、GT 右图、预测右图、红蓝立体图拼接结果
   - TensorBoard 日志

默认输出目录结构为：

```text
../data/exp/<日期>/<前缀_时分秒>/
├── tb_logs/
├── vis/
├── deep3d-0005.pth
├── deep3d-0010.pth
└── deep3d-best.pth
```

#### 5. 推理流程

推理入口在 `inference/run_inference.py`，视频级流程如下：

1. 读取输入视频元信息，包括宽高、帧率、总帧数。
2. 初始化长度为 `2 * alpha + 1` 的时序窗口。
3. 把窗口中的关键帧转换成模型所需的 `x1, x2, x0, x3, x4, x5`。
4. 将 6 帧拼成 18 通道输入，送入 TorchScript 模型。
5. 得到当前时刻预测右视图 `R_hat(t)`。
6. 将 `L(t)` 与 `R_hat(t)` 横向拼接成 side-by-side 立体帧。
7. 把当前 `R_hat(t)` 保存为下一时刻的 `x0`，形成自回归闭环。
8. 循环直到整段视频结束，最后输出无音轨的左右拼接视频。

### 训练与推理阶段的关键区别

- 训练默认采用 `prev_mode=right_gt`，即 `x0=R(t-1)`，属于 teacher forcing，收敛更稳定。
- 推理时无法获得真实 `R(t-1)`，所以 `x0` 改为上一帧预测结果 `R_hat(t-1)`。
- 这意味着推理阶段存在误差累积风险，但同时保留了时间连续性，是视频 2D-to-3D 转换里比较重要的一步。
- 当前推理脚本只处理画面，不做音频抽取、重编码或 mux，因此输出视频默认是静音的。

## 与老版 Deep3D 的异同

仓库根目录下的老版 `deep3d` 代码，主要对应早期 MXNet 版本的 Deep3D 思路：训练入口是 `train.py`，推理入口是 `convert_movie.py`，底层依赖 `data/lmdb` 数据、MXNet 符号图和早期多视差生成方式。`Deep3D_Pro` 则是重新整理后的 PyTorch 版本，但它不是对老版代码的逐文件迁移，而是方法层面已经发生了明显演化。

### 相同点

- 两者目标一致：都希望从左视图重建右视图，用于 2D-to-3D 视频转换。
- 两者都属于“生成右眼图像”路线，而不是直接输出双目深度图后再另做渲染。
- 两者都服务于视频场景，因此最终目标都是生成可视化立体结果，例如 SBS 或 anaglyph。

### 主要不同点

| 维度 | Deep3D_Pro | 老版 deep3d |
|---|---|---|
| 框架 | PyTorch | MXNet |
| 训练数据组织 | 普通文件夹：`片段/left/right` | `LMDB` 数据库 |
| 输入形式 | 6 帧时序 RGB，共 18 通道 | 以单帧为主，`data_frames=1` |
| 时序建模 | 显式引入前后帧与上一帧预测 | 基本按单帧或弱时序方式处理 |
| 历史状态 | 有 `x0`，训练/推理构成时序闭环 | 无明显自回归右视图状态 |
| 几何表达 | 4 组 flow + 3 通道 mask | 更接近传统 Deep3D 的视差/位移分布思想 |
| 输出生成方式 | backward warp 多参考帧后融合 | 依赖旧版网络符号图生成右图 |
| 网络结构 | 4 级 coarse-to-fine refinement | 早期静态符号网络 |
| 工程可维护性 | 目录清晰，训练/推理职责分离 | 历史脚本式组织较重 |
| 推理输出 | 默认直接输出无声 SBS 视频 | 老版脚本更偏研究/demo 工作流 |

### 方法层面的核心差异

#### 1. 单帧视差思路 vs 多帧时序重建思路

老版 Deep3D 更接近论文时代的经典设定：输入一张左图，网络学习一个与视差相关的表示，再据此合成右图。它对每一帧的处理相对独立，视频连续性更多依赖逐帧结果本身是否稳定。

`Deep3D_Pro` 则明确把问题改写成“当前右图由多时刻参考帧联合重建”。这使它不再只依赖 `L(t)`，而是同时利用 `L(t-alpha)`、`L(t-1)`、`L(t)`、`L(t+1)`、`L(t+alpha)` 以及上一时刻右图状态。对于遮挡、运动边缘、瞬时纹理缺失等视频场景，这种多帧建模通常更有优势。

#### 2. 视差分布生成 vs 光流对齐融合

老版 Deep3D 的代表性特点，是通过离散视差或位移分布去“搬运”左图像素，核心更偏向单图视差选择。

`Deep3D_Pro` 的核心则是：

1. 先预测 4 组 flow；
2. 把多个参考图像 warp 到目标右视图坐标；
3. 再用 mask 学习每个像素该信任哪一路参考信息。

这种做法更像“多源候选图像的几何对齐与融合”，比单一视差概率分布更适合处理视频里的运动与时序冗余。

#### 3. 纯图像重建 vs 视频闭环推理

老版方案偏单帧预测，帧间关系不强。

`Deep3D_Pro` 在推理阶段会把上一帧预测结果回灌给下一帧作为 `x0`，因此整个视频形成了真正的递归过程。好处是时序连续性更自然，坏处是如果前面某帧预测偏差较大，也可能向后传播。这也是它和老版方案最本质的差异之一。

#### 4. 数据与工程链路差异

老版代码训练依赖 `data/lmdb`，并使用 MXNet 的 `FeedForward` 与符号式网络定义，今天维护和扩展的门槛都比较高。

`Deep3D_Pro` 使用标准 PyTorch 数据集、DataLoader、checkpoint、TensorBoard 和 TorchScript 推理接口，迁移、调参、断点恢复和二次开发都更直接，也更适合当前工程环境。

### 可以怎样理解两者关系

如果把老版 `deep3d` 看作“论文思想和早期工程验证版”，那么 `Deep3D_Pro` 更像一个“面向视频时序重建的现代化重构版”。它保留了 Deep3D 系列“从左图生成右图”的任务目标，但在输入设计、网络表示、训练方式和推理闭环上，都已经不是原始实现的简单重写，而是一次比较彻底的方法升级。

## 致谢

本仓库参考了以下项目的思路和部分历史工具代码：

- [piiswrong/deep3d](https://github.com/piiswrong/deep3d)
- [HypoX64/DeepMosaics](https://github.com/HypoX64/DeepMosaics)
