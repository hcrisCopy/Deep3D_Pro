## Deep3D_Pro Speed Benchmark

### 1. 目标

- 面向连续帧目录测试 Deep3D_Pro 推理速度、延迟、显存/内存和输出质量。
- 对比 4 条测速路线：Baseline、JIT 优化、Cached Grid 优化、Anchor Skip 优化。
- 输出结果统一落盘，便于 PPT 汇报、表格对比和端侧部署评估。

---

### 2. 输入数据约定

```text
../data/test_set/speed_test/
└── clip_id/
    ├── left/                 # 必需：连续左视图帧，支持 jpg/jpeg/png/bmp
    └── right/                # 可选：右视图真值，用于可视化对比
```

- 每个 `clip_id` 目录会被视为一段视频序列。
- 默认 `--alpha 5`，每帧使用前后时序窗口构造 6 路输入。
- 如果未显式传入 `--output_size`，baseline/JIT/cached-grid/anchor-skip 会在输入不是 1080p 时按 `1920x1080` 进行模拟测速。
- 模型输入尺寸默认从权重名解析，例如 `deep3d_v1.0_1280x720_cuda.pt` 会解析为 `1280 720`。

---

### 3. 输出目录

默认总目录：

```text
../data/test_on_speed/
└── YYYYMMDD/
    └── HHMMSS_<run_type>/
        ├── summary.json
        ├── per_clip_speed.csv
        ├── per_frame_speed.csv
        ├── videos/
        ├── visualizations/
        └── pred_right_frames/        # 仅 --save_pred_frames 时生成
```

核心文件说明：

| 文件/目录 | 作用 |
|---|---|
| `summary.json` | 总体 FPS、延迟分位数、显存/内存、模型体积、端侧估算 |
| `per_clip_speed.csv` | 每个 clip 的 FPS、平均耗时、输出视频路径 |
| `per_frame_speed.csv` | 每帧 wall/model 耗时，含稳态标记 |
| `videos/` | 左右拼接 stereo 视频；默认左图在左，预测右图在右 |
| `visualizations/` | 抽样可视化：Left、Pred Right、Anaglyph、Diff，可选 GT |
| `pred_right_frames/` | 每帧预测右图；需要 `--save_pred_frames` |

---

### 4. 四个测速入口

#### A. Baseline：原始 TorchScript 推理

脚本：

```bash
python inference/benchmark_speed.py \
  --gpu_id 0 \
  --model ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --data_root ../data/test_set/speed_test \
  --out_root ../data/test_on_speed \
  --fps 25
```

默认输出：

```text
../data/test_on_speed/YYYYMMDD/HHMMSS_baseline/
```

定位：

- 作为所有优化方案的基准线。
- 直接加载发布的 TorchScript 权重。
- CUDA 下使用 FP16 推理；CPU 下使用 FP32。
- 模型前向按 `1280x720`，输出再 resize 到 benchmark 输出尺寸。

---

#### B. JIT 优化：推理态整理

脚本：

```bash
python inference/benchmark_speed_jit_cuda_graph.py \
  --gpu_id 0 \
  --model ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --data_root ../data/test_set/speed_test \
  --out_root ../data/test_on_speed \
  --fps 25
```

默认输出：

```text
../data/test_on_speed/YYYYMMDD/HHMMSS_jit_optimized/
```

可关闭单项优化做消融：

```bash
python inference/benchmark_speed_jit_cuda_graph.py \
  --gpu_id 0 \
  --model ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --data_root ../data/test_set/speed_test \
  --out_root ../data/test_on_speed \
  --no_jit_freeze
```

```bash
python inference/benchmark_speed_jit_cuda_graph.py \
  --gpu_id 0 \
  --model ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --data_root ../data/test_set/speed_test \
  --out_root ../data/test_on_speed \
  --no_jit_optimize
```

优化方法：

JIT 优化的目标是在不改变模型输入尺寸、不牺牲画质的前提下，减少推理执行时的额外开销。它更像是对已训练模型做推理态整理：把运行时不再变化的部分固定下来，把只服务训练或动态执行的冗余步骤尽量去掉，让推理过程更接近稳定的、面向部署的执行路径。

这类优化没有减少画面分辨率和主干计算规模，但它的优势是风险低，输出质量理论上应与 baseline 保持一致，适合作为第一组无损加速实验。对汇报来说，可以把它定位为“工程部署侧的轻量优化”，用于说明在不改画质目标的情况下还能获得多少额外收益。

在固定输入尺寸的测试中，底层计算库也更容易复用稳定的执行策略，因此长序列推理时会更有利。需要注意的是，这一路线主要优化推理执行开销，不解决模型本身计算量大的问题；如果瓶颈集中在高分辨率主干计算上，它的提升幅度会相对有限。

适用结论：

- 属于“无画质损失”的轻量推理优化。
- 收益取决于推理执行开销占比，风险低、易部署。
- 推荐作为 baseline 后的第一组无损优化对照。

---

#### C. Cached Grid：缓存固定坐标

脚本：

```bash
python inference/benchmark_speed_cached_grid.py \
  --gpu_id 0 \
  --model ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --data_root ../data/test_set/speed_test \
  --out_root ../data/test_on_speed \
  --fps 25
```

默认输出：

```text
../data/test_on_speed/YYYYMMDD/HHMMSS_cached_grid/
```

优化方法：

Cached Grid 的优化点来自立体视图生成中的反向映射过程。生成右视图时，需要反复根据视差或运动偏移去采样左视图内容；其中有一部分基础坐标在固定分辨率下是恒定不变的，只有每一帧预测出的偏移量会变化。因此可以把这部分固定坐标提前准备好，推理时只叠加每帧变化的偏移信息，减少重复准备坐标的开销。

这条路线不改变输入分辨率，也不改变最终输出尺寸，因此它属于偏无损的细节开销优化。它的收益取决于反向映射在整体耗时中的占比：如果多阶段推理中频繁做图像重采样，缓存固定坐标就更容易体现价值；如果主要瓶颈在主体计算或视频读写上，提升会更有限。

Cached Grid 更适合固定分辨率、批量连续帧推理的场景。它的限制也很清楚：当输入尺寸变化时，固定坐标需要随分辨率重新准备，因此不适合频繁混用多种模型输入尺寸的测速方式。汇报时可以把它定位为“固定尺寸推理下减少重复工作”的优化路线。

适用结论：

- 属于“无画质损失”的细节开销优化。
- 对固定分辨率推理更友好；如果频繁切换输入尺寸，需要为不同尺寸分别准备固定坐标。
- 适合和 JIT 优化做组合路线的后续探索。

---

#### D. Anchor Skip：截断递归 + 锚帧推理

脚本：

```bash
python inference/benchmark_speed_optimized.py \
  --gpu_id 0 \
  --model ../data/pretrained/deep3d_v1.0_1280x720_cuda.pt \
  --data_root ../data/test_set/speed_test \
  --out_root ../data/test_on_speed \
  --fps 25
```

默认输出：

```text
../data/test_on_speed/YYYYMMDD/HHMMSS_optimized/
```

核心思路：

该路线面向“优先提高 `model_fps`”的免训练加速。原始推理中每一帧都需要上一帧预测右图作为 `x0`，因此形成逐帧自回归依赖，模型必须串行前向。Anchor Skip 先把 `x0` 递归依赖截断：默认用近前帧 `x2 = L(t-1)` 替代历史右视图输入，使锚帧输入可以直接由原始左帧窗口构造。

在当前预训练权重上，直接把输入扩展成 `[N, 18, H, W]` 并喂给 TorchScript 不可行，因为发布的 TorchScript 是按 batch=1 trace 出来的，内部存在固定 batch 维度常量。脚本保留了 `--force_python_batch` 用于实验 Python `Deep3DNet` 动态 batch，但实际验证中 Python 后端吞吐低于原 TorchScript。因此默认后端采用 `torchscript_batch1_anchor`：每隔 `--anchor_stride` 帧只对锚帧做一次原 TorchScript 前向，非锚帧不再调用模型。

默认 `--anchor_stride 6`，也就是每 6 帧只做 1 次模型前向。非锚帧默认用 `--skip_fill shifted_left` 做一个轻量右视图近似：把当前左图水平平移 `--skip_shift_px 8` 像素后作为预测右图。这会牺牲非锚帧的深度/遮挡细节，但能让模型前向总耗时约按 stride 降低，直接提升 `model_fps`。

关键参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--anchor_stride` | `6` | 每 N 帧运行一次模型；越大 `model_fps` 越高，非锚帧近似越多 |
| `--x0_mode` | `x2` | 锚帧中替代递归 `x0` 的来源，可选 `x2` 或 `x3` |
| `--skip_fill` | `shifted_left` | 非锚帧填充方式，可选 `shifted_left/current_left/last_pred` |
| `--skip_shift_px` | `8` | `shifted_left` 的水平平移像素数 |
| `--prewarm_model_iters` | `12` | 正式计时前模型预热，避免首轮 CUDA/JIT/cuDNN 开销污染 `model_fps` |
| `--force_python_batch` | 关闭 | 强制使用 Python `Deep3DNet` 做真实 batch 实验，默认不推荐 |

实测结果：

以 `../data/test_on_speed/20260407/032007_baseline/summary.json` 为 baseline，对比本次 `../data/test_on_speed/20260407/131207_optimized/summary.json`：

| 指标 | Baseline | Anchor Skip | 提升 |
|---|---:|---:|---:|
| `model_fps` | `126.23` | `1075.90` | `8.52x` |
| `steady_model_fps` | `162.19` | `1078.58` | `6.65x` |
| `pipeline_fps` | `11.84` | `17.58` | `1.48x` |
| `steady_pipeline_fps` | `12.10` | `17.60` | `1.45x` |

适用结论：

- 属于“速度优先、有画质取舍”的免训练优化路线。
- 能稳定把 `model_fps` 提升到 6 倍以上，但非锚帧不是完整模型输出。
- 适合用于验证模型前向调用频率下降对 `model_fps` 的上限收益；如果要追求画质，需要调小 `--anchor_stride` 或改进非锚帧补偿策略。

---

### 5. 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--gpu_id` | `0` | GPU 编号；传 `-1` 使用 CPU |
| `--model` | `../data/pretrained/deep3d_v1.0_1280x720_cuda.pt` | TorchScript 权重路径 |
| `--data_root` | `../data/test_set/speed_test` | 连续帧测试集根目录 |
| `--out_root` | `../data/test_on_speed` | 结果总目录 |
| `--run_name` | 自动生成 | 手动指定 `YYYYMMDD` 下的运行目录名 |
| `--fps` | `25.0` | 导出视频帧率，不影响模型推理速度统计 |
| `--alpha` | `5` | 时序远帧偏移 |
| `--model_size` | 从模型名解析 | 模型输入尺寸 |
| `--output_size` | 自动策略 | 最终视频和可视化输出尺寸 |
| `--warmup_frames` | `10` | 稳态统计时忽略每个 clip 前 N 帧 |
| `--sample_vis_per_clip` | `3` | 每个 clip 保存的可视化样张数量 |
| `--save_pred_frames` | 关闭 | 额外保存所有预测右图 |
| `--inv` | 关闭 | 交换 stereo 视频中的左右顺序 |
| `--current_device_tops` | 空 | 传入当前设备 TOPS 后，按线性假设估算 4T 设备 FPS |

---

### 6. 汇报看板指标

建议优先看 `summary.json`：

| 字段 | 汇报含义 |
|---|---|
| `throughput.steady_pipeline_fps` | 端到端稳态 FPS，包含读帧预处理、模型、后处理、写视频 |
| `throughput.steady_model_fps` | 纯模型前向稳态 FPS，更适合比较推理优化 |
| `latency.steady_model.avg_ms/p95_ms/p99_ms` | 模型延迟均值和尾延迟 |
| `memory.gpu.max_reserved_mb` | CUDA allocator 峰值保留显存 |
| `memory.process_peak_mb_after` | benchmark 进程宿主内存峰值 |
| `memory.model_file_mb` | 模型文件体积 |
| `competition_estimate.current_measurement` | 是否达到目标 FPS/内存的当前测量 |
| `competition_estimate.tops_linear_estimate` | 传入 `--current_device_tops` 后的 4T 线性估算 |

对比时建议使用：

```text
Baseline steady_model_fps
→ JIT steady_model_fps
→ Cached Grid steady_model_fps
→ Anchor Skip model_fps / steady_model_fps / visual quality
```

---

### 7. 推荐实验顺序

| Step | 实验 | 目的 |
|---:|---|---|
| 1 | `benchmark_speed.py` | 建立 baseline，确认数据、输出视频和可视化正确 |
| 2 | `benchmark_speed_jit_cuda_graph.py` | 验证无损 JIT 优化收益 |
| 3 | `benchmark_speed_cached_grid.py` | 验证固定分辨率缓存收益 |
| 4 | `benchmark_speed_optimized.py` | 验证截断递归和锚帧推理对 `model_fps` 的上限收益 |

一句话结论：

- Baseline：可复现基准。
- JIT：低风险无损优化。
- Cached Grid：固定尺寸下减少反向映射的重复准备工作。
- Anchor Skip：速度优先，默认每 6 帧仅 1 次模型前向，非锚帧用轻量近似补齐。
