## Deep3D_Pro Speed Benchmark

### 1. 目标

- 面向连续帧目录测试 Deep3D_Pro 推理速度、延迟、显存/内存和输出质量。
- 对比 4 条测速路线：Baseline、Fast Mode、JIT 优化、Cached Grid 优化。
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
- 如果未显式传入 `--output_size`，baseline/JIT/cached-grid 会在输入不是 1080p 时按 `1920x1080` 进行模拟测速。
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
        ├── pred_right_frames/        # 仅 --save_pred_frames 时生成
        └── exported_models/          # 仅 fast mode 默认生成
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
| `exported_models/` | Fast Mode 重新导出的低分辨率 TorchScript 模型 |

---


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
