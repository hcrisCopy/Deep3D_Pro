"""
Inference script for author-provided MXNet .params weights.

This script is designed for params like:
    /root/autodl-tmp/deep3d/Deep3D_Pro/exp/deep3d-0050.params

Example:
    python inference_with_params.py \
        --params /root/autodl-tmp/deep3d/Deep3D_Pro/exp/deep3d-0050.params \
        --input demo.jpg \
        --output_dir output
"""

import argparse
import os
import sys
import logging

import cv2
import numpy as np

try:
    import mxnet as mx
except ImportError as exc:
    raise SystemExit(
        "mxnet is required for .params inference. Install it first, e.g. `pip install mxnet-cu121` or `pip install mxnet`."
    ) from exc


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sym import make_upsample_sym


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using MXNet .params weights")
    parser.add_argument(
        "--params",
        type=str,
        default="/root/autodl-tmp/deep3d/Deep3D_Pro/exp/deep3d-0050.params",
        help="Path to .params file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path or directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_params",
        help="Output directory",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id. Use -1 for CPU",
    )
    parser.add_argument(
        "--data_shape",
        type=int,
        nargs=2,
        default=[384, 160],
        help="Model input size: width height (recommended: 384 160)",
    )
    parser.add_argument(
        "--scale_min",
        type=int,
        default=-15,
        help="Minimum disparity",
    )
    parser.add_argument(
        "--scale_max",
        type=int,
        default=17,
        help="Maximum disparity",
    )
    parser.add_argument(
        "--left_mean_npz",
        type=str,
        default=None,
        help="Optional .npz containing key 'left' for mean subtraction",
    )
    parser.add_argument(
        "--save_right",
        action="store_true",
        default=True,
        help="Save predicted right-view image",
    )
    parser.add_argument(
        "--save_anaglyph",
        action="store_true",
        default=True,
        help="Save anaglyph image",
    )
    parser.add_argument(
        "--save_sbs",
        action="store_true",
        default=True,
        help="Save side-by-side image",
    )
    return parser.parse_args()


def choose_ctx(gpu_id):
    if gpu_id >= 0:
        try:
            _ = mx.nd.array([0], ctx=mx.gpu(gpu_id))
            return mx.gpu(gpu_id)
        except mx.base.MXNetError:
            logging.warning("GPU %d not available, fallback to CPU", gpu_id)
    return mx.cpu()


def load_params(params_path):
    if not os.path.isfile(params_path):
        raise FileNotFoundError("Params file not found: %s" % params_path)

    raw = mx.nd.load(params_path)
    arg_params = {}
    aux_params = {}

    for key, value in raw.items():
        if key.startswith("arg:"):
            arg_params[key[4:]] = value
        elif key.startswith("aux:"):
            aux_params[key[4:]] = value

    if not arg_params and not aux_params:
        raise RuntimeError(
            "Failed to parse params file. Expected keys prefixed with 'arg:'/'aux:'."
        )

    return arg_params, aux_params


def build_module(ctx, data_shape, scale):
    width, height = data_shape
    if (width, height) != (384, 160):
        logging.warning(
            "Current symbol has fixed FC shape for 384x160. Using other sizes may fail or produce incorrect results."
        )

    # Old symbols use CuDNNBatchNorm. Map to standard BatchNorm when unavailable.
    if not hasattr(mx.sym, "CuDNNBatchNorm"):
        mx.sym.CuDNNBatchNorm = mx.sym.BatchNorm

    left = mx.sym.Variable(name="left")
    logits, _ = make_upsample_sym(left, scale=scale, fuse="sum", method="multi2")
    sym = mx.sym.SoftmaxActivation(data=logits, mode="channel", name="softmax")

    mod = mx.mod.Module(
        symbol=sym,
        context=ctx,
        data_names=["left"],
        label_names=None,
    )
    mod.bind(
        for_training=False,
        data_shapes=[
            ("left", (1, 3, height, width)),
        ],
        label_shapes=None,
    )
    return mod


def shift_image_chw(img_chw, shift):
    """Shift CHW image horizontally with edge clamping."""
    width = img_chw.shape[2]
    idx = np.arange(width) - int(shift)
    idx = np.clip(idx, 0, width - 1)
    return img_chw[:, :, idx]


def depth_dot_numpy(prob, left_img_chw, scale):
    """Reconstruct right view from depth probability map (NumPy version)."""
    s0, s1 = scale
    output = np.zeros_like(left_img_chw, dtype=np.float32)
    for disp in range(s0, s1):
        d_idx = disp - s0 + 1  # Channel 0 is unused in original implementation.
        if d_idx < 0 or d_idx >= prob.shape[0]:
            continue
        weight = prob[d_idx][np.newaxis, :, :]
        shifted = shift_image_chw(left_img_chw, disp)
        output += weight * shifted
    return output


def load_left_mean(mean_path, data_shape):
    width, height = data_shape
    default_mean = np.zeros((3, height, width), dtype=np.float32)

    if mean_path is None:
        return default_mean
    if not os.path.isfile(mean_path):
        logging.warning("Mean file not found: %s. Use zero mean.", mean_path)
        return default_mean

    data = np.load(mean_path)
    if "left" not in data:
        logging.warning("No 'left' key in %s. Use zero mean.", mean_path)
        return default_mean

    left_mean = data["left"].astype(np.float32)
    if left_mean.shape != (3, height, width):
        logging.warning(
            "left mean shape mismatch: %s vs expected %s. Use zero mean.",
            left_mean.shape,
            (3, height, width),
        )
        return default_mean

    return left_mean


def anaglyph(left_bgr, right_bgr):
    out = np.zeros_like(left_bgr)
    out[:, :, :2] = right_bgr[:, :, :2]
    out[:, :, 2:] = left_bgr[:, :, 2:]
    return out


def sbs(left_bgr, right_bgr):
    out = np.zeros_like(left_bgr)
    sep = left_bgr.shape[1] // 2
    out[:, :sep] = cv2.resize(left_bgr, (sep, left_bgr.shape[0]))
    out[:, sep:] = cv2.resize(right_bgr, (sep, right_bgr.shape[0]))
    return out


def run_inference_on_image(mod, img_bgr, data_shape, left_mean, scale):
    width, height = data_shape
    left0 = cv2.resize(img_bgr, (width, height)).astype(np.float32)

    left0_chw = left0.transpose(2, 0, 1)
    left_chw = left0_chw - left_mean

    batch = mx.io.DataBatch(
        data=[mx.nd.array(left_chw[np.newaxis, ...])],
        label=None,
    )
    mod.forward(batch, is_train=False)
    prob = mod.get_outputs()[0].asnumpy()[0]  # (D, H, W)
    pred = depth_dot_numpy(prob, left0_chw, scale=scale)
    pred = np.clip(pred, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return pred


def collect_images(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError("Input not found: %s" % input_path)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    names = sorted(os.listdir(input_path))
    files = [
        os.path.join(input_path, name)
        for name in names
        if os.path.splitext(name)[1].lower() in exts
    ]
    if not files:
        raise RuntimeError("No image files found in directory: %s" % input_path)
    return files


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    data_shape = tuple(args.data_shape)
    scale = (args.scale_min, args.scale_max)

    ctx = choose_ctx(args.gpu)
    logging.info("Using context: %s", ctx)

    mod = build_module(ctx=ctx, data_shape=data_shape, scale=scale)
    arg_params, aux_params = load_params(args.params)
    mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    logging.info("Loaded params: %s", args.params)

    left_mean = load_left_mean(args.left_mean_npz, data_shape)
    if args.left_mean_npz:
        logging.info("left mean path: %s", args.left_mean_npz)

    image_paths = collect_images(args.input)
    logging.info("Found %d image(s)", len(image_paths))

    for idx, image_path in enumerate(image_paths, start=1):
        img = cv2.imread(image_path)
        if img is None:
            logging.warning("Skip unreadable image: %s", image_path)
            continue

        pred_right_model = run_inference_on_image(
            mod=mod,
            img_bgr=img,
            data_shape=data_shape,
            left_mean=left_mean,
            scale=scale,
        )

        # Keep final outputs at the original input resolution.
        orig_h, orig_w = img.shape[:2]
        pred_right = cv2.resize(pred_right_model, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        if args.save_right:
            cv2.imwrite(os.path.join(args.output_dir, base_name + "_right.png"), pred_right)
        if args.save_anaglyph:
            cv2.imwrite(
                os.path.join(args.output_dir, base_name + "_anaglyph.png"),
                anaglyph(img, pred_right),
            )
        if args.save_sbs:
            cv2.imwrite(
                os.path.join(args.output_dir, base_name + "_sbs.png"),
                sbs(img, pred_right),
            )

        logging.info("[%d/%d] done: %s", idx, len(image_paths), image_path)

    logging.info("All done. Outputs: %s", args.output_dir)


if __name__ == "__main__":
    main()
