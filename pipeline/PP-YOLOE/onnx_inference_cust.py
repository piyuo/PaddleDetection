#!/usr/bin/env python3
"""Inference helper for customized/pruned PP-YOLOE ONNX models."""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from onnx_inference_utils import preprocess_image, draw_and_save
from onnx_inference_ane_model import run_ane_inference


def main():
    parser = argparse.ArgumentParser(description="Run inference on customized/pruned PP-YOLOE ONNX model")
    parser.add_argument(
        "--img",
        default="pipeline/dataset/demo/demo.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--onnx",
        default="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx",
        help="Path to customized ONNX model",
    )
    parser.add_argument(
        "--out",
        default="pipeline/output",
        help="Directory to save visualization",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--network-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Network input size (width height)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs before timing (passed to ANE pipeline)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.img):
        print(f"❌ Error: Image not found: {args.img}")
        sys.exit(1)
    if not os.path.exists(args.onnx):
        print(f"❌ Error: ONNX model not found: {args.onnx}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print("=" * 80)
    print("🚀 PP-YOLOE Pruned Model Inference")
    print("=" * 80)
    print(f"📸 Image: {args.img}")
    print(f"🤖 Model: {args.onnx}")
    print(f"📊 Threshold: {args.thresh}")
    print(f"📐 Network size: {args.network_size}")
    print()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")
    print()

    prep = preprocess_image(args.img, target_size=tuple(args.network_size))
    feed = {"image": prep["image"][np.newaxis, :]}
    if "scale_factor" in input_names:
        feed["scale_factor"] = prep["scale_factor"][np.newaxis, :]
    if "im_shape" in input_names:
        feed["im_shape"] = prep["im_shape"][np.newaxis, :]

    boxes_valid, embs_valid, benchmark = run_ane_inference(
        sess,
        feed,
        args.img,
        float(args.thresh),
        warmup_runs=max(args.warmup, 0),
    )

    detections = boxes_valid
    if detections.size == 0:
        print("⚠️  No detections above threshold. Visualization will still be produced.")

    out_path = os.path.join(args.out, f"{Path(args.img).stem}_cust.jpg")

    # Draw detections on the original image.
    draw_and_save(args.img, detections, float(args.thresh), out_path, label="person")
    print(f"✓ Saved visualization to: {out_path}")
    print()

    print("Benchmark summary:")
    for key, value in benchmark.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    if embs_valid.size > 0:
        print(f"Embeddings available: {embs_valid.shape}")
    else:
        print("Embeddings: none or zero detections")

    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
