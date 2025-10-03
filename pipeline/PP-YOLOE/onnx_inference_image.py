#!/usr/bin/env python3
"""
Run inference on a single image using ONNX Runtime with the exported PP-YOLOE Human model.

This script now focuses on orchestration only; model-specific logic lives in:
  - onnx_inference_utils.py
  - onnx_inference_ane_model.py
  - onnx_inference_original_model.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

from onnx_inference_ane_model import run_ane_inference
from onnx_inference_original_model import run_original_inference
from onnx_inference_utils import (
    default_paths,
    detect_model_type,
    draw_and_save_with_ids,
    get_coreml_version_info,
    get_hardcoded_preprocess,
    preprocess_image,
)


def get_session(onnx_path: str):
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - informative exit
        print(
            "[ERROR] onnxruntime not installed. Install with: pip install onnxruntime",
            file=sys.stderr,
        )
        raise exc

    available = ort.get_available_providers()
    providers = None
    if "CoreMLExecutionProvider" in available:
        coreml_opts = {
            "ModelFormat": "MLProgram",
            "EnableOnSubgraphs": "1",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "1",
        }
        providers = [("CoreMLExecutionProvider", coreml_opts), "CPUExecutionProvider"]
        print("[INFO] Using CoreMLExecutionProvider (Apple Core ML) with options:", coreml_opts)
        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        except Exception as exc:  # pragma: no cover - informative exit
            print("[ERROR] Failed to initialize CoreMLExecutionProvider session:", exc, file=sys.stderr)
            raise
    else:
        print(f"[WARN] CoreMLExecutionProvider not available. Using default providers: {available}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess


def build_feed_dict(sess, inputs_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    feed = {}
    input_names = [i.name for i in sess.get_inputs()]
    for name in input_names:
        if name == "image":
            feed[name] = inputs_map["image"][None, :]
        elif name in inputs_map:
            feed[name] = inputs_map[name][None, :]
        else:
            print(f'[WARN] Model input "{name}" not found in preprocessed data', file=sys.stderr)
    return feed


def print_environment_summary(sess, onnx_path: str):
    try:
        import onnxruntime as ort

        ort_ver = getattr(ort, "__version__", "unknown")
        available = ort.get_available_providers()
    except Exception:
        ort_ver = "unknown"
        available = []

    print("\n[Env] onnxruntime:", ort_ver)
    if available:
        print("      available providers:", available)
    try:
        print("      session providers:", sess.get_providers())
    except Exception:
        pass

    cm = get_coreml_version_info()
    print("[Env] CoreML:", end=" ")
    cmtools = cm.get("coremltools")
    print(f"coremltools={cmtools if cmtools else 'not installed'}", end="; ")
    fw = cm.get("framework")
    if fw:
        short_ver = fw.get("CFBundleShortVersionString") or "unknown"
        bundle_ver = fw.get("CFBundleVersion") or "unknown"
        print(f"framework={short_ver} (bundle {bundle_ver})")
    else:
        print("framework version: unknown")
    if cm.get("macOS"):
        print(f"[Env] macOS: {cm['macOS']}")


# pylint: disable=too-many-locals

def main():
    default_onnx, default_img, default_out = default_paths()

    parser = argparse.ArgumentParser(description="ONNX Runtime inference for PP-YOLOE Human on one image")
    parser.add_argument("--img", default=default_img, help="Path to input image")
    parser.add_argument("--onnx", default=default_onnx, help="Path to ONNX model file")
    parser.add_argument("--out", default=default_out, help="Directory to save visualization")
    parser.add_argument("--thresh", type=float, default=None, help="Score threshold for printing/drawing (default: 0.5)")
    args = parser.parse_args()

    for path, label in ((args.img, "Input image"), (args.onnx, "ONNX model")):
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    draw_threshold, _, label_name = get_hardcoded_preprocess()
    if args.thresh is not None:
        draw_threshold = args.thresh

    sess = get_session(args.onnx)
    print_environment_summary(sess, args.onnx)

    inputs_map = preprocess_image(args.img, target_size=(640, 640), keep_ratio=False)
    feed = build_feed_dict(sess, inputs_map)
    input_names = [i.name for i in sess.get_inputs()]

    print("\n[C++ Porting Info] Model and preprocessing details:")
    print(f"  • Input image size: {args.img} -> resized to 640x640 (keep_ratio=False)")
    print("    ↳ Why keep_ratio=False? Model was trained this way, handles distortion well for humans")
    print("  • Normalization: RGB values /255.0, then (x - mean) / std")
    print("    - mean = [0.485, 0.456, 0.406]")
    print("    - std = [0.229, 0.224, 0.225]")
    print("  • Channel order: RGB (not BGR)")
    print("  • Input tensor: (1, 3, 640, 640) NCHW format, float32")
    print(f"  • Input tensor name: \"{input_names[0] if input_names else 'unknown'}\"")
    print(f"  • Score threshold: {draw_threshold} (filter detections below this)")
    print("  • Post-processing: L2-normalize embeddings, filter detections by score")
    print("  • No dependency on PaddleDetection - standalone preprocessing implementation")

    out_names = [o.name for o in sess.get_outputs()]
    model_type = detect_model_type(out_names)
    if model_type == "original":
        print("\n[INFO] Detected model type: original (with built-in NMS)")
    elif model_type == "ane":
        print("\n[INFO] Detected model type: ANE-optimized (automatic surgery)")
    else:
        print("\n[WARN] Unable to confidently detect model type from outputs; attempting ANE pipeline by default")
        model_type = "ane"

    warmup_runs = 3
    if model_type == "original":
        boxes_valid, embs_valid, benchmark = run_original_inference(
            sess,
            feed,
            draw_threshold,
            warmup_runs=warmup_runs,
        )
    else:
        boxes_valid, embs_valid, benchmark = run_ane_inference(
            sess,
            feed,
            args.img,
            draw_threshold,
            warmup_runs=warmup_runs,
        )
        model_type = "ane"

    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.img))[0]
    vis_path = os.path.join(args.out, f"{base}.jpg")
    try:
        ids_valid = np.arange(boxes_valid.shape[0])
        draw_and_save_with_ids(
            args.img,
            boxes_valid,
            ids_valid,
            float(draw_threshold),
            vis_path,
            label=label_name,
        )
        print("Saved visualization to:", vis_path)
    except Exception as exc:  # pragma: no cover - visualization best effort
        print("[WARN] Failed to save visualization:", exc)

    print(f"\n[Summary] detections kept: {boxes_valid.shape[0]} (threshold {draw_threshold})")
    print(f"[Summary] embeddings shape: {embs_valid.shape}")
    print(f"[Summary] model type: {model_type}")
    if benchmark:
        inference_ms = benchmark.get("inference_ms")
        if inference_ms is not None:
            print(f"[Timing] onnxruntime sess.run (after warmup): {inference_ms:.2f} ms")

    print(f"\n✅ ONNX inference completed! Generated visualization: {vis_path}")


if __name__ == "__main__":
    main()
