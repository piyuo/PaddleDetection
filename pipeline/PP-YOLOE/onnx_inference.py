#!/usr/bin/env python3
"""
Simple PP-YOLOE ONNX inference demo with embedding support.

This script demonstrates:
- Loading and running PP-YOLOE ONNX models
- Preprocessing images for inference
- Extracting detections and embeddings
- Visualizing results

Usage:
    python3 pipeline/PP-YOLOE/onnx_inference.py \
        --img pipeline/dataset/demo/demo.jpg \
        --onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
        --out pipeline/output \
        --thresh 0.5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
import onnx_inference_utils


def main():
    parser = argparse.ArgumentParser(description="Simple ONNX inference demo")
    parser.add_argument(
        "--img",
        default="pipeline/dataset/demo/demo.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--onnx",
        default="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--out",
        default="pipeline/output",
        help="Output directory for visualizations",
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
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.img):
        print(f"❌ Error: Image not found: {args.img}")
        sys.exit(1)
    if not os.path.exists(args.onnx):
        print(f"❌ Error: ONNX model not found: {args.onnx}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print("=" * 80)
    print("🚀 PP-YOLOE ONNX Inference Demo")
    print("=" * 80)
    print(f"📸 Image: {args.img}")
    print(f"🤖 Model: {args.onnx}")
    print(f"📊 Threshold: {args.thresh}")
    print(f"📐 Network size: {args.network_size}")
    print()

    # Load ONNX model
    print("⏳ Loading ONNX model...")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    load_time = (time.perf_counter() - t0) * 1000
    print(f"✓ Model loaded in {load_time:.1f}ms")

    # Get model info
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")

    has_embeddings = any("embed" in name.lower() for name in output_names)
    print(f"  Embeddings: {'✓ Yes' if has_embeddings else '✗ No'}")
    print()

    # Preprocess image
    print("🔧 Preprocessing image...")
    network_size = tuple(args.network_size)
    prep = onnx_inference_utils.preprocess_image(args.img, target_size=network_size)

    # Build feed dict
    feed = {"image": prep["image"][np.newaxis, :]}
    if "scale_factor" in input_names:
        feed["scale_factor"] = prep["scale_factor"][np.newaxis, :]
    if "im_shape" in input_names:
        feed["im_shape"] = prep["im_shape"][np.newaxis, :]

    print(f"  Image shape after preprocessing: {prep['image'].shape}")
    if "scale_factor" in feed:
        print(f"  Scale factor: {prep['scale_factor']}")
    print()

    # Run inference
    print("🔮 Running inference...")
    t0 = time.perf_counter()
    outputs = sess.run(None, feed)
    inference_time = (time.perf_counter() - t0) * 1000
    print(f"✓ Inference completed in {inference_time:.1f}ms")
    print()

    # Parse outputs
    print("📦 Model outputs:")
    detections = None
    embeddings = None

    for i, (name, arr) in enumerate(zip(output_names, outputs)):
        print(f"  [{i}] {name}: {arr.shape}")

        # Find detections (N, 6) array
        if (
            isinstance(arr, np.ndarray)
            and arr.ndim == 2
            and arr.shape[1] >= 6
            and detections is None
        ):
            detections = arr
            print(f"      → Detections: [class_id, score, x0, y0, x1, y1]")

        # Find embeddings
        if "embed" in name.lower():
            embeddings = arr
            print(f"      → Embeddings: per-detection features")

    if detections is None:
        print("❌ Error: Could not find detection output")
        sys.exit(1)

    print()

    # Filter valid detections
    valid_mask = (detections[:, 0] >= 0) & (detections[:, 1] >= args.thresh)
    valid_dets = detections[valid_mask]

    print(f"📊 Detection results:")
    print(f"  Total detections: {len(detections)}")
    print(f"  Valid detections (score ≥ {args.thresh}): {len(valid_dets)}")

    if embeddings is not None:
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Embedding dimension: {embeddings.shape[1] if embeddings.ndim == 2 else 0}")
    print()

    # Print detection details
    if len(valid_dets) > 0:
        print("🎯 Detections:")
        for i, det in enumerate(valid_dets[:10]):  # Show first 10
            cls_id, score, x0, y0, x1, y1 = det[:6]
            w, h = x1 - x0, y1 - y0
            print(f"  [{i}] class={int(cls_id)}, score={score:.3f}, "
                  f"box=[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}], "
                  f"size={w:.1f}×{h:.1f}")

        if len(valid_dets) > 10:
            print(f"  ... and {len(valid_dets) - 10} more")
        print()

        # Show embedding statistics if available
        if embeddings is not None and len(embeddings) > 0:
            norms = np.linalg.norm(embeddings, axis=1)
            print("📐 Embedding statistics:")
            print(f"  L2 norms: min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")

            if len(embeddings) >= 2:
                # Compute pairwise cosine similarities
                cos_sim = embeddings @ embeddings.T
                # Get off-diagonal elements
                triu_indices = np.triu_indices(len(embeddings), k=1)
                pairwise_cos = cos_sim[triu_indices]

                if len(pairwise_cos) > 0:
                    print(f"  Pairwise cosine similarity:")
                    print(f"    min={pairwise_cos.min():.3f}, "
                          f"median={np.median(pairwise_cos):.3f}, "
                          f"max={pairwise_cos.max():.3f}")
            print()
    else:
        print(f"⚠️  No detections above threshold {args.thresh}")
        print(f"💡 Try lowering --thresh (e.g., 0.3)")
        print()

    # Visualize results
    print("🎨 Creating visualization...")
    out_path = os.path.join(args.out, f"{Path(args.img).stem}.jpg")

    # Draw boxes on image
    im = cv2.imread(args.img)
    if im is None:
        print(f"⚠️  Could not load image for visualization: {args.img}")
    else:
        for det in valid_dets:
            cls_id, score, x0, y0, x1, y1 = det[:6]

            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(
                im,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                color,
                2,
            )

            # Draw label
            label = f"person:{score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                im,
                (int(x0), int(y0) - label_size[1] - 4),
                (int(x0) + label_size[0], int(y0)),
                color,
                -1,
            )
            cv2.putText(
                im,
                label,
                (int(x0), int(y0) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(out_path, im)
        print(f"✓ Saved visualization to: {out_path}")

    print()
    print("=" * 80)
    print("✅ Inference complete!")
    print("=" * 80)
    print(f"⏱️  Total time: {inference_time:.1f}ms")
    print(f"📁 Output: {out_path}")

    if has_embeddings and len(valid_dets) > 0:
        print()
        print("💡 This model includes embeddings for tracking!")
        print("   Use with BoT-SORT for multi-object tracking with re-identification.")
    elif not has_embeddings:
        print()
        print("ℹ️  This model does NOT include embeddings (detection only).")
        print("   To add embeddings for tracking, run:")
        print("   ./pipeline/PP-YOLOE/insert_embedding_head.sh")


if __name__ == "__main__":
    main()
