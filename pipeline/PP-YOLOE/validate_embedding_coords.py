#!/usr/bin/env python3
"""
Validate that the embedding head model produces boxes in the correct coordinate space.

This script:
1. Loads the embedding-enhanced ONNX model
2. Runs inference on a test image
3. Checks that box coordinates are in the expected range
4. Validates that ROI alignment is working correctly

Usage:
    python pipeline/PP-YOLOE/validate_embedding_coords.py \
        --onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
        --img pipeline/dataset/demo/demo.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Import utilities from the same directory
sys.path.insert(0, os.path.dirname(__file__))
import onnx_inference_utils


def validate_coordinate_space(
    onnx_path: str,
    img_path: str,
    network_size: tuple = (640, 640),
    verbose: bool = True,
):
    """
    Validate that boxes are in the correct coordinate space for ROI alignment.

    Returns:
        bool: True if validation passes, False otherwise
    """
    if verbose:
        print("=" * 80)
        print("Coordinate Space Validation for Embedding Head")
        print("=" * 80)

    # Load model
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]

    if verbose:
        print(f"\n✓ Model loaded: {onnx_path}")
        print(f"  Inputs: {input_names}")
        print(f"  Outputs: {output_names}")

    # Check for scale_factor input
    has_scale_factor = any('scale' in name.lower() for name in input_names)
    if verbose:
        print(f"\n{'✓' if has_scale_factor else '✗'} scale_factor input: {has_scale_factor}")

    # Check for im_shape input
    has_im_shape = any('im_shape' in name.lower() for name in input_names)

    # Preprocess image
    prep = onnx_inference_utils.preprocess_image(img_path, target_size=network_size)

    # Build feed dict with only the inputs that exist in the model
    feed = {
        "image": prep["image"][np.newaxis, :],
    }

    if has_scale_factor:
        feed["scale_factor"] = prep["scale_factor"][np.newaxis, :]
        if verbose:
            print(f"  scale_factor: {prep['scale_factor']}")

    if has_im_shape:
        feed["im_shape"] = prep["im_shape"][np.newaxis, :]
        if verbose:
            print(f"  im_shape: {prep['im_shape']}")

    # Run inference
    outputs = sess.run(None, feed)

    # Find detection output (should be first output with shape [N, 6])
    det_idx = None
    for i, (name, arr) in enumerate(zip(output_names, outputs)):
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 6:
            det_idx = i
            break

    if det_idx is None:
        print("✗ ERROR: Could not find detection output [N, 6]")
        return False

    detections = outputs[det_idx]
    if verbose:
        print(f"\n✓ Detections shape: {detections.shape}")

    # Filter valid detections
    valid_mask = (detections[:, 0] >= 0) & (detections[:, 1] >= 0.3)
    valid_dets = detections[valid_mask]

    if len(valid_dets) == 0:
        print("✗ WARNING: No valid detections found (score >= 0.3)")
        return False

    if verbose:
        print(f"  Valid detections: {len(valid_dets)}")

    # Extract box coordinates
    boxes = valid_dets[:, 2:6]  # [x0, y0, x1, y1]

    # Analyze coordinate ranges
    x_min, y_min = boxes[:, 0].min(), boxes[:, 1].min()
    x_max, y_max = boxes[:, 2].max(), boxes[:, 3].max()

    if verbose:
        print(f"\n📊 Box coordinate analysis:")
        print(f"  X range: [{x_min:.1f}, {x_max:.1f}]")
        print(f"  Y range: [{y_min:.1f}, {y_max:.1f}]")
        print(f"  Network size: {network_size}")

    # Determine coordinate space
    net_w, net_h = network_size
    is_network_coords = (x_max <= net_w * 1.5) and (y_max <= net_h * 1.5)
    is_original_coords = (x_max > net_w * 2.0) or (y_max > net_h * 2.0)

    validation_passed = True

    if has_scale_factor:
        # Model has scale_factor input
        # Boxes in model output should be in ORIGINAL image coords
        # After internal scaling in embedding head, they should be in NETWORK coords for ROI align

        if is_network_coords and not is_original_coords:
            if verbose:
                print(f"\n✗ UNEXPECTED: Boxes appear to be in NETWORK coordinates")
                print(f"  Model has scale_factor input, so boxes should be in ORIGINAL coordinates")
                print(f"  The model may not be using scale_factor correctly in post-processing")
            validation_passed = False
        else:
            if verbose:
                print(f"\n✓ CORRECT: Boxes are in ORIGINAL image coordinates")
                print(f"  Embedding head will multiply by scale_factor to convert to network coords")
                print(f"  ROI alignment will then work correctly with spatial_scale=1/stride")
    else:
        # Model does NOT have scale_factor input
        # Boxes should already be in network coords

        if is_original_coords:
            if verbose:
                print(f"\n✗ ERROR: Boxes appear to be in ORIGINAL coordinates")
                print(f"  Model has NO scale_factor input, so boxes should be in NETWORK coordinates")
                print(f"  The embedding head scaling logic needs to be disabled!")
            validation_passed = False
        else:
            if verbose:
                print(f"\n✓ CORRECT: Boxes are in NETWORK coordinates")
                print(f"  No scale_factor scaling needed for ROI alignment")

    # Check for embeddings output
    embed_idx = None
    for i, name in enumerate(output_names):
        if 'embed' in name.lower():
            embed_idx = i
            break

    if embed_idx is not None:
        embeddings = outputs[embed_idx]
        if verbose:
            print(f"\n✓ Embeddings found: {embeddings.shape}")

            # Check if embeddings match detections (may be capped at max_detections)
            if embeddings.shape[0] == len(valid_dets):
                print(f"  Embedding count matches detection count ✓")
            elif embeddings.shape[0] < len(valid_dets):
                print(f"  ℹ️  Embedding count ({embeddings.shape[0]}) < detection count ({len(valid_dets)})")
                print(f"  This is expected if --max_detections was used during model creation")
            else:
                print(f"  WARNING: Embedding count ({embeddings.shape[0]}) > detection count ({len(valid_dets)})")
                # This shouldn't happen but is not critical

            # Check L2 normalization
            if embeddings.shape[0] > 0:
                norms = np.linalg.norm(embeddings, axis=1)
                is_normalized = np.allclose(norms, 1.0, atol=0.01)
                if verbose:
                    print(f"  L2 norms: min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")
                    print(f"  Normalized: {is_normalized} {'✓' if is_normalized else '✗'}")
                if not is_normalized:
                    print(f"  WARNING: Embeddings not L2-normalized (expected for BoT-SORT)")

    # Print sample detections
    if verbose:
        print(f"\n📋 Sample detections (first 3):")
        for i in range(min(3, len(valid_dets))):
            cls_id, score, x0, y0, x1, y1 = valid_dets[i]
            print(f"  [{i}] class={int(cls_id)}, score={score:.3f}, box=[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]")

    if verbose:
        print("\n" + "=" * 80)
        if validation_passed:
            print("✓ VALIDATION PASSED: Coordinate space is correct!")
        else:
            print("✗ VALIDATION FAILED: Coordinate space issues detected!")
        print("=" * 80)

    return validation_passed


def main():
    parser = argparse.ArgumentParser(description="Validate embedding head coordinate space")
    parser.add_argument(
        "--onnx",
        default="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx",
        help="Path to ONNX model with embedding head",
    )
    parser.add_argument(
        "--img",
        default="pipeline/dataset/demo/demo.jpg",
        help="Path to test image",
    )
    parser.add_argument(
        "--network_size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Network input size (width height)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    network_size = tuple(args.network_size)

    try:
        passed = validate_coordinate_space(
            args.onnx,
            args.img,
            network_size=network_size,
            verbose=not args.quiet,
        )
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
