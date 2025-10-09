#!/usr/bin/env python3
"""NCNN inference for PP-YOLOE Human model with embedding head.

This script runs inference on a single image using the NCNN runtime.
It works with models that have embeddings directly added by insert_embedding_head.py.

The model should have these outputs:
- out0: boxes (N, 4) - detection boxes in xyxy format
- out1: scores (N, 1) or (N,) - detection confidence scores
- out2: embed (N, D) - L2-normalized appearance embeddings for BoT-SORT

Usage:
    python3 pipeline/PP-YOLOE/ncnn_inference_embed.py \\
        --img pipeline/dataset/demo/demo.jpg \\
        --ncnn_param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_ncnn.param \\
        --ncnn_bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_ncnn.bin \\
        --out pipeline/output \\
        --thresh 0.5
"""
import argparse
import os
import sys
import numpy as np
import cv2

try:
    import ncnn
except ImportError:
    print("ERROR: ncnn not installed. Install with: pip install ncnn", file=sys.stderr)
    sys.exit(1)

# Constants
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_SIZE = (640, 640)


def validate_embedding_for_botsort(embeddings: np.ndarray, boxes: np.ndarray, verbose: bool = True):
    """Validate embedding output for BoT-SORT compatibility.

    Checks:
    1. Embeddings are L2-normalized (norms should be ~1.0)
    2. Pairwise cosine similarities are reasonable (not all identical)
    3. Embeddings have sufficient separation for tracking

    Args:
        embeddings: (N, D) array of embeddings
        boxes: (N, 6) array of detections (class_id, score, x0, y0, x1, y1)
        verbose: Print detailed validation report

    Returns:
        Dict with validation results and statistics
    """
    if embeddings.shape[0] == 0:
        return {
            'valid': True,
            'num_detections': 0,
            'message': 'No detections to validate'
        }

    N, D = embeddings.shape

    # Check L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    norm_mean = norms.mean()
    norm_std = norms.std()
    is_normalized = np.allclose(norms, 1.0, atol=0.01)

    # Compute pairwise cosine similarities
    cosine_sims = []
    ious = []
    if N > 1:
        # Compute pairwise cosines
        for i in range(N):
            for j in range(i + 1, N):
                cos_sim = np.dot(embeddings[i], embeddings[j])
                cosine_sims.append(cos_sim)

                # Compute IoU for this pair
                x0_i, y0_i, x1_i, y1_i = boxes[i, 2:6]
                x0_j, y0_j, x1_j, y1_j = boxes[j, 2:6]

                x0_inter = max(x0_i, x0_j)
                y0_inter = max(y0_i, y0_j)
                x1_inter = min(x1_i, x1_j)
                y1_inter = min(y1_i, y1_j)

                if x1_inter > x0_inter and y1_inter > y0_inter:
                    inter_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)
                    area_i = (x1_i - x0_i) * (y1_i - y0_i)
                    area_j = (x1_j - x0_j) * (y1_j - y0_j)
                    union_area = area_i + area_j - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                else:
                    iou = 0.0

                ious.append(iou)

        cosine_sims = np.array(cosine_sims)
        ious = np.array(ious)

        # Separate high-IoU (likely same person) vs low-IoU (different people)
        high_iou_mask = ious > 0.3
        low_iou_mask = ious <= 0.3

        cos_high_iou = cosine_sims[high_iou_mask] if high_iou_mask.any() else np.array([])
        cos_low_iou = cosine_sims[low_iou_mask] if low_iou_mask.any() else np.array([])
    else:
        cosine_sims = np.array([])
        ious = np.array([])
        cos_high_iou = np.array([])
        cos_low_iou = np.array([])

    # Quality checks
    checks = {
        'l2_normalized': is_normalized,
        'has_variation': len(cosine_sims) == 0 or cosine_sims.std() > 0.01,
        'reasonable_spread': len(cosine_sims) == 0 or (cosine_sims.min() < 0.8 and cosine_sims.max() < 0.95),
    }

    is_valid = all(checks.values())

    result = {
        'valid': is_valid,
        'num_detections': N,
        'embedding_dim': D,
        'checks': checks,
        'norm_mean': float(norm_mean),
        'norm_std': float(norm_std),
        'norm_min': float(norms.min()),
        'norm_max': float(norms.max()),
    }

    if len(cosine_sims) > 0:
        result.update({
            'cosine_mean': float(cosine_sims.mean()),
            'cosine_std': float(cosine_sims.std()),
            'cosine_min': float(cosine_sims.min()),
            'cosine_max': float(cosine_sims.max()),
            'cosine_median': float(np.median(cosine_sims)),
            'cosine_p95': float(np.percentile(cosine_sims, 95)),
        })

        if len(cos_high_iou) > 0:
            result['cosine_high_iou_mean'] = float(cos_high_iou.mean())
            result['cosine_high_iou_max'] = float(cos_high_iou.max())

        if len(cos_low_iou) > 0:
            result['cosine_low_iou_mean'] = float(cos_low_iou.mean())
            result['cosine_low_iou_max'] = float(cos_low_iou.max())

    if verbose and N > 0:
        print("\n" + "="*60)
        print("=== Embedding Validation for BoT-SORT ===")
        print("="*60)
        print(f"Detections: {N}")
        print(f"Embedding dimension: {D}")
        print(f"\nL2 Normalization:")
        print(f"  Norms: mean={norm_mean:.4f}, std={norm_std:.4f}, range=[{norms.min():.4f}, {norms.max():.4f}]")
        print(f"  {'✓' if checks['l2_normalized'] else '✗'} Properly normalized: {checks['l2_normalized']}")

        if len(cosine_sims) > 0:
            print(f"\nPairwise Cosine Similarities ({len(cosine_sims)} pairs):")
            print(f"  Mean:   {cosine_sims.mean():.4f}")
            print(f"  Std:    {cosine_sims.std():.4f}")
            print(f"  Min:    {cosine_sims.min():.4f}")
            print(f"  Max:    {cosine_sims.max():.4f}")
            print(f"  Median: {np.median(cosine_sims):.4f}")
            print(f"  P95:    {np.percentile(cosine_sims, 95):.4f}")

            if len(cos_high_iou) > 0:
                print(f"\n  High IoU pairs (>0.3, likely same person): {len(cos_high_iou)}")
                print(f"    Cosine mean: {cos_high_iou.mean():.4f}, max: {cos_high_iou.max():.4f}")

            if len(cos_low_iou) > 0:
                print(f"  Low IoU pairs (≤0.3, different people): {len(cos_low_iou)}")
                print(f"    Cosine mean: {cos_low_iou.mean():.4f}, max: {cos_low_iou.max():.4f}")

            print(f"\n  {'✓' if checks['has_variation'] else '✗'} Has variation: {checks['has_variation']}")
            print(f"  {'✓' if checks['reasonable_spread'] else '✗'} Reasonable spread: {checks['reasonable_spread']}")

        print(f"\n{'✅' if is_valid else '⚠️ '} Overall: {'VALID for BoT-SORT' if is_valid else 'Issues detected'}")

        # BoT-SORT recommendations
        print("\n=== BoT-SORT Integration Guide ===")
        print("Recommended settings:")
        if len(cos_low_iou) > 0:
            max_low_iou_cos = cos_low_iou.max()
            recommended_threshold = min(0.60, max_low_iou_cos + 0.10)
            print(f"  • Cosine similarity threshold: {recommended_threshold:.2f}")
            print(f"    (If using cosine distance, set max_dist = {1.0 - recommended_threshold:.2f})")
        else:
            print(f"  • Cosine similarity threshold: 0.55-0.60")
            print(f"    (If using cosine distance, set max_dist = 0.40-0.45)")
        print(f"  • Keep IoU gating enabled (IoU threshold ≥ 0.2-0.3)")
        print(f"  • Feature history (nn_budget): 50-100")
        print(f"  • Use EMA smoothing if available")
        print("="*60)

    return result


def load_and_preprocess(img_path, target_size=DEFAULT_SIZE):
    """Load image and create NCNN Mat with normalization."""
    with open(img_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Store original size for scaling
    orig_h, orig_w = im.shape[:2]

    # Resize to model input size
    im_resized = cv2.resize(im, target_size)

    # Create NCNN Mat (BGR to RGB)
    mat = ncnn.Mat.from_pixels(im_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, target_size[0], target_size[1])

    # Apply normalization: (x/255 - mean) / std
    mean_vals = [m * 255.0 for m in MEAN]
    norm_vals = [1.0 / (s * 255.0) for s in STD]
    mat.substract_mean_normalize(mean_vals, norm_vals)

    return mat, (orig_w, orig_h)


def run_inference(net, img_mat, input_names=None, output_names=None):
    """Run NCNN inference and extract outputs.

    Args:
        net: NCNN network
        img_mat: Input image matrix
        input_names: Names of input layers (default: ["in0", "in1"])
        output_names: Names of output layers (default: ["out0", "out1", "out2"])
                     out0=boxes, out1=scores, out2=embed

    Returns:
        Dict of output tensors
    """
    if input_names is None:
        input_names = ["in0", "in1"]
    if output_names is None:
        output_names = ["out0", "out1", "out2"]

    # Create scale_factor input (required by PP-YOLOE)
    scale_factor = ncnn.Mat(np.array([1.0, 1.0], dtype=np.float32))

    # Create extractor and set inputs
    ex = net.create_extractor()
    if "in0" in input_names:
        ex.input("in0", scale_factor)
    if "in1" in input_names:
        ex.input("in1", img_mat)
    else:
        # Fallback: use the image input name
        ex.input(input_names[0], img_mat)

    # Extract standard outputs (detections)
    outputs = {}
    for name in output_names:
        ret, out_mat = ex.extract(name)
        if ret == 0:
            outputs[name] = out_mat.numpy()
        else:
            print(f"[WARN] Failed to extract output: {name}")

    return outputs


def apply_nms(boxes_raw, scores_raw, score_thresh=0.5, nms_thresh=0.5):
    """Apply NMS to raw detection outputs (matching onnx_inference_ane_model.py logic)."""
    # Ensure scores are 1D
    if scores_raw.ndim == 2 and scores_raw.shape[1] == 1:
        person_scores = scores_raw[:, 0]
    elif scores_raw.ndim == 2:
        person_scores = scores_raw[:, 0]
    else:
        person_scores = scores_raw

    # Extract box coordinates
    x0, y0, x1, y1 = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    w = x1 - x0
    h = y1 - y0
    nms_boxes = np.column_stack([x0, y0, w, h]).tolist()

    # Apply NMS (same as ANE version)
    score_threshold = float(score_thresh)
    nms_threshold = float(nms_thresh)
    selected_indices = cv2.dnn.NMSBoxes(nms_boxes, person_scores.tolist(), score_threshold, nms_threshold)

    if len(selected_indices) > 0:
        selected_indices = np.array(selected_indices).flatten()
        boxes_nms = boxes_raw[selected_indices]
        scores_nms = person_scores[selected_indices]
        class_ids = np.zeros_like(scores_nms)
        # Format as (class_id, score, x0, y0, x1, y1)
        return np.column_stack([class_ids, scores_nms, boxes_nms]).astype(np.float32)
    else:
        return np.zeros((0, 6), dtype=np.float32)


def draw_and_save(img_path, boxes, thresh, out_path, orig_size, model_size=DEFAULT_SIZE):
    """Draw boxes on image and save with proper scaling."""
    im = cv2.imread(img_path)
    if im is None:
        print(f"[WARN] Could not read image: {img_path}")
        return

    vis_h, vis_w = im.shape[:2]
    model_w, model_h = model_size
    scale_x = vis_w / float(model_w)
    scale_y = vis_h / float(model_h)

    for b in boxes:
        cls_id, score, x0, y0, x1, y1 = b
        if score < thresh:
            continue

        # Scale coordinates from model input size to original image size
        x0 = int(x0 * scale_x)
        y0 = int(y0 * scale_y)
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)

        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(im, f"{score:.2f}", (x0, max(0, y0-5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, im)


def main():
    parser = argparse.ArgumentParser(description="NCNN inference for PP-YOLOE Human")
    parser.add_argument("--img", required=True, help="Path to input image")
    parser.add_argument("--ncnn_param", required=True, help="Path to NCNN .param file")
    parser.add_argument("--ncnn_bin", required=True, help="Path to NCNN .bin file")
    parser.add_argument("--out", default="pipeline/output", help="Output directory")
    parser.add_argument("--thresh", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--nms-thresh", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings to .npy file")
    args = parser.parse_args()

    # Check files exist
    for path, label in [(args.img, "Image"), (args.ncnn_param, "Param"), (args.ncnn_bin, "Bin")]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    print("="*60)
    print("NCNN Inference - PP-YOLOE Human Detection")
    print("="*60)

    # Load model
    print("\n[1/6] Loading NCNN model...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_arithmetic = True
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_packed = True


    if net.load_param(args.ncnn_param) != 0:
        print(f"[ERROR] Failed to load param file", file=sys.stderr)
        return 1
    if net.load_model(args.ncnn_bin) != 0:
        print(f"[ERROR] Failed to load model file", file=sys.stderr)
        return 1
    print("  ✓ Model loaded")

    # Preprocess image
    print("\n[2/6] Preprocessing image...")
    img_mat, orig_size = load_and_preprocess(args.img, DEFAULT_SIZE)
    orig_w, orig_h = orig_size
    print(f"  ✓ Original size: {orig_w}x{orig_h}")
    print(f"  ✓ Model input: {DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}")
    scale_x = orig_w / float(DEFAULT_SIZE[0])
    scale_y = orig_h / float(DEFAULT_SIZE[1])
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"  ✓ Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

    # Warmup
    if args.warmup > 0:
        print(f"\n[3/6] Warming up ({args.warmup} runs)...")
        for _ in range(args.warmup):
            run_inference(net, img_mat, output_names=["out0", "out1"])
        print(f"  ✓ Warmup complete")

    # Inference
    print("\n[4/6] Running inference...")
    import time
    t0 = time.perf_counter()

    t_forward_start = time.perf_counter()
    outputs = run_inference(net, img_mat, output_names=["out0", "out1", "out2"])
    t_forward_end = time.perf_counter()

    forward_ms = (t_forward_end - t_forward_start) * 1000.0

    if len(outputs) == 0:
        print("[ERROR] No outputs extracted from model", file=sys.stderr)
        return 1

    print(f"  Model outputs:")
    for name, arr in outputs.items():
        print(f"    - {name}: shape={arr.shape}")

    # Post-process
    print("\n[5/6] Post-processing...")
    boxes_raw = outputs.get("out0")
    scores_raw = outputs.get("out1")
    embeddings = outputs.get("out2")  # Embedding output added by insert_embedding_head.py

    if boxes_raw is None or scores_raw is None:
        print("[ERROR] Missing required outputs (out0, out1)", file=sys.stderr)
        return 1

    scores_raw = scores_raw.squeeze()
    boxes_nms = apply_nms(boxes_raw, scores_raw, args.thresh, args.nms_thresh)

    print(f"  ✓ Found {len(boxes_nms)} detections after NMS")

    # Validate embeddings for BoT-SORT
    if embeddings is not None and len(boxes_nms) > 0:
        print(f"\n[Embeddings] Found embedding output: shape={embeddings.shape}")

        # Validate embedding quality for BoT-SORT
        validation_result = validate_embedding_for_botsort(embeddings, boxes_nms, verbose=True)

        if not validation_result['valid']:
            print("\n⚠️  WARNING: Embedding validation detected potential issues!")
            print("    Check the validation report above for details.")
    elif embeddings is None:
        print(f"\n[WARN] No embedding output found (expected 'out2')")
        print(f"       Available outputs: {list(outputs.keys())}")
        print(f"       Make sure the model was processed with insert_embedding_head.py")
    else:
        print(f"\n[INFO] No detections to validate embeddings")

    # Print detections
    if len(boxes_nms) > 0:
        print("\nDetections (class_id score x0 y0 x1 y1):")
        for i, b in enumerate(boxes_nms[:10]):  # Show first 10
            cls_id, score, x0, y0, x1, y1 = b
            print(f"  {i+1}. {int(cls_id)} {score:.4f} {x0:.1f} {y0:.1f} {x1:.1f} {y1:.1f}")
        if len(boxes_nms) > 10:
            print(f"  ... and {len(boxes_nms)-10} more")

    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    print(f"\n[Timing]")
    print(f"  ✓ Forward pass: {forward_ms:.2f}ms")
    print(f"  ✓ End-to-end latency: {total_ms:.2f}ms")

    # Save visualization
    print("\n[6/6] Saving outputs...")
    out_basename = os.path.splitext(os.path.basename(args.img))[0]
    out_path = os.path.join(args.out, out_basename + "_ncnn.jpg")
    draw_and_save(args.img, boxes_nms, args.thresh, out_path, orig_size, DEFAULT_SIZE)
    print(f"  ✓ Visualization: {out_path}")

    # Save embeddings if requested
    if args.save_embeddings and embeddings is not None and len(boxes_nms) > 0:
        emb_path = os.path.join(args.out, out_basename + "_embeddings.npy")
        np.save(emb_path, embeddings)
        print(f"  ✓ Embeddings: {emb_path}")

    print("\n" + "="*60)
    print("✅ NCNN Inference Complete!")
    print(f"   Forward: {forward_ms:.2f}ms")
    print(f"   End-to-end: {total_ms:.2f}ms")
    print(f"   Detections: {len(boxes_nms)}")
    if embeddings is not None and len(boxes_nms) > 0:
        print(f"   Embeddings: {embeddings.shape}")
    print("="*60)
    return 0
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
