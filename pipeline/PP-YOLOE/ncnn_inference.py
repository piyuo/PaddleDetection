#!/usr/bin/env python3
"""NCNN inference for PP-YOLOE Human model.

This script runs inference on a single image using the NCNN runtime.
It handles the pruned model format with raw boxes/scores outputs and
optional feature maps for embeddings.

Usage:
    python3 pipeline/PP-YOLOE/ncnn_inference_image.py \\
        --img pipeline/dataset/demo/demo.jpg \\
        --ncnn_param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \\
        --ncnn_bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \\
        --out pipeline/output \\
        --thresh 0.5
"""
import argparse
import os
import sys
import time
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


def roi_align_pool_multi_scale(
    feat_s8: np.ndarray,
    feat_s16: np.ndarray,
    boxes_xyxy: np.ndarray,
    img_hw: tuple,
    input_size_hw: tuple = (640, 640),
    gp_w: float = 0.2,
    avg_w: float = 1.0,
    max_w: float = 0.0,
    pp_w: float = 0.8,
    pp_k: int = 9,
    pp_stripe_h: int = 2,
    pp_vertical_k: int = 2,
    pp_vertical_stripe_w: int = 2,
    use_inst_norm: bool = True,
    pl_alpha: float = 0.35,
) -> np.ndarray:
    """Multi-scale ROI pooling (matching onnx_inference_ane_model.py logic).

    Extracts embeddings from stride-8 and stride-16 feature maps using
    global pooling and part-based pooling strategies.
    """
    assert feat_s8.ndim == 4 and feat_s8.shape[0] == 1
    assert feat_s16.ndim == 4 and feat_s16.shape[0] == 1

    _, c_s8, h_s8, w_s8 = feat_s8.shape
    _, c_s16, h_s16, w_s16 = feat_s16.shape
    h_img, w_img = img_hw
    h_input, w_input = input_size_hw

    def compute_scales(h_feat, w_feat):
        scale_y = (h_input / float(h_img)) * (h_feat / float(h_input))
        scale_x = (w_input / float(w_img)) * (w_feat / float(w_input))
        return scale_x, scale_y

    scale_x_s8, scale_y_s8 = compute_scales(h_s8, w_s8)
    scale_x_s16, scale_y_s16 = compute_scales(h_s16, w_s16)

    def extract_roi(feat_map, x0, y0, x1, y1, scale_x, scale_y, h_feat, w_feat, channels):
        fx0 = int(max(0, np.floor(x0 * scale_x)))
        fy0 = int(max(0, np.floor(y0 * scale_y)))
        fx1 = int(min(w_feat, np.ceil(x1 * scale_x)))
        fy1 = int(min(h_feat, np.ceil(y1 * scale_y)))
        if fx1 <= fx0 or fy1 <= fy0:
            return np.zeros((channels,), dtype=np.float32)

        roi = feat_map[0, :, fy0:fy1, fx0:fx1]
        _, h_roi, w_roi = roi.shape
        features = []

        # Global pooling
        if gp_w > 0:
            global_feat = np.zeros((channels,), dtype=np.float32)
            if avg_w > 0:
                global_feat += avg_w * roi.mean(axis=(1, 2))
            if max_w > 0:
                global_feat += max_w * roi.max(axis=(1, 2))
            features.append(global_feat * gp_w)

        # Part-based pooling (horizontal stripes)
        if pp_w > 0 and pp_k > 0 and pp_stripe_h > 0:
            stripe_size = max(1, h_roi // pp_k)
            for i in range(pp_k):
                y_start = i * stripe_size
                y_end = min(h_roi, (i + 1) * stripe_size)
                if y_end <= y_start:
                    continue
                sub_stripe_size = max(1, (y_end - y_start) // pp_stripe_h)
                for j in range(pp_stripe_h):
                    sub_y_start = y_start + j * sub_stripe_size
                    sub_y_end = min(y_end, y_start + (j + 1) * sub_stripe_size)
                    if sub_y_end <= sub_y_start:
                        continue
                    stripe = roi[:, sub_y_start:sub_y_end, :]
                    stripe_feat = stripe.mean(axis=(1, 2))
                    features.append(stripe_feat * pp_w / (pp_k * pp_stripe_h))

        # Part-based pooling (vertical stripes)
        if pp_w > 0 and pp_vertical_k > 0 and pp_vertical_stripe_w > 0:
            stripe_size = max(1, w_roi // pp_vertical_k)
            for i in range(pp_vertical_k):
                x_start = i * stripe_size
                x_end = min(w_roi, (i + 1) * stripe_size)
                if x_end <= x_start:
                    continue
                sub_stripe_size = max(1, (x_end - x_start) // pp_vertical_stripe_w)
                for j in range(pp_vertical_stripe_w):
                    sub_x_start = x_start + j * sub_stripe_size
                    sub_x_end = min(x_end, x_start + (j + 1) * sub_stripe_size)
                    if sub_x_end <= sub_x_start:
                        continue
                    stripe = roi[:, :, sub_x_start:sub_x_end]
                    stripe_feat = stripe.mean(axis=(1, 2))
                    features.append(stripe_feat * pp_w / (pp_vertical_k * pp_vertical_stripe_w))

        if not features:
            return np.zeros((channels,), dtype=np.float32)

        combined = np.sum(features, axis=0)

        # Instance normalization
        if use_inst_norm:
            mean = combined.mean()
            std = combined.std()
            if std > 1e-6:
                combined = (combined - mean) / std

        # Power-law transformation
        if pl_alpha != 1.0:
            sign = np.sign(combined)
            combined = sign * np.power(np.abs(combined), pl_alpha)

        return combined

    embs = []
    for x0, y0, x1, y1 in boxes_xyxy:
        feat_s8_vec = extract_roi(feat_s8, x0, y0, x1, y1, scale_x_s8, scale_y_s8, h_s8, w_s8, c_s8)
        feat_s16_vec = extract_roi(feat_s16, x0, y0, x1, y1, scale_x_s16, scale_y_s16, h_s16, w_s16, c_s16)
        combined = np.concatenate([feat_s8_vec, feat_s16_vec])

        # L2 normalization
        norm = np.linalg.norm(combined)
        if norm > 1e-6:
            combined = combined / norm

        embs.append(combined)

    if not embs:
        return np.zeros((0, c_s8 + c_s16), dtype=np.float32)
    return np.stack(embs, axis=0)


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


def run_inference(net, img_mat, num_threads=4, input_names=["in0", "in1"], output_names=["out0", "out1", "out2", "out3"],
                  backbone_layers=None):
    """Run NCNN inference and extract outputs.

    Args:
        net: NCNN network
        img_mat: Input image matrix
        num_threads: Number of threads to use for inference
        input_names: Names of input layers
        output_names: Names of standard output layers (detections)
        backbone_layers: Dict mapping output names to backbone layer names for manual extraction
                        e.g., {"out2": "conv_147", "out3": "conv_159"}
    """
    # Create scale_factor input (required by PP-YOLOE)
    scale_factor = ncnn.Mat(np.array([1.0, 1.0], dtype=np.float32))

    # CRITICAL FIX: Create a fresh extractor for each inference
    # NCNN extractors can only be used once and must be recreated
    ex = net.create_extractor()

    # Set extractor options for better performance
    # Note: In Python bindings, num_threads is set on Net.opt, not on Extractor
    ex.set_light_mode(True)  # Reduce memory usage

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

    # Extract backbone feature maps manually (for optimized models)
    if backbone_layers:
        print(f"\n[Manual Feature Extraction] Attempting to extract from backbone layers...")
        for out_name, layer_name in backbone_layers.items():
            if out_name in outputs:
                continue
            ret, out_mat = ex.extract(layer_name)
            if ret == 0:
                outputs[out_name] = out_mat.numpy()
                print(f"  ✓ Extracted {out_name} from layer '{layer_name}': shape={out_mat.numpy().shape}")
            else:
                print(f"  ✗ Failed to extract {out_name} from layer '{layer_name}' (error code: {ret})")

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
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for inference (default: 4)")
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

    # Core optimizations that work with Python bindings
    net.opt.use_vulkan_compute = True      # Enable Vulkan GPU acceleration
    net.opt.use_fp16_packed = True         # Enable FP16 packed storage
    net.opt.use_fp16_storage = True        # Enable FP16 storage
    net.opt.use_fp16_arithmetic = True     # Enable FP16 arithmetic
    net.opt.use_packing_layout = True      # Use packed tensor layout
    net.opt.use_bf16_storage = False       # Disable BF16 (not widely supported)
    net.opt.num_threads = args.threads     # Set number of threads for inference

    # Note: Some advanced options like use_shader_pack8, use_image_storage,
    # use_winograd_convolution, use_sgemm_convolution are not exposed in
    # the Python bindings but are automatically used by NCNN when beneficial

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
            # Warmup with full pipeline including backbone features for accurate performance
            run_inference(net, img_mat, num_threads=args.threads)
        print(f"  ✓ Warmup complete")

    # Inference
    print("\n[4/6] Running inference...")
    t0 = time.perf_counter()
    # PP-YOLOE backbone layer mappings for feature extraction
    # These layers correspond to stride-8 and stride-16 outputs in typical PP-YOLOE architecture
    # Adjust these layer names based on your specific model architecture
    backbone_feature_layers = {
        "out2": "conv_147",  # Stride-8 feature map (96 channels @ 80x80)
        "out3": "conv_159",  # Stride-16 feature map (192 channels @ 40x40)
    }


    t_forward_start = time.perf_counter()
    # Extract all outputs: out0 (boxes), out1 (scores), out2 (stride-8 features), out3 (stride-16 features)
    outputs = run_inference(net, img_mat, num_threads=args.threads, backbone_layers=backbone_feature_layers)
    t_forward_end = time.perf_counter()

    forward_ms = (t_forward_end - t_forward_start) * 1000.0

    if len(outputs) == 0:
        print("[ERROR] No outputs extracted from model", file=sys.stderr)
        return 1

    for name, arr in outputs.items():
        print(f"    - {name}: shape={arr.shape}")

    # Post-process
    print("\n[5/6] Post-processing...")
    boxes_raw = outputs.get("out0")
    scores_raw = outputs.get("out1")

    if boxes_raw is None or scores_raw is None:
        print("[ERROR] Missing required outputs (out0, out1)", file=sys.stderr)
        return 1

    scores_raw = scores_raw.squeeze()
    boxes_nms = apply_nms(boxes_raw, scores_raw, args.thresh, args.nms_thresh)

    print(f"  ✓ Found {len(boxes_nms)} detections after NMS")

    # Extract embeddings (matching onnx_inference_ane_model.py)
    embeddings = None
    embed_ms = None
    feat_s8 = outputs.get("out2")  # Stride-8 feature map
    feat_s16 = outputs.get("out3")  # Stride-16 feature map

    if feat_s8 is not None and feat_s16 is not None and len(boxes_nms) > 0:
        print(f"\n[Embeddings] Extracting multi-scale features...")
        print(f"  ✓ Stride-8:  out2 (shape={feat_s8.shape})")
        print(f"  ✓ Stride-16: out3 (shape={feat_s16.shape})")

        # Reshape to 4D if needed
        if feat_s8.ndim == 3:
            feat_s8 = feat_s8[np.newaxis, :]
        if feat_s16.ndim == 3:
            feat_s16 = feat_s16[np.newaxis, :]

        # Extract boxes in xyxy format for embedding extraction
        boxes_xyxy = boxes_nms[:, 2:6]  # Skip class_id and score

        t_embed_start = time.perf_counter()
        embeddings = roi_align_pool_multi_scale(
            feat_s8,
            feat_s16,
            boxes_xyxy,
            (orig_h, orig_w),
            input_size_hw=(640, 640),
            gp_w=0.2,
            pp_w=0.8,
            pp_k=9,
            pp_stripe_h=2,
            pp_vertical_k=2,
            pp_vertical_stripe_w=2,
            use_inst_norm=True,
            pl_alpha=0.35,
        )
        print(f"  ✓ Embeddings shape: {embeddings.shape} (dim={embeddings.shape[1]})")

        # Print embedding statistics
        if embeddings.shape[0] > 0:
            emb_norms = np.linalg.norm(embeddings, axis=1)
            print(f"  ✓ Embedding norms: min={emb_norms.min():.4f}, max={emb_norms.max():.4f}, mean={emb_norms.mean():.4f}")

        t_embed_end = time.perf_counter()
        embed_ms = (t_embed_end - t_embed_start) * 1000.0
    elif len(boxes_nms) > 0:
        print(f"\n[WARN] Feature maps not available for embedding extraction")
        print(f"       Available outputs: {list(outputs.keys())}")
    # No detections keeps embed_ms as None

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
    print(f"  ✓ Forward pass: {forward_ms:.2f}ms")
    if embed_ms is not None:
        print(f"  ✓ Embedding extraction: {embed_ms:.2f}ms")
    print(f"  ✓ End-to-end latency: {total_ms:.2f}ms")

    # Save visualization
    print("\n[6/6] Saving outputs...")
    out_basename = os.path.splitext(os.path.basename(args.img))[0]
    out_path = os.path.join(args.out, out_basename + "_ncnn.jpg")
    draw_and_save(args.img, boxes_nms, args.thresh, out_path, orig_size, DEFAULT_SIZE)
    print(f"  ✓ Visualization: {out_path}")

    # Save embeddings if requested
    if args.save_embeddings and embeddings is not None:
        emb_path = os.path.join(args.out, out_basename + "_embeddings.npy")
        np.save(emb_path, embeddings)
        print(f"  ✓ Embeddings: {emb_path}")

    print("\n" + "="*60)
    print("✅ NCNN Inference Complete!")
    print(f"   Forward: {forward_ms:.2f}ms")
    if embed_ms is not None:
        print(f"   Embedding: {embed_ms:.2f}ms")
    print(f"   End-to-end: {total_ms:.2f}ms")
    print(f"   Detections: {len(boxes_nms)}")
    if embeddings is not None:
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
