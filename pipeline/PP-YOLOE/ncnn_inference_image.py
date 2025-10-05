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


def run_inference(net, img_mat, input_names=["in0", "in1"], output_names=["out0", "out1", "out2", "out3"]):
    """Run NCNN inference and extract outputs."""
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
    
    # Extract outputs
    outputs = {}
    for name in output_names:
        ret, out_mat = ex.extract(name)
        if ret == 0:
            outputs[name] = out_mat.numpy()
    
    return outputs


def apply_nms(boxes_raw, scores_raw, score_thresh=0.5, nms_thresh=0.5):
    """Apply NMS to raw detection outputs."""
    # Filter by score threshold
    mask = scores_raw >= score_thresh
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)
    
    boxes_filt = boxes_raw[mask]
    scores_filt = scores_raw[mask]
    
    # Convert boxes to x,y,w,h for OpenCV NMS
    xywh = np.column_stack([
        boxes_filt[:, 0],
        boxes_filt[:, 1],
        boxes_filt[:, 2] - boxes_filt[:, 0],
        boxes_filt[:, 3] - boxes_filt[:, 1],
    ])
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores_filt.tolist(), float(score_thresh), float(nms_thresh))
    if len(indices) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    
    indices = np.array(indices).flatten()
    final_boxes = boxes_filt[indices]
    final_scores = scores_filt[indices]
    
    # Format as (class_id, score, x0, y0, x1, y1)
    cls_ids = np.zeros_like(final_scores)
    return np.column_stack([cls_ids, final_scores, final_boxes]).astype(np.float32)


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
    print("\n[1/5] Loading NCNN model...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
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
    print("\n[2/5] Preprocessing image...")
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
        print(f"\n[3/5] Warming up ({args.warmup} runs)...")
        for _ in range(args.warmup):
            run_inference(net, img_mat, output_names=["out0", "out1"])
        print(f"  ✓ Warmup complete")
    
    # Inference
    print("\n[4/5] Running inference...")
    import time
    t0 = time.perf_counter()
    outputs = run_inference(net, img_mat)
    t1 = time.perf_counter()
    inference_ms = (t1 - t0) * 1000.0
    print(f"  ✓ Inference time: {inference_ms:.2f}ms")
    
    if len(outputs) == 0:
        print("[ERROR] No outputs extracted from model", file=sys.stderr)
        return 1
    
    for name, arr in outputs.items():
        print(f"    - {name}: shape={arr.shape}")
    
    # Post-process
    print("\n[5/5] Post-processing...")
    boxes_raw = outputs.get("out0")
    scores_raw = outputs.get("out1")
    
    if boxes_raw is None or scores_raw is None:
        print("[ERROR] Missing required outputs (out0, out1)", file=sys.stderr)
        return 1
    
    scores_raw = scores_raw.squeeze()
    boxes_nms = apply_nms(boxes_raw, scores_raw, args.thresh, args.nms_thresh)
    
    print(f"  ✓ Found {len(boxes_nms)} detections")
    
    # Print detections
    if len(boxes_nms) > 0:
        print("\nDetections:")
        for i, b in enumerate(boxes_nms[:10]):  # Show first 10
            cls_id, score, x0, y0, x1, y1 = b
            print(f"  {i+1}. score={score:.3f}, bbox=[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]")
        if len(boxes_nms) > 10:
            print(f"  ... and {len(boxes_nms)-10} more")
    
    # Save visualization
    print("\n[Output] Saving visualization...")
    out_path = os.path.join(args.out, os.path.splitext(os.path.basename(args.img))[0] + "_ncnn.jpg")
    draw_and_save(args.img, boxes_nms, args.thresh, out_path, orig_size, DEFAULT_SIZE)
    print(f"  ✓ Saved to: {out_path}")
    
    print("\n" + "="*60)
    print("✅ NCNN Inference Complete!")
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
