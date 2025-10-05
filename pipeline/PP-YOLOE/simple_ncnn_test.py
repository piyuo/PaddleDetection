#!/usr/bin/env python3
"""Simple standalone NCNN inference test for PP-YOLOE"""
import os
import sys
import numpy as np
import cv2

# Make sure we're in the right directory
os.chdir('/Users/cc/Dropbox/PaddleDetection')

try:
    import ncnn
except ImportError:
    print("ERROR: ncnn not installed")
    sys.exit(1)

def main():
    param_path = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param"
    bin_path = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin"
    img_path = "pipeline/dataset/demo/demo.jpg"

    print("="*60)
    print("NCNN Inference Test")
    print("="*60)

    # Load image
    print("\n[1/6] Loading image...")
    with open(img_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    orig_h, orig_w = im.shape[:2]
    print(f"  ✓ Original image size: {orig_w}x{orig_h}")
    im = cv2.resize(im, (640, 640))
    print(f"  ✓ Image loaded: {im.shape}")

    # Create NCNN Mat
    print("\n[2/6] Creating NCNN Mat...")
    mat = ncnn.Mat.from_pixels(im, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 640, 640)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [1.0 / (0.229 * 255), 1.0 / (0.224 * 255), 1.0 / (0.225 * 255)]
    mat.substract_mean_normalize(mean, std)
    scale_factor = ncnn.Mat(np.array([1.0, 1.0], dtype=np.float32))
    print(f"  ✓ Mat created: w={mat.w}, h={mat.h}, c={mat.c}")

    # Load model
    print("\n[3/6] Loading NCNN model...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    if net.load_param(param_path) != 0:
        print(f"  ✗ Failed to load param")
        return 1
    if net.load_model(bin_path) != 0:
        print(f"  ✗ Failed to load model")
        return 1
    print("  ✓ Model loaded")

    # Run inference
    print("\n[4/6] Running inference...")
    ex = net.create_extractor()
    ex.input("in0", scale_factor)
    ex.input("in1", mat)

    # Extract outputs
    print("\n[5/6] Extracting outputs...")
    outputs = {}
    for name in ["out0", "out1", "out2", "out3"]:
        ret, out_mat = ex.extract(name)
        if ret == 0:
            arr = out_mat.numpy()
            outputs[name] = arr
            print(f"  ✓ {name}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  ✗ {name}: extraction failed")

    if len(outputs) != 4:
        print(f"\n✗ Failed to extract all outputs (got {len(outputs)}/4)")
        return 1

    # Post-process
    print("\n[6/6] Post-processing...")
    boxes_raw = outputs["out0"]  # (N, 4)
    scores_raw = outputs["out1"].squeeze()  # (N,)

    # Apply NMS
    thresh = 0.5
    mask = scores_raw >= thresh
    if not mask.any():
        print(f"  ✗ No detections above threshold {thresh}")
        return 0

    boxes_filt = boxes_raw[mask]
    scores_filt = scores_raw[mask]

    # Convert to x,y,w,h for NMS
    xywh = np.column_stack([
        boxes_filt[:, 0],
        boxes_filt[:, 1],
        boxes_filt[:, 2] - boxes_filt[:, 0],
        boxes_filt[:, 3] - boxes_filt[:, 1],
    ])

    indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores_filt.tolist(), float(thresh), 0.5)
    if len(indices) == 0:
        print(f"  ✗ No detections after NMS")
        return 0

    indices = np.array(indices).flatten()
    final_boxes = boxes_filt[indices]
    final_scores = scores_filt[indices]

    print(f"  ✓ Found {len(final_boxes)} detections")
    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
        x0, y0, x1, y1 = box
        print(f"    {i+1}. score={score:.3f}, bbox=[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]")

    # Save visualization
    print("\n[Output] Saving visualization...")
    out_dir = "pipeline/output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ncnn_detections.jpg")

    # Load original image for visualization
    vis_im = cv2.imread(img_path)
    vis_h, vis_w = vis_im.shape[:2]

    # Scale boxes from 640x640 to original image size
    scale_x = vis_w / 640.0
    scale_y = vis_h / 640.0

    print(f"  • Original image: {vis_w}x{vis_h}")
    print(f"  • Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

    for box, score in zip(final_boxes, final_scores):
        x0, y0, x1, y1 = box
        # Scale coordinates to original image size
        x0 = int(x0 * scale_x)
        y0 = int(y0 * scale_y)
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)

        cv2.rectangle(vis_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis_im, f"{score:.2f}", (x0, y0-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(out_path, vis_im)
    print(f"  ✓ Saved to: {out_path}")

    print("\n" + "="*60)
    print("✓ NCNN Inference Complete!")
    print("="*60)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
