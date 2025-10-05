#!/usr/bin/env python3
"""Wrapper to run NCNN inference with better error reporting"""
import sys
import traceback

print("=" * 60)
print("NCNN Inference Debug Wrapper")
print("=" * 60)

try:
    print("\n[1/5] Importing modules...")
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ncnn_inference_image import run_ncnn
    print("✓ Imports successful")

    print("\n[2/5] Setting up parameters...")
    param_path = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param"
    bin_path = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin"
    img_path = "pipeline/dataset/demo/demo.jpg"
    print(f"  - Param: {param_path}")
    print(f"  - Bin: {bin_path}")
    print(f"  - Image: {img_path}")

    print("\n[3/5] Running inference...")
    result = run_ncnn(
        param_path=param_path,
        bin_path=bin_path,
        img_path=img_path,
        thresh=0.5,
        input_name=None,
        output_names=None,
        warmup=3,
        boxes_format="auto",
        nms_threshold=0.5,
        enable_embeddings=True,
    )

    print(f"\n[4/5] Inference complete!")
    print(f"  - Detections: {result.boxes.shape[0]}")
    print(f"  - Inference time: {result.benchmark['inference_ms']:.2f}ms")

    print("\n[5/5] Success!")

except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] User cancelled")
    sys.exit(130)
except Exception as e:
    print(f"\n\n[ERROR] Exception occurred:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
