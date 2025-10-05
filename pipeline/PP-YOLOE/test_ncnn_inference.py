#!/usr/bin/env python3
"""Minimal NCNN inference test"""
import os
import sys
import numpy as np
import cv2

try:
    import ncnn
except ImportError:
    print("ERROR: ncnn not installed. Install with: pip install ncnn")
    sys.exit(1)

def preprocess_image(img_path, target_size=(640, 640)):
    """Load and preprocess image to uint8 BGR"""
    with open(img_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Resize to target size (BGR, uint8)
    im = cv2.resize(im, target_size)
    return im

def ncnn_mat_from_bgr(bgr_img):
    """Convert BGR uint8 image to ncnn.Mat with normalization"""
    h, w = bgr_img.shape[:2]

    # Convert BGR to RGB and create NCNN Mat
    mat = ncnn.Mat.from_pixels(bgr_img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h)

    # Apply normalization: (x/255 - mean) / std
    # NCNN substract_mean_normalize expects: (x - mean*255) * (1 / (std*255))
    # To get (x/255 - mean) / std, we use: (x - mean*255) / (std*255)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [1.0 / (0.229 * 255), 1.0 / (0.224 * 255), 1.0 / (0.225 * 255)]
    mat.substract_mean_normalize(mean, std)

    return mat

def mat_to_numpy(mat):
    """Convert ncnn.Mat to numpy array"""
    # Use built-in numpy() method if available
    if hasattr(mat, 'numpy'):
        return mat.numpy()

    # Fall back to manual conversion
    w = getattr(mat, "w", 0)
    h = getattr(mat, "h", 0)
    c = getattr(mat, "c", 1)

    chans = []
    for i in range(c):
        ch = mat.channel(i)
        arr = np.array(ch).reshape(h, w)
        chans.append(arr)

    if c == 1:
        return chans[0]
    return np.stack(chans, axis=0)

def main():
    param_path = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param"
    bin_path = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin"
    img_path = "pipeline/dataset/demo/demo.jpg"

    print("Loading NCNN model...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False

    if net.load_param(param_path) != 0:
        print(f"ERROR: Failed to load param: {param_path}")
        return 1

    if net.load_model(bin_path) != 0:
        print(f"ERROR: Failed to load model: {bin_path}")
        return 1

    print("Model loaded successfully!")

    print("\nPreprocessing image...")
    bgr_img = preprocess_image(img_path)
    print(f"Image preprocessed: shape={bgr_img.shape}, dtype={bgr_img.dtype}")

    print("\nConverting to NCNN Mat...")
    in_mat = ncnn_mat_from_bgr(bgr_img)
    print(f"Mat created: w={in_mat.w}, h={in_mat.h}, c={in_mat.c}")

    print("\nRunning inference...")
    ex = net.create_extractor()

    # The model has 2 inputs: in0 and in1
    # in1 is the image, in0 is scale_factor
    scale_factor = ncnn.Mat(np.array([1.0, 1.0], dtype=np.float32))

    ex.input("in0", scale_factor)
    ex.input("in1", in_mat)

    # Try to extract all 4 outputs
    outputs = {}
    for out_name in ["out0", "out1", "out2", "out3"]:
        ret, mat = ex.extract(out_name)
        if ret == 0:
            arr = mat_to_numpy(mat)
            outputs[out_name] = arr
            print(f"  ✓ {out_name}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  ✗ {out_name}: extraction failed (ret={ret})")

    if not outputs:
        print("\nERROR: Could not extract any outputs!")
        return 1

    print(f"\n✓ Successfully extracted {len(outputs)} outputs")
    return 0

if __name__ == "__main__":
    sys.exit(main())
