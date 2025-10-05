#!/usr/bin/env python3
"""Check NCNN Mat API"""
import ncnn
import numpy as np

print("NCNN Mat methods:")
mat = ncnn.Mat()
for attr in dir(mat):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\nTrying different Mat constructors:")
# Try different ways to create a Mat
chw = np.random.randn(3, 640, 640).astype(np.float32)

# Method 1: from dims
try:
    mat1 = ncnn.Mat(640, 640, 3)  # w, h, c
    print(f"✓ Mat(w, h, c): {mat1}")
except Exception as e:
    print(f"✗ Mat(w, h, c): {e}")

# Method 2: from pixels (BGR)
try:
    img_bgr = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    mat2 = ncnn.Mat.from_pixels(img_bgr, ncnn.Mat.PixelType.PIXEL_BGR, 640, 640)
    print(f"✓ from_pixels: {mat2}")
except Exception as e:
    print(f"✗ from_pixels: {e}")

# Method 3: from pixels with normalize
try:
    img_rgb = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
    mat3 = ncnn.Mat()
    mat3 = mat3.from_pixels_resize(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 640, 640, 640, 640)
    print(f"✓ from_pixels_resize: {mat3}")
except Exception as e:
    print(f"✗ from_pixels_resize: {e}")
