#!/usr/bin/env python3
"""Standalone ONNX person detection inference script.

This script is completely self-contained and requires only standard Python libraries
plus NumPy, OpenCV, PIL, and ONNXRuntime. No DAMO-YOLO dependencies needed.

Required packages:
    pip install numpy opencv-python pillow onnxruntime

All preprocessing, postprocessing, NMS, and visualization are handled with
NumPy and OpenCV only. Configuration is hardcoded for person detection.

Example:
    python3 pipeline/DAMO-YOLO/onnx_inference_image.py \
        --onnx pipeline/output/damoyolo_tinynasL25_S_person.onnx \
        --image pipeline/dataset/demo/demo.jpg \
        --output pipeline/output \
        --conf 0.5

If the exported ONNX uses a legacy head (person + background), the second
channel (background) is discarded so only the person channel is used.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
