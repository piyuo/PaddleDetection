#!/usr/bin/env python3
import argparse
import os
import sys
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(model_path, output_path):
    print(f"Quantizing model: {model_path}")
    print(f"Output path: {output_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Quantize
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )

    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Reduction: {(1 - quantized_size / original_size) * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8 (Dynamic)")
    parser.add_argument("--model", type=str, required=True, help="Path to input ONNX model")
    parser.add_argument("--out", type=str, required=True, help="Path to output quantized ONNX model")

    args = parser.parse_args()

    quantize_model(args.model, args.out)
