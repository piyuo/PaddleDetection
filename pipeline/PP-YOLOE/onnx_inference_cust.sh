# pipeline/PP-YOLOE/onnx_inference_cust.sh
#!/usr/bin/env bash

# this script runs inference using the customized ONNX model,
# ps. since customized model removed NMS, it uses a different inference script.

# inference
IMG="${1:-pipeline/dataset/demo/demo.jpg}"
ONNX="${2:-pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx}"
OUT="${3:-pipeline/output}"
THRESH="${4:-0.5}"

# Run inference
python3 pipeline/PP-YOLOE/onnx_inference_cust.py \
    --img "$IMG" \
    --onnx "$ONNX" \
    --out "$OUT" \
    --thresh "$THRESH"
