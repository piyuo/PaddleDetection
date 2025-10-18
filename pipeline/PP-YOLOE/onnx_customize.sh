# pipeline/PP-YOLOE/onnx_customize.sh

#!/usr/bin/env bash


# Remove NMS and add feature outputs (stride-8/16)# feature maps for multi-scale embedding extraction.
#         Input:  ppyoloe_crn_s_36e_pphuman.onnx (base model from export_to_onnx.sh)#
#         Output: ppyoloe_crn_s_36e_pphuman_cust.onnx (customized model)# The script will automatically:

BASE_MODEL="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx"
CUST_MODEL="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx"

python3 pipeline/PP-YOLOE/onnx_customize.py \
    --model "$BASE_MODEL" \
    --output "$CUST_MODEL" \
    --auto-discover

# inference
IMG="${1:-pipeline/dataset/demo/demo.jpg}"
ONNX="${2:-pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx}"
OUT="${3:-pipeline/output}"
THRESH="${4:-0.5}"

# Run inference
python3 pipeline/PP-YOLOE/onnx_inference.py \
    --img "$IMG" \
    --onnx "$ONNX" \
    --out "$OUT" \
    --thresh "$THRESH"
