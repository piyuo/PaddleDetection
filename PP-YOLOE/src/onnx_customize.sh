# PP-YOLOE/src/onnx_customize.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source PP-YOLOE/build/venv/bin/activate

# Remove NMS and add feature outputs (stride-8/16)# feature maps for multi-scale embedding extraction.
#         Input:  ppyoloe_crn_s_36e_pphuman.onnx (base model from export_to_onnx.sh)#
#         Output: ppyoloe_crn_s_36e_pphuman_cust.onnx (customized model)# The script will automatically:

BASE_MODEL="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman.onnx"
CUST_MODEL="PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_cust.onnx"

python3 PP-YOLOE/src/onnx_customize.py \
    --model "$BASE_MODEL" \
    --output-model "$CUST_MODEL"
