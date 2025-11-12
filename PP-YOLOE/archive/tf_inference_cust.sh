# pipeline/PP-YOLOE/tf_inference_cust.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv311/bin/activate

# This script runs inference using the customized TFLite model that omits NMS.
# Now uses the no-split version (no Flex ops) by default.

IMG="${1:-pipeline/dataset/demo/demo.jpg}"
TFLITE="${2:-pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_f16.tflite}"
OUT="${3:-pipeline/output}"
THRESH="${4:-0.5}"

python3 pipeline/PP-YOLOE/tf_inference_cust.py \
    --img "$IMG" \
    --tflite "$TFLITE" \
    --out "$OUT" \
    --thresh "$THRESH"
