# pipeline/PP-YOLOE/tf_inference_cust.sh
#!/usr/bin/env bash



# This script runs inference using the customized TFLite model that omits NMS.

IMG="${1:-pipeline/dataset/demo/demo.jpg}"
TFLITE="${2:-pipeline/PP-YOLOE/models/tensorflow/ppyoloe_saved_model/ppyoloe_crn_s_36e_pphuman_cust_tflite_float16.tflite}"
OUT="${3:-pipeline/output}"
THRESH="${4:-0.5}"

python3 pipeline/PP-YOLOE/tf_inference_cust.py \
    --img "$IMG" \
    --tflite "$TFLITE" \
    --out "$OUT" \
    --thresh "$THRESH"