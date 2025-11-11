# PP-YOLOE/src/onnx_inference_image.sh
#!/usr/bin/env bash

echo "🔧 Running ONNX inference..."

# Use python from the (possibly) activated venv
python3 PP-YOLOE/src/onnx_inference_image.py \
    --img PP-YOLOE/build/dataset/demo/demo.jpg \
    --onnx PP-YOLOE/build/models/ppyoloe_crn_s_36e_pphuman_cust_ane_cu.onnx \
    --out PP-YOLOE/build/output \
    --thresh 0.5