# pipeline/PP-YOLOE/onnx_inference_image.sh
#!/usr/bin/env bash

echo "🔧 Running ONNX inference..."

# Use python from the (possibly) activated venv
python3 pipeline/PP-YOLOE/onnx_inference_image.py \
    --img pipeline/dataset/demo/demo.jpg \
    --onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ane.onnx \
    --out pipeline/output \
    --thresh 0.5