# pipeline/PP-YOLOE/onnx_inference_image.sh
#!/bin/bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/venv/bin/activate

echo "🔧 Running ONNX inference..."

python3 pipeline/PP-YOLOE/onnx_inference_image.py

#python3 pipeline/PP-YOLOE/onnx_inference_image.py \
#  --onnx pipeline/output/ppyoloe_crn_s_36e_pphuman.onnx \
#  --image pipeline/dataset/demo/demo.jpg \
#  --output pipeline/output \
#  --conf 0.5 \
#  --debug

echo "✅ Fixed ONNX inference completed! Check pipeline/output/ for result images."