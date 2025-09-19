# pipeline/DAMO-YOLO/onnx_inference_image.sh
#!/bin/bash
# pipeline/DAMO-YOLO/onnx_inference_image.sh

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source .venv/bin/activate

echo "🔧 Running ONNX inference..."

python3 pipeline/DAMO-YOLO/onnx_inference_image.py \
  --onnx pipeline/output/damoyolo_tinynasL25_S_person_feat.onnx \
  --image pipeline/dataset/demo/demo.jpg \
  --output pipeline/output \
  --conf 0.5 \
  --debug

echo "✅ Fixed ONNX inference completed! Check pipeline/output/ for result images."