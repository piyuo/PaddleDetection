#!/usr/bin/env bash
# pipeline/PP-YOLOE/onnx_inference.sh
# Simple demo script for running ONNX inference on an image

set -e

echo "🚀 PP-YOLOE ONNX Inference Demo"
echo ""

# Activate the Python virtual environment if it exists
if [ -d "pipeline/PP-YOLOE/venv" ]; then
    echo "🔧 Activating Python environment..."
    source pipeline/PP-YOLOE/venv/bin/activate
fi

# Default paths
IMG="${1:-pipeline/dataset/demo/demo.jpg}"
ONNX="${2:-pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx}"
OUT="${3:-pipeline/output}"
THRESH="${4:-0.5}"

# Run inference
python3 pipeline/PP-YOLOE/onnx_inference.py \
    --img "$IMG" \
    --onnx "$ONNX" \
    --out "$OUT" \
    --thresh "$THRESH"

echo ""
echo "✅ Done!"
echo ""
echo "💡 Usage with custom parameters:"
echo "   ./pipeline/PP-YOLOE/onnx_inference.sh <image> <model> <output_dir> <threshold>"
echo ""
echo "   Example:"
echo "   ./pipeline/PP-YOLOE/onnx_inference.sh \\"
echo "       pipeline/dataset/demo/demo.jpg \\"
echo "       pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \\"
echo "       pipeline/output \\"
echo "       0.3"
