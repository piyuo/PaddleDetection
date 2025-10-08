#!/usr/bin/env bash
# pipeline/PP-YOLOE/validate_embedding_coords.sh
# Validate that the embedding head model has correct coordinate space handling

set -e

echo "🔍 Validating embedding head coordinate space..."
echo ""

# Activate the Python virtual environment if it exists
if [ -d "pipeline/PP-YOLOE/venv" ]; then
    echo "🔧 Activating Python environment..."
    source pipeline/PP-YOLOE/venv/bin/activate
fi

# Run validation
python3 pipeline/PP-YOLOE/validate_embedding_coords.py \
    --onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
    --img pipeline/dataset/demo/demo.jpg

echo ""
echo "✅ Validation complete!"
