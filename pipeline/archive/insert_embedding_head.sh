# pipeline/PP-YOLOE/insert_embedding_head.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate

# create embedded onnx model for BOT-SORT
echo "🚀 Creating embedding head for PP-YOLOE model..."
python3 pipeline/PP-YOLOE/insert_embedding_head.py \
                --onnx_in pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
                --onnx_out pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx

echo ""
echo "✅ Embedding head created successfully!"
echo ""
echo "💡 To validate coordinate space handling, run:"
echo "   ./pipeline/PP-YOLOE/validate_embedding_coords.sh"


pipeline/PP-YOLOE/validate_embedding_coords.sh