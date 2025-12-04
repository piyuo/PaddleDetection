# PP-YOLOE/src/reid/run_quantize.sh
#!/usr/bin/env bash
set -e

# Paths
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL_DIR="${REPO_ROOT}/PP-YOLOE/build/models"
INPUT_MODEL="${MODEL_DIR}/human_reid.onnx"
OUTPUT_MODEL="${MODEL_DIR}/human_reid_quant.onnx"

# Check if input model exists
if [ ! -f "$INPUT_MODEL" ]; then
    echo "Error: Input model not found at $INPUT_MODEL"
    echo "Please run build_human_reid.sh first."
    exit 1
fi

PREP_MODEL="${MODEL_DIR}/human_reid_prep.onnx"

echo "🔧 Preprocessing model (Constant -> Initializer)..."
python3 "$(dirname "$0")/preprocess_for_quant.py" "$INPUT_MODEL" "$PREP_MODEL"

echo "🔧 Compressing weights (FP32 -> FP16 + Cast)..."
python3 "$(dirname "$0")/compress_weights.py" "$PREP_MODEL" "$OUTPUT_MODEL"

rm "$PREP_MODEL"

echo "✅ Quantized model saved to $OUTPUT_MODEL"
