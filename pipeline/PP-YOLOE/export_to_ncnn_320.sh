# pipeline/PP-YOLOE/export_to_ncnn_320.sh
#!/usr/bin/env bash

set -e

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate


# Export PP-YOLOE model to ONNX format with input size 320x320
pipeline/PP-YOLOE/export_to_onnx.sh \
    --config configs/pphuman/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.yml \
	--weights pipeline/PP-YOLOE/weights/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.pdparams \
    --shape "3,320,320"

# Run customized
python3 pipeline/PP-YOLOE/onnx_customize.py \
    --model pipeline/PP-YOLOE/models/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman.onnx \
    --input-shape "1,3,320,320" \
    --output-model pipeline/PP-YOLOE/models/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman_cust.onnx

MODEL_NAME="ppyoloe_crn_s_36e_pphuman_cust_ncnn"
OUTPUT_MODEL_NAME="ppyoloe_crn_s_36e_pphuman_ncnn"
MODEL="pipeline/PP-YOLOE/models/${MODEL_NAME}.onnx"
TEMP_DIR="pipeline/PP-YOLOE/models/pnnx/"
OUTPUT_DIR="pipeline/PP-YOLOE/models/"


echo "╔════════════════════════════════════════════════════════════╗"
echo "║   NCNN Model Build for Mobile Deployment                  ║"
echo "║   (Preserving all outputs: out0, out1, out2, out3)        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if pnnx is available
if ! command -v pnnx &> /dev/null; then
    echo "❌ pnnx command not found!"
    echo "   Please install pnnx or add it to PATH"
    exit 1
fi

if [ ! -f "pipeline/ncnnoptimize" ]; then
    echo "⚠️  Custom ncnnoptimize not found at pipeline/ncnnoptimize"
    echo "   Please install ncnnoptimize or add it to PATH"
    echo ""
fi

# Convert with maximum optimization for mobile
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: PNNX Conversion to NCNN (Optimized for Mobile)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔧 Running PNNX conversion with optlevel=3 (max optimization)..."

mkdir -p "$TEMP_DIR"
cp "$MODEL" "$TEMP_DIR"


# Model has 2 inputs: scale_factor [1,2] and image [1,3,640,640]
pnnx "${TEMP_DIR}${MODEL_NAME}.onnx"  \
    fp16=1 \
    optlevel=3 \
    device=gpu \
    inputshape="[1,2]f32,[1,3,640,640]f32"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: NCNN Optimize (Preserving Feature Maps)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pipeline/ncnnoptimize \
    "${TEMP_DIR}${MODEL_NAME}.ncnn.param" \
    "${TEMP_DIR}${MODEL_NAME}.ncnn.bin" \
    "${OUTPUT_DIR}${OUTPUT_MODEL_NAME}.param" \
    "${OUTPUT_DIR}${OUTPUT_MODEL_NAME}.bin" \
    1 keep=out2,out3
