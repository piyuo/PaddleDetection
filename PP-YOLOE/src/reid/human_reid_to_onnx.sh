# PP-YOLOE/src/reid/human_reid_to_onnx.sh
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_OPSET=16
DEFAULT_OUT_DIR="${REPO_ROOT}/PP-YOLOE/build/models"
OPSET=${DEFAULT_OPSET}
OUT_DIR="${DEFAULT_OUT_DIR}"
MODEL_NAME="human_reid"
ONNX_FILE="${OUT_DIR}/${MODEL_NAME}.onnx"
TEMP_DIR="PP-YOLOE/build/models/pnnx/"
OUTPUT_MODEL_NAME="human_ncnn"


# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source PP-YOLOE/build/venv/bin/activate

if ! command -v paddle2onnx >/dev/null 2>&1; then
	echo "paddle2onnx not found; installing to user site-packages ..."
	set -x
	"${PY}" -m pip install --user -q paddle2onnx
	set +x
fi

set -x
paddle2onnx \
	--model_dir "${REPO_ROOT}/PP-YOLOE/build/weights/human_reid" \
	--model_filename model.pdmodel \
	--params_filename model.pdiparams \
	--opset_version "${OPSET}" \
    --enable_auto_update_opset False \
    --optimize_tool None \
    --enable_onnx_checker True\
	--save_file "${ONNX_FILE}"
set +x


# convert to ncnn

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   NCNN Model Build for Mobile Deployment                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if pnnx is available
if ! command -v pnnx &> /dev/null; then
    echo "❌ pnnx command not found!"
    echo "   Please install pnnx or add it to PATH"
    exit 1
fi

if [ ! -f "PP-YOLOE/build/ncnnoptimize" ]; then
    echo "⚠️  Custom ncnnoptimize not found at PP-YOLOE/build/ncnnoptimize"
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
cp "$ONNX_FILE" "$TEMP_DIR"


# Model has 2 inputs: scale_factor [1,2] and image [1,3,640,640]
pnnx "${TEMP_DIR}${MODEL_NAME}.onnx"  \
    fp16=1 \
    optlevel=3 \
    device=gpu

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: NCNN Optimize (Preserving Feature Maps)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ncnnoptimize \
    "PP-YOLOE/build/models/pnnx/human_reid.ncnn.param" \
    "PP-YOLOE/build/models/pnnx/human_reid.ncnn.bin" \
    "PP-YOLOE/build/models/human_reid.param" \
    "PP-YOLOE/build/models/human_reid.bin" \
    1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Verify Model Outputs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""


cp -f "${ONNX_FILE}" "../flutter-vision/assets/models/${MODEL_NAME}.onnx"


# and the inference model is in PP-YOLOE/inference_model/ppyoloe_crn_s_36e_pphuman/
echo "Final model is in ../flutter-vision/assets/models/${MODEL_NAME}.onnx, This is reid model for embedding extraction."
