# PP-YOLOE/build_human.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source PP-YOLOE/build/venv/bin/activate

set -e

MODEL_NAME="ppyoloe_crn_s_36e_pphuman"
MODEL_WEIGHT="PP-YOLOE/build/weights/${MODEL_NAME}.pdparams"
MODEL_CONFIG="configs/pphuman/${MODEL_NAME}.yml"
MODEL_PRODUCT_NAME="human"
DEMO_JPG="PP-YOLOE/build/dataset/demo/human.png"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Onnx Model Build.                                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""


PP-YOLOE/src/paddle_to_onnx.sh \
    --config "${MODEL_CONFIG}"  \
	--weights "${MODEL_WEIGHT}"

# Run customized
python3 PP-YOLOE/src/onnx_customize.py \
    --model "PP-YOLOE/build/models/${MODEL_NAME}.onnx" \
    --output-model "PP-YOLOE/build/models/${MODEL_NAME}_cust.onnx"

# ane graph surgery to create ANE optimized onnx model
PP-YOLOE/src/ane_graph_surgery.sh \
    --model "PP-YOLOE/build/models/${MODEL_NAME}_cust.onnx" \
    --output-model "PP-YOLOE/build/models/${MODEL_NAME}_cust_ane.onnx" \
    --img "${DEMO_JPG}"

# cleanup the ANE onnx model
python3 PP-YOLOE/src/onnx_cleanup.py \
    --model "PP-YOLOE/build/models/${MODEL_NAME}_cust_ane.onnx" \
    --output-model "PP-YOLOE/build/models/${MODEL_NAME}_cust_ane_cu.onnx"

# Run inference
python3 PP-YOLOE/src/onnx_inference_image.py \
    --img "${DEMO_JPG}" \
    --onnx "PP-YOLOE/build/models/${MODEL_NAME}_cust_ane_cu.onnx"

# Move final model
cp -f "PP-YOLOE/build/models/${MODEL_NAME}_cust_ane_cu.onnx" "PP-YOLOE/build/output/${MODEL_PRODUCT_NAME}.onnx"
cp -f "PP-YOLOE/build/output/${MODEL_PRODUCT_NAME}.onnx" "../flutter-vision/assets/models/${MODEL_PRODUCT_NAME}.onnx"


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

mkdir -p "PP-YOLOE/build/models/pnnx/"
python3 PP-YOLOE/src/ncnn_graph_surgery.py \
	--model "PP-YOLOE/build/models/${MODEL_NAME}_cust.onnx" \
	--input-shape 1,3,640,640 \
	--fix-input-shapes \
	--fold-iterations 15 \
	--split-concat 4 \
	--fold-static-shapes \
	--rewrite-div \
	--rewrite-pow \
	--rewrite-slice-to-gather \
	--rewrite-slice-range-to-gather \
	--rewrite-resize-to-static \
	--remove-noop-slice \
	--remove-identity \
	--rewrite-reduce-to-globalpool \
	--output-model "PP-YOLOE/build/models/pnnx/${MODEL_NAME}_cust.onnx"

pnnx "PP-YOLOE/build/models/pnnx/${MODEL_NAME}_cust.onnx"  \
    fp16=1 \
    optlevel=3 \
    device=gpu \
    inputshape="[1,2]f32,[1,3,640,640]f32"

PP-YOLOE/build/ncnnoptimize \
    "PP-YOLOE/build/models/pnnx/${MODEL_NAME}_cust.ncnn.param" \
    "PP-YOLOE/build/models/pnnx/${MODEL_NAME}_cust.ncnn.bin" \
    "PP-YOLOE/build/models/${MODEL_NAME}_ncnn.param" \
    "PP-YOLOE/build/models/${MODEL_NAME}_ncnn.bin" \
    1

python3 PP-YOLOE/src/ncnn_inference.py \
    --img "${DEMO_JPG}" \
    --ncnn_param "PP-YOLOE/build/models/${MODEL_NAME}_ncnn.param" \
    --ncnn_bin "PP-YOLOE/build/models/${MODEL_NAME}_ncnn.bin"

cp -f "PP-YOLOE/build/models/${MODEL_NAME}_ncnn.bin" "PP-YOLOE/build/output/${MODEL_PRODUCT_NAME}.bin"
cp -f "PP-YOLOE/build/models/${MODEL_NAME}_ncnn.param" "PP-YOLOE/build/output/${MODEL_PRODUCT_NAME}.param"
cp -f "PP-YOLOE/build/output/${MODEL_PRODUCT_NAME}.bin" "../flutter-vision/assets/models/${MODEL_PRODUCT_NAME}.bin"
cp -f "PP-YOLOE/build/output/${MODEL_PRODUCT_NAME}.param" "../flutter-vision/assets/models/${MODEL_PRODUCT_NAME}.param"


# cleanup temporary files
rm -rf PP-YOLOE/build/models/
mkdir -p PP-YOLOE/build/models/

# show the final output onnx and ncnn files
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Final exported models:"
echo " ONNX Model: PP-YOLOE/build/output/${MODEL_PRODUCT_NAME}.onnx"
echo " NCNN Model: PP-YOLOE/build/output/${MODEL_PRODUCT_NAME}.param and .bin"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "✅ Export completed successfully!"
