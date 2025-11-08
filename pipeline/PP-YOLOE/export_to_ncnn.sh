# pipeline/PP-YOLOE/export_to_ncnn.sh
#!/usr/bin/env bash

set -e

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



echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Verify Model Outputs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 << 'VERIFY_SCRIPT'
import os
import sys

# Check if files exist
param_file = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param"
bin_file = "pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin"

if not os.path.exists(param_file):
    print(f"❌ Param file not found: {param_file}")
    sys.exit(1)

if not os.path.exists(bin_file):
    print(f"❌ Bin file not found: {bin_file}")
    sys.exit(1)

# Parse param file to check outputs
with open(param_file, 'r') as f:
    lines = f.readlines()

# Check for output layers
outputs_found = []
for line in lines:
    if 'out0' in line or 'out1' in line or 'out2' in line or 'out3' in line:
        outputs_found.append(line.strip())

print("✅ Model files verified:")
print(f"   📄 {param_file}")
print(f"   📦 {bin_file}")
print(f"\n📊 Model size:")
param_size = os.path.getsize(param_file) / 1024
bin_size = os.path.getsize(bin_file) / (1024 * 1024)
print(f"   Param: {param_size:.1f} KB")
print(f"   Weights: {bin_size:.2f} MB")

if outputs_found:
    print(f"\n✅ Outputs detected in model (sample):")
    for out in outputs_found[:10]:
        print(f"   {out}")
else:
    print("\n⚠️  Could not detect output layers in param file")

VERIFY_SCRIPT

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Run Inference Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "pipeline/dataset/demo/demo.jpg" ]; then
    echo "🔧 Testing optimized model with inference..."
    ./pipeline/PP-YOLOE/ncnn_inference.sh
else
    echo "⚠️  Test image not found, skipping inference test"
fi


cp -f "${OUTPUT_DIR}${OUTPUT_MODEL_NAME}.bin" "../flutter-vision/assets/models/${OUTPUT_MODEL_NAME}.bin"
cp -f "${OUTPUT_DIR}${OUTPUT_MODEL_NAME}.param" "../flutter-vision/assets/models/${OUTPUT_MODEL_NAME}.param"
rm -rf "$TEMP_DIR"