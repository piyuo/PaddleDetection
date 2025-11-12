# pipeline/PP-YOLOE/export_to_tensorflow.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv311/bin/activate

# Input ONNX model
INPUT_MODEL="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_tflite.onnx"
NOSPLIT_MODEL="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_tflite_no_split.onnx"

# Step 1: Replace Split operations with Slice operations to avoid TFLite Flex ops
echo "🔄 Replacing Split operations with Slice operations..."
python3 pipeline/PP-YOLOE/replace_split_with_slice.py "$INPUT_MODEL" "$NOSPLIT_MODEL"

if [ $? -ne 0 ]; then
  echo "❌ Failed to replace Split operations"
  exit 1
fi

# Step 2: Convert the no-split ONNX model to TensorFlow/TFLite
echo ""
echo "🚀 Converting ONNX to TensorFlow/TFLite..."
python3 pipeline/PP-YOLOE/export_to_tensorflow.py \
  --model "$NOSPLIT_MODEL" \
  --output-dir pipeline/PP-YOLOE/models/tensorflow \
  --verbose

cp pipeline/PP-YOLOE/models/tensorflow/ppyoloe_saved_model/ppyoloe_crn_s_36e_pphuman_cust_tflite_no_split_float16.tflite pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_f16.tflite

#rm -f "$NOSPLIT_MODEL"
#rm -rf pipeline/PP-YOLOE/models/tensorflow