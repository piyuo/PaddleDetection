# pipeline/PP-YOLOE/export_to_tensorflow.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv311/bin/activate

python3 pipeline/PP-YOLOE/export_to_tensorflow.py \
  --model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx \
  --output-dir pipeline/PP-YOLOE/models/tensorflow
  #--convert-tflite --quantize