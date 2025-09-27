# pipeline/PP-YOLOE/onnx_quantize.sh

#!/bin/bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate


# use olive to quantize the onnx model
olive  run --config pipeline/PP-YOLOE/onnx_quantize.json

# Rename the output model to a more descriptive name
#mv -f output/olive/model.onnx output/rtdetrv3_r18vd_6x.onnx

# run inference on demo.jpg
#python3 tools/onnx_inference.py --debug