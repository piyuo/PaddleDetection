# pipeline/PP-YOLOE/onnx_olive.sh

#!/bin/bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate


# use olive to quantize the onnx model
olive  run --config pipeline/PP-YOLOE/onnx_olive.json

# Rename the output model to a more descriptive name
mv -f pipeline/PP-YOLOE/models/olive/model.onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_olive.onnx

# run inference on demo.jpg
pipeline/PP-YOLOE/onnx_inference_image.sh