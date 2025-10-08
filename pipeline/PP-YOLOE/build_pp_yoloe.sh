# pipeline/PP-YOLOE/build_pp_yoloe.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate

# build all PP-YOLOE models, including:
# ane model (pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ane.onnx)
# ncnn model (pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param + .bin)
# onnx model (pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx)


# export to onnx will create base onnx model from PP-YOLOE Paddle model
bash pipeline/PP-YOLOE/export_to_onnx.sh