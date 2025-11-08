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
    --output-model pipeline/PP-YOLOE/models/ppyoloe_plus_crn_t_auxhead_320_60e_pphuman_cust.onnx
