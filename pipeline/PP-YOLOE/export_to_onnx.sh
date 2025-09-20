# pipeline/PP-YOLOE/export_to_onnx.sh
#!/bin/bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/venv/bin/activate

python3 tools/export_model.py \
 -c configs/pphuman/ppyoloe_crn_s_36e_pphuman.yml \
 -o weights=pipeline/PP-YOLOE/weights/ppyoloe_crn_s_36e_pphuman.pdparams  \
 --output_dir pipeline/output
