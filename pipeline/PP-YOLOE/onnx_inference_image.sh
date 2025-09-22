# pipeline/PP-YOLOE/onnx_inference_image.sh
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate the Python virtual environment if it exists
if [[ -f "${REPO_ROOT}/pipeline/PP-YOLOE/venv/bin/activate" ]]; then
    echo "🔧 Activating Python environment..."
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/pipeline/PP-YOLOE/venv/bin/activate"
else
    echo "ℹ️  No venv at pipeline/PP-YOLOE/venv; using system Python"
fi

echo "🔧 Running ONNX inference..."

# Use correct argument names and paths
cd "${REPO_ROOT}"

# Use python from the (possibly) activated venv
python3 pipeline/PP-YOLOE/onnx_inference_image.py \
    --img pipeline/dataset/demo/demo.jpg \
    --onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
    --infer_cfg pipeline/PP-YOLOE/backbone/inference_model/ppyoloe_crn_s_36e_pphuman/infer_cfg.yml \
    --out pipeline/output \
    --check_embed \
    --thresh 0.5

echo "✅ ONNX inference completed! Check pipeline/output/ for visualized result images."