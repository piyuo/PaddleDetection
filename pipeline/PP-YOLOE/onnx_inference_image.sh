# pipeline/PP-YOLOE/onnx_inference_image.sh
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"


echo "🔧 Running ONNX inference..."

# Use correct argument names and paths
cd "${REPO_ROOT}"

# Use python from the (possibly) activated venv
python3 pipeline/PP-YOLOE/onnx_inference_image.py \
    --img pipeline/dataset/demo/demo.jpg \
    --onnx pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
    --out pipeline/output \
    --thresh 0.5