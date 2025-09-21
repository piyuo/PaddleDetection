# pipeline/PP-YOLOE/export_backbone_features.sh
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate

PY="/usr/local/bin/python3"
if ! command -v "${PY}" >/dev/null 2>&1; then
	PY="python3"
fi

cd "${REPO_ROOT}"

ONNX_PATH="pipeline/PP-YOLOE/backbone/ppyoloe_crn_s_36e_pphuman.onnx"
INFER_CFG="pipeline/PP-YOLOE/backbone/inference_model/ppyoloe_crn_s_36e_pphuman/infer_cfg.yml"
IMG_PATH="pipeline/dataset/demo/demo.jpg"
OUT_DIR="pipeline/PP-YOLOE/models"
AUTO_PICK="s8"

echo "🔧 Auto-picking a feature map (${AUTO_PICK})..."
"${PY}" pipeline/PP-YOLOE/export_backbone_features.py \
	--onnx "${ONNX_PATH}" \
	--img "${IMG_PATH}" \
	--infer_cfg "${INFER_CFG}" \
	--auto_pick "${AUTO_PICK}" \
	--roi_from_det \
	--thresh 0.5 \
	--out "${OUT_DIR}"

echo
echo "Tip: To choose explicitly, list candidates:"
echo "  ${PY} pipeline/PP-YOLOE/export_backbone_features.py --onnx ${ONNX_PATH} --list-nodes"

echo "✅ Saved feature map and (optional) embeddings to ${OUT_DIR}"