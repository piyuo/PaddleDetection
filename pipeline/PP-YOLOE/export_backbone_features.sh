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

NODE_NAME="${NODE_NAME:-}"  # allow override via env var
AUTO_PICK="${AUTO_PICK:-s8}" # auto-pick mode when NODE_NAME is empty: largest|s8|s16|s32

if [[ -z "${NODE_NAME}" ]]; then
	echo "ℹ️  No NODE_NAME provided. Auto-picking a feature map (${AUTO_PICK})..."
	"${PY}" pipeline/PP-YOLOE/export_backbone_features.py \
		--onnx "${ONNX_PATH}" \
		--img "${IMG_PATH}" \
		--infer_cfg "${INFER_CFG}" \
		--auto_pick "${AUTO_PICK}" \
		--roi_from_det \
		--thresh 0.5 \
		--export_onnx \
		--out "${OUT_DIR}"
	echo
	echo "Tip: To choose explicitly, list candidates:"
	echo "  ${PY} pipeline/PP-YOLOE/export_backbone_features.py --onnx ${ONNX_PATH} --list-nodes"
	echo "Then run with: NODE_NAME=<tensor> bash pipeline/PP-YOLOE/export_backbone_features.sh"
	exit 0
fi

echo "🔧 Extracting features from tensor: ${NODE_NAME}"
"${PY}" pipeline/PP-YOLOE/export_backbone_features.py \
	--onnx "${ONNX_PATH}" \
	--img "${IMG_PATH}" \
	--infer_cfg "${INFER_CFG}" \
	--node "${NODE_NAME}" \
	--roi_from_det \
	--thresh 0.5 \
	--export_onnx \
	--out "${OUT_DIR}"

echo "✅ Saved feature map and (optional) embeddings to ${OUT_DIR}"