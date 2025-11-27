# PP-YOLOE/src/reid/human_reid_to_onnx.sh
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_OPSET=16
DEFAULT_OUT_DIR="${REPO_ROOT}/PP-YOLOE/build/models"
OPSET=${DEFAULT_OPSET}
OUT_DIR="${DEFAULT_OUT_DIR}"
MODEL_NAME="human_reid"
ONNX_FILE="${OUT_DIR}/${MODEL_NAME}.onnx"


# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source PP-YOLOE/build/venv/bin/activate

if ! command -v paddle2onnx >/dev/null 2>&1; then
	echo "paddle2onnx not found; installing to user site-packages ..."
	set -x
	"${PY}" -m pip install --user -q paddle2onnx
	set +x
fi

set -x
paddle2onnx \
	--model_dir "${REPO_ROOT}/PP-YOLOE/build/weights/human_reid" \
	--model_filename model.pdmodel \
	--params_filename model.pdiparams \
	--opset_version "${OPSET}" \
    --enable_auto_update_opset False \
    --optimize_tool None \
    --enable_onnx_checker True\
	--save_file "${ONNX_FILE}"
set +x

# and the inference model is in PP-YOLOE/inference_model/ppyoloe_crn_s_36e_pphuman/
echo "Final model is in ${ONNX_FILE}, This is reid model for embedding extraction."
