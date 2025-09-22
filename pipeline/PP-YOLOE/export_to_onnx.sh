# pipeline/PP-YOLOE/export_to_onnx.sh
#!/usr/bin/env bash

# Activate the Python virtual environment
echo "🔧 Activating Python environment..."
source pipeline/PP-YOLOE/venv/bin/activate

# Export PP-YOLOE Human model to ONNX for ONNX Runtime use
# - Exports Paddle inference model (tools/export_model.py)
# - Converts to ONNX using paddle2onnx
#
# Defaults target: configs/pphuman/ppyoloe_crn_s_36e_pphuman.yml
#
# Usage:
#   bash pipeline/PP-YOLOE/export_to_onnx.sh [--shape 3,640,640] [--opset 11] \
#        [--config <cfg.yml>] [--weights <model.pdparams>] [--out-dir <dir>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
DEFAULT_CONFIG="${REPO_ROOT}/configs/pphuman/ppyoloe_crn_s_36e_pphuman.yml"
DEFAULT_WEIGHTS="${REPO_ROOT}/pipeline/PP-YOLOE/weights/ppyoloe_crn_s_36e_pphuman.pdparams"
DEFAULT_SHAPE="3,640,640"  # C,H,W
DEFAULT_OPSET=16
DEFAULT_OUT_DIR="${REPO_ROOT}/pipeline/PP-YOLOE/backbone"

CONFIG="${DEFAULT_CONFIG}"
WEIGHTS="${DEFAULT_WEIGHTS}"
SHAPE="${DEFAULT_SHAPE}"
OPSET=${DEFAULT_OPSET}
OUT_DIR="${DEFAULT_OUT_DIR}"

function usage() {
	cat <<EOF
Export PaddleDetection PP-YOLOE Human model to ONNX

Options:
	--config,  -c <file>   Config YAML (default: ${DEFAULT_CONFIG})
	--weights, -w <file>   Weights pdparams (default: ${DEFAULT_WEIGHTS})
	--shape,   -s C,H,W    Fixed input shape (default: ${DEFAULT_SHAPE})
	--opset,   -o <int>    ONNX opset (default: ${DEFAULT_OPSET})
	--out-dir, -d <dir>    Output dir root (default: ${DEFAULT_OUT_DIR})
	-h, --help             Show help

Notes:
	- YOLO-family ONNX export requires fixed shape; batch=1 is implied.
	- Will try to use python3, falling back to python.
	- If pipeline/PP-YOLOE/venv exists, it will be sourced automatically.
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
	case "$1" in
		-c|--config)  CONFIG="$2"; shift 2;;
		-w|--weights) WEIGHTS="$2"; shift 2;;
		-s|--shape)   SHAPE="$2"; shift 2;;
		-o|--opset)   OPSET="$2"; shift 2;;
		-d|--out-dir) OUT_DIR="$2"; shift 2;;
		-h|--help)    usage; exit 0;;
		*) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 2;;
	esac
done

MODEL_NAME="$(basename "${CONFIG}")"; MODEL_NAME="${MODEL_NAME%.yml}"
INFER_ROOT="${OUT_DIR}/inference_model"
MODEL_DIR="${INFER_ROOT}/${MODEL_NAME}"
ONNX_FILE="${OUT_DIR}/${MODEL_NAME}.onnx"

echo "Repo root : ${REPO_ROOT}"
echo "Config    : ${CONFIG}"
echo "Weights   : ${WEIGHTS}"
echo "Shape     : ${SHAPE}"
echo "Opset     : ${OPSET}"
echo "Infer dir : ${MODEL_DIR}"
echo "ONNX file : ${ONNX_FILE}"
echo


# Choose python executable
PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
	if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; else echo "[ERROR] python not found" >&2; exit 1; fi
fi

# Validate inputs
[[ -f "${CONFIG}" ]]  || { echo "[ERROR] Config not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${WEIGHTS}" ]] || { echo "[ERROR] Weights not found: ${WEIGHTS}" >&2; exit 1; }

mkdir -p "${INFER_ROOT}" "${OUT_DIR}"

echo "[1/2] Exporting Paddle inference model ..."
set -x
"${PY}" "${REPO_ROOT}/tools/export_model.py" \
	-c "${CONFIG}" \
	-o weights="${WEIGHTS}" TestReader.inputs_def.image_shape="[${SHAPE}]" use_gpu=false \
	--output_dir "${INFER_ROOT}"
set +x

if [[ ! -f "${MODEL_DIR}/model.pdmodel" || ! -f "${MODEL_DIR}/model.pdiparams" ]]; then
	echo "[ERROR] Export failed; expected model files not found under ${MODEL_DIR}" >&2
	exit 1
fi

echo "[2/2] Converting to ONNX with paddle2onnx ..."
if ! command -v paddle2onnx >/dev/null 2>&1; then
	echo "paddle2onnx not found; installing to user site-packages ..."
	set -x
	"${PY}" -m pip install --user -q paddle2onnx
	set +x
fi

set -x
paddle2onnx \
	--model_dir "${MODEL_DIR}" \
	--model_filename model.pdmodel \
	--params_filename model.pdiparams \
	--opset_version "${OPSET}" \
    --enable_auto_update_opset False \
    --optimize_tool None \
    --enable_onnx_checker True\
	--save_file "${ONNX_FILE}"
set +x

if [[ -f "${ONNX_FILE}" ]]; then
	echo
	echo "✅ Export successful"
	echo "- Paddle inference: ${MODEL_DIR}/ (model.pdmodel, model.pdiparams, infer_cfg.yml)"
	echo "- ONNX model      : ${ONNX_FILE}"
	echo
	echo "Optional: quick onnxruntime sanity-check (CPU)"
	echo "  ${PY} deploy/third_engine/onnx/infer.py \\\n+      --infer_cfg ${MODEL_DIR}/infer_cfg.yml \\\n+      --onnx_file ${ONNX_FILE} \\\n+      --image_file demo/000000014439.jpg"
else
	echo "[ERROR] ONNX file not generated: ${ONNX_FILE}" >&2
	exit 1
fi

#./pipeline/PP-YOLOE/export_backbone_features.sh
python3 pipeline/PP-YOLOE/insert_embedding_head.py \
    --onnx_in pipeline/PP-YOLOE/backbone/ppyoloe_crn_s_36e_pphuman.onnx \
	--onnx_out pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
	--use_inst_norm \
	--gp_w 0.2 \
	--pp_w 0.8 \
	--color_gain 0.0 \
	--avg_w 1.0 \
	--max_w 0.0 \
	--pp_k 9 \
	--pp_stripe_h 2 \
	--pp_vertical_k 2 \
	--pp_vertical_stripe_w 2 \
	--pooled_hw 16 \
	--pl_alpha 0.35 \
	--pre_norm_scales \
	--sampling_ratio 2 \
	--max_probe 20
