
# pipeline/PP-YOLOE/olive_run_fp16.sh
#!/usr/bin/env bash
set -euo pipefail

ROOT="pipeline/PP-YOLOE"
CONF="$ROOT/onnx_optimize_fp16.json"
OUT_DIR="$ROOT/models/olive_fp16"
DST="$ROOT/models/ppyoloe_crn_s_36e_pphuman_embed_olive.onnx"

echo "[Olive] Running peephole + FP16 conversion…"
olive run --config "$CONF"

# Find the most recent ONNX in output_dir
FP16_ONNX=$(ls -t "$OUT_DIR"/*.onnx 2>/dev/null | head -n 1 || true)
if [[ -z "${FP16_ONNX}" ]]; then
  echo "[ERROR] No ONNX produced in $OUT_DIR" >&2
  exit 1
fi

cp -f "$FP16_ONNX" "$DST"
echo "[Olive] Prepared FP16 model: $DST"
echo "Now run: $ROOT/onnx_inference_image.sh"
