# pipeline/PP-YOLOE/olive_run_no_quant.sh
#!/usr/bin/env bash
set -euo pipefail

CONF="pipeline/PP-YOLOE/onnx_optimize_no_quant.json"
SRC="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx"
DST="pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_olive.onnx"

echo "[Olive] Running peephole-only optimization (no quant)…"
if command -v olive >/dev/null 2>&1; then
  olive run --config "$CONF" || echo "[Olive] Warning: olive run did not produce an updated model; falling back to copying original."
else
  echo "[Olive] olive CLI not found; skipping run and copying original model."
fi

echo "[Olive] Preparing model for inference script: $DST"
cp -f "$SRC" "$DST"
echo "[Olive] Done. Now run: pipeline/PP-YOLOE/onnx_inference_image.sh"

# run inference on demo.jpg
pipeline/PP-YOLOE/onnx_inference_image.sh