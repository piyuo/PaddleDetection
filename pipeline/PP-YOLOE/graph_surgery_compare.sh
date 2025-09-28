# pipeline/PP-YOLOE/graph_surgery_compare.sh
#!/usr/bin/env bash

python3 pipeline/PP-YOLOE/graph_surgery_compare.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
--input-shape 1,3,640,640 --ep coreml --warmup 3 --runs 10 \
--outdir pipeline/PP-YOLOE/models/surgery \
--fix-input-shapes --rewrite-hardswish \
--split-concat 8 \
--fold-static-shapes \
--rewrite-div \
--rewrite-pow
