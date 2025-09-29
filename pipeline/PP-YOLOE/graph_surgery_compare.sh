# pipeline/PP-YOLOE/graph_surgery_compare.sh
#!/usr/bin/env bash

python3 pipeline/PP-YOLOE/graph_surgery_compare.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
--input-shape 1,3,640,640 --ep coreml --warmup 10 --runs 20 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--ort-profile --ort-profile-dir pipeline/output \
--fix-input-shapes \
--split-concat 8 \
--fold-static-shapes \
--rewrite-div \
--rewrite-pow \
--rewrite-hardswish \
--rewrite-swish \
--rewrite-hardsigmoid \
--rewrite-slice-to-gather
