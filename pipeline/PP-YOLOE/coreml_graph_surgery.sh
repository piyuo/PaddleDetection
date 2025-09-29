# pipeline/PP-YOLOE/coreml_graph_surgery.sh
#!/usr/bin/env bash

#--ort-profile --ort-profile-dir pipeline/output \

python3 pipeline/PP-YOLOE/coreml_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
--input-shape 1,3,640,640 --ep coreml --warmup 10 --runs 20 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_ane \
--fix-input-shapes \
--split-concat 8 \
--fold-static-shapes \
--rewrite-div \
--rewrite-pow \
--rewrite-hardswish \
--rewrite-swish \
--rewrite-hardsigmoid \
--rewrite-slice-to-gather
