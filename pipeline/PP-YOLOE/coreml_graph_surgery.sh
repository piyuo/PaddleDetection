# pipeline/PP-YOLOE/coreml_graph_surgery.sh
#!/usr/bin/env bash

#--ort-profile --ort-profile-dir pipeline/output \
#--fp16

python3 pipeline/PP-YOLOE/coreml_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx \
--input-shape 1,3,640,640 --ep coreml --warmup 20 --runs 100 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--fix-input-shapes \
--split-concat 4 \
--fold-static-shapes \
--rewrite-div \
--rewrite-pow \
--rewrite-hardsigmoid \
--rewrite-slice-to-gather \
--rewrite-resize-to-static \
--remove-noop-slice \
--rewrite-reduce-to-globalpool \
--fp16 \
--rewrite-slice-range-to-gather \
--ort-profile --ort-profile-dir pipeline/output \
--output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_ane.onnx

rm -rf pipeline/PP-YOLOE/models/surgery
