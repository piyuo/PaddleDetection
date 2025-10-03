# pipeline/PP-YOLOE/coreml_graph_surgery.sh
#!/usr/bin/env bash

# Pipeline PP-YOLOE CoreML Graph Surgery
# This script removes NMS (NonMaxSuppression) and embedding head from the model to enable full ANE acceleration
# NMS and embedding extraction will be performed in the inference script instead

# Note: We remove 'embed' output because it depends on post-NMS boxes via RoiAlign.
# The inference script will compute embeddings using simple feature map pooling after NMS.

python3 pipeline/PP-YOLOE/coreml_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
--input-shape 1,3,640,640 --ep coreml --warmup 20 --runs 100 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--fix-input-shapes \
--keep-outputs "p2o.pd_op.divide.0.0,p2o.pd_op.concat.14.0,p2o.pd_op.batch_norm_.13.0,p2o.pd_op.conv2d.45.0" \
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
--output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ane.onnx

rm -rf pipeline/PP-YOLOE/models/surgery


python3 pipeline/PP-YOLOE/coreml_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx \
--input-shape 1,3,640,640 --ep coreml --warmup 20 --runs 100 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--find-nms