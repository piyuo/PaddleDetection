# pipeline/PP-YOLOE/ane_graph_surgery.sh
#!/usr/bin/env bash

# Pipeline PP-YOLOE Apple Neural Engine Graph Surgery
# This script automatically discovers and removes NMS (NonMaxSuppression) from the model
# to enable full ANE acceleration. It also identifies optimal stride-8 and stride-16
# feature maps for multi-scale embedding extraction.
#
# The script will automatically:
# - Find NMS inputs (boxes, scores)
# - Discover stride-8 feature map for fine-grained embeddings
# - Discover stride-16 feature map for semantic embeddings
# - Print comprehensive output guide for inference script integration
#
# NMS and embedding extraction will be performed in the inference script.

python3 pipeline/PP-YOLOE/ane_graph_surgery.py \
--model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust.onnx \
--input-shape 1,3,640,640 --warmup 20 --runs 80 \
--img pipeline/dataset/demo/demo.jpg \
--outdir pipeline/PP-YOLOE/models/surgery \
--fix-input-shapes \
--fold-iterations 15 \
--split-concat 4 \
--fold-static-shapes \
--rewrite-div \
--rewrite-pow \
--rewrite-hardsigmoid \
--rewrite-slice-to-gather \
--rewrite-slice-range-to-gather \
--rewrite-resize-to-static \
--remove-noop-slice \
--rewrite-reduce-to-globalpool \
--output-model pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_cust_ane.onnx

rm -rf pipeline/PP-YOLOE/models/surgery
