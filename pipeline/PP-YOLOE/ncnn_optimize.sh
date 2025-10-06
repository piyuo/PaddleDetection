# pipeline/PP-YOLOE/ncnn_optimize.sh
#!/usr/bin/env bash


python3 pipeline/PP-YOLOE/ncnn_optimize.py \
    --input-param pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param \
    --input-bin pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.bin \
    --output-param pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.param \
    --output-bin pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_ncnn.bin \
    --keep p2o.pd_op.batch_norm_.13.0,p2o.pd_op.batch_norm_.19.0
