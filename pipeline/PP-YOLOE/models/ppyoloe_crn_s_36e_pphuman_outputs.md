# Model outputs

Total outputs: 3

- 0: fetch_name_0  shape: [dynamic/unknown]  — Detections: [class_id, score, x_min, y_min, x_max, y_max] per row
- 1: fetch_name_1  shape: [1]  — Original model output
- 2: p2o.pd_op.conv2d.3.0  shape: [dynamic/unknown]  — Backbone/neck feature map (NCHW) for downstream embedding/association

Notes:
- Shapes may be dynamic depending on opset and preprocessing; when unknown, infer at runtime.
- The feature map is useful for ROI pooling or computing appearance embeddings for tracking (e.g., BoT-SORT).