#!/usr/bin/env python3
"""List all blob names in NCNN model to find feature maps."""

import sys
sys.path.insert(0, 'pipeline/PP-YOLOE')
from ncnn_optimize import NCNNOptimizer

opt = NCNNOptimizer()
opt.parse_param_file('pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param')

print(f"Total blobs: {len(opt.blobs)}\n")

# Search for the specific blobs mentioned
search_patterns = ['batch_norm', '13.0', '19.0', 'p2o', 'pd_op']

print("Searching for batch_norm related blobs:")
print("-" * 80)
for name in sorted(opt.blobs.keys()):
    if any(pattern in name for pattern in search_patterns):
        producer_idx = opt.blobs[name].producer
        if producer_idx >= 0 and producer_idx < len(opt.layers):
            layer = opt.layers[producer_idx]
            print(f"  {name} (from {layer.type})")

print("\n\nAll blob names (first 50):")
print("-" * 80)
for i, name in enumerate(sorted(opt.blobs.keys())[:50]):
    print(f"  {i+1:3d}. {name}")
