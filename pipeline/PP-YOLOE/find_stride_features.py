#!/usr/bin/env python3
"""Find feature maps by their characteristics (channels and spatial dimensions)."""

import sys
import re
sys.path.insert(0, 'pipeline/PP-YOLOE')
from ncnn_optimize import NCNNOptimizer

opt = NCNNOptimizer()
opt.parse_param_file('pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param')

print("Looking for feature maps with specific characteristics:")
print("  Target 1: ~128 channels, stride-8 (fine-grained)")
print("  Target 2: ~256 channels, stride-16 (semantic)")
print("=" * 80)

# Look for Convolution layers that might produce these feature maps
candidates_s8 = []
candidates_s16 = []

for i, layer in enumerate(opt.layers):
    # Check convolution layers with output channels
    if layer.type in ['Convolution', 'ConvolutionDepthWise']:
        # Parse num_output parameter
        num_output = layer.params.get('0', layer.params.get('num_output', ''))

        if num_output:
            try:
                channels = int(num_output)
                out_blob = layer.output_blobs[0] if layer.output_blobs else None

                # Look for 128-channel outputs (stride-8)
                if 120 <= channels <= 136 and out_blob:
                    consumers = len(opt.blobs[out_blob].consumers) if out_blob in opt.blobs else 0
                    candidates_s8.append((i, layer.name, out_blob, channels, consumers))

                # Look for 256-channel outputs (stride-16)
                elif 240 <= channels <= 270 and out_blob:
                    consumers = len(opt.blobs[out_blob].consumers) if out_blob in opt.blobs else 0
                    candidates_s16.append((i, layer.name, out_blob, channels, consumers))
            except:
                pass

print(f"\nFound {len(candidates_s8)} candidate(s) for stride-8 (128ch) feature map:")
print("-" * 80)
for idx, name, blob, ch, cons in candidates_s8[-10:]:  # Last 10
    print(f"  Layer {idx:3d}: {name:<35} → blob '{blob}' ({ch}ch, {cons} consumers)")

print(f"\nFound {len(candidates_s16)} candidate(s) for stride-16 (256ch) feature map:")
print("-" * 80)
for idx, name, blob, ch, cons in candidates_s16[-10:]:  # Last 10
    print(f"  Layer {idx:3d}: {name:<35} → blob '{blob}' ({ch}ch, {cons} consumers)")

# Recommend the last few before detection head
if candidates_s8 and candidates_s16:
    print("\n" + "=" * 80)
    print("RECOMMENDED BLOB NAMES FOR ncnn_optimize.sh:")
    print("=" * 80)

    # Take candidates from middle-to-late part (before final head)
    # Usually the best features are after neck but before reshape/permute
    s8_idx, s8_name, s8_blob, s8_ch, s8_cons = candidates_s8[-3] if len(candidates_s8) >= 3 else candidates_s8[-1]
    s16_idx, s16_name, s16_blob, s16_ch, s16_cons = candidates_s16[-3] if len(candidates_s16) >= 3 else candidates_s16[-1]

    print(f"\nStride-8 feature (fine-grained):")
    print(f"  Blob name: '{s8_blob}'")
    print(f"  From layer {s8_idx}: {s8_name} ({s8_ch} channels)")

    print(f"\nStride-16 feature (semantic):")
    print(f"  Blob name: '{s16_blob}'")
    print(f"  From layer {s16_idx}: {s16_name} ({s16_ch} channels)")

    print(f"\nUpdate your ncnn_optimize.sh to:")
    print(f"--keep \"{s8_blob},{s16_blob}\"")
