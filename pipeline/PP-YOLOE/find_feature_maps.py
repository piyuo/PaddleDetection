#!/usr/bin/env python3#!/usr/bin/env python3

# -*- coding: utf-8 -*-"""

"""Find suitable intermediate feature maps in PP-YOLOE model for embedding extraction.

Find potential feature map extraction points for embeddings in PP-YOLOE model."""

"""import onnx

import sys

import sys

sys.path.insert(0, 'pipeline/PP-YOLOE')def find_feature_maps(model_path):

from ncnn_optimize import NCNNOptimizer    """Find candidate feature maps suitable for ROI pooling to generate embeddings."""

    m = onnx.load(model_path)

    g = m.graph

def main():

    opt = NCNNOptimizer()    print('=== Analyzing PP-YOLOE Model for Feature Maps ===')

    opt.parse_param_file('pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param')    print(f'Model: {model_path}')

    print(f'Total nodes: {len(g.node)}\n')

    print('=' * 80)

    print('Looking for backbone feature extraction points...')    # Strategy: Find Conv/BN outputs with reasonable spatial dimensions

    print('=' * 80)    # Typical: C3/C4/C5 features (e.g., 80x80, 40x40, 20x20) with 256-512 channels

        candidates = []

    # Look for layers before the detection head (typically before reshape/permute operations)

    # In YOLO models, good feature points are usually:    for node in g.node:

    # 1. After backbone (before neck)        if node.op_type in ['Conv', 'BatchNormalization', 'Relu', 'Add']:

    # 2. After neck (before head)            for out_name in node.output:

    # 3. Intermediate feature maps with spatial information                # Find shape info

                    shape = None

    print('\nLayers 200-250 (likely backbone/neck transition):')                for vi in g.value_info:

    print('-' * 80)                    if vi.name == out_name:

    for i in range(200, min(250, len(opt.layers))):                        if vi.type.tensor_type.shape.dim:

        layer = opt.layers[i]                            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]

        out_blob = layer.output_blobs[0] if layer.output_blobs else 'none'                        break

        consumers = len(opt.blobs[out_blob].consumers) if out_blob in opt.blobs else 0

                        # Look for 4D tensors with spatial dims 10-80 and channels 128-512

        # Highlight convolution layers                if shape and len(shape) == 4:

        marker = '★' if layer.type in ['Convolution', 'ConvolutionDepthWise'] else ' '                    N, C, H, W = shape

        print(f'{marker} {i:3d}. [{layer.type:<20}] {layer.name:<30} → {out_blob:<15} (c:{consumers})')                    if 10 <= H <= 80 and 10 <= W <= 80 and 128 <= C <= 512:

                            candidates.append({

    print('\n' + '=' * 80)                            'name': out_name,

    print('Concat layers (feature fusion points - good for embeddings):')                            'shape': shape,

    print('=' * 80)                            'op': node.op_type,

    for i, layer in enumerate(opt.layers):                            'node_name': node.name

        if layer.type == 'Concat':                        })

            out_blob = layer.output_blobs[0] if layer.output_blobs else 'none'

            consumers = len(opt.blobs[out_blob].consumers) if out_blob in opt.blobs else 0    # Print candidates

            in_blobs = ', '.join(layer.input_blobs[:3]) + ('...' if len(layer.input_blobs) > 3 else '')    print(f'Found {len(candidates)} candidate feature maps:\n')

            print(f'{i:3d}. {layer.name:<30} → {out_blob:<15} (inputs: {len(layer.input_blobs)}, consumers: {consumers})')    for i, c in enumerate(candidates[:20]):  # Show top 20

            print(f"{i+1}. {c['name']}")

    print('\n' + '=' * 80)        print(f"   Shape: {c['shape']} (NCHW) - {c['shape'][1]} channels, {c['shape'][2]}x{c['shape'][3]} spatial")

    print('Recommended blob names for feature extraction (for BOT-SORT):')        print(f"   Op: {c['op']}, Node: {c['node_name']}")

    print('=' * 80)        print()

    print('Based on typical PP-YOLOE architecture, good feature extraction points are:')

    print('1. Before the detection head (preserves spatial + semantic info)')    # Recommendation

    print('2. After feature fusion in the neck')    if candidates:

    print()        print(f"\n=== Recommendation ===")

            # Prefer medium resolution (20x20 or 40x40) with high channels (256+)

    # Find the last few concat operations before final outputs        best = None

    concat_layers = [(i, l) for i, l in enumerate(opt.layers) if l.type == 'Concat']        for c in candidates:

    if len(concat_layers) >= 2:            H, W, C = c['shape'][2], c['shape'][3], c['shape'][1]

        # Typically want features from neck before head            if 15 <= H <= 45 and C >= 256:

        idx1, layer1 = concat_layers[-3] if len(concat_layers) >= 3 else concat_layers[-2]                best = c

        idx2, layer2 = concat_layers[-2]                break

                if not best and candidates:

        blob1 = layer1.output_blobs[0] if layer1.output_blobs else None            best = candidates[len(candidates)//2]  # Middle candidate

        blob2 = layer2.output_blobs[0] if layer2.output_blobs else None

                if best:

        print(f'Suggested feature map 1: {blob1} (from layer {idx1}: {layer1.name})')            print(f"Best candidate: {best['name']}")

        print(f'Suggested feature map 2: {blob2} (from layer {idx2}: {layer2.name})')            print(f"  • Shape: {best['shape']}")

        print()            print(f"  • Channels: {best['shape'][1]} (embedding dimension after ROI pooling)")

        print('Usage in ncnn_optimize.sh:')            print(f"  • Spatial: {best['shape'][2]}x{best['shape'][3]} (enough resolution for ROI pooling)")

        print(f'--keep "{blob1},{blob2}"')            print(f"\nTo use this feature map, update coreml_graph_surgery.sh:")

            print(f'  --keep-outputs "p2o.pd_op.divide.0.0,p2o.pd_op.concat.14.0,{best["name"]}"')

    else:

if __name__ == '__main__':        print('\nNo suitable feature maps found. Showing all Conv outputs:')

    main()        for node in g.node[:50]:

            if node.op_type == 'Conv':
                print(f'  {node.output[0]} (from {node.op_type})')

    return candidates


if __name__ == '__main__':
    model_path = 'pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    try:
        candidates = find_feature_maps(model_path)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
