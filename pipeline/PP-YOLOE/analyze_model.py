#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze NCNN model structure to find potential feature extraction points.
"""

import sys
from collections import Counter
sys.path.insert(0, 'pipeline/PP-YOLOE')
from ncnn_optimize import NCNNOptimizer


def main():
    opt = NCNNOptimizer()
    opt.parse_param_file('pipeline/PP-YOLOE/models/ncnn/ppyoloe_crn_s_36e_pphuman_ncnn.ncnn.param')

    print(f'Total layers: {len(opt.layers)}')
    print(f'Total blobs: {len(opt.blobs)}')
    print()

    # Count layer types
    layer_types = Counter(layer.type for layer in opt.layers)
    print('Layer type distribution:')
    print('-' * 80)
    for ltype, count in sorted(layer_types.items(), key=lambda x: -x[1])[:15]:
        print(f'  {ltype:<30} {count:>3}')

    print()
    print('=' * 80)
    print('Potential feature extraction points (last 30 layers):')
    print('=' * 80)

    start_idx = max(0, len(opt.layers) - 30)
    for i in range(start_idx, len(opt.layers)):
        layer = opt.layers[i]
        out_blobs = ', '.join(layer.output_blobs) if layer.output_blobs else 'none'
        num_consumers = 0
        if layer.output_blobs and layer.output_blobs[0] in opt.blobs:
            num_consumers = len(opt.blobs[layer.output_blobs[0]].consumers)

        print(f'{i:3d}. [{layer.type:<20}] {layer.name:<30} → {out_blobs:<20} (consumers: {num_consumers})')

    print()
    print('=' * 80)
    print('Convolution/Pooling layers with multiple consumers (backbone features):')
    print('=' * 80)

    for i, layer in enumerate(opt.layers):
        if layer.type in ['Convolution', 'ConvolutionDepthWise', 'Pooling']:
            for out_blob in layer.output_blobs:
                if out_blob in opt.blobs:
                    num_consumers = len(opt.blobs[out_blob].consumers)
                    if num_consumers > 1:
                        print(f'{i:3d}. {layer.name:<40} → {out_blob:<20} (consumers: {num_consumers})')


if __name__ == '__main__':
    main()
