#!/usr/bin/env python3
"""
Find suitable intermediate feature maps in PP-YOLOE model for embedding extraction.
"""
import onnx
import sys

def find_feature_maps(model_path):
    """Find candidate feature maps suitable for ROI pooling to generate embeddings."""
    m = onnx.load(model_path)
    g = m.graph

    print('=== Analyzing PP-YOLOE Model for Feature Maps ===')
    print(f'Model: {model_path}')
    print(f'Total nodes: {len(g.node)}\n')

    # Strategy: Find Conv/BN outputs with reasonable spatial dimensions
    # Typical: C3/C4/C5 features (e.g., 80x80, 40x40, 20x20) with 256-512 channels
    candidates = []

    for node in g.node:
        if node.op_type in ['Conv', 'BatchNormalization', 'Relu', 'Add']:
            for out_name in node.output:
                # Find shape info
                shape = None
                for vi in g.value_info:
                    if vi.name == out_name:
                        if vi.type.tensor_type.shape.dim:
                            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                        break

                # Look for 4D tensors with spatial dims 10-80 and channels 128-512
                if shape and len(shape) == 4:
                    N, C, H, W = shape
                    if 10 <= H <= 80 and 10 <= W <= 80 and 128 <= C <= 512:
                        candidates.append({
                            'name': out_name,
                            'shape': shape,
                            'op': node.op_type,
                            'node_name': node.name
                        })

    # Print candidates
    print(f'Found {len(candidates)} candidate feature maps:\n')
    for i, c in enumerate(candidates[:20]):  # Show top 20
        print(f"{i+1}. {c['name']}")
        print(f"   Shape: {c['shape']} (NCHW) - {c['shape'][1]} channels, {c['shape'][2]}x{c['shape'][3]} spatial")
        print(f"   Op: {c['op']}, Node: {c['node_name']}")
        print()

    # Recommendation
    if candidates:
        print(f"\n=== Recommendation ===")
        # Prefer medium resolution (20x20 or 40x40) with high channels (256+)
        best = None
        for c in candidates:
            H, W, C = c['shape'][2], c['shape'][3], c['shape'][1]
            if 15 <= H <= 45 and C >= 256:
                best = c
                break
        if not best and candidates:
            best = candidates[len(candidates)//2]  # Middle candidate

        if best:
            print(f"Best candidate: {best['name']}")
            print(f"  • Shape: {best['shape']}")
            print(f"  • Channels: {best['shape'][1]} (embedding dimension after ROI pooling)")
            print(f"  • Spatial: {best['shape'][2]}x{best['shape'][3]} (enough resolution for ROI pooling)")
            print(f"\nTo use this feature map, update coreml_graph_surgery.sh:")
            print(f'  --keep-outputs "p2o.pd_op.divide.0.0,p2o.pd_op.concat.14.0,{best["name"]}"')
    else:
        print('\nNo suitable feature maps found. Showing all Conv outputs:')
        for node in g.node[:50]:
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
