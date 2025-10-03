#!/usr/bin/env python3
"""
Probe PP-YOLOE model to find stride-8 and stride-16 feature maps.
"""
import onnx
import onnxruntime as ort
import numpy as np

def probe_model(model_path, img_hw=(640, 640)):
    """Run inference and find suitable feature maps for embedding extraction."""

    print(f'=== Probing Model: {model_path} ===\n')

    # Load model
    m = onnx.load(model_path)

    # Find candidate feature nodes (Conv, BN outputs)
    candidates = []
    for node in m.graph.node:
        if node.op_type in ['Conv', 'BatchNormalization']:
            for out in node.output:
                if not any(bad in out for bad in ['.w_', '.b_', 'constant', 'scale', 'bias']):
                    candidates.append((out, node.op_type))

    print(f'Found {len(candidates)} candidate nodes to probe\n')

    # Create temp models with extra outputs and probe
    probed = []
    H, W = img_hw

    for i, (name, op_type) in enumerate(candidates[:80]):  # Limit to 80 probes
        try:
            # Add this as an output
            m_temp = onnx.load(model_path)
            found = False
            for vi in list(m_temp.graph.value_info) + list(m_temp.graph.output):
                if vi.name == name:
                    m_temp.graph.output.append(vi)
                    found = True
                    break

            if not found:
                # Create generic output
                vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [])
                m_temp.graph.output.append(vi)

            # Save and run
            temp_path = f'/tmp/probe_{i}.onnx'
            onnx.save(m_temp, temp_path)

            sess = ort.InferenceSession(temp_path, providers=['CPUExecutionProvider'])

            # Create dummy input
            feed = {}
            for inp in sess.get_inputs():
                if 'image' in inp.name.lower():
                    feed[inp.name] = np.random.randn(1, 3, H, W).astype(np.float32)
                elif 'shape' in inp.name.lower():
                    feed[inp.name] = np.array([[H, W]], dtype=np.float32)
                elif 'scale' in inp.name.lower():
                    feed[inp.name] = np.array([[1.0, 1.0]], dtype=np.float32)

            # Run
            outputs = sess.run(None, feed)

            # Find the probed output
            out_names = [o.name for o in sess.get_outputs()]
            if name in out_names:
                idx = out_names.index(name)
                shape = list(outputs[idx].shape)

                # Only keep 4D feature maps
                if len(shape) == 4 and shape[0] == 1:
                    _, C, Hf, Wf = shape
                    if 10 <= Hf <= 80 and 10 <= Wf <= 80 and 64 <= C <= 512:
                        stride_h = H / Hf
                        stride_w = W / Wf
                        stride_avg = (stride_h + stride_w) / 2
                        probed.append({
                            'name': name,
                            'shape': shape,
                            'stride': stride_avg,
                            'op': op_type
                        })
                        print(f'✓ {name}: {shape} (stride≈{stride_avg:.1f})')

        except Exception as e:
            pass

    # Organize by stride
    print(f'\n=== Found {len(probed)} suitable feature maps ===\n')

    # Find stride 8 and stride 16 candidates
    stride8_candidates = [f for f in probed if 6 <= f['stride'] <= 10]
    stride16_candidates = [f for f in probed if 12 <= f['stride'] <= 20]

    print(f'Stride-8 candidates ({len(stride8_candidates)}):')
    for f in stride8_candidates:
        print(f"  {f['name']}: {f['shape']} (stride={f['stride']:.1f})")

    print(f'\nStride-16 candidates ({len(stride16_candidates)}):')
    for f in stride16_candidates:
        print(f"  {f['name']}: {f['shape']} (stride={f['stride']:.1f})")

    # Pick best candidates
    best_s8 = stride8_candidates[-1]['name'] if stride8_candidates else None
    best_s16 = stride16_candidates[-1]['name'] if stride16_candidates else None

    print(f'\n=== Recommendation ===')
    if best_s8:
        s8_info = [f for f in probed if f['name'] == best_s8][0]
        print(f'Stride-8:  {best_s8}')
        print(f'  Shape: {s8_info["shape"]} ({s8_info["shape"][1]} channels)')
    if best_s16:
        s16_info = [f for f in probed if f['name'] == best_s16][0]
        print(f'Stride-16: {best_s16}')
        print(f'  Shape: {s16_info["shape"]} ({s16_info["shape"][1]} channels)')

    if best_s8 and best_s16:
        total_dim = s8_info['shape'][1] + s16_info['shape'][1]
        print(f'\nTotal embedding dimension (multi-scale): {total_dim}')
        print(f'\nTo use in surgery script:')
        print(f'  --keep-outputs "p2o.pd_op.divide.0.0,p2o.pd_op.concat.14.0,{best_s8},{best_s16}"')

    return probed

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman.onnx'
    probe_model(model_path)
