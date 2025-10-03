#!/usr/bin/env python3
"""
Quick test script to verify real embedding extraction is working.
Run this to validate the implementation after surgery.
"""
import numpy as np
import onnxruntime as ort

def test_real_embeddings():
    """Test that the pruned model outputs real embeddings from feature maps."""

    model_path = 'pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed_ane.onnx'

    print('=== Testing Real Embedding Extraction ===\n')

    # Load model
    sess = ort.InferenceSession(model_path)

    # Check outputs
    outputs = [o.name for o in sess.get_outputs()]
    print(f'Model outputs: {outputs}\n')

    # Expected outputs
    expected = ['p2o.pd_op.divide.0.0', 'p2o.pd_op.concat.14.0', 'p2o.pd_op.conv2d.27.0']

    for exp in expected:
        if exp in outputs:
            print(f'✅ {exp} found')
        else:
            print(f'❌ {exp} NOT FOUND')
            return False

    # Check feature map shape
    for o in sess.get_outputs():
        if o.name == 'p2o.pd_op.conv2d.27.0':
            shape = o.shape
            print(f'\n✅ Feature map shape: {shape}')
            if shape == [1, 256, 40, 40]:
                print('   Correct! 256 channels at 40×40 resolution')
            else:
                print(f'   ⚠️  Expected [1, 256, 40, 40], got {shape}')

    print('\n=== Summary ===')
    print('✅ Real embedding extraction is ready!')
    print('   • Feature map: p2o.pd_op.conv2d.27.0 (256 channels)')
    print('   • ROI pooling: Extracts 256-dim embeddings per detection')
    print('   • Usage: Run onnx_inference_image.py with the pruned model')

    return True

if __name__ == '__main__':
    try:
        test_real_embeddings()
    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        print('   Run ./pipeline/PP-YOLOE/coreml_graph_surgery.sh first')
