import onnx, sys
from onnx import numpy_helper

path = 'pipeline/PP-YOLOE/models/ppyoloe_crn_s_36e_pphuman_embed.onnx'
try:
    m = onnx.load(path)
except Exception as e:
    print('ERR loading', e)
    sys.exit(1)
print('ir_version:', m.ir_version)
print('opset_import:', [(o.domain or 'ai.onnx', o.version) for o in m.opset_import])
print('inputs:')
for i in m.graph.input:
    t = i.type.tensor_type
    shp = []
    for d in t.shape.dim:
        if d.dim_value:
            shp.append(d.dim_value)
        else:
            shp.append(d.dim_param or '?')
    print(' -', i.name, shp, t.elem_type)
print('outputs:')
for o in m.graph.output:
    t = o.type.tensor_type
    shp = []
    for d in t.shape.dim:
        if d.dim_value:
            shp.append(d.dim_value)
        else:
            shp.append(d.dim_param or '?')
    print(' -', o.name, shp, t.elem_type)
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print('node_count:', len(m.graph.node))
print('unique_ops:', len(ops))
print('top ops:')
for k,v in sorted(ops.items(), key=lambda kv: -kv[1])[:25]:
    print(f'  {k}: {v}')