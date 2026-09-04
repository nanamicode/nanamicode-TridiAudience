#!/usr/bin/env python3
import sys
import onnx
p = sys.argv[1]
m = onnx.load(p, load_external_data=False)
print("MODEL", p)
print("IR", m.ir_version, "opsets", [(o.domain, o.version) for o in m.opset_import])
for kind, vals in (("INPUT", m.graph.input), ("OUTPUT", m.graph.output)):
    for v in vals:
        tt = v.type.tensor_type
        dims = [d.dim_value if d.dim_value else (d.dim_param or -1) for d in tt.shape.dim]
        print(kind, v.name, tt.elem_type, dims)
