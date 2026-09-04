#!/usr/bin/env python3
import os
import shutil
import sys
import onnx

root, out = sys.argv[1], sys.argv[2]
candidates = []
for dp, _, files in os.walk(root):
    for fn in files:
        if not fn.lower().endswith(".onnx"):
            continue
        p = os.path.join(dp, fn)
        try:
            m = onnx.load(p, load_external_data=False)
            inputs = []
            for v in m.graph.input:
                dims = []
                tt = v.type.tensor_type
                for d in tt.shape.dim:
                    dims.append(d.dim_value if d.dim_value else -1)
                inputs.append((v.name, dims))
            outputs = [(v.name, [d.dim_value if d.dim_value else -1 for d in v.type.tensor_type.shape.dim])
                       for v in m.graph.output]
            shapes = [x[1] for x in inputs]
            ok = len(inputs) == 3 and sum(1 for s in shapes if s == [1,3,60,60]) >= 2 and any(s == [1,3] for s in shapes)
            candidates.append((ok, os.path.getsize(p), p, inputs, outputs))
        except Exception as e:
            print("SKIP", p, repr(e))

for row in sorted(candidates, reverse=True):
    print("CANDIDATE", row)

valid = [r for r in candidates if r[0]]
if not valid:
    raise SystemExit("No gaze ONNX with expected 3-input signature was found")

# Prefer the smallest valid static model if the bundle contains several equivalent variants.
valid.sort(key=lambda x: x[1])
chosen = valid[0]
print("SELECTED", chosen[2], chosen[3], chosen[4])
os.makedirs(os.path.dirname(out), exist_ok=True)
shutil.copy2(chosen[2], out)
