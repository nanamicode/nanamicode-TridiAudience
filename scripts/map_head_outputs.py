#!/usr/bin/env python3
import itertools
import json
import sys
import numpy as np
import openvino as ov

xml_path, onnx_path, out_json = sys.argv[1:4]
core = ov.Core()
orig_model = core.read_model(xml_path)
onnx_model = core.read_model(onnx_path)
orig = core.compile_model(orig_model, "CPU")
conv = core.compile_model(onnx_model, "CPU")

def friendly_outputs(model):
    out=[]
    results=model.get_results()
    for i,o in enumerate(model.outputs):
        names=[]
        try: names=list(o.get_names())
        except Exception: pass
        fr=""
        try: fr=results[i].get_friendly_name()
        except Exception: pass
        out.append((i, fr, names))
    return out

orig_meta=friendly_outputs(orig_model)
conv_meta=friendly_outputs(onnx_model)
print("ORIGINAL OUTPUTS",orig_meta)
print("CONVERTED OUTPUTS",conv_meta)

rng=np.random.default_rng(1220)
a=[[] for _ in orig.outputs]
b=[[] for _ in conv.outputs]
for _ in range(8):
    x=rng.uniform(0.0,255.0,size=(1,3,60,60)).astype(np.float32)
    ro=orig([x]); rc=conv([x])
    for i in range(len(a)): a[i].append(float(np.asarray(ro[orig.output(i)]).reshape(-1)[0]))
    for j in range(len(b)): b[j].append(float(np.asarray(rc[conv.output(j)]).reshape(-1)[0]))

best=None
for perm in itertools.permutations(range(len(a)), len(b)):
    # perm[j] = original output corresponding to converted output j
    err=sum(float(np.mean((np.asarray(b[j])-np.asarray(a[perm[j]]))**2)) for j in range(len(b)))
    if best is None or err<best[0]: best=(err,perm)
print("OUTPUT MATCH",best)
if best is None or best[0] > 1e-3:
    raise SystemExit(f"head pose conversion does not numerically match original IR: {best}")

sem={}
for i,fr,names in orig_meta:
    s=(fr+" "+" ".join(names)).lower()
    if "angle_y" in s or "yaw" in s: sem[i]="yaw"
    elif "angle_p" in s or "pitch" in s: sem[i]="pitch"
    elif "angle_r" in s or "roll" in s: sem[i]="roll"
if set(sem.values())!={"yaw","pitch","roll"}:
    raise SystemExit(f"could not identify original head outputs: {orig_meta} -> {sem}")

mapping={f"out{j}":sem[best[1][j]] for j in range(len(b))}
print("CANONICAL HEAD MAPPING",mapping)
with open(out_json,"w") as f: json.dump(mapping,f,indent=2,sort_keys=True)
