#!/usr/bin/env python3
import os, shutil, sys
import onnx

mode, root, out = sys.argv[1:4]

def dims(v):
    return [d.dim_value if d.dim_value else -1 for d in v.type.tensor_type.shape.dim]

rows=[]
for dp, _, fs in os.walk(root):
    for fn in fs:
        if not fn.lower().endswith(".onnx"): continue
        p=os.path.join(dp,fn)
        try:
            m=onnx.load(p, load_external_data=False)
            ins=[(v.name,dims(v)) for v in m.graph.input]
            outs=[(v.name,dims(v)) for v in m.graph.output]
            ok=False
            if mode=="head":
                ok=(len(ins)==1 and ins[0][1]==[1,3,60,60] and
                    sum(max(1, __import__("math").prod([x for x in s if x>0])) for _,s in outs)>=3)
            elif mode=="landmarks":
                ok=(len(ins)==1 and ins[0][1]==[1,3,60,60] and
                    any(__import__("math").prod([x for x in s if x>0])==70 for _,s in outs))
            rows.append((ok,os.path.getsize(p),p,ins,outs))
        except Exception as e:
            print("SKIP",p,repr(e))
for row in sorted(rows,key=lambda x:(not x[0],x[1])):
    print("CANDIDATE",row)
valid=[x for x in rows if x[0]]
if not valid: raise SystemExit(f"No valid {mode} ONNX found under {root}")
# Prefer explicit float32/static files, then smallest.
valid.sort(key=lambda x:(0 if ("float32" in x[2].lower() or "model_float32" in x[2].lower()) else 1,x[1]))
chosen=valid[0]
print("SELECTED",chosen)
os.makedirs(os.path.dirname(out),exist_ok=True)
shutil.copy2(chosen[2],out)
