#!/usr/bin/env python3
import sys
import numpy as np
import openvino as ov

core=ov.Core()

def compile(path):
    m=core.read_model(path)
    return m, core.compile_model(m,"CPU")

def lname(port):
    try:
        ns=sorted(port.get_names())
        if ns: return ns[0]
    except Exception: pass
    try: return port.get_any_name()
    except Exception: return ""

def run_landmarks(irxml, onnx):
    mi,ci=compile(irxml); mo,co=compile(onnx)
    rng=np.random.default_rng(3502)
    worst=0.0
    for _ in range(4):
        x=rng.uniform(0,255,(1,3,60,60)).astype(np.float32)
        a=np.asarray(ci([x])[ci.output(0)]).reshape(-1)
        b=np.asarray(co([x])[co.output(0)]).reshape(-1)
        if a.shape!=b.shape: raise SystemExit(f"landmarks shape mismatch {a.shape} {b.shape}")
        worst=max(worst,float(np.max(np.abs(a-b))))
    print("LANDMARKS_MAX_ABS",worst)
    if worst>2e-3: raise SystemExit(f"landmarks conversion mismatch {worst}")

def classify_inputs(compiled):
    d={}
    for i,p in enumerate(compiled.inputs):
        n=(" ".join(sorted(p.get_names())) if hasattr(p,"get_names") else "").lower()
        shape=list(p.shape)
        if "left" in n: d["left"]=i
        elif "right" in n: d["right"]=i
        elif "head" in n or "pose" in n or shape==[1,3]: d["pose"]=i
    if set(d)!={"left","right","pose"}:
        # Fallback by shapes + order used by the official model.
        eyes=[i for i,p in enumerate(compiled.inputs) if list(p.shape)==[1,3,60,60]]
        pose=[i for i,p in enumerate(compiled.inputs) if list(p.shape)==[1,3]]
        if len(eyes)==2 and len(pose)==1:
            d={"left":eyes[0],"right":eyes[1],"pose":pose[0]}
    return d

def run_gaze(irxml, onnx):
    mi,ci=compile(irxml); mo,co=compile(onnx)
    ii=classify_inputs(ci); oi=classify_inputs(co)
    print("GAZE_ORIG_INPUT_MAP",ii,[(i,lname(p),list(p.shape)) for i,p in enumerate(ci.inputs)])
    print("GAZE_ONNX_INPUT_MAP",oi,[(i,lname(p),list(p.shape)) for i,p in enumerate(co.inputs)])
    if set(ii)!={"left","right","pose"} or set(oi)!={"left","right","pose"}:
        raise SystemExit("could not map gaze inputs")
    rng=np.random.default_rng(2002)
    worst=0.0
    cos_worst=1.0
    for _ in range(8):
        left=rng.uniform(0,255,(1,3,60,60)).astype(np.float32)
        right=rng.uniform(0,255,(1,3,60,60)).astype(np.float32)
        pose=rng.uniform(-30,30,(1,3)).astype(np.float32)
        def call(c,mapping):
            args=[None]*len(c.inputs)
            args[mapping["left"]]=left; args[mapping["right"]]=right; args[mapping["pose"]]=pose
            return np.asarray(c(args)[c.output(0)]).reshape(-1)[:3]
        a=call(ci,ii); b=call(co,oi)
        worst=max(worst,float(np.max(np.abs(a-b))))
        na=np.linalg.norm(a); nb=np.linalg.norm(b)
        if na>1e-9 and nb>1e-9:
            cos=float(np.dot(a,b)/(na*nb)); cos_worst=min(cos_worst,cos)
    print("GAZE_MAX_ABS",worst,"GAZE_MIN_COS",cos_worst)
    # Direction is the critical invariant. Values should also closely match.
    if cos_worst<0.999 or worst>0.02:
        raise SystemExit(f"gaze ONNX mismatch max_abs={worst} min_cos={cos_worst}")

if __name__=="__main__":
    run_landmarks(sys.argv[1],sys.argv[2])
    run_gaze(sys.argv[3],sys.argv[4])
