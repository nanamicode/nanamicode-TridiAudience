#!/usr/bin/env python3
import json
import sys

head_param,lm_param,gaze_param,mapping_json=sys.argv[1:5]
mapping=json.load(open(mapping_json))

def rewrite(path, mapping):
    lines=open(path,encoding="utf-8").read().splitlines()
    out=[]
    for line in lines:
        if not line.strip() or line.strip()=="7767517" or len(out)==1:
            out.append(line); continue
        parts=line.split()
        parts=[mapping.get(x,x) for x in parts]
        out.append(" ".join(parts))
    open(path,"w",encoding="utf-8").write("\n".join(out)+"\n")

rewrite(head_param, {"in0":"head_image", **mapping})
rewrite(lm_param, {"in0":"face_image","out0":"landmarks35"})
rewrite(gaze_param, {
    "in0":"left_eye_image",
    "in1":"right_eye_image",
    "in2":"head_pose_angles",
    "out0":"gaze_vector",
})
print(open(mapping_json).read())
