#!/usr/bin/env python3
import os
import re
import sys

root = sys.argv[1]
expected = {
    "head_pose.param": 3,
    "landmarks35.param": 1,
    "gaze.param": 1,
}
for fn, min_outputs in expected.items():
    p = os.path.join(root, fn)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    text = open(p, encoding="utf-8", errors="replace").read()
    if not text.startswith("7767517"):
        raise SystemExit(f"{fn}: invalid NCNN param magic")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if len(lines) < 4:
        raise SystemExit(f"{fn}: suspiciously short")
    # Output count is inferred by blobs that are produced but never consumed.
    layer_lines = lines[2:]
    produced = set()
    consumed = set()
    for line in layer_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            ni, no = int(parts[2]), int(parts[3])
        except ValueError:
            continue
        start = 4
        for x in parts[start:start+ni]:
            consumed.add(x)
        for x in parts[start+ni:start+ni+no]:
            produced.add(x)
    outputs = sorted(produced - consumed)
    print(fn, "terminal_blobs=", outputs)
    if len(outputs) < min_outputs:
        raise SystemExit(f"{fn}: expected at least {min_outputs} terminal blobs, got {outputs}")
    bin_path = p[:-6] + ".bin"
    if not os.path.exists(bin_path) or os.path.getsize(bin_path) < 1024:
        raise SystemExit(f"{fn}: missing/small bin {bin_path}")
print("OK")
