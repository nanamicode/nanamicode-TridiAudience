#!/usr/bin/env python3
import sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

def replace_initializer(graph, name, arr):
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer.remove(init)
            graph.initializer.insert(i, numpy_helper.from_array(arr.astype(np.int64), name=name))
            return True
    return False

def fix_constant_node(graph, name):
    for node in graph.node:
        if name not in node.output or node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                arr = numpy_helper.to_array(attr.t)
                if np.issubdtype(arr.dtype, np.floating):
                    attr.t.CopyFrom(numpy_helper.from_array(np.rint(arr).astype(np.int64)))
                    return True
    return False

def main(path):
    m = onnx.load(path)
    fixed = []
    for node in m.graph.node:
        if node.op_type == "Reshape":
            # vino2onnx hard-codes allowzero=1, but OpenVINO Reshape uses 0 as
            # "copy this source dimension" for these ADAS graphs.
            found = False
            for attr in node.attribute:
                if attr.name == "allowzero":
                    attr.i = 0
                    found = True
            if not found:
                from onnx import helper
                node.attribute.append(helper.make_attribute("allowzero", 0))
        if node.op_type not in ("Reshape", "Unsqueeze", "Squeeze", "Expand", "Tile"):
            continue
        if len(node.input) < 2:
            continue
        name = node.input[1]
        init = next((x for x in m.graph.initializer if x.name == name), None)
        if init is not None:
            arr = numpy_helper.to_array(init)
            if np.issubdtype(arr.dtype, np.floating):
                replace_initializer(m.graph, name, np.rint(arr))
                fixed.append((node.op_type, name, "initializer"))
                continue
        if fix_constant_node(m.graph, name):
            fixed.append((node.op_type, name, "Constant"))

    # vino2onnx 0.1 emits shape constants as f32 for a few ops. ONNX requires
    # integer shape/axes tensors. The values themselves are integral.
    print("FIXED", path, fixed)
    onnx.checker.check_model(m)
    onnx.save(m, path)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
