#!/usr/bin/env python3
"""Make a Slovo MViTv2 ONNX model runnable on ONNX Runtime Web's WebGPU backend.

The WebGPU backend can't run grouped (depthwise) 3D convolutions, which MViTv2
uses for attention pooling. Each one is rewritten as an exact equivalent: a
depthwise 3D conv with a k_d x k_h x k_w kernel equals the sum, over the k_d
temporal taps, of depthwise 2D convs applied to time-shifted slices.

Usage:
    pip install onnx
    python tools/convert_webgpu.py web/models/mvit16-4.onnx web/models/mvit16-4-webgpu.onnx
"""

import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def static_shapes(model):
    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = v.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") for d in dims):
            shapes[v.name] = [d.dim_value for d in dims]
    return shapes


def rewrite_depthwise_conv3d(node, x_shape, weight, bias_name, add_init):
    """Return replacement nodes for one grouped Conv3d node."""
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    n, c, d, h, w = x_shape
    c_out, c_per_group, kd, kh, kw = weight.shape
    group = attrs.get("group", 1)
    if group != c or c_out != c or c_per_group != 1:
        raise ValueError(f"{node.name}: only depthwise Conv3d is supported")
    if any(v != 1 for v in attrs.get("dilations", [1, 1, 1])):
        raise ValueError(f"{node.name}: dilation is not supported")
    sd, sh, sw = attrs.get("strides", [1, 1, 1])
    pd0, ph0, pw0, pd1, ph1, pw1 = attrs.get("pads", [0] * 6)
    if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
        raise ValueError(f"{node.name}: auto_pad is not supported")

    d_out = (d + pd0 + pd1 - kd) // sd + 1
    h_out = (h + ph0 + ph1 - kh) // sh + 1
    w_out = (w + pw0 + pw1 - kw) // sw + 1
    p = node.name or node.output[0]
    nodes = []

    # [N, C, D, H, W] -> pad time -> [N, D+pad, C, H, W]
    x = node.input[0]
    if pd0 or pd1:
        pads = add_init(f"{p}_tpads", np.array([0, 0, pd0, 0, 0, 0, 0, pd1, 0, 0], dtype=np.int64))
        nodes.append(helper.make_node("Pad", [x, pads], [f"{p}_tpad"], name=f"{p}_tpad"))
        x = f"{p}_tpad"
    nodes.append(helper.make_node("Transpose", [x], [f"{p}_ndchw"], perm=[0, 2, 1, 3, 4], name=f"{p}_ndchw"))

    to_2d = add_init(f"{p}_to2d", np.array([n * d_out, c, h, w], dtype=np.int64))
    axis = add_init(f"{p}_axis", np.array([1], dtype=np.int64))
    step = add_init(f"{p}_step", np.array([sd], dtype=np.int64))
    taps = []
    for t in range(kd):
        start = add_init(f"{p}_s{t}", np.array([t], dtype=np.int64))
        end = add_init(f"{p}_e{t}", np.array([t + (d_out - 1) * sd + 1], dtype=np.int64))
        wt = add_init(f"{p}_w{t}", np.ascontiguousarray(weight[:, :, t]))
        inputs_2d = [f"{p}_r{t}", wt] + ([bias_name] if bias_name and t == 0 else [])
        nodes += [
            helper.make_node("Slice", [f"{p}_ndchw", start, end, axis, step], [f"{p}_sl{t}"], name=f"{p}_sl{t}"),
            helper.make_node("Reshape", [f"{p}_sl{t}", to_2d], [f"{p}_r{t}"], name=f"{p}_r{t}"),
            helper.make_node(
                "Conv", inputs_2d, [f"{p}_c{t}"], name=f"{p}_c{t}",
                group=c, kernel_shape=[kh, kw], strides=[sh, sw], pads=[ph0, pw0, ph1, pw1],
            ),
        ]
        taps.append(f"{p}_c{t}")

    # Chained Adds rather than Sum: WebGPU has no Sum kernel, and a CPU fallback
    # would copy tensors between GPU and CPU for every pooling layer.
    summed = taps[0]
    for t in range(1, len(taps)):
        nodes.append(helper.make_node("Add", [summed, taps[t]], [f"{p}_add{t}"], name=f"{p}_add{t}"))
        summed = f"{p}_add{t}"
    to_5d = add_init(f"{p}_to5d", np.array([n, d_out, c, h_out, w_out], dtype=np.int64))
    nodes += [
        helper.make_node("Reshape", [summed, to_5d], [f"{p}_r5"], name=f"{p}_r5"),
        helper.make_node("Transpose", [f"{p}_r5"], [node.output[0]], perm=[0, 2, 1, 3, 4], name=f"{p}_ncdhw"),
    ]
    return nodes


def convert(src, dst):
    model = onnx.load(src)
    shapes = static_shapes(model)
    inits = {i.name: i for i in model.graph.initializer}

    def add_init(name, array):
        model.graph.initializer.append(numpy_helper.from_array(array, name))
        return name

    new_nodes = []
    replaced = 0
    for node in model.graph.node:
        is_grouped_3d = (
            node.op_type == "Conv"
            and any(a.name == "group" and a.i > 1 for a in node.attribute)
            and node.input[1] in inits
            and len(inits[node.input[1]].dims) == 5
        )
        if not is_grouped_3d:
            new_nodes.append(node)
            continue
        if node.input[0] not in shapes:
            raise ValueError(f"{node.name}: input shape is not static")
        weight = numpy_helper.to_array(inits[node.input[1]])
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        new_nodes += rewrite_depthwise_conv3d(node, shapes[node.input[0]], weight, bias, add_init)
        replaced += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"rewrote {replaced} grouped Conv3d nodes -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    convert(args.src, args.dst)
