"""Make the fast copy of a YOLOX model: half precision, and an input of the size asked for.

    python -m tools.make_fast_models <models folder> [--width 640] [--out fast]

The published model files take a square picture (640x640), and a 16:9 frame fills only the top of
it. The copy takes a picture as wide as asked and only as tall as a 16:9 frame of that width needs
(rounded up to 32), with its weights at half precision (float32 in and out). Written to
<models>/<out>/<same name>. Needs the `onnx` package (the service itself does not).
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def convert(src: pathlib.Path, dst: pathlib.Path, width: int) -> str:
    model = onnx.load(str(src))
    g = model.graph
    dim = g.input[0].type.tensor_type.shape.dim
    height = math.ceil((width * 9 / 16) / 32) * 32
    reshapes = [numpy_helper.to_array(next(i for i in g.initializer if i.name == n.input[1])).tolist() for n in g.node if n.op_type == "Reshape"]
    if not all(r[-1] == -1 for r in reshapes):
        return f"{src.name} skipped: a fixed Reshape"
    del g.value_info[:]
    dim[2].dim_value, dim[3].dim_value = height, width
    g.output[0].type.tensor_type.shape.dim[1].dim_value = sum((height // s) * (width // s) for s in (8, 16, 32))
    keep = {n.input[k] for n in g.node if n.op_type == "Resize" for k in (1, 2) if k < len(n.input)}
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT and t.name not in keep:
            t.CopyFrom(numpy_helper.from_array(np.clip(numpy_helper.to_array(t), -65504, 65504).astype(np.float16), t.name))
    inp, outp = g.input[0].name, g.output[0].name
    for n in g.node:
        for k, name in enumerate(n.input):
            if name == inp:
                n.input[k] = inp + "_h"
        for k, name in enumerate(n.output):
            if name == outp:
                n.output[k] = outp + "_h"
    g.node.insert(0, helper.make_node("Cast", [inp], [inp + "_h"], to=TensorProto.FLOAT16, name="to_half"))
    g.node.append(helper.make_node("Cast", [outp + "_h"], [outp], to=TensorProto.FLOAT, name="from_half"))
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return f"{src.name} -> {dst.parent.name}/{dst.name}: {width}x{height}, {dst.stat().st_size // 2**20} MB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", default="fast")
    ap.add_argument("--only", help="one size only, e.g. m")
    args = ap.parse_args()
    models = pathlib.Path(args.models)
    for src in sorted(models.glob("yolox_*.onnx")):
        if args.only and src.stem != f"yolox_{args.only}":
            continue
        print(convert(src, models / args.out / src.name, args.width))


if __name__ == "__main__":
    main()
