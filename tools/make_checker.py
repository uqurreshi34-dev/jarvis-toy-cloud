"""Makes the cloud's checker (toy/check.py): a YOLO model exported to ONNX, and its notes.

    pip install ultralytics onnx onnxslim
    python tools/make_checker.py                    # yolo26s at 640 -> checker/yolo26s_640.onnx and .json
    python tools/make_checker.py yolo26n.pt 640     # a smaller one (TOY_CHECK_MODEL), if memory is short

Run on a computer, not on Render: Ultralytics brings PyTorch, far too big for the cloud. The .onnx and
.json are committed, so Render has them as it builds. End-to-end (YOLO26 needs no NMS): one row per
thing, [x1, y1, x2, y2, score, class]. Its weights are kept in half precision, each with a Cast back
to full precision that onnxruntime folds away as it loads: the same model and speed, half the file
(19 MB for yolo26s, under GitHub's and the toy bridge's limits).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from ultralytics import YOLO

weights = sys.argv[1] if len(sys.argv) > 1 else "yolo26s.pt"
size = int(sys.argv[2]) if len(sys.argv) > 2 else 640
folder = Path(__file__).resolve().parents[1] / "checker"
folder.mkdir(exist_ok=True)

model = YOLO(weights)
made = Path(model.export(format="onnx", imgsz=size, opset=17, simplify=True, batch=1))
target = folder / f"{Path(weights).stem}_{size}.onnx"
shutil.move(str(made), target)

# Weights kept in half precision: each float initializer but the smallest becomes float16, and a Cast
# back to float gives the nodes that use it what they had (folded away by onnxruntime as it loads).
graph = onnx.load(str(target))
casts = []

for initializer in list(graph.graph.initializer):
    values = numpy_helper.to_array(initializer)

    if values.dtype != np.float32 or values.size < 1024:
        continue

    half = numpy_helper.from_array(values.astype(np.float16), initializer.name + "_half")
    initializer.CopyFrom(half)
    casts.append(helper.make_node("Cast", [half.name], [half.name[:-len("_half")]], to=onnx.TensorProto.FLOAT,
                                  name=half.name + "_cast"))

nodes = casts + list(graph.graph.node)
del graph.graph.node[:]
graph.graph.node.extend(nodes)
onnx.checker.check_model(graph)
onnx.save(graph, str(target))
names = [model.names[number] for number in range(len(model.names))]
target.with_suffix(".json").write_text(json.dumps({"size": size, "names": names}, indent=1) + "\n")
print(f"{target} and {target.with_suffix('.json').name}: {len(names)} names, {size} x {size}")
