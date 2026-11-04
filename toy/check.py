"""The cloud's second look: a bigger brain checks what the toy's small one thinks it saw.

The toy's own brain (YOLO26n, squeezed to int8, at 256 x 256 on its chip) is quick but imagines
things: an airplane in a bright ceiling, a suitcase in a pillow (October and November 2026, three
rounds of retraining could not stop it without losing beds). So when something new comes into view,
the cloud asks the toy for that look's picture (POST /camera/check, the JPEG as the camera made it)
and looks again with a bigger brain, YOLO26s at 640 x 640 in full precision (four times the toy's
brain, at six times its pixels). Only what both agree on is said. People are not checked here: their
faces are (toy/faces.py).

The checker is an ONNX model and its notes (input size and names, in the order of the model's
classes) beside it, run by onnxruntime, on one thread and without its memory arena:

    checker/yolo26s_640.onnx   YOLO26s (Ultralytics, AGPL-3.0) exported at 640, end-to-end (no NMS)
    checker/yolo26s_640.json   {"size": 640, "names": [...]}
    checker/yolo26n_640.onnx   the toy's own YOLO26n, but at 640 in full precision: the smaller choice

Made by tools/make_checker.py. Measured here under gunicorn as Render runs it (4 November 2026): the
whole service 245 MB before a look, 395 MB after with yolo26s (315 MB with yolo26n), of the free plan's
512; 0.2 s a look on one full CPU (0.1 s for yolo26n), so about two seconds on the free plan's tenth of
one. On 79 of the toy's own pictures it saw none of the 24 airplanes or 27 suitcases the toy imagined;
it also missed some real things the toy saw (half the laptops, two of three phones, the bed), so the
toy says less, but what it says is there. yolo26n saw the same false things as nothing, and fewer real.

TOY_CHECK              "0" turns the check off: everything is said as the toy saw it, as before
TOY_CHECK_MODEL        another checker's .onnx (its .json beside it)
TOY_CHECK_SURE         how sure the checker must be to agree (default 0.25)

The picture is looked at and let go: nothing of it is kept.
"""

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# onnxruntime is imported by the first look, in the process that answers the toy: imported here, the
# app's import would start its thread in gunicorn's parent, which then forks the worker (see faces.py).

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "checker" / "yolo26s_640.onnx"
DEFAULT_SURE = 0.25

_lock = threading.Lock()
_loaded = {"path": None, "session": None, "size": None, "names": None, "error": None, "seconds": None}


def model_path():
    return Path(os.getenv("TOY_CHECK_MODEL") or DEFAULT_MODEL)


def sure():
    try:
        return min(max(float(os.getenv("TOY_CHECK_SURE") or DEFAULT_SURE), 0.05), 0.95)
    except ValueError:
        return DEFAULT_SURE


def ready():
    """True when the check is on and its model is here (loaded on the first look, not now)."""
    path = model_path()
    return os.getenv("TOY_CHECK", "1") != "0" and path.is_file() and path.with_suffix(".json").is_file() \
        and _loaded["error"] is None


def status():
    """For /health?detail=1: "off", "missing <file>", "not loaded yet", "ready (2.1 s a look)" or "failed: why"."""
    if os.getenv("TOY_CHECK", "1") == "0":
        return "off"

    if _loaded["error"]:
        return f"failed: {_loaded['error']}"

    path = model_path()

    if not path.is_file() or not path.with_suffix(".json").is_file():
        return f"missing {path.name} or its .json"

    if _loaded["session"] is None:
        return "not loaded yet"

    seconds = _loaded["seconds"]
    return f"ready ({seconds:.1f} s a look)" if seconds is not None else "ready"


def _load():
    path = model_path()

    if _loaded["session"] is not None and _loaded["path"] == path:
        return

    try:
        notes = json.loads(path.with_suffix(".json").read_text())
        size, names = int(notes["size"]), [str(name) for name in notes["names"]]
        import onnxruntime

        # One thread (Render's free plan has a tenth of a CPU), and no arena: it would keep the most a
        # look ever needed for good, where without it the memory goes back after each look.
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    except Exception as error:   # noqa: BLE001 - a broken checker turns the check off, never the cloud
        _loaded["error"] = f"{type(error).__name__}: {error}"
        print(f"[toy-cloud] check: {path.name} would not load: {_loaded['error']}", flush=True)
        raise

    _loaded.update(path=path, session=session, size=size, names=names, error=None)
    print(f"[toy-cloud] check: {path.name} loaded, {size} x {size}, {len(names)} names", flush=True)


def _letterboxed(picture, size):
    """The picture as Ultralytics trains on it: the aspect kept, grey bars, 0 to 1, red first."""
    height, width = picture.shape[:2]
    scale = size / max(height, width)
    shrunk = cv2.resize(picture, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - shrunk.shape[0]) // 2, (size - shrunk.shape[1]) // 2
    canvas[top:top + shrunk.shape[0], left:left + shrunk.shape[1]] = shrunk
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def found(jpeg):
    """What the checker is sure of in [jpeg]: {name: how many}. Raises when it cannot look."""
    picture = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    if picture is None:
        raise ValueError("not a picture")

    with _lock:   # one look at a time: two at once would double the memory
        _load()
        started = time.monotonic()
        session = _loaded["session"]
        rows = session.run(None, {session.get_inputs()[0].name: _letterboxed(picture, _loaded["size"])})[0][0]
        _loaded["seconds"] = time.monotonic() - started
        names = _loaded["names"]

    least = sure()
    counts = {}

    for row in rows:   # end-to-end: [x1, y1, x2, y2, score, class] for up to 300 things, no NMS needed
        number = int(row[5])

        if row[4] >= least and 0 <= number < len(names):
            counts[names[number]] = counts.get(names[number], 0) + 1

    return counts


def forget():
    """The loaded model, as after a restart (for tests)."""
    _loaded.update(path=None, session=None, size=None, names=None, error=None, seconds=None)
