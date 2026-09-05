import os
import urllib.request
from collections import namedtuple

import numpy as np
import cv2

MODELS = {
    # both BSD-3-Clause, same licence as the upstream Real-ESRGAN weights
    "anime6b": ("https://huggingface.co/RekluzLabs/realesrgan_anime6b.onnx/"
                "resolve/main/realesrgan_anime6b.onnx", 4),
    "x4plus": ("https://huggingface.co/SceneWorks/real-esrgan-onnx/"
               "resolve/main/real_esrgan_x4.onnx", 4),
}
MODEL_DIR = "models"

# src_box/dst_box are the tile core (native / hi coords). halo is the core's offset
# inside hi_rgb. pad_box + reflect describe exactly what was sliced and mirrored to
# build the patch, so a caller can reproduce the same geometry for its own arrays.
TileView = namedtuple("TileView", "hi_rgb src_box dst_box halo pad_box reflect backend")

_session = None
_session_key = None


def _model_path(name):
    return os.path.join(MODEL_DIR, name + ".onnx")


def ensure_model(name="anime6b", timeout=600):
    """Download the ONNX weights on first use. Returns the path, or raises."""
    path = _model_path(name)
    if os.path.exists(path):
        return path
    url = MODELS[name][0]
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "lineart2psd"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if os.path.getsize(tmp) < 1 << 20:
        os.remove(tmp)
        raise RuntimeError("downloaded model looks truncated: " + url)
    os.replace(tmp, path)
    return path


def _get_session(name):
    """Cached InferenceSession plus the model's own spatial constraint.

    Published Real-ESRGAN exports are not consistent about this: some carry dynamic
    height/width, others are frozen at a fixed tile. Read it off the model rather
    than assuming, otherwise the first inference fails on a shape mismatch.
    """
    global _session, _session_key
    if _session_key == name and _session is not None:
        return _session
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ensure_model(name), opts,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dims = inp.shape[2:4]
    fixed = [d if isinstance(d, int) and d > 0 else None for d in dims]
    _session = (sess, inp.name, sess.get_outputs()[0].name, fixed)
    _session_key = name
    return _session


def _run_onnx(name, tile_rgb, scale):
    """Run at the model's own scale, then resample to the requested one.

    The network has a fixed factor (4 for both bundled models); asking for 2x by
    slicing its 4x output would silently return the top-left quarter of the tile.
    """
    sess, in_name, out_name, fixed = _get_session(name)
    model_scale = MODELS[name][1]
    h, w = tile_rgb.shape[:2]
    ph = fixed[0] or h
    pw = fixed[1] or w
    src = tile_rgb
    if ph != h or pw != w:
        src = cv2.copyMakeBorder(tile_rgb, 0, max(0, ph - h), 0, max(0, pw - w),
                                 cv2.BORDER_REFLECT)[:ph, :pw]
    x = src.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    y = sess.run([out_name], {in_name: x})[0]
    y = np.clip(y[0].transpose(1, 2, 0), 0.0, 1.0)
    out = (y * 255.0 + 0.5).astype(np.uint8)[: h * model_scale, : w * model_scale]
    if scale != model_scale:
        out = cv2.resize(out, (w * scale, h * scale),
                         interpolation=cv2.INTER_AREA if scale < model_scale
                         else cv2.INTER_LANCZOS4)
    return out


def _lanczos(tile_rgb, scale):
    h, w = tile_rgb.shape[:2]
    return cv2.resize(tile_rgb, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def resolve_backend(backend="auto", name="anime6b"):
    """-> ('onnx'|'lanczos', reason). Never raises: a missing model or a missing
    onnxruntime downgrades the quality, it does not break the app."""
    if backend == "lanczos":
        return "lanczos", "requested"
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return "lanczos", "onnxruntime not installed (pip install -r requirements-sr.txt)"
    try:
        ensure_model(name)
    except Exception as e:
        return "lanczos", "model unavailable: %s" % str(e)[:120]
    return "onnx", "ok"


def iter_tiles(rgb, scale=4, tile=256, pad=16, backend="auto", name="anime6b"):
    """Yield upscaled tiles one at a time instead of returning a whole 8192px image.

    The halo is kept on the yielded tile rather than cropped here: the matting stage
    needs context beyond the tile core to find a nearest sure-foreground pixel.
    """
    h, w = rgb.shape[:2]
    mode, reason = resolve_backend(backend, name)
    if mode == "onnx":
        _, _, _, fixed = _get_session(name)
        if fixed[0]:
            tile = min(tile, fixed[0] - 2 * pad)
    tile = max(32, int(tile))

    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            th = min(tile, h - y0)
            tw = min(tile, w - x0)
            sy0, sx0 = max(0, y0 - pad), max(0, x0 - pad)
            sy1, sx1 = min(h, y0 + th + pad), min(w, x0 + tw + pad)
            patch = rgb[sy0:sy1, sx0:sx1]

            top, left = y0 - sy0, x0 - sx0
            bot, right = sy1 - (y0 + th), sx1 - (x0 + tw)
            # reflect only where the image itself ran out, so every tile sees `pad`
            # of context and the model never meets a hard edge mid-image
            reflect = (pad - top, pad - bot, pad - left, pad - right)
            if any(reflect):
                patch = cv2.copyMakeBorder(patch, *reflect, cv2.BORDER_REFLECT)
                top = left = pad

            if mode == "onnx":
                try:
                    hi = _run_onnx(name, patch, scale)
                except Exception as e:
                    mode, reason = "lanczos", "inference failed: %s" % str(e)[:120]
                    hi = _lanczos(patch, scale)
            else:
                hi = _lanczos(patch, scale)

            yield TileView(
                hi_rgb=hi,
                src_box=(x0, y0, tw, th),
                dst_box=(x0 * scale, y0 * scale, tw * scale, th * scale),
                halo=(left * scale, top * scale),
                pad_box=(sx0, sy0, sx1 - sx0, sy1 - sy0),
                reflect=reflect,
                backend="%s (%s)" % (mode, reason) if reason != "ok" else "onnx",
            )


def take_patch(arr, pad_box, reflect):
    """Slice + mirror `arr` exactly the way iter_tiles built its image patch, so
    per-tile auxiliary data lines up with hi_rgb pixel for pixel."""
    x0, y0, w, h = pad_box
    p = arr[y0:y0 + h, x0:x0 + w]
    if not any(reflect):
        return p
    if p.dtype == bool:  # copyMakeBorder has no bool overload
        return cv2.copyMakeBorder(p.astype(np.uint8), *reflect,
                                  cv2.BORDER_REFLECT).astype(bool)
    return cv2.copyMakeBorder(p, *reflect, cv2.BORDER_REFLECT)


def upscale(rgb, scale=4, tile=256, pad=16, backend="auto", name="anime6b"):
    """Whole-image convenience wrapper. The pipeline uses iter_tiles instead."""
    h, w = rgb.shape[:2]
    out = np.empty((h * scale, w * scale, 3), np.uint8)
    used = "lanczos"
    for t in iter_tiles(rgb, scale, tile, pad, backend, name):
        X, Y, W, H = t.dst_box
        lx, ty = t.halo
        out[Y:Y + H, X:X + W] = t.hi_rgb[ty:ty + H, lx:lx + W]
        used = t.backend
    return out, used
