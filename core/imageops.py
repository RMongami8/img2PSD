import numpy as np
import cv2


def load_rgb(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"failed to decode image: {path}")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha = a.astype(np.float32) / 255.0
        white = np.full_like(b, 255)
        b = (b.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha + white.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        bgr = cv2.merge([b, g, r])
    else:
        bgr = img

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def to_square(rgb: np.ndarray, size: int = 2048, fill: str = "auto", return_box: bool = False):
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)

    if fill == "white":
        fill_color = np.array([255, 255, 255], dtype=np.uint8)
    elif fill == "black":
        fill_color = np.array([0, 0, 0], dtype=np.uint8)
    else:
        border = np.concatenate([
            rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]
        ], axis=0)
        fill_color = np.median(border, axis=0).astype(np.uint8)

    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:, :] = fill_color

    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    box = (left, top, new_w, new_h)
    return (canvas, box) if return_box else canvas
