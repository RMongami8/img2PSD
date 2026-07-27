import numpy as np
import cv2


def extract_ink(
    rgb: np.ndarray,
    radius: int = 3,
    bias: int = -15,
    min_area: int = 24,
    ss: int = 2,
    line_weight: float = 0.0,
    aa: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    s = rgb.shape[0]
    hi = s * ss

    if ss != 1:
        up = cv2.resize(rgb, (hi, hi), interpolation=cv2.INTER_LANCZOS4)
    else:
        up = rgb

    gray = cv2.cvtColor(up, cv2.COLOR_RGB2GRAY).astype(np.float32)

    rad = max(1, round(radius * ss))
    kernel = np.ones((2 * rad + 1, 2 * rad + 1), np.uint8)
    dil = cv2.dilate(gray, kernel)

    line = np.clip(gray / np.maximum(dil, 1e-6) * 255.0, 0, 255)

    lo, hi_v = line.min(), line.max()
    line = (line - lo) * 255.0 / max(1.0, hi_v - lo)

    line_u8 = line.astype(np.uint8)
    otsu, _ = cv2.threshold(line_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(250, max(5, otsu + bias))

    ink = np.where(line < t, 255, 0).astype(np.uint8)

    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if min_area > 0:
        area_px = round(min_area * ss * ss)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < area_px:
                ink[labels == i] = 0

    w = line_weight * ss

    d_in = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_out = cv2.distanceTransform(255 - ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sdf = d_out - d_in
    del d_in, d_out

    ink_hi = np.where(sdf < w, 255, 0).astype(np.uint8)

    if aa:
        cov = np.clip(0.5 - (sdf - w), 0.0, 1.0)
        alpha_hi = (cov * 255.0).astype(np.uint8)
    else:
        alpha_hi = ink_hi
    del sdf
    alpha = cv2.resize(alpha_hi, (s, s), interpolation=cv2.INTER_AREA)

    return ink_hi, alpha
