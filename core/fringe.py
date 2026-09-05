import numpy as np
import cv2


def adjust(alpha_u8, choke=0.0, feather=0.0, gamma=1.0):
    """Shape-only tuning of the cut matte.

    choke works on a signed distance field rather than erode/dilate so it moves the
    edge in sub-pixel steps, matching how lineart.extract_ink handles line weight.

    Deliberately never touches colour: decontamination runs off alpha_source, so
    moving this slider cannot shift the recovered foreground colour.
    """
    a = alpha_u8
    if abs(choke) > 1e-6:
        binary = np.where(a > 127, 255, 0).astype(np.uint8)
        d_in = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d_out = cv2.distanceTransform(255 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sdf = d_out - d_in
        del d_in, d_out
        cov = np.clip(0.5 - (sdf - float(choke)), 0.0, 1.0)
        del sdf
        a = (cov * 255.0 + 0.5).astype(np.uint8)

    if feather > 1e-6:
        k = int(max(1, round(feather * 3.0)) * 2 + 1)
        a = cv2.GaussianBlur(a, (k, k), float(feather))

    if abs(gamma - 1.0) > 1e-6:
        x = a.astype(np.float32) / 255.0
        a = (np.power(x, 1.0 / max(gamma, 1e-3)) * 255.0 + 0.5).astype(np.uint8)

    return a


def compose_rgba(fg_rgb, alpha_u8):
    h, w = alpha_u8.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = fg_rgb
    out[..., 3] = alpha_u8
    return out
