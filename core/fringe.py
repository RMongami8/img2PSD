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
        # Fractional grayscale morphology preserves soft coverage and avoids the
        # discontinuity caused by thresholding the entire matte at alpha 0.5.
        # Keep the existing sign convention: positive expands, negative shrinks.
        amount = abs(float(choke))
        lo = int(np.floor(amount))
        frac = amount - lo
        op = cv2.dilate if choke > 0 else cv2.erode
        def shifted(radius):
            if radius == 0:
                return alpha_u8.astype(np.float32)
            kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
            return op(alpha_u8, kernel).astype(np.float32)
        a = np.clip((1.0 - frac) * shifted(lo) + frac * shifted(lo + 1)
                    + 0.5, 0, 255).astype(np.uint8)

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
