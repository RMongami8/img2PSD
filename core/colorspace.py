import numpy as np

_I = np.arange(256, dtype=np.float32) / 255.0
SRGB_TO_LINEAR_LUT = np.where(
    _I <= 0.04045, _I / 12.92, ((_I + 0.055) / 1.055) ** 2.4
).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def to_working(rgb_u8: np.ndarray, space: str = "srgb") -> np.ndarray:
    """uint8 sRGB -> float32 [0,1] in the requested working space."""
    if space == "linear":
        return SRGB_TO_LINEAR_LUT[rgb_u8]
    return rgb_u8.astype(np.float32) / 255.0


def from_working(x: np.ndarray, space: str = "srgb") -> np.ndarray:
    """float32 [0,1] working space -> uint8 sRGB."""
    if space == "linear":
        x = _linear_to_srgb(x)
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
