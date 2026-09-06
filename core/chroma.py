import numpy as np
import cv2

from core import colorspace

KEY_PRESETS = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
    "blue": (0, 0, 255),
}

# ||F - K||^2 below which the projection is numerically unstable (F sits on top of K).
# This guard does NOT detect an F mis-propagated from another part: such an F is far
# from K and passes right through here. The residual guard is what catches that.
# Measured on benchmark/ edge pixels: sRGB min 42436 / p1 47524, linear min 0.6335.
DENOM_MIN = {"srgb": 2500.0, "linear": 0.04}
ALPHA_MARGIN = 0.1
# Measured on benchmark edge pixels: residual median 5.2, p99 38.2 with a good F,
# while an F taken from the wrong part lands around 144. 50 keeps 99.9% of correct
# estimates and still rejects mis-propagation with room to spare.
RESIDUAL_MAX = {"srgb": 50.0, "linear": 0.19}

# F must come from pixels that are genuinely opaque, not merely "not background".
# Sweeping this on the benchmark: 90 -> alpha_mae_edge ~30, 120 -> ~14, 140 -> ~11.4,
# 180 -> ~24. Loose thresholds let key-contaminated pixels seed F and bias alpha.
# Hue distance at which a pixel is considered unrelated to the key. Measured on a
# real generated frame: flat background p95 0.019, genuine partial-alpha pixels p1
# 0.499, certain subject p1 0.694. 0.35 sits 18x above the background side and 2x
# below the subject side.
HUE_FULL = 0.35

STRICT_FG_DEFAULT = 140.0
STRICT_BG_DEFAULT = 8.0
MIN_SURE_FG_FRACTION = 0.02


def estimate_key_color(rgb: np.ndarray, preset: str = "green") -> np.ndarray:
    """Median of the pixels that already look like the preset key colour.

    Sampling the image border instead would be cheaper, but it silently returns a
    hair or skin colour whenever the subject reaches the frame edge -- and a wrong K
    does not degrade the matte, it destroys it. Anchoring on the preset keeps the
    estimate correct no matter where the background sits, while still absorbing the
    few-LSB drift of a generated background away from exact #00FF00.
    """
    base = np.array(KEY_PRESETS.get(preset, KEY_PRESETS["green"]), np.float32)
    d = np.linalg.norm(rgb.astype(np.float32) - base, axis=-1)
    near = d < 90.0
    if near.mean() < 0.002:
        return base
    return np.median(rgb[near].astype(np.float32), axis=0)


def key_hue_distance(rgb: np.ndarray, key_rgb: np.ndarray) -> np.ndarray:
    """Distance in the normalised chromaticity plane -- hue only, luminance removed.

    This is what recognises a *darkened* key colour as still being the key. It is the
    piece the CbCr distance cannot supply, because CbCr magnitude scales with
    luminance and therefore reports a dark green as far from a bright green.
    """
    f = rgb.astype(np.float32)
    s = f.sum(axis=-1, keepdims=True) + 1e-6
    k = np.asarray(key_rgb, np.float32).reshape(3)
    ck = k / max(float(k.sum()), 1e-6)
    return np.linalg.norm(f / s - ck, axis=-1)


def key_distance(rgb: np.ndarray, key_rgb: np.ndarray) -> np.ndarray:
    """CbCr-plane distance, damped where the pixel's hue is the key's own.

    The CbCr term is insensitive to luminance in the sense that a dark line against a
    saturated key separates cleanly -- but it is *proportional* to chroma magnitude,
    so a darkened key colour also sits far from a bright key and reads as foreground.
    Generated art is full of exactly that: the model renders fine hair as a darkened
    background rather than as hair, giving pixels like [0,90,0] that are 78 away from
    a [8,243,3] key in CbCr yet are unmistakably background by hue.

    Damping by hue is free on genuine partial-alpha pixels (measured p1 hue 0.499,
    weight already 1.0) and collapses the distance for those impostors, which fixes
    every downstream consumer at once: the fallback alpha, the trimap, the seeds, and
    the per-tile sure-foreground test in maskgen.
    """
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    k = cv2.cvtColor(
        np.clip(key_rgb, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2YCrCb
    ).astype(np.float32).reshape(3)
    d = ycc[..., 1:] - k[1:]
    dist = np.sqrt(np.einsum("...i,...i->...", d, d))
    w = np.clip(key_hue_distance(rgb, key_rgb) / HUE_FULL, 0.0, 1.0)
    return dist * w


def distance_alpha(dist: np.ndarray, tol: float, soft: float) -> np.ndarray:
    """Plain distance key. Baseline, and the fallback when matting is not trustworthy."""
    return np.clip((dist - tol) / max(soft, 1e-3), 0.0, 1.0).astype(np.float32)


def make_trimap(dist: np.ndarray, tol: float, soft: float):
    """-> (trimap, candidate_bg, candidate_fg); trimap: 0=bg, 1=fg, 2=unknown.

    candidate_* deliberately stay narrow: seeding region growth over the whole
    unknown band lets a single seed leak across a line the upscaler has blurred.
    """
    bg = dist <= tol
    fg = dist >= tol + soft
    unknown = ~(bg | fg)
    trimap = np.where(bg, 0, np.where(fg, 1, 2)).astype(np.uint8)
    candidate_bg = bg | (unknown & (dist < tol * 1.5))
    candidate_fg = fg | (unknown & (dist > tol + soft * 0.5))
    return trimap, candidate_bg, candidate_fg


def native_seeds(rgb: np.ndarray, key_rgb: np.ndarray,
                 strict_bg: float = STRICT_BG_DEFAULT,
                 strict_fg: float = STRICT_FG_DEFAULT, dist: np.ndarray = None):
    """Strict-threshold certainties at native resolution.

    No erosion on either side. On the background side it would delete exactly the
    1px gaps between hair strands this mechanism exists to protect; on the
    foreground side it measurably degrades F (benchmark: 10.9 -> 13.4 at one
    erosion step), because it discards good interior pixels for nothing.

    The foreground threshold relaxes if it would leave too few sources for F -- a
    desaturated subject can otherwise come back empty, since near-grey sits only
    ~137 away from green in the CbCr plane, just inside the default threshold.
    -> (sure_bg, sure_fg, effective_fg_threshold)
    """
    if dist is None:
        dist = key_distance(rgb, key_rgb)
    sure_bg = dist <= strict_bg
    # Key-hued pixels are barred from seeding F outright. key_distance already damps
    # them, but the bar has to be absolute: an F taken from a key-coloured pixel makes
    # the projection self-consistent (C == F gives a_raw = 1 with zero residual), so
    # the reconstruction guard cannot catch it. This is the single mechanism behind
    # the opaque key-coloured rim seen on real generated art.
    not_key = key_hue_distance(rgb, key_rgb) >= HUE_FULL
    thr = strict_fg
    sure_fg = (dist >= thr) & not_key
    while sure_fg.mean() < MIN_SURE_FG_FRACTION and thr > 20.0:
        thr *= 0.8
        sure_fg = (dist >= thr) & not_key
    return sure_bg, sure_fg, thr


def seed_to_sr(seed_patch: np.ndarray, scale: int, candidate: np.ndarray):
    """Map native certainties to SR space as single centre points, then grow them
    inside `candidate` only. -> (confirmed_mask, seed_miss_count)

    `seed_patch` must be the native seed mask sliced with upscale.take_patch, so it
    shares an origin with `candidate` (which spans the padded tile, halo included).

    Blowing each native pixel up to a scale x scale block would be the naive route;
    the centre point plus constrained growth keeps 1px gaps alive without letting a
    seed claim area the upscaler resolved differently.
    """
    hh, ww = candidate.shape[:2]
    ys, xs = np.nonzero(seed_patch)
    if ys.size == 0:
        return np.zeros((hh, ww), bool), 0

    off = scale // 2
    py = np.clip(ys * scale + off, 0, hh - 1)
    px = np.clip(xs * scale + off, 0, ww - 1)

    num, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    hit = labels[py, px]
    keep = np.zeros(num, bool)
    keep[hit[hit > 0]] = True

    confirmed = keep[labels]
    miss = int(np.count_nonzero(hit == 0))
    if miss:
        # The upscaler closed the gap this seed sits in. Pin the point itself so the
        # evidence is not lost entirely, and report it.
        m = hit == 0
        confirmed[py[m], px[m]] = True
    return confirmed, miss


def nearest_color_lut(rgb: np.ndarray, sure_mask: np.ndarray):
    """Propagate the colour of the nearest 'sure' pixel to every pixel.

    cv2.distanceTransformWithLabels hands back label IDs, not flat indices, so the
    colours have to be routed through a LUT keyed by label.
    -> (F, has_F)
    """
    h, w = sure_mask.shape[:2]
    if not sure_mask.any():
        return np.zeros((h, w, 3), np.uint8), False

    src = np.where(sure_mask, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
    lut[labels[sure_mask]] = rgb[sure_mask]
    F = lut[labels]
    return cv2.medianBlur(F, 3), True


def matte_known_bg(rgb, key_rgb, F, has_F, dist_alpha, space="srgb"):
    """Known-background matting: solve C = a*F + (1-a)*K for a.

    `key_rgb` may be one colour or an (H,W,3) map, which is what lets the same
    solver run against an image's own locally-estimated background.

    -> (alpha float32, stats dict). The guards are evaluated on the *unclamped*
    a_raw; clipping first would bury a failed estimate as solid foreground or
    solid background instead of surfacing it.
    """
    C = colorspace.to_working(rgb, space)
    k_arr = np.clip(np.asarray(key_rgb, np.float32), 0, 255).astype(np.uint8)
    K = colorspace.to_working(k_arr.reshape(1, 1, 3) if k_arr.ndim == 1 else k_arr, space)
    Fw = colorspace.to_working(F, space)
    if space == "srgb":
        C, K, Fw = C * 255.0, K * 255.0, Fw * 255.0

    d = Fw - K
    denom = np.einsum("...i,...i->...", d, d)
    a_raw = np.einsum("...i,...i->...", C - K, d) / np.maximum(denom, 1e-9)

    C_hat = a_raw[..., None] * Fw + (1.0 - a_raw[..., None]) * K
    resid = np.linalg.norm(C - C_hat, axis=-1)

    bad_denom = denom < DENOM_MIN[space]
    bad_range = (a_raw < -ALPHA_MARGIN) | (a_raw > 1.0 + ALPHA_MARGIN)
    bad_resid = resid > RESIDUAL_MAX[space]
    bad_no_fg = np.zeros_like(bad_denom) if has_F else np.ones_like(bad_denom)

    bad = bad_denom | bad_range | bad_resid | bad_no_fg
    alpha = np.where(bad, dist_alpha, np.clip(a_raw, 0.0, 1.0)).astype(np.float32)

    # Rates are reported over the pixels a projection actually matters on -- the
    # soft band. Counting guard hits over every pixel inflates the numerator with
    # solid interior pixels (where falling back is harmless) and, on a mostly-flat
    # crop, dilutes the denominator until a real anomaly disappears.
    band = (dist_alpha > 0.0) & (dist_alpha < 1.0)
    attempted = int(np.count_nonzero(band))
    stats = {
        "attempted": attempted,
        "fallback_denom": int(np.count_nonzero(bad_denom & band)),
        "fallback_alpha_range": int(np.count_nonzero(bad_range & band)),
        "fallback_residual": int(np.count_nonzero(bad_resid & band)),
        "fallback_no_foreground": int(np.count_nonzero(bad_no_fg & band)),
    }
    return alpha, stats


def decontaminate(rgb, alpha, key_rgb, strength=1.0, space="srgb", band=(0.03, 0.98)):
    """Recover the true foreground colour at partial-alpha pixels by undoing the
    known composite. This is what removes the background-coloured halo.

    `key_rgb` may be a single colour or an (H,W,3) map of per-pixel background
    colours, which is what lets the same routine clean an edge against a background
    that is merely flat *locally* -- the original artwork behind the subject, say,
    rather than a synthetic key.

    Only the boundary band is touched: below `band[0]` the division explodes for no
    visible gain, above `band[1]` the pixel is opaque and must not be altered.
    """
    lo, hi = band
    a = alpha.astype(np.float32)
    work = (a > lo) & (a < hi)
    if not work.any() or strength <= 0.0:
        return rgb.copy()

    C = colorspace.to_working(rgb, space)
    k_arr = np.clip(np.asarray(key_rgb, np.float32), 0, 255).astype(np.uint8)
    if k_arr.ndim == 1:
        k_arr = k_arr.reshape(1, 1, 3)
    K = colorspace.to_working(k_arr, space)
    if K.shape[:2] == (1, 1):
        K = K.reshape(3)

    a3 = a[..., None]
    F = np.clip((C - (1.0 - a3) * K) / np.maximum(a3, lo), 0.0, 1.0)
    if strength < 1.0:
        F = C + (F - C) * strength

    out = np.where(work[..., None], F, C)
    return colorspace.from_working(out, space)


def despill(rgb, alpha, key_rgb, strength=1.0, balance=0.5, band=(0.03, 0.98)):
    """Limit whatever key-direction component survived decontamination.

    Runs after decontaminate, over the same boundary band only.
    """
    lo, hi = band
    a = alpha.astype(np.float32)
    work = (a > lo) & (a < hi)
    if not work.any() or strength <= 0.0:
        return rgb.copy()

    key_ch = int(np.argmax(np.asarray(key_rgb, np.float32)))
    others = [i for i in range(3) if i != key_ch]
    x = rgb.astype(np.float32)
    o1, o2 = x[..., others[0]], x[..., others[1]]
    limit = np.maximum(o1, o2) * (1.0 - balance) + (o1 + o2) * 0.5 * balance

    ch = x[..., key_ch]
    limited = np.minimum(ch, limit + (1.0 - strength) * np.maximum(ch - limit, 0.0))
    out = x.copy()
    out[..., key_ch] = np.where(work, limited, ch)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
