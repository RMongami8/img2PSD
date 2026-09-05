import numpy as np
import cv2

from core import chroma, upscale


def _resize_alpha(alpha_hi, size_wh):
    if alpha_hi.shape[:2] == (size_wh[1], size_wh[0]):
        return alpha_hi.copy()
    return cv2.resize(alpha_hi, size_wh, interpolation=cv2.INTER_AREA)


def build_matte(gen_rgb, key_preset="green", scale=2, tile=256, pad=16,
                backend="auto", tol=30.0, soft=20.0,
                strict_bg=chroma.STRICT_BG_DEFAULT, strict_fg=chroma.STRICT_FG_DEFAULT,
                min_noise_area=0, matte_space="srgb", raster_from="binary",
                smooth=0.25, model="anime6b", speck_repair_area=0):
    """Key the generated image against its flat background and return two mattes.

    alpha_source is the analytic matte and is what colour recovery must use.
    alpha_cut_base is the shape the PSD gets, and is what the fringe controls act on.
    """
    # Non-square input is supported: the app always feeds a squared canvas, but the
    # module is also driven directly from tools and tests.
    h, w = gen_rgb.shape[:2]
    scale = max(1, int(scale))
    hi_h, hi_w = h * scale, w * scale

    key_rgb = chroma.estimate_key_color(gen_rgb, key_preset)
    native_dist = chroma.key_distance(gen_rgb, key_rgb)
    # native_seeds may relax the foreground threshold on a desaturated subject; the
    # per-tile test below has to use that same effective value, or every tile decides
    # it has no sure foreground and falls back to F_native for the whole image.
    sure_bg_n, sure_fg_n, eff_fg = chroma.native_seeds(gen_rgb, key_rgb, strict_bg,
                                                       strict_fg, dist=native_dist)

    # One global pass at native resolution: when a tile holds no sure foreground of
    # its own this is the fallback for F. A tile-local mean colour would be worse
    # than useless -- on a tile mixing skin, black line and cloth it invents a colour
    # that belongs to nothing, and its distance from the key is large enough to sail
    # through the denominator guard.
    F_native, has_F_native = chroma.nearest_color_lut(gen_rgb, sure_fg_n)

    is_key_native = native_dist <= tol

    alpha_hi = np.zeros((hi_h, hi_w), np.uint8)

    stats = {"attempted": 0, "fallback_denom": 0, "fallback_alpha_range": 0,
             "fallback_residual": 0, "fallback_no_foreground": 0,
             "seed_miss": 0, "seed_override": 0, "f_native_used": 0, "specks_filled": 0,
             "backend": "lanczos"}

    for t in upscale.iter_tiles(gen_rgb, scale, tile, pad, backend, model):
        stats["backend"] = t.backend
        tile_rgb = t.hi_rgb
        lx, ty = t.halo
        X, Y, W, H = t.dst_box

        dist = chroma.key_distance(tile_rgb, key_rgb)
        da = chroma.distance_alpha(dist, tol, soft)
        _, cand_bg, _ = chroma.make_trimap(dist, tol, soft)

        sure_fg_t = dist >= eff_fg
        if sure_fg_t.mean() >= chroma.MIN_SURE_FG_FRACTION:
            F, has_F = chroma.nearest_color_lut(tile_rgb, sure_fg_t)
        elif has_F_native:
            patch = upscale.take_patch(F_native, t.pad_box, t.reflect)
            F = cv2.resize(patch, (tile_rgb.shape[1], tile_rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            has_F = True
            stats["f_native_used"] += int(tile_rgb.shape[0] * tile_rgb.shape[1])
        else:
            F, has_F = np.zeros_like(tile_rgb), False

        alpha_t, st = chroma.matte_known_bg(tile_rgb, key_rgb, F, has_F, da, matte_space)
        for k in ("attempted", "fallback_denom", "fallback_alpha_range",
                  "fallback_residual", "fallback_no_foreground"):
            stats[k] += st[k]

        # Native certainty overrides whatever the upscaler decided: a hair gap that
        # was unambiguously background at native resolution stays background even if
        # the model painted over it. Two limits, both measured rather than assumed:
        #
        # Background only. A symmetric foreground constraint sounds tidy but is
        # destructive -- candidate_fg spans ~94% of a typical frame as one connected
        # component, so growing seeds through it pins the entire subject, soft edge
        # pixels and enclosed background holes included, to alpha 1.
        #
        # And only where the result actually contradicts the evidence. Zeroing the
        # whole confirmed region also clips genuinely soft pixels that fall inside
        # it, costing 5.4 alpha_mae_edge (19.3 vs 14.0) while recovering no holes.
        bg_patch = upscale.take_patch(sure_bg_n, t.pad_box, t.reflect)
        conf_bg, miss_bg = chroma.seed_to_sr(bg_patch, scale, cand_bg)
        stats["seed_miss"] += miss_bg
        overridden = conf_bg & (alpha_t > 0.5)
        stats["seed_override"] += int(np.count_nonzero(overridden))
        alpha_t = np.where(overridden, 0.0, alpha_t)

        core = alpha_t[ty:ty + H, lx:lx + W]
        alpha_hi[Y:Y + H, X:X + W] = np.clip(core * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if speck_repair_area > 0:
        alpha_hi, stats["specks_filled"] = _repair_specks(
            alpha_hi, is_key_native, int(round(speck_repair_area * scale * scale)))

    if min_noise_area > 0:
        alpha_hi = _denoise(alpha_hi, int(round(min_noise_area * scale * scale)))

    alpha_source = _resize_alpha(alpha_hi, (w, h))
    binary_hi = np.where(alpha_hi > 127, 255, 0).astype(np.uint8)

    if raster_from == "soft":
        alpha_cut_base = alpha_source.copy()
    elif raster_from == "path":
        alpha_cut_base = _raster_from_path(binary_hi, (w, h), scale, smooth)
    else:
        alpha_cut_base = _resize_alpha(binary_hi, (w, h))

    return {
        "alpha_source": alpha_source,
        "alpha_cut_base": alpha_cut_base,
        "alpha_hi": alpha_hi,
        "key_rgb": key_rgb,
        "hi_size": alpha_hi.shape[0],
        "scale": scale,
        "stats": stats,
    }


def _repair_specks(alpha_hi, is_key_native, max_area, key_frac_max=0.02):
    """Close transparent specks that the key colour does not justify. Off by default.

    Because F is estimated from a neighbouring pixel rather than the pixel itself,
    the projection returns alpha slightly under 1 across subject interiors. On real
    generated art the average is invisible (~3/255 of background bleeding through)
    but a tail of 0.07-0.28% of interior pixels falls below half opacity.

    This helps that tail only marginally -- measured 0.283% -> 0.259% -- while
    costing benchmark edge accuracy once max_area grows past a few hundred pixels,
    so it is opt-in rather than on by default. The discriminator is colour, not
    geometry: a real gap between hair strands contains key-coloured pixels, a
    mis-estimated speck contains none. Gating on distance to background instead
    would be far worse, costing 12.5 alpha_mae_edge on the benchmark, because dense
    fine hair sits far from any fully key-coloured pixel.
    -> (alpha_hi, filled_px)
    """
    holes = (alpha_hi < 128).astype(np.uint8)
    if not holes.any():
        return alpha_hi, 0
    key_hi = is_key_native
    if key_hi.shape != alpha_hi.shape:
        key_hi = cv2.resize(is_key_native.astype(np.uint8),
                            (alpha_hi.shape[1], alpha_hi.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)

    num, labels, st, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    # a component is genuine background if ANY of it looks like the key. Requiring
    # a majority instead swallows thin hair gaps, which are made almost entirely of
    # partial pixels and so rarely read as key-coloured outright.
    key_frac = np.bincount(labels.ravel(), weights=key_hi.ravel(),
                           minlength=num) / np.maximum(st[:, cv2.CC_STAT_AREA], 1)
    bogus = np.zeros(num, bool)
    for i in range(1, num):
        if st[i, cv2.CC_STAT_AREA] <= max_area and key_frac[i] <= key_frac_max:
            bogus[i] = True
    if not bogus.any():
        return alpha_hi, 0
    mask = bogus[labels]
    alpha_hi[mask] = 255
    return alpha_hi, int(mask.sum())


def _denoise(alpha_hi, area_px):
    """Drop small speckles of *uncertain* matte only.

    Enclosed background is never filled in, whatever its area: a 1px gap between two
    hair strands is shape, not noise, and this is precisely the operation that would
    destroy it.
    """
    if area_px <= 0:
        return alpha_hi
    uncertain = ((alpha_hi > 0) & (alpha_hi < 255)).astype(np.uint8)
    num, labels, st, _ = cv2.connectedComponentsWithStats(uncertain, connectivity=8)
    out = alpha_hi
    for i in range(1, num):
        if st[i, cv2.CC_STAT_AREA] < area_px:
            out[labels == i] = 0
    return out


def _contours_with_depth(binary_hi):
    cs, hier = cv2.findContours(binary_hi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, c in enumerate(cs):
        d, p = 0, hier[i][3]
        while p != -1:
            d += 1
            p = hier[p][3]
        out.append((c, d))
    return out


def _raster_from_path(binary_hi, size_wh, scale, smooth):
    """Kept for comparison only. Rebuilding the raster through findContours/fillPoly
    adds up to a pixel of drift from the contour coordinate convention, so the
    shipped path downsamples the binary directly instead."""
    aa = 2
    w, h = size_wh
    canvas = np.zeros((h * aa, w * aa), np.uint8)
    k = float(h * aa) / binary_hi.shape[0]
    eps = max(0.01, smooth * scale)
    for c, d in sorted(_contours_with_depth(binary_hi), key=lambda t: t[1]):
        a = cv2.approxPolyDP(c, eps, True)
        if len(a) < 3:
            continue
        cv2.fillPoly(canvas, [np.round(a.reshape(-1, 2) * k).astype(np.int32)],
                     255 if d % 2 == 0 else 0)
    return cv2.resize(canvas, size_wh, interpolation=cv2.INTER_AREA)


def to_svg(alpha_hi, out_size=2048, scale=2, smooth=0.25, color="#000000"):
    """Vector artifact. Nested holes are expressed with fill-rule evenodd, and the
    depth walk (RETR_TREE, not RETR_CCOMP) keeps shapes nested more than two deep.

    out_size names the output width; the height follows the mask's aspect ratio.
    """
    binary = np.where(alpha_hi > 127, 255, 0).astype(np.uint8)
    k = float(out_size) / binary.shape[1]
    out_h = int(round(binary.shape[0] * k))
    eps = max(0.01, smooth * scale)
    subpaths = []
    for c, _ in _contours_with_depth(binary):
        a = cv2.approxPolyDP(c, eps, True)
        if len(a) < 3:
            continue
        pts = a.reshape(-1, 2)
        coords = ["%s %s" % (round(float(x) * k, 2), round(float(y) * k, 2))
                  for x, y in pts]
        subpaths.append("M " + " L ".join(coords) + " Z")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{out_size}" height="{out_h}" '
        f'viewBox="0 0 {out_size} {out_h}" shape-rendering="geometricPrecision">\n'
        f'<path fill="{color}" fill-rule="evenodd" d="{" ".join(subpaths)}"/>\n'
        '</svg>\n'
    )
