import numpy as np
import cv2

from core import chroma, matting, upscale

# Minimum share of a tile that must be certain foreground before the tile is trusted
# to source F on its own. Below this it falls back to F_native, which is nearest-
# neighbour upscaled and therefore blocky, so the bar wants to be low: a handful of
# genuine pixels propagated by distance transform beats a whole tile of blocks.
TILE_FG_MIN = 0.02

# Tolerance on "the base image agrees this is background", in RGB L2, used when
# pinning a gap. It only has to absorb the few-LSB noise of a flat background, so it
# sits far below the distance to any drawn content -- hair against a light ground
# measures in the hundreds -- and a strand that moved between the two Gemini passes
# therefore fails the test instead of being pinned transparent.
PIN_TOL = 24.0


def _resize_alpha(alpha_hi, size_wh):
    if alpha_hi.shape[:2] == (size_wh[1], size_wh[0]):
        return alpha_hi.copy()
    return cv2.resize(alpha_hi, size_wh, interpolation=cv2.INTER_AREA)


def build_matte(gen_rgb, key_preset="green", scale=2, tile=256, pad=16,
                backend="auto", tol=30.0, soft=20.0,
                strict_bg=chroma.STRICT_BG_DEFAULT, strict_fg=chroma.STRICT_FG_DEFAULT,
                min_noise_area=0, matte_space="srgb", raster_from="soft",
                smooth=0.25, model="anime6b", speck_repair_area=0,
                rim_hue=chroma.HUE_FULL):
    """Key the generated image against its flat background and return two mattes.

    alpha_source is the analytic matte and is what colour recovery must use.
    alpha_cut_base is the shape the PSD gets, and is what the fringe controls act on.

    rim_hue removes the opaque key-coloured rim that generated art produces (0 to
    disable). Colour is the discriminator, not geometry: a genuine partial pixel is a
    mixture and so sits away from the key's own hue, while the rim is the key hue at
    a different luminance.
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
             "rim_removed": 0, "backend": "lanczos"}

    for t in upscale.iter_tiles(gen_rgb, scale, tile, pad, backend, model):
        stats["backend"] = t.backend
        tile_rgb = t.hi_rgb
        lx, ty = t.halo
        X, Y, W, H = t.dst_box

        dist = chroma.key_distance(tile_rgb, key_rgb)
        da = chroma.distance_alpha(dist, tol, soft)
        _, cand_bg, _ = chroma.make_trimap(dist, tol, soft)

        hue_t = chroma.key_hue_distance(tile_rgb, key_rgb)
        # key_distance already damps key-hued pixels, but F sourcing gets an explicit
        # bar as well: an F lifted from a key-coloured pixel satisfies the projection
        # exactly (C == F) and so is invisible to the reconstruction guard.
        sure_fg_t = (dist >= eff_fg) & (hue_t >= chroma.HUE_FULL)
        if sure_fg_t.mean() >= TILE_FG_MIN:
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

        if rim_hue > 0.0:
            rim = (alpha_t > 0.5) & (hue_t < float(rim_hue))
            alpha_t = np.where(rim, 0.0, alpha_t)
            stats["rim_removed"] += int(np.count_nonzero(rim[ty:ty + H, lx:lx + W]))

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
        # Un-eroded and straight off the key image, which is the point: every matte
        # downstream of here has already lost the 1px gaps it is meant to protect.
        "key_bg": sure_bg_n,
        "key_rgb": key_rgb,
        "hi_size": alpha_hi.shape[0],
        "scale": scale,
        "stats": stats,
    }


def estimate_local_bg(base_rgb, alpha_source, ds=8, radius=6, min_weight=0.02,
                      std_max=18.0):
    """Per-pixel background colour of `base_rgb`, read where the matte says background.

    The key image says *which* pixels are background; this asks what colour those
    pixels have in the image we are actually cutting out. That is what lets
    chroma.decontaminate clean an edge against ordinary artwork, whose background is
    flat only locally, instead of against a synthetic key.

    Estimated on a downscaled copy: a box filter there spans ds * radius pixels of the
    original for the cost of a small one, and the result is smooth by construction.
    Where too few background pixels are in reach, or where the ones in reach disagree
    with each other, the estimate is marked unreliable rather than guessed at.
    -> (bg_map uint8 (H,W,3), reliable bool (H,W))
    """
    h, w = alpha_source.shape[:2]
    sw, sh = max(2, w // ds), max(2, h // ds)
    m = (alpha_source == 0).astype(np.float32)
    x = base_rgb.astype(np.float32)

    ms = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    c1 = cv2.resize(x * m[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
    c2 = cv2.resize(x * x * m[..., None], (sw, sh), interpolation=cv2.INTER_AREA)

    k = 2 * int(radius) + 1
    wt = cv2.boxFilter(ms, -1, (k, k))
    s1 = cv2.boxFilter(c1, -1, (k, k))
    s2 = cv2.boxFilter(c2, -1, (k, k))

    safe = np.maximum(wt, 1e-6)[..., None]
    mean = s1 / safe
    std = np.sqrt(np.maximum(s2 / safe - mean * mean, 0.0)).max(axis=-1)
    ok = (wt >= min_weight) & (std <= std_max)

    if m.any():
        flat = x[m > 0]
        fallback = np.median(flat[::max(1, len(flat) // 100000)], axis=0)
    else:
        fallback = np.full(3, 255.0, np.float32)
    mean = np.where(ok[..., None], mean, fallback)

    bg = cv2.resize(np.clip(mean, 0, 255), (w, h), interpolation=cv2.INTER_LINEAR)
    rel = cv2.resize(ok.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return bg.astype(np.uint8), rel.astype(bool)


def _global_bg(base_rgb, alpha_key, spread_max=18.0, sample=100000):
    """The base image's single background colour, when it has one.

    estimate_local_bg already computes this median, but it hands it back marked
    unreliable exactly where a narrow gap needs it: a 3px hole holds too few
    background pixels for the box filter to reach `min_weight`, so the local route
    has no opinion there at all. The same colour is therefore offered again here,
    gated on the background being uniform enough for one colour to describe it.
    -> (colour float32 (3,), uniform bool)
    """
    m = alpha_key == 0
    if not m.any():
        return np.full(3, 255.0, np.float32), False
    flat = base_rgb[m].astype(np.float32)
    flat = flat[::max(1, len(flat) // sample)]
    med = np.median(flat, axis=0)
    spread = float(np.percentile(np.linalg.norm(flat - med, axis=-1), 90))
    return med.astype(np.float32), bool(spread <= spread_max)


def pin_background(base_rgb, alpha_key, key_bg, d, rel, tol=PIN_TOL, keep=1):
    """Hold the centre of a narrow background gap at certain background.

    Three conditions, and each one is load-bearing:

    `key_bg` is read from the key image directly -- chroma.native_seeds, which is
    deliberately not eroded -- rather than from the matte derived from it. Over pure
    key colour the matte still comes back at alpha 10-32, above the `bg_max` of 8
    that build_trimap calls certain, so by the time the trimap is built the evidence
    that the gap was ever background has already been thrown away. On the 256px
    synthetic that is the difference between all three gaps filling and none.

    The base image has to agree, against its own background colour. The two Gemini
    passes register only to about a pixel, so pinning on the key image alone deletes
    hair that moved between them; requiring the base to look like background as well
    means a shifted strand fails the test and is solved normally instead. The local
    estimate is used where it is reliable and the global one fills in where it is
    not, which is the case in every gap small enough for this to matter.

    And only the centre is pinned -- one erosion step, never `band`. The rim of a
    gap is a genuine mixture and has to stay in the unknown band to be solved as
    one, so a 3px gap keeps its middle pixel and gives both edges to the solver.

    Against ground truth the pin costs 2.4 / 10.3 / 7.5 alpha_mae_edge on crops
    A / B / C at zero shift -- where the benchmark's key image is an exact composite
    and its matte cannot be improved on -- and buys 7.4 / 6.3 / 10.3 at one pixel of
    shift, which is the regime two Gemini passes are actually in. Over the same
    sweep it takes the key-coloured rim from 9 / 25 / 81 px to 0 / 2 / 0.
    -> (pin bool (H,W) or None, stats)
    """
    if key_bg is None:
        return None, {}
    match = rel & (d <= tol)
    colour, uniform = _global_bg(base_rgb, alpha_key)
    if uniform:
        match |= np.linalg.norm(base_rgb.astype(np.float32) - colour, axis=-1) <= tol
    k = np.ones((2 * int(keep) + 1,) * 2, np.uint8)
    pin = cv2.erode((np.asarray(key_bg, bool) & match).astype(np.uint8), k).astype(bool)
    return pin, {"pinned": int(pin.sum()), "pin_uniform_bg": uniform}


def refine_with_base(alpha_key, base_rgb, bg_map, rel, space="srgb", sure=250):
    """Re-solve the boundary coverage on the image actually being cut out.

    The key pass decides *what* is background -- which region, which enclosed holes --
    and it is the only thing that can, since the original's background is not
    separable by colour. But it decides sub-pixel coverage on its own geometry, and
    the two Gemini passes agree only to about a pixel. Measured on a thin hair strand:
    the key mask gave alpha 0.43 and 0.71 to the pure-white pixels either side of the
    strand and 0.00 to the dark core between them. Those wrongly-opaque background
    pixels are the pale outline left along every strand once the green is gone.

    So the same known-background solver runs a second time, on the original against
    its own local background, seeded from what the key mask calls certain subject.
    Coverage is then the smaller of the two: the key pass can only ever remove.

    Two details are what make it work, both measured on that strand:

    Seeds come from the whole certain region, not just the part with a known local
    background -- restricted to the latter the nearest seed to a boundary pixel is
    often itself, F comes back equal to C, and the projection returns 1.0 with zero
    residual. That is the same self-consistency trap key-coloured pixels spring in
    the key pass.

    And the seed region is eroded by one pixel. Without it a boundary pixel still
    seeds itself (alpha 1.00 where the truth is 0.29); with it F comes from the
    strand's interior and the answer is 0.29. This is the opposite of the key pass,
    where erosion measurably hurts -- there the seed set is huge and interior, here
    it is the boundary itself that must not be trusted.

    The solver's own guards handle the rest: where the subject is barely separable
    from its background -- white cloth on white paper, ||F-B||^2 below DENOM_MIN --
    the projection is refused and the key mask stands.
    -> (alpha uint8, refined_px)
    """
    seed = alpha_key >= sure
    seed = cv2.erode(seed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not seed.any():
        return alpha_key, 0
    F, has_F = chroma.nearest_color_lut(base_rgb, seed)
    ak = alpha_key.astype(np.float32) / 255.0
    a_base, _ = chroma.matte_known_bg(base_rgb, bg_map, F, has_F, ak, space)
    out = np.where(rel, np.minimum(ak, a_base), ak)
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # Count only changes worth a log line. Almost every boundary pixel moves by a
    # level or two, which says nothing; a drop of 8 means the two passes disagreed.
    return out, int(np.count_nonzero(out.astype(np.int16) < alpha_key.astype(np.int16) - 8))


def apply_to_base(base_rgb, alpha_key, method="matting", decontam=1.0, space="srgb",
                  trimap_band=3, budget_gb=4.0, disagree_radius=4, key_bg=None,
                  band_authority=True, **kw):
    """Carry the key-derived mask over to the image being cut out.

    This is the whole point of keeping the two images apart: the mask says where the
    background is, the original supplies every colour. Nothing the key pass did to
    the subject can reach the output, so a key-coloured edge is impossible by
    construction rather than by correction.

    method:
      analytic    the earlier route -- min(alpha_key, alpha solved on the base) plus
                  an unpremultiply against the local background. Kept as the baseline
                  every change is judged against
      foreground  alpha_key unchanged, colours from multi-level foreground estimation
      matting     alpha re-solved from a trimap, then foreground estimation

    `key_bg` is the key pass's own un-eroded certain-background mask (build_matte's
    "key_bg"). It feeds pin_background, and only the `matting` route: that is the one
    that re-solves alpha, so it is the only one a recovered background seed reaches.

    `band_authority` hands the whole unknown band to the solver instead of only the
    neighbourhood of measured disagreement. On by default: it is what clears the white
    background left in the gaps between hair strands (204 -> 5 px at alpha>128 on the
    real pair). Turn it off to keep the key matte's coverage and lose ~900 px less of
    thin hair -- see matting.blend_by_disagreement for why the two disagree.

    Only `matting` can put back hair the key pass deleted; the other two can subtract
    but never add. Measured on the real pair, lost-hair candidates 3113 (analytic) /
    2875 (foreground) / 1341 (matting), hair-region mean alpha 0.709 / 0.730 / 0.849,
    white-edge candidates 251 / 897 / 335 -- and on the benchmark, where the key image
    is an exact composite and its alpha is already right, matting costs only 0.26
    alpha_mae_edge because it defers to the key matte away from disagreement.
    -> (fg_rgb uint8, alpha uint8, stats)
    """
    bg, rel = estimate_local_bg(base_rgb, alpha_key, **kw)
    d = np.linalg.norm(base_rgb.astype(np.float32) - bg.astype(np.float32), axis=-1)
    stats = {}

    if method == "analytic":
        alpha, stats["refined"] = refine_with_base(alpha_key, base_rgb, bg, rel, space)
        a = alpha.astype(np.float32) / 255.0
        if decontam > 0.0:
            fg = np.where(rel[..., None],
                          chroma.decontaminate(base_rgb, a, bg, decontam, space),
                          base_rgb)
        else:
            fg = base_rgb.copy()
        stats["solver"] = "analytic"
        # Only this route recovers colour by unpremultiplying against the estimated
        # background, so only here does a region without one change the result. The
        # other two take their colours from the foreground estimator, which needs no
        # background map, and reporting it there would be noise.
        stats["no_local_bg"] = int(np.count_nonzero(
            ~rel & (alpha_key > 8) & (alpha_key < 250)))
        return fg, alpha, stats

    # The trimap is built for both remaining routes, even though `foreground` does not
    # re-solve alpha: interior_weight comes from it, so the colour path is identical
    # and the comparison isolates the change to alpha.
    pin, pst = pin_background(base_rgb, alpha_key, key_bg, d, rel)
    trimap, disagree, tst = matting.build_trimap(alpha_key, d, rel, band=int(trimap_band),
                                                 pin_bg=pin)
    stats.update(tst)
    stats.update(pst)
    a = alpha_key.astype(np.float32) / 255.0

    if method == "matting":
        a_cf, info = matting.estimate_alpha(base_rgb, trimap, budget_gb)
        a = matting.blend_by_disagreement(a, a_cf, disagree, int(disagree_radius),
                                          trimap=trimap if band_authority else None)
        stats.update(info)
        stats["disagree_px"] = int(disagree.sum())
    else:
        stats["solver"] = "none (alpha unchanged)"

    # alpha stays float from here into foreground estimation -- on thin hair a small
    # alpha difference moves the unpremultiplied colour a long way, so there is no 8
    # bit round trip in between.
    F = matting.estimate_foreground(base_rgb, a)
    w = matting.interior_weight(trimap)[..., None]
    fg = np.clip(w * base_rgb.astype(np.float32) + (1.0 - w) * F + 0.5, 0, 255).astype(np.uint8)
    return fg, np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8), stats

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
