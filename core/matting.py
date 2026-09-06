"""Alpha and foreground estimation on the image that is actually being cut out.

The key-background pass says *which region* is background -- it is the only thing
that can, since the original's background is not separable by colour. But it decides
sub-pixel coverage on its own geometry, and the two Gemini passes agree only to about
a pixel. Measured on a thin hair strand: the key mask gave alpha 0.43 and 0.71 to the
pure-white pixels either side of the strand and 0.00 to the dark core between them.

So the key mask is demoted to a trimap -- certain foreground, certain background, and
an unknown band that deliberately includes every place the two images disagree -- and
alpha is re-solved on the original, followed by a foreground colour estimate that
separates the background that is mixed into the boundary pixels.

Alpha stays float32 from the solver through to foreground estimation. Rounding to 8
bits in between costs precision exactly where it matters: on thin hair, a small alpha
difference moves the unpremultiplied foreground colour a lot.
"""
import contextlib
import io

import numpy as np
import cv2

from pymatting import estimate_alpha_cf, estimate_alpha_lkm, estimate_foreground_ml

from core import upscale

# float64 values + int64 indices, 25 entries per pixel, from pymatting's cf_laplacian
# (v1.1.15: indices = np.zeros(n * (4*radius+1)**2, int64), values = (n,5,5) float64).
# Measured against that: 2048x2048 with a 5.5% unknown band peaks at +1.28GB, i.e.
# 319 bytes/pixel against the 400 the arrays alone imply -- pymatting is handed
# is_known and skips windows that are entirely known, which more than pays for the
# L_U/R slicing and the preconditioner. 1.2 keeps a margin over that measurement
# without pushing a solve that comfortably fits onto the slower tiled path.
_BYTES_PER_PIXEL = 25 * (8 + 8)
_HEADROOM = 1.2

FG, BG, UNKNOWN = 1.0, 0.0, 0.5


def laplacian_gb(n_pixels: int) -> float:
    """Rough peak for a full-image closed-form solve, in GB."""
    return n_pixels * _BYTES_PER_PIXEL * _HEADROOM / (1024.0 ** 3)


def _k(n):
    return np.ones((2 * int(n) + 1, 2 * int(n) + 1), np.uint8)


def build_trimap(alpha_key, d, rel, band=3, bg_max=8, fg_min=247,
                 content=90.0, empty=12.0, touch=2, pin_bg=None):
    """Demote the key mask to certain / certain / unknown.

    `d` is the original's distance from its own locally-estimated background and
    `rel` says where that estimate is usable (both from maskgen.estimate_local_bg).
    `d` is a colour difference, not ground truth about what is background, so it is
    only ever used to move a pixel *into* the unknown band -- never to decide an
    answer directly.

    Three thresholds are measured rather than assumed:

    `bg_max` is 8, not 0. The key matte carries a faint haze over the whole
    background -- mean 0.81/255, p99 4 -- so `== 0` claims only 44% of a frame whose
    background is 73%, and the unknown band balloons to 64%.

    `empty` only applies within `2 * band` of the foreground boundary. Applied
    globally it unmarks the interior of white clothing on a white ground, which is
    precisely the region the trimap has to protect.

    `touch` is the one that matters most, and it is not obvious. A wrongly-opaque
    pixel is on the *outside* of the silhouette and therefore adjacent to the key's
    own background; a merely light-coloured pixel a few px inside the subject is not.
    Without this condition the benchmark's light clothing trips the test on 35,335
    pixels whose ground-truth alpha is 0.999 -- solid foreground handed to the solver
    as unknown, which is where its hole_false_pos came from. With it: 60.

    `pin_bg` (maskgen.pin_background) marks pixels the key image itself, cross-checked
    against the base, still calls background. It does three things at once, and it
    needs all three: it overrides both erosions, and it counts as a disagreement.

    Nothing else can speak for a gap between hair strands narrower than 2*band+1 --
    the erosion below empties it, so the solver gets no seed and closes it. Measured
    on a 256px synthetic with the gap held at alpha_key 0, 3px and 5px gaps fill
    solid while a 9px one keeps its seeds and solves to mean 5.4/255; when the matte
    reads 10-32 over pure key colour, which it does, `empty_m` is empty and even the
    9px gap fills.

    And a translucent rim is the same failure seen from the other side: the pixel is
    neither `lost` nor `false`, so it never enters the disagreement mask and
    blend_by_disagreement leaves the key matte's answer standing. Pinning it is what
    lets the solver overrule that, without handing it the whole band.
    -> (trimap float32 in {0, 0.5, 1}, disagreement bool, stats dict)
    """
    solid = (alpha_key >= fg_min).astype(np.uint8)
    empty_m = (alpha_key <= bg_max).astype(np.uint8)

    near = cv2.distanceTransform(solid, cv2.DIST_L2, 3) <= 2.0 * band
    touch_bg = cv2.dilate(empty_m, _k(touch)) > 0
    lost = (alpha_key <= bg_max) & (d > content) & rel
    false = (alpha_key >= fg_min) & (d < empty) & rel & near & touch_bg

    fg = cv2.erode(solid, _k(band)).astype(bool) & ~false
    bg = cv2.erode(empty_m, _k(band)).astype(bool) & ~lost

    disagree = lost | false
    if pin_bg is not None:
        # Outranks `solid` as well, not just the erosion: where the matte filled a
        # gap outright it reads >= fg_min there, so leaving fg alone would hand the
        # pinned pixel straight back as certain foreground.
        pin_bg = np.asarray(pin_bg, bool)
        fg &= ~pin_bg
        bg |= pin_bg
        disagree |= pin_bg

    trimap = np.full(alpha_key.shape, UNKNOWN, np.float32)
    trimap[bg] = BG
    trimap[fg] = FG
    unknown = ~(fg | bg)
    return trimap, disagree, {
        "fg": float(fg.mean()), "bg": float(bg.mean()), "unknown": float(unknown.mean()),
        "unknown_px": int(unknown.sum()),
        "lost_px": int(lost.sum()), "false_px": int(false.sum()),
        "lost_in_unknown": float(unknown[lost].mean()) if lost.any() else 1.0,
        "false_in_unknown": float(unknown[false].mean()) if false.any() else 1.0,
    }


def blend_by_disagreement(alpha_key, alpha_cf, disagree, radius=4, trimap=None):
    """Take the solver's answer through the unknown band, or only near disagreement.

    With `trimap`, the solver is authoritative: certain labels are honoured and the
    whole unknown band comes back from closed-form matting. Without it, the key matte
    stands except near measured disagreement. The benchmark prefers the second, the
    real pair needs the first, and the reason they disagree is worth writing down.

    Re-solving the whole boundary is wrong, and the benchmark says so with ground
    truth: where the key image is an exact composite its alpha is already excellent
    in the unknown band (MAE 5.9/255) and closed-form matting on the *other* image
    returns 18.0 -- dense dark hair over white is a harder problem than a flat green
    key, so the solver loses whenever it is asked to redo work that was already right.

    Where the two passes are misaligned the ordering reverses. Sweeping a synthetic
    shift over the benchmark, alpha_mae_edge for the key matte is 14.3 / 26.0 / 42.7 /
    55.5 at 0 / 0.5 / 1.0 / 1.5 px while the solver sits flat at ~33.5: it does not
    care about a misregistration it never sees.

    So the key matte is trusted by default and the solver only overrides it in a
    neighbourhood of measured disagreement. At radius 4 that costs 0.26 alpha_mae on
    the pristine benchmark and buys 4.0 at one pixel of shift, and on the real pair
    it cuts white-edge candidates 897 -> 335 and lost-hair candidates 2875 -> 1341.

    All of which is measured on a benchmark whose key image is an *exact composite*
    of the base: perfectly registered, and its alpha is ground truth by construction.
    That is the one situation the two Gemini passes are never in, and the failure it
    cannot express is the one that matters -- the model redraws fine hair between
    passes, so the key matte's coverage is right for a silhouette the base does not
    have. Shifting the benchmark simulates a translation; it cannot simulate redrawn
    hair.

    On the real pair it is the other way round, and not by a little. Counting the
    base's own white background (flood-filled from the frame) that survives into the
    cutout, all of it inside gaps between hair strands:

        key matte alone     3018 px at alpha>128 / 10020 at >64
        disagreement only    204 / 905
        whole band + pin       5 /  63

    So the band is handed to the solver by default. It costs roughly 900 px of the
    subject -- thin hair edges dropping below half opacity, ~0.09% of the figure --
    which is why `band_authority` in maskgen.apply_to_base can turn it off.
    """
    if trimap is not None:
        return np.where(trimap == FG, 1.0,
                        np.where(trimap == BG, 0.0, alpha_cf)).astype(np.float32)
    if radius <= 0 or not disagree.any():
        return alpha_key
    w = cv2.dilate(disagree.astype(np.uint8) * 255, _k(radius)).astype(np.float32) / 255.0
    w = np.clip(cv2.GaussianBlur(w, (0, 0), max(radius * 0.6, 0.8)), 0.0, 1.0)
    return (1.0 - w) * alpha_key + w * alpha_cf


def interior_weight(trimap, r_inner=3.0, r_outer=6.0):
    """How much of the original's own colour to keep, by distance -- not by alpha.

    A high alpha does not mean a pixel is interior. Boundary pixels routinely solve
    to alpha near 1 while still holding a mixture, and handing those back the
    original RGB puts the background colour straight back into the edge. Distance
    into the certain-foreground region is the property actually wanted, and ramping
    it over (r_inner, r_outer) leaves no visible seam where the two sources meet.
    """
    dt = cv2.distanceTransform((trimap >= FG).astype(np.uint8), cv2.DIST_L2, 3)
    span = max(float(r_outer) - float(r_inner), 1e-3)
    return np.clip((dt - float(r_inner)) / span, 0.0, 1.0).astype(np.float32)


def _cf(rgb, trimap, notes=None):
    """pymatting prints a multi-line PERFORMANCE WARNING to stdout each time its
    incomplete-Cholesky preconditioner has to retry with a larger shift. It retries
    on its own and the solve still succeeds, so the text is noise in the app log --
    but the count is worth keeping, since it is the difference between a fast solve
    and a slow one."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a = estimate_alpha_cf(rgb.astype(np.float64) / 255.0, trimap.astype(np.float64))
    if notes is not None:
        notes["ichol_retries"] = notes.get("ichol_retries", 0) + buf.getvalue().count(
            "incomplete Cholesky")
    return np.asarray(a, np.float32)


def _seed_counts(trimap):
    return int((trimap >= FG).sum()), int((trimap <= BG).sum())


def _solve_tiled(rgb, trimap, tile, pad, min_seed, info_notes=None):
    """Fallback for images too large to build one Laplacian for.

    Tiles with no unknown pixel are copied through rather than solved -- the unknown
    band is a few percent of a frame, so most tiles cost nothing. A tile whose halo
    holds too few certain pixels is retried with double the halo: starving a tile of
    seeds thins hair without necessarily leaving a visible seam at the join, so the
    check is on the seeds themselves, not on the output.
    """
    h, w = trimap.shape[:2]
    out = trimap.astype(np.float32).copy()
    overlap_max = 0.0
    solved = widened = 0
    seen = np.zeros((h, w), np.float32)
    have = np.zeros((h, w), bool)

    for t in upscale.iter_tiles(rgb, 1, tile, pad, "lanczos"):
        tri = upscale.take_patch(trimap, t.pad_box, t.reflect)
        if not (tri == UNKNOWN).any():
            continue
        img, cur_pad = t.hi_rgb, pad
        n_fg, n_bg = _seed_counts(tri)
        if min(n_fg, n_bg) < min_seed:
            x0, y0, tw, th = t.src_box
            cur_pad = pad * 2
            sy0, sx0 = max(0, y0 - cur_pad), max(0, x0 - cur_pad)
            sy1, sx1 = min(h, y0 + th + cur_pad), min(w, x0 + tw + cur_pad)
            img = rgb[sy0:sy1, sx0:sx1]
            tri = trimap[sy0:sy1, sx0:sx1]
            t = t._replace(halo=(x0 - sx0, y0 - sy0))
            widened += 1

        a = _cf(img, tri, info_notes)
        solved += 1

        lx, ty = t.halo
        X, Y, W, H = t.dst_box
        core = a[ty:ty + H, lx:lx + W]
        prev = have[Y:Y + H, X:X + W]
        if prev.any():
            overlap_max = max(overlap_max,
                              float(np.abs(core[prev] - seen[Y:Y + H, X:X + W][prev]).max()))
        seen[Y:Y + H, X:X + W] = core
        have[Y:Y + H, X:X + W] = True
        out[Y:Y + H, X:X + W] = core

    return out, {"tiles_solved": solved, "tiles_widened": widened,
                 "overlap_alpha_max": overlap_max}


def estimate_alpha(rgb, trimap, budget_gb=4.0, tile=512, pad=64, min_seed=2048):
    """Re-solve alpha on `rgb` from `trimap`. -> (alpha float32 0..1, info dict)

    The route is chosen from a memory estimate up front rather than by letting an
    allocation fail: a MemoryError part-way through a 2GB build is a bad way to find
    out, and on Windows it may arrive as a hard stop rather than an exception.
    """
    n = trimap.size
    need = laplacian_gb(n)
    info = {"need_gb": need, "budget_gb": float(budget_gb)}

    if need <= budget_gb:
        try:
            a = _cf(rgb, trimap, info)
            if np.isfinite(a).all():
                info["solver"] = "cf"
                return np.clip(a, 0.0, 1.0), info
            info["note"] = "cf produced non-finite alpha"
        except (MemoryError, ValueError) as e:
            info["note"] = "cf failed: %s" % str(e)[:120]
    else:
        info["note"] = "estimated %.1fGB over the %.1fGB budget" % (need, budget_gb)

    try:
        a, tinfo = _solve_tiled(rgb, trimap, tile, pad, min_seed, info)
        if np.isfinite(a).all():
            info.update(tinfo)
            info["solver"] = "cf-tiled"
            return np.clip(a, 0.0, 1.0), info
        info["note"] = "tiled cf produced non-finite alpha"
    except (MemoryError, ValueError) as e:
        info["note"] = "tiled cf failed: %s" % str(e)[:120]

    a = np.asarray(estimate_alpha_lkm(rgb.astype(np.float64) / 255.0,
                                      trimap.astype(np.float64)), np.float32)
    info["solver"] = "lkm"
    return np.clip(np.nan_to_num(a), 0.0, 1.0), info


def estimate_foreground(rgb, alpha):
    """Separate the background that is mixed into the boundary pixels.

    Multi-level, so it costs O(N) and needs no linear system. -> float32 RGB 0..255
    """
    f = estimate_foreground_ml(rgb.astype(np.float64) / 255.0,
                               np.clip(alpha, 0.0, 1.0).astype(np.float64))
    return (np.asarray(f, np.float32) * 255.0).clip(0.0, 255.0)
