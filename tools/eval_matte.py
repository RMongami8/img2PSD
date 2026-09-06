"""Matting evaluation against benchmark/ (ground-truth alpha and foreground).

Primary metrics are edge-limited: solid foreground plus solid background account for
95% of the benchmark, so whole-image means say almost nothing about matte quality.

Modes:
  distance  plain distance key -- the baseline every change is judged against
  oracle    known-background matting fed the ground-truth F. A ceiling, not an
            ablation: it uses information unavailable at runtime
  nearest   known-background matting with F propagated from sure-foreground pixels;
            this is what actually ships

Usage:
  python tools/eval_matte.py --crops A,B,C,D
  python tools/eval_matte.py --full --modes distance,oracle,nearest
  python tools/eval_matte.py --two-pass --methods analytic,foreground,matting

--two-pass evaluates what actually ships: the mask comes from the key-background
image and every colour comes from a different one. white_reference is built from the
same F and alpha as green_input, so it stands in exactly for the original artwork --
and unlike real data it comes with ground truth for both alpha and F.
"""
import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core import chroma, maskgen, matting  # noqa: E402

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark")

# Chosen from the full-resolution labelling below, each covering a different failure
# mode. Enclosed-background counts must come from the whole image: measured inside a
# crop they under-report by 3-5x, because the crop edge fakes a connection to outside.
CROPS = {
    "A": (256, 512),    # densest fine hair          16.8% fractional alpha
    "B": (1536, 512),   # closed hair holes          32,556 enclosed bg px
    "C": (1536, 1536),  # thick outline + skin/cloth 334 distinct fg colours
    "D": (512, 1792),   # control: flat interior     0 enclosed, 0.9% fractional
}


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in
        ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
         "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
         "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def peak_mb():
    """Peak working set. argtypes matter here: without them ctypes passes the
    pseudo-handle as a 32-bit int and the call quietly returns zero."""
    try:
        k32 = ctypes.WinDLL("kernel32")
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return float("nan")


def _imread(path, flags=cv2.IMREAD_UNCHANGED):
    img = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    if img is None:
        raise SystemExit("cannot read " + path)
    return img


def load_benchmark():
    meta = json.load(open(os.path.join(BENCH_DIR, "anime_hair_benchmark.json"),
                          encoding="utf-8"))
    f = meta["files"]
    green = cv2.cvtColor(_imread(os.path.join(BENCH_DIR, f["green_input"]),
                                 cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    white = cv2.cvtColor(_imread(os.path.join(BENCH_DIR, f["white_reference"]),
                                 cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fg_bgra = _imread(os.path.join(BENCH_DIR, f["foreground_rgba"]))
    fg = cv2.cvtColor(fg_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    alpha = _imread(os.path.join(BENCH_DIR, f["alpha_ground_truth"]), cv2.IMREAD_GRAYSCALE)
    return meta, green, white, fg, alpha


def label_enclosed(alpha_gt):
    """Background not reachable from the image border, labelled on the FULL image.

    Doing this per-crop is wrong in both directions: background that escapes through
    the crop edge looks enclosed, and a genuine hole touching the edge looks open.
    """
    h, w = alpha_gt.shape
    bg = (alpha_gt == 0).astype(np.uint8)
    ff = bg.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = ([(0, x) for x in np.flatnonzero(bg[0])]
             + [(h - 1, x) for x in np.flatnonzero(bg[h - 1])]
             + [(y, 0) for y in np.flatnonzero(bg[:, 0])]
             + [(y, w - 1) for y in np.flatnonzero(bg[:, w - 1])])
    for y, x in seeds:
        if ff[y, x] == 1:
            cv2.floodFill(ff, mask, (int(x), int(y)), 2, flags=8)
    return (bg == 1) & (ff != 2)


# ---------------------------------------------------------------- modes
def run_mode(mode, rgb, key_rgb, tol, soft, strict_bg, strict_fg, space, fg_gt=None,
             scale=2, backend="lanczos", tile=128):
    """Modes ending in a scale/backend run the shipping pipeline; the rest isolate
    the matting equation at native resolution so the ablation stays interpretable."""
    if mode.startswith("pipeline"):
        r = maskgen.build_matte(rgb, scale=scale, tile=tile, backend=backend,
                                tol=tol, soft=soft, strict_bg=strict_bg,
                                strict_fg=strict_fg, matte_space=space)
        alpha = r["alpha_source"].astype(np.float32) / 255.0
        fg_rgb = chroma.decontaminate(rgb, alpha, r["key_rgb"], 1.0, space)
        return alpha, fg_rgb, r["stats"]

    dist = chroma.key_distance(rgb, key_rgb)
    da = chroma.distance_alpha(dist, tol, soft)
    stats = {"attempted": int(np.count_nonzero((da > 0) & (da < 1))),
             "fallback_denom": 0, "fallback_alpha_range": 0,
             "fallback_residual": 0, "fallback_no_foreground": 0}

    if mode == "distance":
        alpha = da
    else:
        if mode == "oracle":
            F, has_F = fg_gt, True
        else:
            _, sure_fg, _ = chroma.native_seeds(rgb, key_rgb, strict_bg, strict_fg)
            F, has_F = chroma.nearest_color_lut(rgb, sure_fg)
        alpha, stats = chroma.matte_known_bg(rgb, key_rgb, F, has_F, da, space)

    fg_rgb = chroma.decontaminate(rgb, alpha, key_rgb, 1.0, space)
    return alpha, fg_rgb, stats


# ---------------------------------------------------------------- metrics
def metrics(alpha, fg_rgb, alpha_gt, fg_gt, white_gt, enclosed, stats, key_hue=None):
    """key_hue is the per-pixel hue distance to the key. rim_px counts pixels the
    matte calls foreground while their hue says they are the key at a different
    luminance -- the failure mode real generated art has and the benchmark does
    not, so it reads 0 here and only earns its keep on real inputs."""
    a = alpha.astype(np.float32)
    agt = alpha_gt.astype(np.float32) / 255.0
    edge = (alpha_gt > 0) & (alpha_gt < 255)
    n_edge = int(edge.sum())

    d_alpha = np.abs(a - agt) * 255.0
    white = a[..., None] * fg_rgb.astype(np.float32) + (1.0 - a[..., None]) * 255.0
    d_white = np.abs(white - white_gt.astype(np.float32)).mean(axis=-1)
    pm = np.abs(a[..., None] * fg_rgb.astype(np.float32)
                - agt[..., None] * fg_gt.astype(np.float32)).mean(axis=-1)
    hi_a = edge & (alpha_gt >= 32)
    d_fg = np.abs(fg_rgb.astype(np.float32) - fg_gt.astype(np.float32)).mean(axis=-1)

    solid_fg = alpha_gt == 255
    att = max(stats["attempted"], 1)
    out = {
        "rim_px": int(((a > 0.5) & (key_hue < 0.12)).sum()) if key_hue is not None else 0,
        "alpha_mae_edge": d_alpha[edge].mean() if n_edge else 0.0,
        "alpha_p95_edge": np.percentile(d_alpha[edge], 95) if n_edge else 0.0,
        "white_mae_edge": d_white[edge].mean() if n_edge else 0.0,
        "white_p95_edge": np.percentile(d_white[edge], 95) if n_edge else 0.0,
        "premul_rgb_mae_edge": pm[edge].mean() if n_edge else 0.0,
        "edge_rgb_mae_a32": d_fg[hi_a].mean() if hi_a.any() else 0.0,
        "hole_bg_recall": ((a[enclosed] < 0.5).mean() if enclosed.any() else float("nan")),
        "hole_false_pos": int(((a < 0.5) & solid_fg).sum()),
        "fb_denom%": 100.0 * stats["fallback_denom"] / att,
        "fb_range%": 100.0 * stats["fallback_alpha_range"] / att,
        "fb_resid%": 100.0 * stats["fallback_residual"] / att,
        "fb_nofg%": 100.0 * stats["fallback_no_foreground"] / att,
        "n_edge": n_edge,
    }
    return out


HDR = ["alpha_mae_edge", "alpha_p95_edge", "white_mae_edge", "white_p95_edge",
       "premul_rgb_mae_edge", "edge_rgb_mae_a32", "hole_bg_recall", "hole_false_pos",
       "rim_px", "fb_denom%", "fb_range%", "fb_resid%", "fb_nofg%"]


def _shift(img, dx):
    """Offset the key image against the base by a sub-pixel amount.

    The benchmark's green image is an exact composite of the same F and alpha as
    white_reference, so at zero shift its matte is ground truth by construction and
    nothing can improve on it. That is not the situation real inputs are in: two
    Gemini passes agree only to about a pixel. Shifting restores the failure mode the
    benchmark is missing, while keeping the ground truth that real data lacks.
    """
    m = np.float32([[1, 0, dx], [0, 1, dx * 0.5]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def two_pass(green, white, fg_gt, alpha_gt, enclosed, key_rgb, methods, band, shift=0.0,
             band_authority=True):
    """Mask from the key image, colours from a different image -- with ground truth.

    The foreground estimator is also scored on its own, fed the ground-truth alpha.
    Without that split an improvement in F and a regression in alpha cancel out and
    the table says nothing about either.
    """
    print("%-12s" % "method" + "".join("%13s" % k for k in HDR) + "%9s%9s" % ("ms", "peakMB"))
    f_ideal = matting.estimate_foreground(white, alpha_gt.astype(np.float32) / 255.0)
    edge = (alpha_gt > 0) & (alpha_gt < 255)
    print("%-12s" % "F|gt-alpha" + " " * (13 * 5)
          + "%13.3f" % np.abs(f_ideal - fg_gt.astype(np.float32)).mean(-1)[edge].mean()
          + "   (edge_rgb_mae_a32 column: foreground estimator alone)")

    src = _shift(green, shift) if shift else green
    for m in methods:
        t0 = time.time()
        r = maskgen.build_matte(src, scale=2, tile=128, backend="lanczos")
        # key_bg and band_authority go through as well, or the benchmark scores a
        # configuration the app never runs. Note this benchmark cannot see what
        # band_authority is for: its green image is an exact composite of `white`, so
        # the redrawn-hair disagreement the real pair has does not exist here, and
        # the setting reads as pure cost. Judge it on real output, not on this table.
        fg, a8, st = maskgen.apply_to_base(white, r["alpha_source"], m, trimap_band=band,
                                           key_bg=r["key_bg"],
                                           band_authority=band_authority)
        ms = (time.time() - t0) * 1000.0
        alpha = a8.astype(np.float32) / 255.0
        stats = {"attempted": max(int(edge.sum()), 1), "fallback_denom": 0,
                 "fallback_alpha_range": 0, "fallback_residual": 0,
                 "fallback_no_foreground": 0}
        mt = metrics(alpha, fg, alpha_gt, fg_gt, white, enclosed, stats,
                     chroma.key_hue_distance(green, key_rgb))
        print("%-12s" % m
              + "".join(("%13d" % mt[k]) if isinstance(mt[k], (int, np.integer))
                        else ("%13.3f" % float(mt[k])) for k in HDR)
              + "%9.0f%9.0f" % (ms, peak_mb())
              + ("   solver=%s" % st.get("solver", "?")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crops", default="A,B,C,D")
    p.add_argument("--full", action="store_true")
    p.add_argument("--modes", default="distance,oracle,nearest")
    p.add_argument("--space", default="srgb", choices=["srgb", "linear"])
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--backend", default="lanczos", choices=["lanczos", "auto"])
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--tol", type=float, default=30.0)
    p.add_argument("--soft", type=float, default=20.0)
    p.add_argument("--strict-bg", type=float, default=8.0)
    p.add_argument("--strict-fg", type=float, default=140.0)
    p.add_argument("--two-pass", action="store_true")
    p.add_argument("--methods", default="analytic,foreground,matting")
    p.add_argument("--trimap-band", type=int, default=3)
    p.add_argument("--shift", default="0",
                   help="comma-separated sub-pixel offsets of the key image vs the base")
    p.add_argument("--no-band-authority", action="store_true",
                   help="score the disagreement-only blend instead of the shipped default")
    args = p.parse_args()

    meta, green, white, fg_gt, alpha_gt = load_benchmark()
    key_rgb = np.array(meta["key_rgb"], np.float32)
    enclosed_full = label_enclosed(alpha_gt)
    print("benchmark %dx%d  key=%s  space=%s  enclosed_bg=%d px"
          % (green.shape[1], green.shape[0], meta["key_rgb"], args.space,
             int(enclosed_full.sum())))

    if args.two_pass:
        for name in args.crops.split(","):
            x0, y0 = CROPS[name.strip()]
            sl = (slice(y0, y0 + 256), slice(x0, x0 + 256))
            for sv in [float(v) for v in args.shift.split(",")]:
                print("\n=== two-pass crop %s (%d,%d) edge=%d  key shift %.1fpx ==="
                      % (name.strip(), x0, y0,
                         int(((alpha_gt[sl] > 0) & (alpha_gt[sl] < 255)).sum()), sv))
                two_pass(green[sl], white[sl], fg_gt[sl], alpha_gt[sl], enclosed_full[sl],
                         key_rgb, [m.strip() for m in args.methods.split(",")],
                         args.trimap_band, sv, not args.no_band_authority)
        return

    regions = []
    if args.full:
        regions.append(("FULL", (0, 0, green.shape[1], green.shape[0])))
    else:
        for name in args.crops.split(","):
            x0, y0 = CROPS[name.strip()]
            regions.append((name.strip(), (x0, y0, 256, 256)))

    for rname, (x0, y0, w, h) in regions:
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        print("\n=== crop %s (%d,%d) %dx%d  edge=%d ==="
              % (rname, x0, y0, w, h, int(((alpha_gt[sl] > 0) & (alpha_gt[sl] < 255)).sum())))
        print("%-11s" % "mode" + "".join("%13s" % k for k in HDR)
                  + "%9s%9s" % ("ms", "peakMB"))
        for mode in args.modes.split(","):
            mode = mode.strip()
            t0 = time.time()
            alpha, fg_rgb, st = run_mode(
                mode, green[sl], key_rgb, args.tol, args.soft,
                args.strict_bg, args.strict_fg, args.space, fg_gt[sl],
                args.scale, args.backend, args.tile)
            ms = (time.time() - t0) * 1000.0
            m = metrics(alpha, fg_rgb, alpha_gt[sl], fg_gt[sl], white[sl],
                        enclosed_full[sl], st, chroma.key_hue_distance(green[sl], key_rgb))
            print("%-11s" % mode
                  + "".join(("%13d" % m[k]) if isinstance(m[k], (int, np.integer))
                            else ("%13.3f" % float(m[k]))
                            for k in HDR)
                  + "%9.0f%9.0f" % (ms, peak_mb()))


if __name__ == "__main__":
    main()
