"""REST-friendly wrappers around the same core/*.py building blocks app.py's
do_* functions call.

Deliberately does not import app.py: the do_* functions are wired for Gradio's
calling convention (gr.Error, gr.State-shaped return tuples matched positionally
to component lists) and touching their signatures risks the working Gradio UI.
This module sits at the same level of abstraction -- thin orchestration over
core/* -- for a caller that isn't Gradio.

Session state (the numpy arrays a later stage needs) lives in an in-process
dict keyed by an opaque session_id the frontend mints once per tab. This is
the same role Gradio's gr.State plays for the classic UI, just explicit: fine
for a single-user local tool, not meant to survive a process restart.
"""
import os
import time
from datetime import datetime

import numpy as np
import cv2

from core import imageops, lineart, svgout, gemini, psd_writer
from core import maskgen, fringe

OUT_DIR = "out"

# Defaults mirror the classic Gradio UI's own defaults (app.py), so a run with
# no overrides produces the same output as "▶ 一括実行" there.
DEFAULTS = {
    "radius": 3, "bias": -15, "min_area": 24, "smooth": 1, "scale": 2,
    "pad": "auto", "line_weight": -0.4,
    "crop_on": True, "use_lineart_ref": False,
    "bg_remove": True, "key_color": "green",
    "sr_backend": "lanczos", "sr_scale": 2, "sr_tile": 256,
    "tol": 30.0, "soft": 20.0, "strict_bg": 8.0, "strict_fg": 140.0,
    "noise_area": 0, "raster_from": "soft", "mask_smooth": 0.25,
    "matte_space": "srgb", "rim_hue": 0.35, "edge_method": "matting",
    "trimap_band": 3, "decontam": 1.0, "band_authority": True,
    "choke": 0.0, "feather": 0.0, "matte_gamma": 1.0,
}

SESSIONS: dict[str, dict] = {}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _p(params: dict, key: str):
    return params.get(key, DEFAULTS[key])


def new_session(session_id: str) -> None:
    SESSIONS[session_id] = {}


def _session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise ValueError("unknown session_id -- reload the page")
    return SESSIONS[session_id]


def run_preprocess(session_id: str, image_path: str, params: dict) -> dict:
    """-> {"square_path", "lineart_path", "lineart_svg_path", "elapsed"}"""
    if not image_path:
        raise ValueError("画像を選択してください")
    t0 = time.time()
    ss = int(_p(params, "scale"))

    rgb = imageops.load_rgb(image_path)
    square, box = imageops.to_square(rgb, 2048, _p(params, "pad"), return_box=True)
    ink_hi, alpha = lineart.extract_ink(
        square, radius=int(_p(params, "radius")), bias=int(_p(params, "bias")),
        min_area=int(_p(params, "min_area")), ss=ss,
        line_weight=float(_p(params, "line_weight")))
    svg_str = svgout.ink_to_svg(ink_hi, out_size=2048, ss=ss, smooth=int(_p(params, "smooth")))

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = _ts()
    square_path = os.path.join(OUT_DIR, f"{ts}_square.png")
    lineart_path = os.path.join(OUT_DIR, f"{ts}_lineart.png")
    svg_path = os.path.join(OUT_DIR, f"{ts}_lineart.svg")

    cv2.imwrite(square_path, cv2.cvtColor(square, cv2.COLOR_RGB2BGR))
    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = alpha
    cv2.imwrite(lineart_path, rgba)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)

    st = _session(session_id)
    st.update(square=square, alpha=alpha, box=box)

    return {
        "square_path": square_path, "lineart_path": lineart_path,
        "lineart_svg_path": svg_path, "elapsed": time.time() - t0,
        "shape": list(square.shape),
    }


def run_generate(session_id: str, api_key: str, prompt: str, use_lineart_ref: bool) -> dict:
    """-> {"generated_path", "elapsed"}"""
    st = _session(session_id)
    if st.get("square") is None:
        raise ValueError("先に画像を読み込んでください")
    if not api_key:
        raise ValueError("Gemini API キーを入力してください")
    if not prompt:
        raise ValueError("プロンプトを入力してください")

    t0 = time.time()
    square_rgb, alpha = st["square"], st["alpha"]
    lineart_ref = None
    if use_lineart_ref and alpha is not None:
        h, w = alpha.shape
        lineart_ref = np.full((h, w, 3), 255, np.uint8)
        lineart_ref[alpha > 127] = (0, 0, 0)

    gen_rgb = gemini.generate_image(api_key, prompt, square_rgb, lineart_ref)
    if gen_rgb.shape[:2] != square_rgb.shape[:2]:
        gen_rgb = imageops.to_square(gen_rgb, square_rgb.shape[0], "auto")

    os.makedirs(OUT_DIR, exist_ok=True)
    gen_path = os.path.join(OUT_DIR, f"{_ts()}_generated.png")
    cv2.imwrite(gen_path, cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR))

    st["gen_rgb"] = gen_rgb
    return {"generated_path": gen_path, "elapsed": time.time() - t0}


def run_keygen(session_id: str, api_key: str, key_color: str) -> dict:
    """-> {"keybg_path", "elapsed"}"""
    st = _session(session_id)
    if st.get("square") is None:
        raise ValueError("先に画像を読み込んでください")
    if not api_key:
        raise ValueError("Gemini API キーを入力してください")

    t0 = time.time()
    square_rgb = st["square"]
    key_img = gemini.generate_image(api_key, gemini.key_only_prompt(key_color), square_rgb)
    if key_img.shape[:2] != square_rgb.shape[:2]:
        key_img = imageops.to_square(key_img, square_rgb.shape[0], "auto")

    os.makedirs(OUT_DIR, exist_ok=True)
    keybg_path = os.path.join(OUT_DIR, f"{_ts()}_keybg.png")
    cv2.imwrite(keybg_path, cv2.cvtColor(key_img, cv2.COLOR_RGB2BGR))

    st["key_img"] = key_img
    return {"keybg_path": keybg_path, "elapsed": time.time() - t0}


def run_mask(session_id: str, params: dict) -> dict:
    """-> {"mask_path", "mask_svg_path", "elapsed", "stats"}"""
    st = _session(session_id)
    key_img, base_rgb = st.get("key_img"), st.get("square")
    if key_img is None or base_rgb is None:
        raise ValueError("先に①②④-aを実行してください")

    t0 = time.time()
    key_color = _p(params, "key_color")
    r = maskgen.build_matte(
        key_img, key_preset=key_color, scale=int(_p(params, "sr_scale")),
        tile=int(_p(params, "sr_tile")), backend=_p(params, "sr_backend"),
        tol=float(_p(params, "tol")), soft=float(_p(params, "soft")),
        strict_bg=float(_p(params, "strict_bg")), strict_fg=float(_p(params, "strict_fg")),
        min_noise_area=int(_p(params, "noise_area")), matte_space=_p(params, "matte_space"),
        raster_from=_p(params, "raster_from"), smooth=float(_p(params, "mask_smooth")),
        rim_hue=float(_p(params, "rim_hue")))

    method = _p(params, "edge_method")
    fg_rgb, alpha_src, bst = maskgen.apply_to_base(
        base_rgb, r["alpha_source"], method, float(_p(params, "decontam")),
        _p(params, "matte_space"), trimap_band=int(_p(params, "trimap_band")),
        key_bg=r["key_bg"], band_authority=bool(_p(params, "band_authority")))

    raster_from = _p(params, "raster_from")
    cut_base = (alpha_src if method == "matting" or str(raster_from).startswith("soft")
                else np.minimum(r["alpha_cut_base"], alpha_src))

    svg_str = maskgen.to_svg(r["alpha_hi"], key_img.shape[0], r["scale"], float(_p(params, "mask_smooth")))

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = _ts()
    mask_path = os.path.join(OUT_DIR, f"{ts}_mask.png")
    mask_svg_path = os.path.join(OUT_DIR, f"{ts}_mask.svg")
    cv2.imwrite(mask_path, cut_base)
    with open(mask_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)

    st.update(fg_rgb=fg_rgb, cut_base=cut_base)
    return {
        "mask_path": mask_path, "mask_svg_path": mask_svg_path,
        "elapsed": time.time() - t0,
        "stats": {"backend": r["stats"]["backend"], "scale": r["scale"],
                  "solver": bst.get("solver"), "pinned": bst.get("pinned", 0)},
    }


def run_fringe(session_id: str, choke: float, feather: float, gamma: float) -> dict:
    """-> {"cutout_path", "elapsed"}. Cheap: no Gemini call, no matte re-solve."""
    st = _session(session_id)
    fg_rgb, alpha_cut_base = st.get("fg_rgb"), st.get("cut_base")
    if alpha_cut_base is None:
        raise ValueError("先に背景マスク生成を実行してください")

    t0 = time.time()
    a = fringe.adjust(alpha_cut_base, float(choke), float(feather), float(gamma))
    rgba = fringe.compose_rgba(fg_rgb, a)

    os.makedirs(OUT_DIR, exist_ok=True)
    cut_path = os.path.join(OUT_DIR, f"{_ts()}_cutout.png")
    cv2.imwrite(cut_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    st["cut_alpha"] = a
    return {"cutout_path": cut_path, "elapsed": time.time() - t0, "alpha_mean": float(a.mean())}


def _composite(layers: list) -> np.ndarray:
    comp = np.full_like(layers[0]["rgb"], 255, np.uint8).astype(np.float32)
    for layer in layers:
        a = layer["alpha"].astype(np.float32)[..., None] / 255.0
        comp = layer["rgb"].astype(np.float32) * a + comp * (1 - a)
    return comp.astype(np.uint8)


def run_psd(session_id: str, params: dict) -> dict:
    """-> {"psd_path", "elapsed", "size_mb"}"""
    st = _session(session_id)
    square_rgb, alpha = st.get("square"), st.get("alpha")
    if square_rgb is None or alpha is None:
        raise ValueError("先に画像を読み込んでください")

    t0 = time.time()
    sh, sw = square_rgb.shape[:2]
    cut_alpha, fg_rgb, gen_rgb = st.get("cut_alpha"), st.get("fg_rgb"), st.get("gen_rgb")
    cut = np.full((sh, sw), 255, np.uint8) if cut_alpha is None else cut_alpha

    layers = [{"name": "original", "rgb": fg_rgb if fg_rgb is not None else square_rgb, "alpha": cut}]
    if gen_rgb is not None:
        layers.append({"name": "generated", "rgb": gen_rgb, "alpha": cut})
    lineart_alpha = alpha if cut_alpha is None else (
        (alpha.astype(np.uint16) * cut.astype(np.uint16) // 255).astype(np.uint8))
    layers.append({"name": "lineart", "rgb": np.zeros((sh, sw, 3), np.uint8), "alpha": lineart_alpha})

    comp = _composite(layers)
    out_w, out_h = sw, sh

    box = st.get("box")
    if bool(_p(params, "crop_on")) and box is not None:
        x0, y0, w, h = box
        crop = lambda a: a[y0:y0 + h, x0:x0 + w]
        for layer in layers:
            layer["rgb"] = crop(layer["rgb"])
            layer["alpha"] = crop(layer["alpha"])
        comp = crop(comp)
        out_w, out_h = w, h

    os.makedirs(OUT_DIR, exist_ok=True)
    psd_path = os.path.join(OUT_DIR, f"{_ts()}_layers.psd")
    psd_writer.write_psd(psd_path, out_w, out_h, layers, comp)

    return {
        "psd_path": psd_path, "elapsed": time.time() - t0,
        "size_mb": os.path.getsize(psd_path) / (1024 * 1024),
    }
