import os
import time
from datetime import datetime

import numpy as np
import cv2
import gradio as gr

from core import imageops, lineart, svgout, gemini, psd_writer
from core import chroma, maskgen, fringe

OUT_DIR = "out"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _warn(text: str) -> str:
    return "\n\n　※ " + text


def _append_log(log_text: str, line: str) -> str:
    return (log_text + "\n\n" + line).strip() if log_text else line


def _lineart_rgba(alpha: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = alpha
    return rgba


def _composite(layers: list) -> np.ndarray:
    """Flatten over white.

    Starts from a white canvas and blends every layer including the first, because
    once the background is removed layer 0 is no longer opaque. With all-opaque
    layers this is identical to compositing straight onto layer 0. The PSD image
    data section carries no alpha, so the flattened preview has to be baked onto
    something -- white matches how Photoshop shows a transparent document.
    """
    comp = np.full_like(layers[0]["rgb"], 255, np.uint8).astype(np.float32)
    for layer in layers:
        a = layer["alpha"].astype(np.float32)[..., None] / 255.0
        comp = layer["rgb"].astype(np.float32) * a + comp * (1 - a)
    return comp.astype(np.uint8)


def _parse_leading_int(choice: str, default: int) -> int:
    try:
        return int(str(choice).split()[0])
    except (ValueError, IndexError):
        return default


def do_preprocess(image_path, radius, bias, min_area, smooth, scale_choice, pad_choice, log_text,
                   line_weight=0.0):
    if not image_path:
        raise gr.Error("画像を選択してください")

    t0 = time.time()
    if scale_choice.startswith("3"):
        ss = 3
    elif scale_choice.startswith("2"):
        ss = 2
    else:
        ss = 1

    rgb = imageops.load_rgb(image_path)
    square, box = imageops.to_square(rgb, 2048, pad_choice, return_box=True)
    ink_hi, alpha = lineart.extract_ink(square, radius=int(radius), bias=int(bias),
                                        min_area=int(min_area), ss=ss, line_weight=float(line_weight))
    svg_str = svgout.ink_to_svg(ink_hi, out_size=2048, ss=ss, smooth=int(smooth))

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = _ts()
    square_path = os.path.join(OUT_DIR, f"{ts}_square.png")
    lineart_path = os.path.join(OUT_DIR, f"{ts}_lineart.png")
    svg_path = os.path.join(OUT_DIR, f"{ts}_lineart.svg")

    cv2.imwrite(square_path, cv2.cvtColor(square, cv2.COLOR_RGB2BGR))
    rgba = _lineart_rgba(alpha)
    cv2.imwrite(lineart_path, rgba)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)

    elapsed = time.time() - t0
    line = f"① 前処理+線画抽出 完了 ({elapsed:.1f}s) square={square.shape} svg={len(svg_str)}bytes"
    new_log = _append_log(log_text, line)

    return square, rgba, svg_path, square, ink_hi, alpha, new_log, box


def do_generate(square_rgb, alpha, api_key, prompt, use_lineart_ref, log_text):
    if square_rgb is None:
        raise gr.Error("先に「① 前処理+線画抽出」を実行してください")
    if not api_key:
        raise gr.Error("Gemini API キーを入力してください")
    if not prompt:
        raise gr.Error("プロンプトを入力してください")

    t0 = time.time()
    lineart_ref = None
    if use_lineart_ref and alpha is not None:
        h, w = alpha.shape
        lineart_ref = np.full((h, w, 3), 255, np.uint8)
        lineart_ref[alpha > 127] = (0, 0, 0)

    gen_rgb = gemini.generate_image(api_key, prompt, square_rgb, lineart_ref)
    if gen_rgb.shape[:2] != square_rgb.shape[:2]:
        gen_rgb = imageops.to_square(gen_rgb, square_rgb.shape[0], "auto")

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = _ts()
    gen_path = os.path.join(OUT_DIR, f"{ts}_generated.png")
    cv2.imwrite(gen_path, cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR))

    elapsed = time.time() - t0
    line = f"② Gemini生成 完了 ({elapsed:.1f}s) shape={gen_rgb.shape}"
    new_log = _append_log(log_text, line)

    return gen_rgb, gen_rgb, new_log


def do_keygen(square_rgb, api_key, key_color, log_text):
    """Generate the key-background version of the padded original.

    Its only job is to say where the background is. The colours of the cutout come
    from the original, so nothing the model does to the subject here can reach the
    output -- which is exactly why this is a separate request from the colourisation.
    """
    if square_rgb is None:
        raise gr.Error("先に「① 前処理+線画抽出」を実行してください")
    if not api_key:
        raise gr.Error("Gemini API キーを入力してください")

    t0 = time.time()
    key_img = gemini.generate_image(api_key, gemini.key_only_prompt(key_color), square_rgb)
    if key_img.shape[:2] != square_rgb.shape[:2]:
        key_img = imageops.to_square(key_img, square_rgb.shape[0], "auto")

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"{_ts()}_keybg.png"),
                cv2.cvtColor(key_img, cv2.COLOR_RGB2BGR))

    line = f"④-a キー背景版を生成 完了 ({time.time() - t0:.1f}s) key={key_color}"
    return key_img, key_img, _append_log(log_text, line)


def _align_shift(a_rgb, b_rgb, ds=256, rad=4):
    """Coarse registration between two images, in pixels of the originals.

    The mask is derived from one Gemini pass and applied to a different image, so the
    one assumption worth checking at runtime is that the two still line up. Reported,
    never corrected: a silent shift would be worse than a warning.
    """
    def edges(x):
        g = cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (ds, ds),
                       interpolation=cv2.INTER_AREA)
        return np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3))
    A, B = edges(a_rgb), edges(b_rgb)
    err, dy, dx = min((float(np.abs(A - np.roll(np.roll(B, y, 0), x, 1)).mean()), y, x)
                      for y in range(-rad, rad + 1) for x in range(-rad, rad + 1))
    k = a_rgb.shape[0] / float(ds)
    return int(round(dy * k)), int(round(dx * k))


def do_mask(key_img, base_rgb, key_color, sr_backend, sr_scale, sr_tile, tol, soft,
            strict_bg, strict_fg, noise_area, raster_from, mask_smooth,
            matte_space, rim_hue, edge_method, trimap_band, decontam, band_mode,
            log_text):
    if key_img is None:
        raise gr.Error("先に「④-a キー背景版を生成」を実行してください")
    if base_rgb is None:
        raise gr.Error("先に「① 前処理+線画抽出」を実行してください")

    t0 = time.time()
    backend = "lanczos" if str(sr_backend).startswith("Lanczos") else "auto"
    scale = _parse_leading_int(sr_scale, 2)
    rf = str(raster_from).split()[0]

    r = maskgen.build_matte(
        key_img, key_preset=key_color, scale=scale, tile=int(sr_tile),
        backend=backend, tol=float(tol), soft=float(soft),
        strict_bg=float(strict_bg), strict_fg=float(strict_fg),
        min_noise_area=int(noise_area), matte_space=matte_space,
        raster_from=rf, smooth=float(mask_smooth), rim_hue=float(rim_hue))

    # The key image has done its job and is dropped here. Colour comes from the image
    # actually being cut out, so no key colour can survive into the output; the edge
    # is cleaned against that image's own local background instead.
    # Colour recovery always uses alpha_source, never the choked matte: undoing the
    # composite with a shrunken alpha over-corrects the edge and whitens dark lines.
    method = str(edge_method).split()[0].lower()
    # key_bg goes along separately from the matte: it is the key image's own
    # un-eroded verdict, and it is the only thing left that can still speak for a
    # gap between hair strands too narrow to survive the trimap's erosion.
    fg_rgb, alpha_src, bst = maskgen.apply_to_base(
        base_rgb, r["alpha_source"], method, float(decontam), matte_space,
        trimap_band=int(trimap_band), key_bg=r["key_bg"],
        band_authority=str(band_mode).startswith("不明帯全体"))
    # The cut matte inherits the refinement too, so the shape the PSD gets is the one
    # the preview showed.
    cut_base = (alpha_src if method == "matting" or str(raster_from).startswith("soft")
                else np.minimum(r["alpha_cut_base"], alpha_src))

    svg_str = maskgen.to_svg(r["alpha_hi"], key_img.shape[0], r["scale"],
                             float(mask_smooth))

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = _ts()
    mask_path = os.path.join(OUT_DIR, f"{ts}_mask.png")
    svg_path = os.path.join(OUT_DIR, f"{ts}_mask.svg")
    cv2.imwrite(mask_path, cut_base)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)

    st = r["stats"]
    att = max(st["attempted"], 1)
    line = (
        f"④ 背景マスク生成 完了 ({time.time() - t0:.1f}s) backend={st['backend']} "
        f"scale={r['scale']} hi={r['hi_size']} key={r['key_rgb'].astype(int).tolist()}\n\n"
        f"　fallback: denom={100 * st['fallback_denom'] / att:.2f}% "
        f"range={100 * st['fallback_alpha_range'] / att:.2f}% "
        f"resid={100 * st['fallback_residual'] / att:.2f}% "
        f"nofg={100 * st['fallback_no_foreground'] / att:.2f}% / "
        f"seed_miss={st['seed_miss']} f_native={st['f_native_used']} "
        f"rim_removed={st['rim_removed']}" + "\n\n"
        f"　境界: {bst['solver']} "
        + (f"trimap fg/bg/unknown = {100 * bst['fg']:.1f}/{100 * bst['bg']:.1f}/"
           f"{100 * bst['unknown']:.2f}% ({bst['unknown_px']}px) "
           f"救済 消えた髪{bst['lost_px']} 白フチ{bst['false_px']}"
           if "unknown" in bst else f"refined={bst.get('refined', 0)}")
    )
    for k, label in (("tiles_solved", "タイル"), ("overlap_alpha_max", "重なりα差"),
                     ("ichol_retries", "ichol再試行"), ("pinned", "背景固定")):
        if bst.get(k):
            line += f" {label}={bst[k]}"
    dy, dx = _align_shift(key_img, base_rgb)
    if dy or dx:
        line += _warn(f"キー背景版が元画像に対して {dx:+d},{dy:+d} px ずれています。"
                      "マスクの位置が合っていない可能性があります。④-a をやり直してください。")
    # Only the analytic route reports this: it is the only one whose colours come from
    # unpremultiplying against the estimated background, so it is the only one a
    # region without such an estimate changes.
    if bst.get("no_local_bg"):
        line += _warn(f"元画像の背景が一様でない領域が {bst['no_local_bg']} px あり、"
                      "そこではエッジ色浄化を行っていません。")
    if st["seed_miss"] > 0:
        line += "\n\n　※ seed_miss>0: 超解像がネイティブ解像度の背景の隙間を埋めています。" \
                "SR倍率を下げるか Lanczos に切り替えてください。"

    # alpha_cut_base is returned twice: once as the cached base the fringe controls
    # act on, once as the live cut so PSD export works even if step 5 is skipped
    return (cut_base, svg_path, alpha_src, cut_base,
            cut_base, fg_rgb, r["key_rgb"], _append_log(log_text, line))


def do_fringe(fg_rgb, alpha_cut_base, choke, feather, gamma, log_text):
    if alpha_cut_base is None:
        raise gr.Error("先に「④ 背景マスク生成」を実行してください")
    t0 = time.time()
    a = fringe.adjust(alpha_cut_base, float(choke), float(feather), float(gamma))
    rgba = fringe.compose_rgba(fg_rgb, a)

    os.makedirs(OUT_DIR, exist_ok=True)
    cut_path = os.path.join(OUT_DIR, f"{_ts()}_cutout.png")
    cv2.imwrite(cut_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    line = (f"⑤ フリンジ調整 ({time.time() - t0:.2f}s) choke={choke} "
            f"feather={feather} gamma={gamma} alpha_mean={a.mean():.1f}")
    return rgba, a, _append_log(log_text, line)


def do_psd(square_rgb, alpha, gen_rgb, log_text, box=None, crop_on=True,
           cut_alpha=None, fg_rgb=None):
    if square_rgb is None or alpha is None:
        raise gr.Error("先に「① 前処理+線画抽出」を実行してください")

    t0 = time.time()
    sh, sw = square_rgb.shape[:2]
    cut = np.full((sh, sw), 255, np.uint8) if cut_alpha is None else cut_alpha

    # fg_rgb is the original with its edge cleaned against its own background, so it
    # replaces the original layer -- not the generated one, which is a different image.
    layers = [{"name": "original",
               "rgb": fg_rgb if fg_rgb is not None else square_rgb, "alpha": cut}]
    if gen_rgb is not None:
        layers.append({"name": "generated", "rgb": gen_rgb, "alpha": cut})
    lineart_alpha = alpha if cut_alpha is None else (
        (alpha.astype(np.uint16) * cut.astype(np.uint16) // 255).astype(np.uint8))
    layers.append({"name": "lineart", "rgb": np.zeros((sh, sw, 3), np.uint8),
                   "alpha": lineart_alpha})

    comp = _composite(layers)
    out_w, out_h = sw, sh

    if crop_on and box is not None:
        x0, y0, w, h = box
        crop = lambda a: a[y0:y0 + h, x0:x0 + w]
        for layer in layers:
            layer["rgb"] = crop(layer["rgb"])
            layer["alpha"] = crop(layer["alpha"])
        comp = crop(comp)
        out_w, out_h = w, h

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = _ts()
    psd_path = os.path.join(OUT_DIR, f"{ts}_layers.psd")
    psd_writer.write_psd(psd_path, out_w, out_h, layers, comp)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(psd_path) / (1024 * 1024)
    line = (f"③ PSD書き出し 完了 ({elapsed:.1f}s) {size_mb:.1f}MB layers={len(layers)} "
            f"PSD: {out_w}×{out_h}")
    new_log = _append_log(log_text, line)

    return psd_path, new_log


def do_all(image_path, api_key, prompt, use_lineart_ref, radius, bias, min_area, smooth,
           scale_choice, pad_choice, log_text, line_weight, crop_on,
           bg_remove, key_color, sr_backend, sr_scale, sr_tile,
           tol, soft, strict_bg, strict_fg, noise_area, raster_from, mask_smooth,
           matte_space, rim_hue, edge_method, trimap_band, decontam, band_mode,
           choke, feather, matte_gamma):
    square, rgba, svg_path, sq_state, ink_hi_state, alpha_state, log1, box_state = do_preprocess(
        image_path, radius, bias, min_area, smooth, scale_choice, pad_choice, log_text, line_weight)
    gen_disp, gen_state, log2 = do_generate(sq_state, alpha_state, api_key, prompt,
                                            use_lineart_ref, log1)

    key_disp = mask_disp = cut_disp = mask_svg = None
    key_state = alpha_src = cut_base = cut_alpha = fg_rgb = key_rgb = None
    log5 = log2
    if bg_remove:
        key_disp, key_state, log3 = do_keygen(sq_state, api_key, key_color, log2)
        (mask_disp, mask_svg, alpha_src, cut_base, cut_alpha, fg_rgb, key_rgb,
         log4) = do_mask(key_state, sq_state, key_color, sr_backend, sr_scale, sr_tile,
                         tol, soft, strict_bg, strict_fg, noise_area, raster_from,
                         mask_smooth, matte_space, rim_hue, edge_method, trimap_band,
                         decontam, band_mode, log3)
        cut_disp, cut_alpha, log5 = do_fringe(fg_rgb, cut_base, choke, feather,
                                              matte_gamma, log4)

    psd_path, log6 = do_psd(sq_state, alpha_state, gen_state, log5, box_state, crop_on,
                            cut_alpha, fg_rgb)
    return (square, rgba, svg_path, gen_disp, key_disp, mask_disp, cut_disp, mask_svg,
            psd_path, sq_state, ink_hi_state, alpha_state, gen_state, key_state,
            alpha_src, cut_base, cut_alpha, fg_rgb, key_rgb, log6, box_state)


with gr.Blocks(title="lineart2psd") as demo:
    state_square = gr.State(None)
    state_ink_hi = gr.State(None)
    state_alpha = gr.State(None)
    state_gen = gr.State(None)
    state_key_img = gr.State(None)
    state_log = gr.State("")
    state_box = gr.State(None)
    state_alpha_source = gr.State(None)
    state_cut_base = gr.State(None)
    state_cut_alpha = gr.State(None)
    state_fg_rgb = gr.State(None)
    state_key_rgb = gr.State(None)

    gr.Markdown("[✨ 新しいUIを試す](/)")

    with gr.Row():
        with gr.Column():
            in_image = gr.Image(type="filepath", label="入力画像")
            in_api_key = gr.Textbox(label="Gemini API キー", type="password")
            in_prompt = gr.Textbox(label="プロンプト", lines=5, value="")
            in_use_lineart_ref = gr.Checkbox(label="線画も参照画像として送る", value=False)

            with gr.Accordion("線画パラメータ", open=False):
                in_radius = gr.Slider(1, 8, value=3, step=1, label="線の太さ想定 (radius)")
                in_bias = gr.Slider(-80, 80, value=-15, step=1, label="しきい値バイアス")
                in_min_area = gr.Slider(0, 200, value=24, step=2, label="最小面積(ノイズ除去)")
                in_smooth = gr.Slider(0, 5, value=1, step=1, label="SVG 滑らかさ")
                in_scale = gr.Dropdown(["3 (最高品質)", "2 (推奨)", "1 (高速)"], value="2 (推奨)",
                                        label="内部倍率")
                in_pad = gr.Dropdown(["auto", "white", "black"], value="auto", label="パディング色")
                in_line_weight = gr.Slider(-2.0, 2.0, value=-0.4, step=0.1,
                                            label="線の太さ微調整（−で細く）")
                gr.Markdown(
                    "| 用途 | radius | bias | 最小面積 | 滑らかさ | 内部倍率 | 太さ微調整 |\n"
                    "|---|---|---|---|---|---|---|\n"
                    "| 標準（現状より少し細い） | 3 | -15 | 24 | 1 | 2 | **-0.4** |\n"
                    "| 細線・高品質 | 2 | -20 | 16 | 0 | 3 | **-0.6** |\n"
                    "| 極細 | 2 | -25 | 16 | 0 | 3 | **-1.0** |\n"
                    "| 太らせたい | 3 | -10 | 24 | 1 | 2 | +0.5 |\n\n"
                    "注記: 線が途切れる場合は太さ微調整を +0.2 ずつ戻す。かすれが増える場合は最小面積を下げる。"
                )

            with gr.Accordion("背景透過", open=False):
                in_bg_remove = gr.Checkbox(label="背景透過を有効にする", value=True)
                in_key_color = gr.Dropdown(["green", "magenta", "blue"], value="green",
                                            label="キー色（キャラに使われていない色を選ぶ）")
                gr.Markdown(
                    "キー背景版は ②とは別に生成します。同じ生成に統合すると、"
                    "色を取り出す画像そのものにキー色が入ってしまい、"
                    "モデルが細い髪を「暗いキー色」で描いた縁は原理的に取り切れません。"
                )
                in_sr_backend = gr.Dropdown(
                    ["Lanczos (推奨)", "Real-ESRGAN (ONNX/CPU)"], value="Lanczos (推奨)",
                    label="超解像バックエンド")
                in_sr_scale = gr.Dropdown(["4 (高負荷)", "2 (推奨)", "1 (SRなし)"],
                                           value="2 (推奨)", label="超解像倍率")
                in_sr_tile = gr.Slider(128, 512, value=256, step=64, label="タイルサイズ")
                in_key_tol = gr.Slider(0, 100, value=30, step=1, label="キー距離しきい値")
                in_key_soft = gr.Slider(1, 100, value=20, step=1, label="未知帯の幅")
                in_strict_bg = gr.Slider(0, 40, value=8, step=1,
                                          label="確実背景シード厳格度")
                in_strict_fg = gr.Slider(40, 200, value=140, step=5,
                                          label="確実前景シード厳格度（前景色の推定元）")
                in_noise_area = gr.Slider(0, 500, value=0, step=10,
                                           label="ノイズ除去面積（2048px基準・0推奨）")
                in_raster_from = gr.Dropdown(
                    ["soft (連続)", "binary (シャープ)", "path (旧仕様)"],
                    value="soft (連続)", label="マスクの作り方")
                in_edge_method = gr.Dropdown(
                    ["matting (α再推定＋前景色推定・推奨)", "foreground (前景色推定のみ)",
                     "analytic (旧・解析的)"],
                    value="matting (α再推定＋前景色推定・推奨)", label="境界の処理")
                in_trimap_band = gr.Slider(1, 12, value=3, step=1,
                                            label="trimap の不明バンド幅 (px)")
                in_band_mode = gr.Dropdown(
                    ["不明帯全体を再推定（隙間が綺麗・推奨）",
                     "不一致のみ再推定（髪を優先して残す）"],
                    value="不明帯全体を再推定（隙間が綺麗・推奨）",
                    label="不明帯の扱い")
                gr.Markdown(
                    "実測（実画像）: 元画像の白背景が髪の隙間に残る量は "
                    "**不明帯全体で 5px、不一致のみで 204px**（α>128）。"
                    "代わりに全体再推定は細い髪を約900px（図の0.09%）失います。"
                    "髪が痩せて見える場合だけ「不一致のみ」に切り替えてください。")
                in_rim_hue = gr.Slider(
                    0.0, 0.6, value=0.35, step=0.05,
                    label="キー色リム除去（0で無効・キー色の服がある時は下げる）")
                in_mask_smooth = gr.Slider(0, 3, value=0.25, step=0.25,
                                            label="SVG 輪郭簡略化")
                in_matte_space = gr.Dropdown(["srgb", "linear"], value="srgb",
                                              label="マット計算の色空間")
                gr.Markdown(
                    "**フリンジ調整**（マスク生成後、スライダーを離すと即反映）。"
                    "形状だけを動かし、前景色は再推定しません。"
                    "そのため choke を動かしても色は変化しません。")
                in_choke = gr.Slider(-3.0, 3.0, value=0.0, step=0.1,
                                      label="エッジ収縮/膨張 (px)")
                in_feather = gr.Slider(0, 5, value=0.0, step=0.1, label="エッジぼかし (px)")
                in_matte_gamma = gr.Slider(0.2, 3.0, value=1.0, step=0.05,
                                            label="マット濃度 (ガンマ)")
                in_decontam = gr.Slider(
                    0, 1, value=1.0, step=0.05,
                    label="エッジ色浄化（元画像の背景を境界画素から差し引く）")
                gr.Markdown(
                    "ベンチマーク実測: Lanczos の方が Real-ESRGAN より "
                    "**マット精度が約2倍良く、約7倍速い**（alpha_mae_edge 19.2 vs 36.4）。"
                    "ESRGAN はエッジを描き直すため、前景と背景の混合比という"
                    "マッティングに必要な情報を壊します。倍率は 2 で十分（1 より良く、4 と同等）。"
                )

            in_crop_psd = gr.Checkbox(label="PSDを元画像の比率でクロップする", value=True)

            btn_preprocess = gr.Button("① 前処理+線画抽出")
            btn_generate = gr.Button("② Gemini で生成")
            btn_keygen = gr.Button("④-a キー背景版を生成")
            btn_mask = gr.Button("④-b 背景マスク生成")
            btn_fringe = gr.Button("⑤ フリンジ調整")
            btn_psd = gr.Button("③ PSD 書き出し")
            btn_all = gr.Button("▶ 一括実行")

        with gr.Column():
            out_square = gr.Image(label="2048×2048 パディング済み")
            out_lineart = gr.Image(label="線画プレビュー(透過)")
            out_svg = gr.File(label="線画SVG")
            out_gen = gr.Image(label="生成画像")
            out_key = gr.Image(label="キー背景版（マスク元・出力には使いません）")
            out_mask = gr.Image(label="背景マスク(白黒)")
            out_cutout = gr.Image(label="背景透過プレビュー", image_mode="RGBA")
            out_mask_svg = gr.File(label="マスクSVG")
            out_psd = gr.File(label="PSD")
            out_log = gr.Markdown(label="ログ")

    btn_preprocess.click(
        fn=do_preprocess,
        inputs=[in_image, in_radius, in_bias, in_min_area, in_smooth, in_scale, in_pad, state_log,
                in_line_weight],
        outputs=[out_square, out_lineart, out_svg, state_square, state_ink_hi, state_alpha,
                 state_log, state_box],
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)

    btn_generate.click(
        fn=do_generate,
        inputs=[state_square, state_alpha, in_api_key, in_prompt, in_use_lineart_ref,
                state_log],
        outputs=[out_gen, state_gen, state_log],
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)

    btn_keygen.click(
        fn=do_keygen,
        inputs=[state_square, in_api_key, in_key_color, state_log],
        outputs=[out_key, state_key_img, state_log],
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)

    mask_inputs = [state_key_img, state_square, in_key_color, in_sr_backend, in_sr_scale,
                   in_sr_tile, in_key_tol, in_key_soft, in_strict_bg, in_strict_fg,
                   in_noise_area, in_raster_from, in_mask_smooth, in_matte_space,
                   in_rim_hue, in_edge_method, in_trimap_band, in_decontam,
                   in_band_mode, state_log]
    mask_outputs = [out_mask, out_mask_svg, state_alpha_source, state_cut_base,
                    state_cut_alpha, state_fg_rgb, state_key_rgb, state_log]
    btn_mask.click(
        fn=do_mask, inputs=mask_inputs, outputs=mask_outputs,
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)

    fringe_inputs = [state_fg_rgb, state_cut_base, in_choke, in_feather,
                     in_matte_gamma, state_log]
    fringe_outputs = [out_cutout, state_cut_alpha, state_log]
    btn_fringe.click(
        fn=do_fringe, inputs=fringe_inputs, outputs=fringe_outputs,
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)

    # release rather than change: the fringe pass is cheap but not free, and change
    # fires on every intermediate value while the handle is being dragged
    for slider in (in_choke, in_feather, in_matte_gamma):
        slider.release(fn=do_fringe, inputs=fringe_inputs, outputs=fringe_outputs)

    btn_psd.click(
        fn=do_psd,
        inputs=[state_square, state_alpha, state_gen, state_log, state_box, in_crop_psd,
                state_cut_alpha, state_fg_rgb],
        outputs=[out_psd, state_log],
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)

    btn_all.click(
        fn=do_all,
        inputs=[in_image, in_api_key, in_prompt, in_use_lineart_ref, in_radius, in_bias,
                in_min_area, in_smooth, in_scale, in_pad, state_log, in_line_weight,
                in_crop_psd, in_bg_remove, in_key_color, in_sr_backend,
                in_sr_scale, in_sr_tile, in_key_tol, in_key_soft, in_strict_bg,
                in_strict_fg, in_noise_area, in_raster_from, in_mask_smooth,
                in_matte_space, in_rim_hue, in_edge_method, in_trimap_band,
                in_decontam, in_band_mode, in_choke, in_feather,
                in_matte_gamma],
        outputs=[out_square, out_lineart, out_svg, out_gen, out_key, out_mask,
                 out_cutout, out_mask_svg, out_psd,
                 state_square, state_ink_hi, state_alpha, state_gen, state_key_img,
                 state_alpha_source, state_cut_base, state_cut_alpha, state_fg_rgb,
                 state_key_rgb, state_log, state_box],
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)


if __name__ == "__main__":
    import webbrowser

    import gradio
    import uvicorn

    from web.server import create_app

    # The new UI at "/" and the REST/SSE API it talks to are a separate FastAPI
    # app; the classic Gradio UI is mounted onto it unchanged so both are
    # served from one process/port and can link to each other with a plain
    # <a href>. See web/pipeline.py's module docstring for why the new UI's
    # backend does not import do_preprocess/do_generate/etc. from this file.
    fastapi_app = create_app()
    fastapi_app = gradio.mount_gradio_app(fastapi_app, demo, path="/gradio")

    webbrowser.open("http://127.0.0.1:7860/")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=7860)
