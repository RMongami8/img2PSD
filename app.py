import os
import time
from datetime import datetime

import numpy as np
import cv2
import gradio as gr

from core import imageops, lineart, svgout, gemini, psd_writer

OUT_DIR = "out"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _append_log(log_text: str, line: str) -> str:
    return (log_text + "\n\n" + line).strip() if log_text else line


def _lineart_rgba(alpha: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 3] = alpha
    return rgba


def _composite(layers: list) -> np.ndarray:
    comp = layers[0]["rgb"].astype(np.float32)
    for layer in layers[1:]:
        a = layer["alpha"].astype(np.float32)[..., None] / 255.0
        comp = layer["rgb"].astype(np.float32) * a + comp * (1 - a)
    return comp.astype(np.uint8)


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


def do_psd(square_rgb, alpha, gen_rgb, log_text, box=None, crop_on=True):
    if square_rgb is None or alpha is None:
        raise gr.Error("先に「① 前処理+線画抽出」を実行してください")

    t0 = time.time()
    s = square_rgb.shape[0]
    layers = [{"name": "original", "rgb": square_rgb, "alpha": np.full((s, s), 255, np.uint8)}]
    if gen_rgb is not None:
        layers.append({"name": "generated", "rgb": gen_rgb, "alpha": np.full((s, s), 255, np.uint8)})
    layers.append({"name": "lineart", "rgb": np.zeros((s, s, 3), np.uint8), "alpha": alpha})

    comp = _composite(layers)
    out_w, out_h = s, s

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
           scale_choice, pad_choice, log_text, line_weight=0.0, crop_on=True):
    square, rgba, svg_path, sq_state, ink_hi_state, alpha_state, log1, box_state = do_preprocess(
        image_path, radius, bias, min_area, smooth, scale_choice, pad_choice, log_text, line_weight)
    gen_disp, gen_state, log2 = do_generate(sq_state, alpha_state, api_key, prompt,
                                            use_lineart_ref, log1)
    psd_path, log3 = do_psd(sq_state, alpha_state, gen_state, log2, box_state, crop_on)
    return (square, rgba, svg_path, gen_disp, psd_path,
            sq_state, ink_hi_state, alpha_state, gen_state, log3, box_state)


with gr.Blocks(title="lineart2psd") as demo:
    state_square = gr.State(None)
    state_ink_hi = gr.State(None)
    state_alpha = gr.State(None)
    state_gen = gr.State(None)
    state_log = gr.State("")
    state_box = gr.State(None)

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

            in_crop_psd = gr.Checkbox(label="PSDを元画像の比率でクロップする", value=True)

            btn_preprocess = gr.Button("① 前処理+線画抽出")
            btn_generate = gr.Button("② Gemini で生成")
            btn_psd = gr.Button("③ PSD 書き出し")
            btn_all = gr.Button("▶ 一括実行")

        with gr.Column():
            out_square = gr.Image(label="2048×2048 パディング済み")
            out_lineart = gr.Image(label="線画プレビュー(透過)")
            out_svg = gr.File(label="線画SVG")
            out_gen = gr.Image(label="生成画像")
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
        inputs=[state_square, state_alpha, in_api_key, in_prompt, in_use_lineart_ref, state_log],
        outputs=[out_gen, state_gen, state_log],
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)

    btn_psd.click(
        fn=do_psd,
        inputs=[state_square, state_alpha, state_gen, state_log, state_box, in_crop_psd],
        outputs=[out_psd, state_log],
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)

    btn_all.click(
        fn=do_all,
        inputs=[in_image, in_api_key, in_prompt, in_use_lineart_ref, in_radius, in_bias,
                in_min_area, in_smooth, in_scale, in_pad, state_log, in_line_weight, in_crop_psd],
        outputs=[out_square, out_lineart, out_svg, out_gen, out_psd,
                 state_square, state_ink_hi, state_alpha, state_gen, state_log, state_box],
    ).then(fn=lambda log: log, inputs=state_log, outputs=out_log)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
