import base64
import json
import urllib.request
import urllib.error

import numpy as np
import cv2

MODEL_ID = "gemini-3.1-flash-image"
INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_ID}:generateContent"
)


def _rgb_to_png_b64(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to encode png")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_interactions_payload(prompt: str, source_rgb: np.ndarray,
                                lineart_rgb: np.ndarray = None) -> dict:
    input_blocks = [
        {"type": "text", "text": prompt},
        {"type": "image", "mime_type": "image/png", "data": _rgb_to_png_b64(source_rgb)},
    ]
    if lineart_rgb is not None:
        input_blocks.append(
            {"type": "image", "mime_type": "image/png", "data": _rgb_to_png_b64(lineart_rgb)}
        )

    return {
        "model": MODEL_ID,
        "input": input_blocks,
        "response_format": {
            "type": "image",
            "mime_type": "image/png",
            "aspect_ratio": "1:1",
            "image_size": "2K",
        },
    }


def build_generate_content_payload(prompt: str, source_rgb: np.ndarray,
                                    lineart_rgb: np.ndarray = None) -> dict:
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": _rgb_to_png_b64(source_rgb)}},
    ]
    if lineart_rgb is not None:
        parts.append(
            {"inline_data": {"mime_type": "image/png", "data": _rgb_to_png_b64(lineart_rgb)}}
        )

    return {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": "1:1", "imageSize": "2K"},
        },
    }


def _find_first_image(node):
    if isinstance(node, dict):
        mime = node.get("mime_type") or node.get("mimeType")
        data = node.get("data")
        if isinstance(mime, str) and mime.startswith("image/") and isinstance(data, str):
            return data
        for v in node.values():
            found = _find_first_image(v)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_first_image(item)
            if found is not None:
                return found
    return None


def _post_json(url: str, api_key: str, payload: dict, timeout: int = 300) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1500]
        raise RuntimeError(f"Gemini API {e.code}: {detail}") from None


def generate_image(api_key: str, prompt: str, source_rgb: np.ndarray,
                    lineart_rgb: np.ndarray = None) -> np.ndarray:
    payload = build_interactions_payload(prompt, source_rgb, lineart_rgb)
    try:
        resp = _post_json(INTERACTIONS_URL, api_key, payload)
    except RuntimeError as e:
        if "404" in str(e) or "400" in str(e):
            payload = build_generate_content_payload(prompt, source_rgb, lineart_rgb)
            resp = _post_json(GENERATE_CONTENT_URL, api_key, payload)
        else:
            raise

    b64 = _find_first_image(resp)
    if b64 is None:
        raise RuntimeError(f"no image found in Gemini response: {json.dumps(resp)[:1500]}")

    raw = base64.b64decode(b64)
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode generated image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
