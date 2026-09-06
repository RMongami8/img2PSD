"""FastAPI app for the new frontend: REST/SSE endpoints over web/pipeline.py,
plus the static files it needs and the mounted classic Gradio UI.

Kept separate from app.py's do_* functions on purpose -- see pipeline.py's
module docstring.
"""
import asyncio
import json
import os
import queue
import tempfile
import threading
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from web import pipeline

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Mirrors the classic UI's own numbered steps (README/app.py), so the two
# front ends narrate the pipeline identically.
STEP_LABELS = {
    "preprocess": "① 前処理・線画抽出",
    "generate": "② AI彩色",
    "keygen": "④-a キー背景生成",
    "mask": "④-b 背景マスク生成",
    "fringe": "⑤ フリンジ調整",
    "psd": "③ PSD書き出し",
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_pipeline_sync(q: "queue.Queue", session_id: str, image_path: str,
                        api_key: str, prompt: str, params: dict, bg_remove: bool):
    """Runs on a worker thread; pushes SSE-ready dicts onto `q`. Blocking calls
    (cv2, the Gemini HTTP request, the matting solve) are fine here -- this is
    exactly why it is not run on the asyncio event loop."""
    try:
        def step(name, fn):
            q.put(("status", {"step": name, "label": STEP_LABELS[name], "state": "running"}))
            result = fn()
            q.put(("status", {"step": name, "label": STEP_LABELS[name], "state": "done", **result}))
            return result

        step("preprocess", lambda: pipeline.run_preprocess(session_id, image_path, params))
        step("generate", lambda: pipeline.run_generate(
            session_id, api_key, prompt, bool(params.get("use_lineart_ref", False))))

        if bg_remove:
            step("keygen", lambda: pipeline.run_keygen(session_id, api_key, params.get("key_color", "green")))
            step("mask", lambda: pipeline.run_mask(session_id, params))
            step("fringe", lambda: pipeline.run_fringe(
                session_id, params.get("choke", 0.0), params.get("feather", 0.0),
                params.get("matte_gamma", 1.0)))

        psd = step("psd", lambda: pipeline.run_psd(session_id, params))
        q.put(("done", {"urls": {"psd_url": _out_url(psd["psd_path"])}}))
    except Exception as e:  # noqa: BLE001 -- reported to the client, not raised in the thread
        q.put(("error", {"message": str(e)}))
    finally:
        q.put((None, None))


def _out_url(path: str) -> str:
    return "/outputs/" + os.path.basename(path)


def create_app() -> FastAPI:
    app = FastAPI()

    @app.post("/api/run")
    async def run(image: UploadFile = File(...), session_id: str = Form(...),
                  api_key: str = Form(...), prompt: str = Form(...),
                  params: str = Form("{}")):
        parsed = json.loads(params) if params else {}
        bg_remove = bool(parsed.pop("bg_remove", True))
        pipeline.new_session(session_id)

        suffix = os.path.splitext(image.filename or "")[1] or ".png"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(await image.read())

        q: "queue.Queue" = queue.Queue()
        threading.Thread(
            target=_run_pipeline_sync,
            args=(q, session_id, tmp_path, api_key, prompt, parsed, bg_remove),
            daemon=True,
        ).start()

        async def gen():
            loop = asyncio.get_event_loop()
            try:
                while True:
                    event, data = await loop.run_in_executor(None, q.get)
                    if event is None:
                        break
                    for k in list(data.keys()):
                        if k.endswith("_path"):
                            data[k[:-5] + "_url"] = _out_url(data.pop(k))
                    yield _sse(event, data)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/fringe")
    async def fringe_tune(session_id: str = Form(...), choke: float = Form(0.0),
                          feather: float = Form(0.0), gamma: float = Form(1.0)):
        try:
            r = pipeline.run_fringe(session_id, choke, feather, gamma)
        except ValueError as e:
            return JSONResponse({"message": str(e)}, status_code=400)
        r["cutout_url"] = _out_url(r.pop("cutout_path"))
        return r

    @app.get("/api/session")
    async def new_session():
        sid = uuid.uuid4().hex
        pipeline.new_session(sid)
        return {"session_id": sid}

    os.makedirs(pipeline.OUT_DIR, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=pipeline.OUT_DIR), name="outputs")

    @app.get("/")
    async def index():
        # no-store: this is the app shell for a locally-served dev tool, and a
        # cached copy silently hides every change made to it.
        return FileResponse(os.path.join(STATIC_DIR, "index.html"),
                            headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
