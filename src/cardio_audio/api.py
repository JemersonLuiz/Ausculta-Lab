"""Aplicação local: arquivos transitórios, sem persistência dos uploads."""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .audio import (AudioValidationError, CLASS_NAMES, MAX_DURATION_S, MAX_UPLOAD_BYTES,
                    decode_audio, extract_features, quality_warnings, select_segment, spectrogram_png)
from .predictor import ModelUnavailableError, Predictor

LIMITATIONS = [
    "Protótipo acadêmico, sem validação para uso clínico.",
    "Classifica um ciclo delimitado; não detecta automaticamente respirações em áudio contínuo.",
    "Pontuações de softmax não são probabilidades clínicas calibradas.",
    "Ausência de sibilos/estertores não significa ausência de doença.",
    "O módulo de arritmia não foi implementado: faltam dados de áudio com confirmação de ritmo.",
    "Redimensionar o tempo do ciclo pode alterar a representação de eventos curtos.",
]


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def create_app(project_root: Path | None = None, model_dir: Path | None = None) -> FastAPI:
    root = Path(project_root or os.environ.get("AUSCULTA_ROOT", Path(__file__).resolve().parents[2])).resolve()
    model_path = Path(model_dir or root / "models").resolve()
    application = FastAPI(title="Ausculta Lab", version=__version__, docs_url="/api/docs", redoc_url=None)
    application.state.predictor = None
    application.state.model_error = None
    application.state.prediction_lock = Lock()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    def get_predictor():
        with application.state.prediction_lock:
            if application.state.predictor is None:
                application.state.predictor = Predictor(model_path)
            return application.state.predictor

    application.state.get_predictor = get_predictor

    @application.middleware("http")
    async def local_headers(request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in {f"http://{request.headers.get('host')}", f"https://{request.headers.get('host')}"}:
            return JSONResponse({"detail": "Origem externa não permitida nesta aplicação local."}, status_code=403)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_UPLOAD_BYTES + 65536:
                    return JSONResponse({"detail": "O envio excede o limite de 20 MB."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Cabeçalho de tamanho inválido."}, status_code=400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/api/status")
    def status():
        ready = True
        try:
            predictor = get_predictor()
            metadata = predictor.metadata
        except ModelUnavailableError:
            ready, metadata = False, {}
        audit = _read_json(root / "data" / "audit.json", {})
        if not audit:
            audit = _read_json(root / "data" / "raw" / "icbhi" / "audit.json", {})
        # Metadados distribuídos no pacote sem necessidade de toda a base.
        audit = audit or _read_json(model_path / "dataset.json", {})
        comparison = _read_json(model_path / "metrics.json", {})
        return {"version": __version__, "model_ready": ready,
                "model": {"name": metadata.get("name", "CNN respiratória"), "task": "Classificação de ciclos respiratórios",
                          "classes": CLASS_NAMES, "trained_at": metadata.get("trained_at"),
                          "metrics": comparison.get("cnn", {}).get("test", {}), "comparison": comparison,
                          "limitations": LIMITATIONS, "metadata": metadata},
                "dataset": audit,
                "limits": {"max_upload_mb": 20, "max_duration_s": MAX_DURATION_S,
                           "min_segment_s": 0.1, "max_segment_s": 20}}

    @application.get("/api/examples")
    def examples():
        rows = _read_json(root / "examples" / "index.json", [])
        return [{key: value for key, value in row.items() if key in {"id", "name", "label", "start", "end"}} for row in rows]

    @application.get("/api/examples/{example_id}/audio")
    def example_audio(example_id: str):
        rows = _read_json(root / "examples" / "index.json", [])
        row = next((row for row in rows if row.get("id") == example_id), None)
        if row is None:
            raise HTTPException(404, "Exemplo não encontrado.")
        examples_root = (root / "examples").resolve()
        path = (examples_root / row.get("file", "")).resolve()
        if not path.is_relative_to(examples_root) or not path.is_file() or path.suffix.lower() != ".wav":
            raise HTTPException(404, "Exemplo não disponível.")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @application.post("/api/analyze")
    async def analyze(file: UploadFile = File(...), start: float = Form(...), end: float = Form(...)):
        try:
            payload = await file.read(MAX_UPLOAD_BYTES + 1)
        finally:
            await file.close()
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "O arquivo excede o limite de 20 MB.")
        try:
            audio = await run_in_threadpool(decode_audio, payload)
            selected = select_segment(audio, start, end)
            predictor = get_predictor()

            def compute():
                features = extract_features(selected, audio.sample_rate)
                with application.state.prediction_lock:
                    prediction = predictor.predict_features(features)
                return prediction, spectrogram_png(features)

            prediction, png = await run_in_threadpool(compute)
        except AudioValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"prediction": prediction,
                "audio": {"duration_s": audio.duration_s, "sample_rate": audio.sample_rate, "channels": audio.channels},
                "segment": {"start": start, "end": end, "duration_s": end - start},
                "quality": {"warnings": quality_warnings(selected, audio.sample_rate)},
                "spectrogram_png": png, "model": {"name": predictor.metadata.get("name", "CNN respiratória"),
                                                 "version": predictor.metadata.get("version", __version__)},
                "limitations": LIMITATIONS}

    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    web = root / "web"
    if web.is_dir():
        application.mount("/", StaticFiles(directory=web, html=True), name="interface")
    return application


app = create_app()
