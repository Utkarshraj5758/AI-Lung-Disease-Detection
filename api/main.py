"""FastAPI inference service for the chest X-ray classifier. Research and education only.

    uvicorn api.main:app --host 127.0.0.1 --port 8000

``GET /health`` reports the loaded model. ``POST /predict`` takes a multipart field ``file``
(PNG, JPEG or DICOM, at most 10 MB); ``?explain=true`` adds a base64 PNG Grad-CAM overlay.
Interactive docs are at ``/docs``.

One Predictor is loaded per process at startup. The bundle is ``$LUNGXRAY_WEIGHTS`` or
``<artifacts_dir>/best.pt``, else it is downloaded from the Hugging Face repo in
``$LUNGXRAY_HF_REPO``; ``$LUNGXRAY_DEVICE`` picks the torch device (default: CUDA if
available).
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from lungxray import __version__, configure_logging
from lungxray.config import load_config
from lungxray.gradcam import overlay
from lungxray.inference import DISCLAIMER, Predictor, load_image, output_summary
from lungxray.model import bundle_source, resolve_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # boundaries and part headers around the file
ACCEPTED_TYPES = {"image/png", "image/jpeg", "application/dicom"}
DICOM_SUFFIXES = (".dcm", ".dicom")  # application/octet-stream is accepted for these
OVERLAY_ALPHA = 0.4

log = logging.getLogger("lungxray.api")


class ModelDetails(BaseModel):
    name: str
    git_commit: str | None
    created_at: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str = Field(description="Model name plus a hash of the bundle file")
    device: str
    threshold: float
    temperature: float
    model: ModelDetails
    service_version: str


class PredictResponse(BaseModel):
    file_sha256: str = Field(description="SHA-256 of the uploaded bytes; the file name is unused")
    label: str = Field(description="Predicted class, e.g. NORMAL or PNEUMONIA")
    prob_pneumonia: float = Field(
        ge=0, le=1, description="Temperature-scaled P(pneumonia); the threshold applies to this"
    )
    prob_pneumonia_raw: float = Field(ge=0, le=1, description="Softmax of the raw logits")
    temperature: float
    threshold: float
    threshold_source: str
    borderline: bool = Field(
        description="P(pneumonia) lies inside the log-odds borderline band around the threshold"
    )
    borderline_band: tuple[float, float]
    input_warnings: list[str] = Field(
        description="Basic input sanity check (near-constant, colour, tiny); not an X-ray detector"
    )
    model_version: str
    model: ModelDetails
    generated_at: str
    disclaimer: str
    gradcam_png_base64: str | None = Field(
        default=None,
        description="Base64 PNG Grad-CAM overlay for the predicted class on the model's "
        "center crop; only with explain=true",
    )


class ErrorResponse(BaseModel):
    detail: str


class BodySizeLimitMiddleware:
    """Reject bodies over ``max_body_bytes`` (413) or of unknown length (411) before parsing."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            declared = dict(scope["headers"]).get(b"content-length")
            if declared is None:
                response = JSONResponse({"detail": "Content-Length header required."}, 411)
                await response(scope, receive, send)
                return
            try:
                too_large = int(declared) > self.max_body_bytes
            except ValueError:
                too_large = None
            if too_large is None:
                await JSONResponse({"detail": "Invalid Content-Length header."}, 400)(
                    scope, receive, send
                )
                return
            if too_large:
                detail = f"Upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                await JSONResponse({"detail": detail}, 413)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(*, bundle_path: str | Path | None = None, device: str | None = None) -> FastAPI:
    """Build the service; the bundle is loaded once when the app starts (lifespan)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        if bundle_path is None:
            bundle = resolve_bundle(*bundle_source(load_config(CONFIG_PATH), root=ROOT))
        else:
            bundle = Path(bundle_path)
        app.state.predictor = Predictor(bundle, device=device)
        log.info(
            "Loaded %s from %s on %s",
            app.state.predictor.model_version,
            bundle,
            app.state.predictor.device,
        )
        yield

    app = FastAPI(
        title="Chest X-ray pneumonia classifier (research use only)",
        version=__version__,
        description=f"{DISCLAIMER} Uploads are processed in memory and not stored.",
        lifespan=lifespan,
    )
    app.add_middleware(
        BodySizeLimitMiddleware, max_body_bytes=MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
    )
    app.add_exception_handler(Exception, _internal_error)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        predictor: Predictor = request.app.state.predictor
        return HealthResponse(
            model_version=predictor.model_version,
            device=str(predictor.device),
            threshold=predictor.threshold,
            temperature=predictor.temperature,
            model=_model_details(predictor),
            service_version=__version__,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            400: {"model": ErrorResponse},
            411: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"description": "Unreadable image or invalid request"},
        },
    )
    def predict(
        request: Request,
        file: Annotated[UploadFile, File(description="Chest X-ray: PNG, JPEG or DICOM, <= 10 MB")],
        explain: Annotated[
            bool, Query(description="Include a base64 PNG Grad-CAM overlay")
        ] = False,
    ) -> PredictResponse:
        predictor: Predictor = request.app.state.predictor
        _check_content_type(file)
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
        if not data:
            raise HTTPException(422, "The uploaded file is empty.")
        file_sha256 = hashlib.sha256(data).hexdigest()
        try:
            image = load_image(data, filename=file.filename)
        except ValueError as e:
            log.info("Rejected upload sha256=%s: %s", file_sha256[:12], e)
            raise HTTPException(422, str(e)) from e

        log.info("Running inference for upload sha256=%s", file_sha256[:12])
        prediction = predictor.predict(image)
        summary = output_summary(
            predictor,
            prediction,
            file_sha256=file_sha256,
            input_warnings=predictor.check_input(image),
        )
        gradcam = None
        if explain:
            rgb = overlay(predictor.model_view(image), predictor.explain(image), OVERLAY_ALPHA)
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            gradcam = base64.b64encode(buffer.getvalue()).decode("ascii")
        return PredictResponse(
            **summary,
            borderline_band=predictor.borderline_band(prediction.threshold),
            gradcam_png_base64=gradcam,
        )

    return app


def _check_content_type(upload: UploadFile) -> None:
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    name = (upload.filename or "").lower()
    if content_type in ACCEPTED_TYPES:
        return
    if content_type == "application/octet-stream" and name.endswith(DICOM_SUFFIXES):
        return
    raise HTTPException(
        415,
        f"Unsupported content type {content_type or 'none'!r}. Upload image/png, image/jpeg or "
        "application/dicom (application/octet-stream is accepted for .dcm files).",
    )


def _model_details(predictor: Predictor) -> ModelDetails:
    meta = predictor.meta
    return ModelDetails(
        name=meta["model_name"], git_commit=meta["git_commit"], created_at=meta["created_at"]
    )


async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse({"detail": "Internal error while processing the request."}, 500)


app = create_app(device=os.environ.get("LUNGXRAY_DEVICE"))
