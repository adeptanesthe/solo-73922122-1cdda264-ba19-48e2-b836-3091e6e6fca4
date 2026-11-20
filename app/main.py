"""FastAPI application: multimodal troubleshooting evidence correlation."""
from __future__ import annotations

import base64
import binascii
import io
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.blink.engine import (
    MAX_FRAMES as BLINK_MAX_FRAMES,
    MAX_IMAGE_SIDE as BLINK_MAX_IMAGE_SIDE,
    MAX_PERIOD_MS,
    MIN_FRAMES as BLINK_MIN_FRAMES,
    MIN_PERIOD_MS,
    analyze_blink,
)
from app.config import get_settings
from app.correlate.engine import correlate
from app.loganalysis.analyzer import analyze_logs
from app.schemas import (
    AlertBlinkConstantHypothesis,
    AlertBlinkFrame,
    AlertBlinkPeriodSolution,
    AlertBlinkResponse,
    BackendInfo,
    BoundingBox,
    DiagnoseJsonRequest,
    DiagnoseResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.vision.regions import detect_regions
from app.vision.vlm import HeuristicVisionBackend, build_backend


def _backend_info(backend) -> BackendInfo:
    return BackendInfo(
        vision_backend=backend.name,
        vision_model=getattr(backend, "model_id", None),
        vision_model_loaded=not isinstance(backend, HeuristicVisionBackend),
        fallback_reason=getattr(backend, "_reason", None),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.vision_backend = build_backend(settings)
    yield


app = FastAPI(
    title="Multimodal Troubleshooting Evidence Correlation API",
    description=(
        "Accepts a fault screenshot, a troubleshooting question and related log "
        "snippets; locates key screenshot regions and log anomalies with a "
        "locally loaded vision-language model (heuristic fallback), then emits "
        "numbered evidence, ranked hypotheses, support/contradiction relations "
        "and next-step suggestions. Returns `status=uncertain` with reasons "
        "when no reliable judgment can be formed."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz() -> HealthResponse:
    return HealthResponse()


@app.get("/v1/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info(request: Request) -> ModelInfoResponse:
    settings = request.app.state.settings
    return ModelInfoResponse(
        backend=_backend_info(request.app.state.vision_backend),
        settings={
            "vlm_backend": settings.vlm_backend,
            "vlm_model_id": settings.vlm_model_id,
            "vlm_local_files_only": settings.vlm_local_files_only,
            "confidence_threshold": settings.confidence_threshold,
            "max_regions": settings.max_regions,
        },
    )


def _decode_image(data: bytes, settings) -> Image.Image:
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, f"screenshot too large (>{settings.max_image_bytes} bytes)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(422, f"cannot decode screenshot as an image: {exc}") from exc
    return img.convert("RGB")


def _run_diagnosis(request: Request, image: Image.Image, question: str, logs: str) -> DiagnoseResponse:
    settings = request.app.state.settings
    backend = request.app.state.vision_backend
    if len(logs) > settings.max_log_chars:
        logs = logs[: settings.max_log_chars]
    regions = detect_regions(image, max_regions=settings.max_regions)
    vision = backend.analyze(image, question, regions)
    log_analysis = analyze_logs(logs)
    return correlate(
        question=question,
        regions=regions,
        vision=vision,
        logs=log_analysis,
        settings=settings,
        backend=_backend_info(backend),
    )


@app.post("/v1/diagnose", response_model=DiagnoseResponse, tags=["diagnosis"])
async def diagnose(
    request: Request,
    screenshot: UploadFile = File(..., description="fault screenshot (PNG/JPEG)"),
    question: str = Form(..., description="troubleshooting question"),
    logs: str = Form("", description="related log snippet (plain text)"),
) -> DiagnoseResponse:
    if not question.strip():
        raise HTTPException(422, "question must not be empty")
    data = await screenshot.read()
    image = _decode_image(data, request.app.state.settings)
    # CPU-bound work (CV + VLM inference) must not block the event loop
    return await run_in_threadpool(_run_diagnosis, request, image, question.strip(), logs)


@app.post("/v1/diagnose/json", response_model=DiagnoseResponse, tags=["diagnosis"])
def diagnose_json(request: Request, body: DiagnoseJsonRequest) -> DiagnoseResponse:
    if not body.question.strip():
        raise HTTPException(422, "question must not be empty")
    try:
        data = base64.b64decode(body.screenshot_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, f"screenshot_base64 is not valid base64: {exc}") from exc
    image = _decode_image(data, request.app.state.settings)
    return _run_diagnosis(request, image, body.question.strip(), body.logs)


def _decode_png(data: bytes, settings) -> Image.Image:
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, f"frame too large (>{settings.max_image_bytes} bytes)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(422, f"cannot decode frame as an image: {exc}") from exc
    if img.format != "PNG":
        raise HTTPException(422, f"all frames must be PNG, got {img.format}")
    if img.width > BLINK_MAX_IMAGE_SIDE or img.height > BLINK_MAX_IMAGE_SIDE:
        raise HTTPException(
            422,
            f"frame side must be <= {BLINK_MAX_IMAGE_SIDE}px, got {img.width}x{img.height}",
        )
    return img.convert("RGB")


def _parse_lamp_box(raw: str) -> tuple[int, int, int, int]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"lamp_box must be JSON [x, y, width, height]: {exc}") from exc
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(v, int) for v in value):
        raise HTTPException(422, "lamp_box must be four integers [x, y, width, height]")
    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise HTTPException(422, "lamp_box x/y must be >= 0 and width/height must be positive")
    return x, y, width, height


def _parse_windows(raw: str, frame_count: int) -> list[tuple[int, int]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"sample_windows_ms must be JSON: {exc}") from exc
    if not isinstance(value, list) or len(value) != frame_count:
        raise HTTPException(
            422,
            f"sample_windows_ms must contain one [t_start, t_end] pair per frame "
            f"({frame_count} pairs expected, got {len(value) if isinstance(value, list) else 'non-list'})",
        )
    windows: list[tuple[int, int]] = []
    for i, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, int) for v in pair):
            raise HTTPException(422, f"sample_windows_ms[{i}] must be two integers [t_start, t_end]")
        t_start, t_end = pair
        if t_start < 0 or t_end < t_start:
            raise HTTPException(422, f"sample_windows_ms[{i}] must satisfy 0 <= t_start <= t_end")
        windows.append((t_start, t_end))
    return windows


def _run_alert_blink(
    images: list[Image.Image],
    box: tuple[int, int, int, int],
    windows: list[tuple[int, int]],
    period_min: int,
    period_max: int,
) -> AlertBlinkResponse:
    width, height = images[0].size
    x, y, box_w, box_h = box
    if x + box_w > width or y + box_h > height:
        raise HTTPException(
            422,
            f"lamp_box ({x},{y},{box_w},{box_h}) extends outside the {width}x{height} frames",
        )
    result = analyze_blink(images, box, windows, period_min, period_max)
    return AlertBlinkResponse(
        conclusion=result.conclusion,
        reasons=result.reasons,
        lamp_box=BoundingBox(x=x, y=y, width=box_w, height=box_h),
        period_range_ms=(period_min, period_max),
        frames=[
            AlertBlinkFrame(
                index=i + 1,
                state=frame.state,
                red_fraction=round(frame.red_fraction, 6),
                sample_window_ms=(frame.t_start, frame.t_end),
            )
            for i, frame in enumerate(result.frames)
        ],
        feasible_periods_ms=[s.period_ms for s in result.periods],
        period_solutions=[
            AlertBlinkPeriodSolution(
                period_ms=s.period_ms,
                lit_start_phase_ms=s.lit_start_phase_ms,
                feasible_phase_count=s.feasible_phase_count,
                sample_times_ms=s.sample_times_ms,
            )
            for s in result.periods
        ],
        always_lit=AlertBlinkConstantHypothesis(
            feasible=result.always_lit.feasible,
            reason=result.always_lit.reason,
            sample_times_ms=result.always_lit.sample_times_ms,
        ),
        always_off=AlertBlinkConstantHypothesis(
            feasible=result.always_off.feasible,
            reason=result.always_off.reason,
            sample_times_ms=result.always_off.sample_times_ms,
        ),
    )


@app.post("/v1/alert-blink", response_model=AlertBlinkResponse, tags=["diagnosis"])
async def alert_blink(
    request: Request,
    frames: list[UploadFile] = File(
        ..., description=f"{BLINK_MIN_FRAMES}-{BLINK_MAX_FRAMES} same-size PNG snapshots of the same lamp box"
    ),
    lamp_box: str = Form(..., description="JSON [x, y, width, height] of the lamp box in pixel coordinates"),
    sample_windows_ms: str = Form(
        ..., description="JSON list of closed sampling intervals [[t_start, t_end], ...], one per frame, integer ms"
    ),
    period_min_ms: int = Form(MIN_PERIOD_MS, description="smallest candidate period, even, >= 200"),
    period_max_ms: int = Form(MAX_PERIOD_MS, description="largest candidate period, even, <= 4000"),
) -> AlertBlinkResponse:
    """Infer whether an alert lamp is steady (always lit/off) or blinking.

    Red-pixel rule inside ``lamp_box``: >= 60% red = lit, <= 10% red = off,
    otherwise unclear. For every even period in the requested range the
    endpoint checks whether a single integer-ms phase can explain all clear
    frames, each sampled anywhere inside its own closed interval.
    """
    if not BLINK_MIN_FRAMES <= len(frames) <= BLINK_MAX_FRAMES:
        raise HTTPException(
            422,
            f"expected between {BLINK_MIN_FRAMES} and {BLINK_MAX_FRAMES} frames, got {len(frames)}",
        )
    if not (MIN_PERIOD_MS <= period_min_ms <= period_max_ms <= MAX_PERIOD_MS):
        raise HTTPException(
            422,
            f"require {MIN_PERIOD_MS} <= period_min_ms <= period_max_ms <= {MAX_PERIOD_MS}, "
            f"got {period_min_ms}..{period_max_ms}",
        )
    if period_min_ms % 2 or period_max_ms % 2:
        raise HTTPException(422, "period_min_ms and period_max_ms must be even")

    box = _parse_lamp_box(lamp_box)
    settings = request.app.state.settings
    images: list[Image.Image] = []
    for upload in frames:
        images.append(_decode_png(await upload.read(), settings))
    sizes = {img.size for img in images}
    if len(sizes) != 1:
        raise HTTPException(422, f"all frames must have identical dimensions, got {sorted(sizes)}")
    windows = _parse_windows(sample_windows_ms, len(images))
    return await run_in_threadpool(_run_alert_blink, images, box, windows, period_min_ms, period_max_ms)
