"""Pydantic schemas for the diagnosis API."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DiagnosisStatus(str, Enum):
    OK = "ok"
    UNCERTAIN = "uncertain"


class EvidenceSource(str, Enum):
    SCREENSHOT = "screenshot"
    LOG = "log"


class BoundingBox(BaseModel):
    x: int = Field(..., description="left, pixels")
    y: int = Field(..., description="top, pixels")
    width: int
    height: int


class Evidence(BaseModel):
    id: str = Field(..., description="evidence id, e.g. E1")
    source: EvidenceSource
    kind: str = Field(..., description="e.g. red_alert_region, log_anomaly, vlm_observation")
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    region: BoundingBox | None = None
    log_lines: list[int] = Field(default_factory=list, description="1-based line numbers")
    matched_keywords: list[str] = Field(default_factory=list)


class RelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"  # evidence gap: needed to confirm, currently missing


class EvidenceRelation(BaseModel):
    evidence_id: str
    hypothesis_id: str
    relation: RelationType
    rationale: str


class Hypothesis(BaseModel):
    id: str = Field(..., description="hypothesis id, e.g. H1")
    statement: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(
        default_factory=list, description="what evidence would be needed to confirm"
    )


class BackendInfo(BaseModel):
    vision_backend: str
    vision_model: str | None = None
    vision_model_loaded: bool = False
    fallback_reason: str | None = None


class DiagnoseResponse(BaseModel):
    status: DiagnosisStatus
    uncertainty_reasons: list[str] = Field(default_factory=list)
    question: str
    evidence: list[Evidence]
    hypotheses: list[Hypothesis]
    relations: list[EvidenceRelation]
    next_steps: list[str]
    backend: BackendInfo


class DiagnoseJsonRequest(BaseModel):
    question: str = Field(..., min_length=1)
    logs: str = ""
    screenshot_base64: str = Field(..., description="base64-encoded PNG/JPEG image")
    filename: str | None = None


class ModelInfoResponse(BaseModel):
    backend: BackendInfo
    settings: dict[str, str | int | float | bool]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


# ---------------------------------------------------------------------------
# /v1/alert-blink
# ---------------------------------------------------------------------------


class AlertBlinkFrame(BaseModel):
    index: int = Field(..., description="1-based frame index, matching upload order")
    state: Literal["lit", "off", "unclear"]
    red_fraction: float = Field(..., ge=0.0, le=1.0, description="red-pixel fraction inside the lamp box")
    sample_window_ms: tuple[int, int] = Field(..., description="closed sampling interval [t_start, t_end]")


class AlertBlinkPeriodSolution(BaseModel):
    period_ms: int
    lit_start_phase_ms: int = Field(
        ..., description="example phase p: lit half-cycles start at p, p+T, p+2T, ..."
    )
    feasible_phase_count: int = Field(..., description="number of integer phases that work for this period")
    sample_times_ms: list[int | None] = Field(
        ..., description="one valid sample time per frame (null for unclear frames)"
    )


class AlertBlinkConstantHypothesis(BaseModel):
    feasible: bool
    reason: str
    sample_times_ms: list[int | None]


class AlertBlinkResponse(BaseModel):
    conclusion: Literal["always_lit", "always_off", "blink", "ambiguous", "no_hypothesis_fits"]
    reasons: list[str]
    lamp_box: BoundingBox
    period_range_ms: tuple[int, int]
    frames: list[AlertBlinkFrame]
    feasible_periods_ms: list[int]
    period_solutions: list[AlertBlinkPeriodSolution]
    always_lit: AlertBlinkConstantHypothesis
    always_off: AlertBlinkConstantHypothesis
