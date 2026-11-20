"""Deterministic blink-period inference for an alert lamp.

Given several snapshots of the same lamp box, each tagged with a closed
sampling-time interval, decide per frame whether the lamp is lit, off or
unclear from the fraction of red pixels, then find every integer-millisecond
blink period (and phase) that can explain all clear frames simultaneously.

Blink model
-----------
For period ``T`` (even, 200..4000 ms) the lamp alternates lit/off, each lasting
``T/2`` ms. A single integer phase ``p`` (0 <= p < T) marks the start of a lit
half-cycle: at integer time ``t`` the lamp is lit iff

    (t - p) mod T < T / 2

Switch instants belong to the *new* state (at t = p + T/2 the lamp is already
off, at t = p + T it is lit again). Every frame may pick its own sample time
inside its closed interval, but one period must use the *same* phase for all
frames.

Constant (always-lit / always-off) hypotheses are checked as well and need no
phase: they fit when every clear frame has the matching state.

Arithmetic on the phase circle
------------------------------
For a clear frame with closed interval [a, b] (span = b-a) the feasible
phases form one contiguous integer arc of length ``min(T, span + T/2)``:

* LIT frame: start ``(a - T/2 + 1) mod T``  (phases for which some t in
  [a, b] lands in the lit half-cycle)
* OFF frame: start ``(a + 1) mod T``

Two arcs intersect iff ``(start2-start1) mod T < len1`` or vice-versa;
intersecting all frame arcs with numpy boolean masks gives the full feasible
phase set for a period in O(T + k*T) time with no sampling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image

from app.vision.regions import red_pixel_mask

LampState = Literal["lit", "off", "unclear"]
LAMP_LIT: LampState = "lit"
LAMP_OFF: LampState = "off"
LAMP_UNCLEAR: LampState = "unclear"

# Red-pixel fraction thresholds inside the lamp box.
LIT_MIN_FRACTION = 0.60
OFF_MAX_FRACTION = 0.10

MIN_PERIOD_MS = 200
MAX_PERIOD_MS = 4000
MIN_FRAMES = 3
MAX_FRAMES = 12
MAX_IMAGE_SIDE = 512


@dataclass
class LampFrame:
    """One snapshot plus its closed sampling interval [t_start, t_end] (ms)."""

    image: Image.Image
    t_start: int
    t_end: int
    red_fraction: float = 0.0
    state: LampState = LAMP_UNCLEAR


@dataclass
class FrameConstraint:
    """Integer circular arc of feasible phases for one clear frame."""

    index: int
    state: LampState
    t_start: int
    t_end: int
    arc_start: int  # phase-circle start, in [0, T)
    arc_length: int  # number of integer phases, in [1, T]


@dataclass
class PeriodSolution:
    """A feasible blink period with one concrete phase/sample-time witness."""

    period_ms: int
    lit_start_phase_ms: int
    sample_times_ms: list[int | None]
    feasible_phase_count: int


@dataclass
class ConstantHypothesis:
    kind: Literal["always_lit", "always_off"]
    feasible: bool
    reason: str
    sample_times_ms: list[int | None]


@dataclass
class BlinkAnalysis:
    frames: list[LampFrame]
    periods: list[PeriodSolution]
    always_lit: ConstantHypothesis
    always_off: ConstantHypothesis
    conclusion: str
    reasons: list[str] = field(default_factory=list)


def classify_lamp_frame(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[LampState, float]:
    """Classify one snapshot from red-pixel fraction inside the lamp box.

    ``box`` is (x, y, width, height). >=60% red -> lit, <=10% -> off,
    anything in between -> unclear.
    """
    x, y, w, h = box
    arr = np.asarray(image.convert("RGB"))
    crop = arr[y : y + h, x : x + w]
    red_fraction = float(red_pixel_mask(crop).mean()) if crop.size else 0.0
    if red_fraction >= LIT_MIN_FRACTION:
        return LAMP_LIT, red_fraction
    if red_fraction <= OFF_MAX_FRACTION:
        return LAMP_OFF, red_fraction
    return LAMP_UNCLEAR, red_fraction


def _build_constraints(frames: list[LampFrame], period: int) -> list[FrameConstraint]:
    """Per-frame feasible-phase arcs for one candidate period."""
    half = period // 2
    constraints: list[FrameConstraint] = []
    for i, frame in enumerate(frames):
        if frame.state == LAMP_UNCLEAR:
            continue
        # Every integer t in [a, b] is a possible sample; the union of their
        # lit (or off) half-cycle phase windows is one arc whose length grows
        # with the interval span and covers the whole circle once span >= half.
        length = min(period, frame.t_end - frame.t_start + half)
        if frame.state == LAMP_LIT:
            # lit at t requires (t-p) mod T in [0, half) -> p in (t-half, t]
            start = (frame.t_start - half + 1) % period
        else:
            # off at t requires (t-p) mod T in [half, T) -> p in (t-T, t-half]
            start = (frame.t_start + 1) % period
        constraints.append(
            FrameConstraint(
                index=i,
                state=frame.state,
                t_start=frame.t_start,
                t_end=frame.t_end,
                arc_start=start,
                arc_length=length,
            )
        )
    return constraints


def _feasible_phase_mask(period: int, constraints: list[FrameConstraint]) -> np.ndarray:
    """Intersect all frame arcs on the integer phase circle [0, period)."""
    phases = np.arange(period, dtype=np.int64)
    feasible = np.ones(period, dtype=bool)
    for c in constraints:
        d = (phases - c.arc_start) % period
        feasible &= d < c.arc_length
        if not feasible.any():
            break
    return feasible


def _pick_sample_time(frame: LampFrame, period: int, phase: int) -> int:
    """Return an integer time in the frame's interval matching the lamp state.

    Only called for clear frames of a period/phase already proven feasible,
    so a witness always exists within the first period of residues.
    """
    half = period // 2
    for r in range(half):
        t = frame.t_start + r
        if t > frame.t_end:
            break
        rel = (t - phase) % period
        lit = rel < half
        if (frame.state == LAMP_LIT) == lit:
            return t
    # Defensive fallback: scan the whole interval (keeps the witness honest).
    for t in range(frame.t_start, frame.t_end + 1):
        rel = (t - phase) % period
        lit = rel < half
        if (frame.state == LAMP_LIT) == lit:
            return t
    raise RuntimeError("feasible phase produced no sample time")  # pragma: no cover


def _constant_hypothesis(kind: Literal["always_lit", "always_off"], frames: list[LampFrame]) -> ConstantHypothesis:
    wanted = LAMP_LIT if kind == "always_lit" else LAMP_OFF
    clear = [f for f in frames if f.state != LAMP_UNCLEAR]
    contradict = [i for i, f in enumerate(frames) if f.state not in (wanted, LAMP_UNCLEAR)]
    sample_times = [
        (f.t_start if f.state != LAMP_UNCLEAR else None) for f in frames
    ]
    if not clear:
        return ConstantHypothesis(
            kind=kind,
            feasible=True,
            reason="no clear frames: nothing contradicts this hypothesis, but it cannot be confirmed either",
            sample_times_ms=sample_times,
        )
    if contradict:
        joined = ", ".join(f"frame {i + 1} ({frames[i].state})" for i in contradict)
        return ConstantHypothesis(
            kind=kind,
            feasible=False,
            reason=f"contradicted by {joined}",
            sample_times_ms=sample_times,
        )
    return ConstantHypothesis(
        kind=kind,
        feasible=True,
        reason=f"all {len(clear)} clear frame(s) are observed {wanted}; each can be sampled at its interval start",
        sample_times_ms=sample_times,
    )


def analyze_blink(
    images: list[Image.Image],
    box: tuple[int, int, int, int],
    windows: list[tuple[int, int]],
    period_min: int = MIN_PERIOD_MS,
    period_max: int = MAX_PERIOD_MS,
) -> BlinkAnalysis:
    """Classify every frame and enumerate every feasible blink period/phase.

    Candidate periods are the even integers in [period_min, period_max].
    """
    frames: list[LampFrame] = []
    for image, (t_start, t_end) in zip(images, windows):
        state, fraction = classify_lamp_frame(image, box)
        frames.append(LampFrame(image=image, t_start=t_start, t_end=t_end,
                                red_fraction=fraction, state=state))

    clear_frames = [f for f in frames if f.state != LAMP_UNCLEAR]
    always_lit = _constant_hypothesis("always_lit", frames)
    always_off = _constant_hypothesis("always_off", frames)

    solutions: list[PeriodSolution] = []
    if clear_frames:
        for period in range(period_min, period_max + 1, 2):
            constraints = _build_constraints(frames, period)
            mask = _feasible_phase_mask(period, constraints)
            if not mask.any():
                continue
            phase = int(np.flatnonzero(mask)[0])  # smallest feasible phase
            sample_times: list[int | None] = []
            for i, frame in enumerate(frames):
                if frame.state == LAMP_UNCLEAR:
                    sample_times.append(None)
                else:
                    sample_times.append(_pick_sample_time(frame, period, phase))
            solutions.append(
                PeriodSolution(
                    period_ms=period,
                    lit_start_phase_ms=phase,
                    sample_times_ms=sample_times,
                    feasible_phase_count=int(mask.sum()),
                )
            )

    reasons: list[str] = []
    n_unclear = sum(f.state == LAMP_UNCLEAR for f in frames)
    if n_unclear:
        reasons.append(
            f"{n_unclear} of {len(frames)} frame(s) are unclear "
            f"(red fraction between {OFF_MAX_FRACTION:.0%} and {LIT_MIN_FRACTION:.0%}); "
            "they do not constrain period or phase"
        )
    if not clear_frames:
        reasons.append("no clear frames: every period in range is vacuously compatible and none can be confirmed")
    elif not solutions:
        reasons.append(
            "no blink period in range can satisfy every clear frame with one shared phase; "
            "the lamp may not blink on a fixed 50/50 schedule, or the sampling windows miss the transitions"
        )

    feasible_kinds = []
    if always_lit.feasible:
        feasible_kinds.append("always_lit")
    if always_off.feasible:
        feasible_kinds.append("always_off")
    if solutions:
        feasible_kinds.append("blink")
    if len(feasible_kinds) > 1:
        reasons.append(
            "multiple hypotheses remain possible ("
            + ", ".join(feasible_kinds)
            + "); the snapshots cannot distinguish them"
        )
    if not feasible_kinds:
        conclusion = "no_hypothesis_fits"
    elif len(feasible_kinds) == 1:
        conclusion = feasible_kinds[0]
    else:
        conclusion = "ambiguous"

    return BlinkAnalysis(
        frames=frames,
        periods=solutions,
        always_lit=always_lit,
        always_off=always_off,
        conclusion=conclusion,
        reasons=reasons,
    )
