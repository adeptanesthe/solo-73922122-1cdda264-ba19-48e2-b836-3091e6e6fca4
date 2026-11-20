"""Tests for /v1/alert-blink: lamp classification and blink-period inference."""
from __future__ import annotations

import io
import json
import random

import pytest
from PIL import Image

from app.blink.engine import (
    LAMP_LIT,
    LAMP_OFF,
    LAMP_UNCLEAR,
    analyze_blink,
    classify_lamp_frame,
)

BOX = (10, 10, 20, 20)  # 400 px box
W, H = 64, 64
RED = (200, 30, 30)
DARK = (10, 10, 12)


def _png(state: str, red_fraction: float | None = None) -> bytes:
    """Build a PNG whose lamp box has a controlled red-pixel fraction."""
    img = Image.new("RGB", (W, H), (40, 40, 44))
    px = img.load()
    if red_fraction is None:
        red_fraction = {"lit": 1.0, "off": 0.0, "unclear": 0.35}[state]
    total = BOX[2] * BOX[3]
    red_count = round(total * red_fraction)
    k = 0
    for y in range(BOX[1], BOX[1] + BOX[3]):
        for x in range(BOX[0], BOX[0] + BOX[2]):
            px[x, y] = RED if k < red_count else DARK
            k += 1
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _img(state: str, red_fraction: float | None = None) -> Image.Image:
    return Image.open(io.BytesIO(_png(state, red_fraction))).convert("RGB")


def _lamp_state_at(t: int, period: int, phase: int) -> str:
    return LAMP_LIT if (t - phase) % period < period // 2 else LAMP_OFF


def _point_blink_frames(period: int, phase: int, times: list[int]) -> list:
    return [
        (_img(_lamp_state_at(t, period, phase)), (t, t))
        for t in times
    ]


# ---------------------------------------------------------------------------
# classification thresholds
# ---------------------------------------------------------------------------


def test_classify_thresholds():
    lit, frac = classify_lamp_frame(_img("lit", 0.60), BOX)
    assert lit == LAMP_LIT and frac == pytest.approx(0.60)
    off, frac = classify_lamp_frame(_img("off", 0.10), BOX)
    assert off == LAMP_OFF and frac == pytest.approx(0.10)
    unclear, frac = classify_lamp_frame(_img("unclear", 0.59), BOX)
    assert unclear == LAMP_UNCLEAR
    assert classify_lamp_frame(_img("unclear", 0.11), BOX)[0] == LAMP_UNCLEAR


# ---------------------------------------------------------------------------
# point-window inference on a known blinking ground truth
# ---------------------------------------------------------------------------


def test_blink_ground_truth_period_is_feasible_and_steady_hypotheses_fail():
    T, phase = 400, 50
    times = [0, 100, 300, 500, 900, 1300]
    frames = _point_blink_frames(T, phase, times)
    images = [im for im, _ in frames]
    windows = [w for _, w in frames]
    result = analyze_blink(images, BOX, windows, 200, 1200)

    assert [f.state for f in result.frames] == [
        _lamp_state_at(t, T, phase) for t in times
    ]
    feasible = [s.period_ms for s in result.periods]
    assert T in feasible
    assert result.always_lit.feasible is False
    assert result.always_off.feasible is False
    assert result.conclusion == "blink"

    # every returned solution is self-consistent with its witness sample times
    for sol in result.periods:
        assert 1 <= sol.feasible_phase_count <= sol.period_ms
        for i, (frame, t) in enumerate(zip(result.frames, sol.sample_times_ms)):
            assert t is not None
            assert windows[i][0] <= t <= windows[i][1]
            assert _lamp_state_at(t, sol.period_ms, sol.lit_start_phase_ms) == frame.state


def test_switch_instant_counts_as_new_state():
    # phase=0, T=400: t=200 is the first OFF instant; t=400 is lit again
    assert _lamp_state_at(200, 400, 0) == LAMP_OFF
    assert _lamp_state_at(400, 400, 0) == LAMP_LIT
    frames = _point_blink_frames(400, 0, [0, 200, 400, 600])
    images = [im for im, _ in frames]
    windows = [w for _, w in frames]
    result = analyze_blink(images, BOX, windows, 200, 800)
    sol400 = next(s for s in result.periods if s.period_ms == 400)
    # phase 0 is the canonical explanation and fits the boundary frames
    assert sol400.lit_start_phase_ms == 0
    assert sol400.sample_times_ms == [0, 200, 400, 600]


def test_impossible_point_windows_fit_no_period():
    # lit at t=0, off at t=3: no even period >=200 can switch that fast
    images = [_img("lit"), _img("off"), _img("lit")]
    windows = [(0, 0), (3, 3), (8, 8)]
    result = analyze_blink(images, BOX, windows, 200, 4000)
    assert result.periods == []
    assert result.always_lit.feasible is False
    assert result.always_off.feasible is False
    assert result.conclusion == "no_hypothesis_fits"
    assert result.reasons


def test_always_lit_frames_support_steady_and_slow_blinks():
    images = [_img("lit") for _ in range(4)]
    windows = [(0, 0), (500, 500), (1000, 1000), (1500, 1500)]
    result = analyze_blink(images, BOX, windows, 200, 4000)
    assert result.always_lit.feasible is True
    assert result.always_off.feasible is False
    # Points at 0,500,1000,1500 all lit. One lit half-cycle (T/2) must cover
    # the whole 1500 ms sampling span: T=4000 can, T=2000 and T=200 cannot.
    feasible = [s.period_ms for s in result.periods]
    assert 4000 in feasible
    assert 2000 not in feasible
    assert 200 not in feasible
    assert result.conclusion == "ambiguous"
    assert any("multiple hypotheses" in r for r in result.reasons)
    # witness sample times stay inside each window and reproduce the states
    for sol in result.periods:
        for frame, t in zip(result.frames, sol.sample_times_ms):
            assert t is not None
            assert _lamp_state_at(t, sol.period_ms, sol.lit_start_phase_ms) == frame.state


def test_always_off_only():
    images = [_img("off") for _ in range(3)]
    windows = [(0, 0), (100, 100), (250, 250)]
    result = analyze_blink(images, BOX, windows, 200, 4000)
    assert result.always_off.feasible is True
    assert result.always_lit.feasible is False
    assert result.always_off.sample_times_ms == [0, 100, 250]


def test_unclear_frames_are_ignored_by_period_search():
    images = [_img("lit"), _img("unclear", 0.3), _img("off")]
    windows = [(0, 0), (10, 10), (700, 700)]
    result = analyze_blink(images, BOX, windows, 200, 4000)
    assert [f.state for f in result.frames] == [LAMP_LIT, LAMP_UNCLEAR, LAMP_OFF]
    assert any("unclear" in r for r in result.reasons)
    for sol in result.periods:
        assert sol.sample_times_ms[1] is None
        assert sol.sample_times_ms[0] is not None and sol.sample_times_ms[2] is not None


def test_all_unclear_frames_leave_everything_open():
    images = [_img("unclear", 0.3) for _ in range(3)]
    windows = [(0, 0), (10, 10), (20, 20)]
    result = analyze_blink(images, BOX, windows, 200, 400)
    assert result.periods == []
    assert result.always_lit.feasible is True
    assert result.always_off.feasible is True
    assert result.conclusion == "ambiguous"


# ---------------------------------------------------------------------------
# brute-force cross-check of the arc-intersection engine
# ---------------------------------------------------------------------------


def test_engine_matches_brute_force_on_random_cases():
    rng = random.Random(20261120)
    states_pool = [LAMP_LIT, LAMP_OFF, LAMP_UNCLEAR]
    for _ in range(40):
        n = rng.randint(3, 8)
        frames, windows = [], []
        for _ in range(n):
            state = rng.choice(states_pool)
            frames.append(_img(state))
            a = rng.randint(0, 5000)
            windows.append((a, a + rng.randint(0, 900)))
        period = rng.choice([200, 250, 400, 777 * 2, 1234, 4000])
        result = analyze_blink(frames, BOX, windows, period, period)

        expected_phases = []
        for p in range(period):
            ok = True
            for frame, (a, b) in zip(result.frames, windows):
                if frame.state == LAMP_UNCLEAR:
                    continue
                if not any(
                    _lamp_state_at(t, period, p) == frame.state
                    for t in range(a, b + 1)
                ):
                    ok = False
                    break
            if ok:
                expected_phases.append(p)

        if not expected_phases:
            assert result.periods == []
            continue
        sol = result.periods[0]
        assert sol.period_ms == period
        assert sol.feasible_phase_count == len(expected_phases)
        assert sol.lit_start_phase_ms == expected_phases[0]
        for frame, t in zip(result.frames, sol.sample_times_ms):
            if frame.state == LAMP_UNCLEAR:
                assert t is None
            else:
                assert frame.t_start <= t <= frame.t_end
                assert _lamp_state_at(t, period, sol.lit_start_phase_ms) == frame.state


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def _upload(client, states, windows, box=BOX, period_min=200, period_max=1200, extra=None):
    files = [("frames", (f"f{i}.png", _png(s), "image/png")) for i, s in enumerate(states)]
    data = {
        "lamp_box": json.dumps(list(box)),
        "sample_windows_ms": json.dumps([list(w) for w in windows]),
        "period_min_ms": str(period_min),
        "period_max_ms": str(period_max),
    }
    if extra:
        data.update(extra)
    return client.post("/v1/alert-blink", files=files, data=data)


def test_alert_blink_api_end_to_end(client):
    T, phase = 400, 50
    times = [0, 100, 300, 500, 900]
    states = [_lamp_state_at(t, T, phase) for t in times]
    r = _upload(client, states, [(t, t) for t in times])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["conclusion"] == "blink"
    assert len(body["frames"]) == 5
    assert body["frames"][0]["state"] in ("lit", "off")
    assert body["lamp_box"] == {"x": 10, "y": 10, "width": 20, "height": 20}
    assert body["period_range_ms"] == [200, 1200]
    assert 400 in body["feasible_periods_ms"]
    sol = next(s for s in body["period_solutions"] if s["period_ms"] == 400)
    assert len(sol["sample_times_ms"]) == 5
    assert body["always_lit"]["feasible"] is False
    assert body["always_off"]["feasible"] is False


def test_alert_blink_rejects_wrong_frame_count(client):
    r = _upload(client, ["lit", "off"], [(0, 0), (1, 1)])
    assert r.status_code == 422
    assert "3 and 12" in r.text


def test_alert_blink_rejects_size_mismatch(client):
    files = [
        ("frames", ("f0.png", _png("lit"), "image/png")),
        ("frames", ("f1.png", _png("off"), "image/png")),
    ]
    small = io.BytesIO()
    Image.new("RGB", (32, 32), (0, 0, 0)).save(small, format="PNG")
    files.append(("frames", ("f2.png", small.getvalue(), "image/png")))
    data = {
        "lamp_box": json.dumps(list(BOX)),
        "sample_windows_ms": json.dumps([[0, 0], [1, 1], [2, 2]]),
    }
    r = client.post("/v1/alert-blink", files=files, data=data)
    assert r.status_code == 422
    assert "identical dimensions" in r.text


def test_alert_blink_rejects_non_png(client):
    buf = io.BytesIO()
    Image.new("RGB", (W, H)).save(buf, format="JPEG")
    files = [("frames", (f"f{i}.jpg", buf.getvalue() if i == 0 else _png(s), "image/jpeg"))
             for i, s in enumerate(["lit", "off", "lit"])]
    data = {
        "lamp_box": json.dumps(list(BOX)),
        "sample_windows_ms": json.dumps([[0, 0], [1, 1], [2, 2]]),
    }
    r = client.post("/v1/alert-blink", files=files, data=data)
    assert r.status_code == 422
    assert "PNG" in r.text


def test_alert_blink_rejects_oversized_image(client):
    big = io.BytesIO()
    Image.new("RGB", (513, 256)).save(big, format="PNG")
    files = [("frames", (f"f{i}.png", big.getvalue() if i == 0 else _png(s), "image/png"))
             for i, s in enumerate(["lit", "off", "lit"])]
    data = {
        "lamp_box": json.dumps(list(BOX)),
        "sample_windows_ms": json.dumps([[0, 0], [1, 1], [2, 2]]),
    }
    r = client.post("/v1/alert-blink", files=files, data=data)
    assert r.status_code == 422
    assert "512" in r.text


def test_alert_blink_rejects_box_outside_frame(client):
    r = _upload(client, ["lit"] * 3, [(0, 0), (1, 1), (2, 2)], box=(50, 50, 30, 30))
    assert r.status_code == 422
    assert "outside" in r.text


def test_alert_blink_rejects_bad_window_pair(client):
    r = _upload(client, ["lit"] * 3, [(0, 0), (5, 1), (2, 2)])
    assert r.status_code == 422
    assert "t_start <= t_end" in r.text


def test_alert_blink_rejects_odd_period_bounds(client):
    r = _upload(client, ["lit"] * 3, [(0, 0), (1, 1), (2, 2)], period_min=201, period_max=400)
    assert r.status_code == 422
    assert "even" in r.text


def test_alert_blink_rejects_period_out_of_range(client):
    r = _upload(client, ["lit"] * 3, [(0, 0), (1, 1), (2, 2)], period_min=100, period_max=400)
    assert r.status_code == 422
    r = _upload(client, ["lit"] * 3, [(0, 0), (1, 1), (2, 2)], period_min=200, period_max=5000)
    assert r.status_code == 422
