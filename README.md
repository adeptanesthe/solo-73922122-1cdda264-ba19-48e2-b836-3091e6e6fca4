# Multimodal Troubleshooting Evidence Correlation API

A locally runnable FastAPI service that correlates **fault screenshots**, a
**troubleshooting question** and **log snippets** into numbered evidence,
ranked fault hypotheses, support/contradiction relations and next-step
suggestions. When the evidence is too weak or contradictory, it answers with
`status=uncertain` and explicit reasons instead of guessing.

## How it works

```
screenshot ──┬─ CV region detection (red alerts / amber warnings / dialog blocks)
             └─ local VLM (SmolVLM-256M) ── full-image observation + region captions
logs ────────┴─ rule-based anomaly extraction (severity + category, line numbers)
                              │
                    correlation engine
                              │
   evidence E1..En ─ hypotheses H1..Hn ─ relations (supports/contradicts/insufficient)
                              │
              status ok|uncertain + reasons + next steps
```

* **Vision-language model** — `HuggingFaceTB/SmolVLM-256M-Instruct` loaded
  locally via transformers (CPU-friendly, ~500 MB weights). If the model cannot
  be loaded (no weights, no network), the service degrades to a deterministic
  heuristic backend and **says so in every response** (`backend` field).
* **Region localization** — classical CV (numpy) finds error-colored and
  high-contrast regions; the VLM describes the full screenshot and the top
  regions. Bounding boxes are returned in pixel coordinates.
* **Log analysis** — regex rules map lines to severity + fault category
  (network / database / auth / resource / dependency / application / config)
  and detect recovery signals used for contradiction checks.
* **Correlation** — deterministic scoring per fault category; every hypothesis
  lists supporting/contradicting/missing evidence, every relation carries a
  rationale. No hidden state, fully explainable output.

## Quick start (Docker Compose)

```bash
docker compose up --build
# open http://localhost:8000/docs
```

By default the image is built **without model weights** and the service runs
the deterministic heuristic/CV backend, so `docker compose up --build` works
fully offline and every endpoint — including `/v1/alert-blink` — is
demonstrable from `/docs` with no download. To use the locally loaded
SmolVLM model instead, set `PRELOAD_MODEL: "true"` in `docker-compose.yml`
and `VLM_BACKEND: auto` (the build then pre-downloads the weights into the
image; if huggingface.co is unreachable, set the `HF_ENDPOINT` build arg to
a mirror). With `PRELOAD_MODEL=false` and `VLM_BACKEND=auto`, weights are
downloaded on first start into the `models` volume.

## Quick start (local, no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt

# generate a demo screenshot + logs
python scripts/make_sample.py --out samples

# generate alert-lamp blink/steady demo snapshots (no model needed)
python scripts/make_blink_sample.py --out samples/blink_demo

# run (heuristic backend — no model download needed)
VLM_BACKEND=heuristic uvicorn app.main:app --port 8000

# or run with the real local VLM (downloads ~500 MB once into ./.cache/models)
MODEL_CACHE_DIR=.cache/models VLM_BACKEND=auto uvicorn app.main:app --port 8000
```

> CPU note: on a 2-core container, one diagnosis with the VLM takes roughly
> 1–3 minutes (one full-image pass + up to `VLM_REGION_CAPTION_LIMIT` region
> captions). The heuristic backend answers in milliseconds. The app
> auto-detects the cgroup CPU quota and matches torch threads to it.

## API

| Method | Path             | Purpose                                             |
|--------|------------------|-----------------------------------------------------|
| GET    | `/healthz`       | liveness                                            |
| GET    | `/v1/model/info` | active vision backend + settings                    |
| POST   | `/v1/diagnose`   | multipart: `screenshot` (file), `question`, `logs`  |
| POST   | `/v1/diagnose/json` | JSON: `screenshot_base64`, `question`, `logs`    |
| POST   | `/v1/alert-blink` | multipart: 3–12 lamp snapshots + sampling windows  |
| GET    | `/docs`          | Swagger UI (try the endpoints interactively)        |

### `/v1/alert-blink` — is the alert lamp steady or blinking?

Several snapshots of the same lamp box show it lit. This endpoint tells
whether the lamp is always on, always off, or blinking on a fixed period —
and, critically, returns **every** period compatible with the evidence so a
handful of lucky lit-only shots cannot be mistaken for steady light.

**Inputs** (multipart form):

* `frames` — 3 to 12 same-size PNG files, each side at most 512 px, showing
  the same lamp box;
* `lamp_box` — JSON `[x, y, width, height]` of the lamp box in pixels;
* `sample_windows_ms` — JSON list of closed sampling intervals
  `[[t_start, t_end], ...]`, one pair per frame, integer milliseconds
  (each frame may have been exposed at any time in its own interval);
* `period_min_ms`, `period_max_ms` — candidate period range; both even,
  within 200–4000 ms.

**Per-frame rule** (red pixels inside the box, same redness test as the CV
region detector): red fraction `>= 0.60` → `lit`, `<= 0.10` → `off`,
anything in between → `unclear`. Unclear frames are reported but do not
constrain the search.

**Blink model**: for period `T` the lamp is lit for `T/2` ms and off for
`T/2` ms; a switch instant belongs to the new state. One period is feasible
only if a **single integer-ms phase** explains every clear frame at once,
each frame picking its own sample time inside its interval. All even
periods in the requested range are checked exhaustively via integer
arc-intersection on the phase circle (no sampling, no approximation).

```bash
python scripts/make_blink_sample.py --out samples/blink_demo
curl -s http://localhost:8000/v1/alert-blink \
  -F "frames=@samples/blink_demo/blink/frame1.png" \
  -F "frames=@samples/blink_demo/blink/frame2.png" \
  -F "frames=@samples/blink_demo/blink/frame3.png" \
  -F "frames=@samples/blink_demo/blink/frame4.png" \
  -F "frames=@samples/blink_demo/blink/frame5.png" \
  -F "frames=@samples/blink_demo/blink/frame6.png" \
  -F 'lamp_box=[44,44,40,40]' \
  -F 'sample_windows_ms=[[0,0],[300,300],[600,600],[450,550],[1600,1600],[2100,2100]]' \
  -F 'period_min_ms=200' -F 'period_max_ms=2000' | jq
```

Response (abridged):

```json
{
  "conclusion": "blink",
  "reasons": [],
  "lamp_box": {"x": 44, "y": 44, "width": 40, "height": 40},
  "period_range_ms": [200, 2000],
  "frames": [
    {"index": 1, "state": "lit", "red_fraction": 0.78, "sample_window_ms": [0, 0]}
  ],
  "feasible_periods_ms": [902, 904, "...", 1000, "...", 1098],
  "period_solutions": [
    {"period_ms": 1000, "lit_start_phase_ms": 0, "feasible_phase_count": 50,
     "sample_times_ms": [0, 300, 600, 450, 1600, 2100]}
  ],
  "always_lit": {"feasible": false, "reason": "contradicted by frame 3 (off), ..."},
  "always_off": {"feasible": false, "reason": "..."}
}
```

* `period_solutions[].lit_start_phase_ms` is one example phase (the smallest
  feasible one); `feasible_phase_count` says how many integer phases work;
  `sample_times_ms` gives a concrete valid sample time per clear frame
  (`null` for unclear ones). Always-lit and always-off are checked as
  separate hypotheses. `conclusion` is one of `always_lit`, `always_off`,
  `blink`, `ambiguous` (several hypotheses survive) or
  `no_hypothesis_fits`, with human-readable `reasons` whenever the evidence
  is insufficient or contradictory.

### Example

```bash
curl -s http://localhost:8000/v1/diagnose \
  -F "screenshot=@samples/fault_screenshot.png" \
  -F "question=Checkout page shows 502 and payments time out. Likely cause?" \
  -F "logs=$(cat samples/app_logs.txt)" | jq
```

Response (abridged):

```json
{
  "status": "ok",
  "uncertainty_reasons": [],
  "evidence": [
    {"id": "E1", "source": "screenshot", "kind": "red_alert",
     "region": {"x": 0, "y": 60, "width": 960, "height": 60}, "confidence": 0.98, ...},
    {"id": "E6", "source": "log", "kind": "log_anomaly",
     "log_lines": [3], "confidence": 0.8, ...}
  ],
  "hypotheses": [
    {"id": "H1", "category": "dependency", "confidence": 0.8,
     "supporting_evidence": ["E1", "E2", "E6", ...],
     "contradicting_evidence": ["E10"], "missing_evidence": []}
  ],
  "relations": [
    {"evidence_id": "E6", "hypothesis_id": "H1", "relation": "supports", "rationale": "..."},
    {"evidence_id": "E10", "hypothesis_id": "H1", "relation": "contradicts", "rationale": "..."}
  ],
  "next_steps": ["Check health/status pages of the upstream dependency ...", ...],
  "backend": {"vision_backend": "smolvlm", "vision_model_loaded": true, ...}
}
```

When the evidence is insufficient, the response becomes:

```json
{"status": "uncertain",
 "uncertainty_reasons": ["Top hypothesis confidence 0.31 is below the reliability threshold 0.45.", ...],
 "hypotheses": [...], "next_steps": [...]}
```

## Configuration (environment variables)

| Variable                   | Default                                  | Meaning                                   |
|----------------------------|------------------------------------------|-------------------------------------------|
| `VLM_BACKEND`              | `auto`                                   | `auto` / `smolvlm` / `heuristic`          |
| `VLM_MODEL_ID`             | `HuggingFaceTB/SmolVLM-256M-Instruct`    | any transformers image-text-to-text model |
| `MODEL_CACHE_DIR`          | `/models` (image) / `.cache/models` (default) | local weights cache                  |
| `VLM_LOCAL_FILES_ONLY`     | `true` (image) / `false` (defaults)      | forbid runtime downloads                  |
| `VLM_MAX_NEW_TOKENS`       | `160`                                    | generation budget per VLM call            |
| `VLM_REGION_CAPTION_LIMIT` | `2`                                      | region crops captioned per request        |
| `VLM_MAX_IMAGE_SIDE`       | `512`                                    | downscale bound for VLM input             |
| `CONFIDENCE_THRESHOLD`     | `0.45`                                   | min confidence for `status=ok`            |
| `MAX_IMAGE_BYTES`          | `15728640`                               | upload size limit                         |
| `MAX_LOG_CHARS`            | `200000`                                 | log snippet size limit                    |

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v        # uses the deterministic heuristic backend
```

## Known limitations

* SmolVLM-256M is a small model: it reads prominent dialog/banner text well
  but can miss fine print; treat its observations as hints, the CV/log signals
  as the backbone.
* CPU inference is slow (see note above); use a GPU or the heuristic backend
  for interactive latency.
* Log rules are English-keyword oriented.
