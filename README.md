# Sanjeevani Triage

A multilingual, offline-first health-triage assistant built on **Gemma 4**, for the
**Build with Gemma 4 — AI Durg** hackathon (Sanjeevani track).

> **Not a diagnostic tool.** This is a hackathon prototype. In a real emergency,
> call your local emergency number immediately. See [Limitations](#honest-limitations-read-this-first).

## The problem

Rural and low-connectivity clinics in India often have one health worker
covering a large population, patients who describe symptoms in regional
languages or dialects, and no reliable internet for cloud-based tools.
Sanjeevani Triage is a first pass at closing that gap: a conversational
triage agent that runs on-device (Gemma 4 E2B/E4B via Ollama), understands
multiple languages, asks clarifying questions instead of guessing, and
hands off a structured note a health worker can act on quickly.

## Architecture

```
patient text/voice
        │
        ▼
┌───────────────────┐
│ symptom_extractor  │  Gemma 4 → structured JSON (symptoms, duration,
│   (Gemma call #1)  │  severity, age group, or a follow-up question)
└─────────┬──────────┘
          │
          ▼
┌───────────────────┐
│    red_flags.py    │  Deterministic keyword/pattern rules, runs
│  (no LLM, no net)  │  independently of the model — the safety net
└─────────┬──────────┘
          │
   EMERGENCY? ──yes──► skip LLM reasoning, escalate immediately
          │no
          ▼
┌───────────────────┐
│  triage_engine.py  │  Gemma 4 → reasoned urgency tier + rationale
│   (Gemma call #2)  │  (clamped: LLM can only escalate the rule tier
└─────────┬──────────┘   up, never down)
          │
          ▼
   structured handoff note (app.py / Gradio UI)
```

**The key design decision** (defend this in the demo): the rule-based
red-flag layer is deliberately kept separate from and authoritative over
the LLM. Gemma handles language understanding, follow-up questions, and
nuanced reasoning — but a hard-coded danger-sign match can never be
"talked down" by the model, and the model can never talk a rule-flagged
emergency down to self-care either (see `_clamp_up_only` in
`triage_engine.py`). This matters specifically because generative models
can be inconsistent, and a triage tool's failure mode of most concern is
under-triage (missing something serious), not over-triage.

## Quick start

```bash
pip install -r requirements.txt

# Option A (recommended - offline, matches the "edge" pitch):
ollama pull gemma4:e4b
ollama run gemma4:e4b &        # leave running, or just `ollama serve`

# Option B (hosted, free, no card required - OpenRouter):
export OPENROUTER_API_KEY=your_key_here    # from openrouter.ai -> Settings -> Keys
export GEMMA_BACKEND=openrouter            # optional; auto-detects otherwise

# Option C (hosted, free, no card required - Google AI Studio):
export GEMMA_API_KEY=your_key_here         # from aistudio.google.com
export GEMMA_BACKEND=gemini                # optional; auto-detects otherwise

python app.py
```

Backend auto-detection order (when `GEMMA_BACKEND` isn't set explicitly):
`ollama` (if reachable) → `gemini` (if `GEMMA_API_KEY` set) → `openrouter`
(if `OPENROUTER_API_KEY` set) → `mock` (fallback, no model reachable).

Open the local URL Gradio prints. On Kaggle/Colab, set `GRADIO_SHARE=1`
before running `app.py` to get a public tunnel link, since the local port
isn't reachable from outside the notebook:
```bash
export GRADIO_SHARE=1
python app.py
```

To sanity-check the pipeline without the UI:

```bash
python run_tests.py
```

If none of Ollama, `GEMMA_API_KEY`, or `OPENROUTER_API_KEY` are available,
the client falls back to a `mock` backend so the app still runs end-to-end
for structural testing — clearly labeled as mock output (`raw_backend:
"mock"` in the result), never presented as a real triage result.

## Repo layout

```
app.py                     Gradio demo UI
run_tests.py                Test-case runner
triage/
  gemma_client.py           Backend-agnostic Gemma 4 wrapper (ollama/gemini/mock)
  symptom_extractor.py      Free text -> structured symptom JSON
  red_flags.py               Deterministic danger-sign rules
  triage_engine.py           Orchestration + handoff note generation
data/test_cases.json        Hand-written multilingual test cases
```

## Honest limitations (read this first)

- **The red-flag keyword lists and tier thresholds in `red_flags.py` and
  `triage_engine.py` are engineering estimates for this prototype**, not
  clinically validated protocol. They follow the general shape of
  publicly-taught triage concepts (danger-sign screening, stroke
  face/speech signs, emergency breathing/chest-pain signs) but have not
  been reviewed by a licensed clinician or tested against a real
  labeled dataset.
- **No held-out evaluation against real patient transcripts yet.** The
  test cases in `data/test_cases.json` are hand-written by the author,
  not sourced from real triage encounters — treat pass/fail on them as a
  regression check, not a validation study.
- **Multilingual coverage is intentionally small and demo-scoped.** The
  keyword rules currently cover English, Hindi, and Romanized Hindi with
  a few Chhattisgarhi terms; broader dialect coverage needs testing with
  real speakers, not machine translation.
- **Voice input** is not wired into `app.py` in this version — the
  pipeline accepts text; adding ASR (e.g. Whisper or Gemma 4's native
  audio input on E2B/E4B) is the next step, not yet implemented.
- **This tool does not replace a health worker.** It is designed to
  triage urgency and produce a clean handoff note, not to diagnose or
  prescribe treatment.

## Why Gemma 4 E4B

Chosen over the larger 26B/31B variants specifically for the offline/edge
pitch this track calls for — E4B is small enough to run on modest local
hardware (a rural clinic laptop, not a GPU server), while still handling
structured extraction and multi-turn reasoning well enough for this scope.
The architecture is backend-agnostic (`gemma_client.py`), so swapping to
the 26B MoE or hosted Gemini API for a stronger-reasoning deployment is a
one-line config change.

## License

Apache 2.0, matching the Gemma 4 model license.
