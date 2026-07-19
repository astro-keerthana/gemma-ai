# Sanjeevani Triage — Kaggle Write-up Draft

*Fill in the [bracketed] parts with your specifics before submitting — a
write-up with your own voice and real testing notes will read far more
credibly to judges than a generic template.*

---

## Problem

[1-2 sentences: e.g. "In rural India, one health worker often serves a
population of thousands, patients describe symptoms in regional languages
or dialects, and connectivity is unreliable. Existing digital health tools
assume steady internet and English/Hindi-only input."]

## What we built

Sanjeevani Triage is a multilingual, offline-first triage assistant built
on Gemma 4. A patient describes symptoms in their own language (English,
Hindi, Romanized Hindi, or Chhattisgarhi in this prototype); the system
extracts structured symptom data, asks a clarifying question if
information is missing, and produces an urgency classification
(EMERGENCY / URGENT / ROUTINE / SELF_CARE) plus a structured handoff note
a health worker can act on immediately.

## Why this approach

The core design decision is separating **safety-critical rule logic**
from **language understanding**. A deterministic red-flag layer
(keyword/pattern matching on danger signs — chest pain + breathing
difficulty, stroke face/speech signs, infant danger signs, etc.) runs
independently of the LLM and acts as a ceiling: the model can escalate a
case's urgency but never downgrade a rule-flagged emergency. This matters
because the failure mode we most want to avoid in a triage tool is
under-triage — missing something serious because a generative model
smoothed over it in conversation.

## Gemma 4 usage

- **Variant:** Gemma 4 [E4B / 26B / whichever you actually tested with] —
  chosen for [offline/edge deployment on modest hardware / stronger
  reasoning via hosted API — pick what's true].
- **How it's used:** two calls per turn — (1) structured symptom
  extraction from free text into JSON, including an optional follow-up
  question when information is incomplete; (2) urgency-tier reasoning
  over the structured summary, clamped by the rule-based ceiling.
- **Multimodal/native tool use:** [if you added voice/ASR or function
  calling, describe it here — if not, say so honestly rather than
  implying it's there].

## Architecture

[Paste or link the diagram from README.md — the pipeline: patient text →
symptom_extractor.py (Gemma call) → red_flags.py (deterministic rules) →
triage_engine.py (Gemma call, clamped) → handoff note.]

## What we tested

We ran [N] hand-written multilingual test cases (`data/test_cases.json`)
covering emergency, urgent, self-care, and ambiguous/needs-more-info
scenarios across English and Hindi. [State pass rate honestly, and note
what failed and why if anything did — judges respond better to honest
gaps than to a suspiciously perfect scorecard.]

## Honest limitations

- Red-flag keyword lists and urgency thresholds are engineering estimates
  for this prototype, not clinically validated protocol — not reviewed by
  a licensed clinician.
- No evaluation against real patient transcripts, only hand-written test
  cases.
- Multilingual coverage is currently narrow (a small keyword set per
  language) and would need testing with real speakers of each dialect to
  harden.
- [Add anything else you know is rough — e.g. voice input not wired up,
  no persistence/logging, UI is a bare Gradio demo, etc.]

## What's next

[e.g. clinician review of the red-flag rule set, ASR integration for true
voice-first input, broader dialect coverage tested with real speakers,
integration with an actual ASHA-worker workflow.]

## Links

- GitHub: [your public repo URL]
- Demo video (3 min): [YouTube link]
- Live demo: [Gradio share link / hosted URL]
