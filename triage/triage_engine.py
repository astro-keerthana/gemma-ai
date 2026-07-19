"""
triage_engine.py

Orchestrates one triage turn:
  1. Extract structured symptoms from patient text (Gemma).
  2. Run deterministic red-flag rules on BOTH raw text and structured
     symptoms (see red_flags.py for why this runs independently of the LLM).
  3. If rules say EMERGENCY, that wins immediately -- no further LLM call,
     no ambiguity, fastest possible path to "go get help now".
  4. Otherwise, ask Gemma for a reasoned urgency tier + rationale, but
     clamp it so the LLM can only escalate the rule-based tier, never
     downgrade it. (An LLM should never be able to talk a red flag down
     to "self care".)
  5. Emit a structured handoff note -- the artifact a health worker or
     the patient would actually see, not just a chat reply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from .gemma_client import GemmaClient
from .red_flags import UrgencyTier, check_red_flags
from .symptom_extractor import SymptomRecord, extract_symptoms

_TIER_ORDER = [
    UrgencyTier.SELF_CARE,
    UrgencyTier.ROUTINE,
    UrgencyTier.URGENT,
    UrgencyTier.EMERGENCY,
]


def _clamp_up_only(rule_tier: UrgencyTier, llm_tier: UrgencyTier) -> UrgencyTier:
    """LLM may only move the tier UP the urgency scale, never down."""
    if rule_tier == UrgencyTier.NEEDS_MORE_INFO:
        return llm_tier
    try:
        rule_idx = _TIER_ORDER.index(rule_tier)
        llm_idx = _TIER_ORDER.index(llm_tier)
    except ValueError:
        return rule_tier
    return _TIER_ORDER[max(rule_idx, llm_idx)]


REASONING_SYSTEM_PROMPT = """You are assisting a health-triage tool. You will
receive a structured symptom summary. Classify urgency into exactly one of:
SELF_CARE, ROUTINE, URGENT, EMERGENCY.

Rules:
- If genuinely uncertain, prefer the MORE urgent tier (never under-triage).
- Return ONLY valid JSON: {"tier": "...", "rationale": "<one sentence, plain language>"}
"""


@dataclass
class TriageResult:
    tier: UrgencyTier
    rationale: str
    rule_reason: str
    symptom_record: SymptomRecord
    handoff_note: str
    needs_follow_up: bool


def _llm_tier_classification(record: SymptomRecord, client: GemmaClient) -> tuple[UrgencyTier, str]:
    payload = {
        "symptoms": record.symptoms,
        "duration": record.duration,
        "severity_self_reported": record.severity_self_reported,
        "age_group": record.age_group,
    }
    resp = client.generate(
        REASONING_SYSTEM_PROMPT,
        f"Symptom summary: {json.dumps(payload)}\n\nReturn the JSON now.",
        temperature=0.1, max_tokens=200,
    )
    text = re.sub(r"^```(json)?|```$", "", resp.text.strip()).strip()
    try:
        parsed = json.loads(text)
        tier = UrgencyTier(parsed.get("tier", "ROUTINE"))
        rationale = parsed.get("rationale", "")
    except Exception:
        tier, rationale = UrgencyTier.ROUTINE, "Fallback: could not parse model output; defaulting to ROUTINE for review."
    return tier, rationale


def _build_handoff_note(record: SymptomRecord, tier: UrgencyTier, rationale: str, rule_reason: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "=== SANJEEVANI TRIAGE - HANDOFF NOTE (prototype, not clinically validated) ===",
        f"Generated: {ts}",
        f"Detected language: {record.detected_language}",
        f"Age group: {record.age_group}",
        f"Reported severity: {record.severity_self_reported}",
        f"Duration: {record.duration or 'not specified'}",
        f"Symptoms: {', '.join(record.symptoms) if record.symptoms else 'not yet captured'}",
        "",
        f"URGENCY TIER: {tier.value}",
        f"Rule-based check: {rule_reason}",
        f"Model rationale: {rationale}",
        "",
        f"Patient summary: {record.patient_summary or 'n/a'}",
        "",
        "NOTE: This is a hackathon prototype. Tier thresholds are engineering",
        "estimates, not clinically validated protocol. A qualified health",
        "worker should review before any action is taken on EMERGENCY/URGENT cases.",
    ]
    return "\n".join(lines)


def run_triage(patient_text: str, conversation_history: Optional[List[str]] = None,
               client: Optional[GemmaClient] = None) -> TriageResult:
    client = client or GemmaClient()

    record = extract_symptoms(patient_text, conversation_history, client=client)

    rule_match = check_red_flags(patient_text, record.symptoms)

    if rule_match.tier == UrgencyTier.EMERGENCY:
        # Skip the LLM reasoning call entirely -- speed matters here, and
        # we don't want an emergency escalation to depend on a second
        # network round-trip succeeding.
        note = _build_handoff_note(record, UrgencyTier.EMERGENCY, "Immediate escalation - danger sign matched.", rule_match.reason)
        return TriageResult(
            tier=UrgencyTier.EMERGENCY,
            rationale="Immediate escalation - danger sign matched.",
            rule_reason=rule_match.reason,
            symptom_record=record,
            handoff_note=note,
            needs_follow_up=False,
        )

    if record.follow_up_question and not record.symptoms:
        # Not enough info yet -- surface the follow-up instead of guessing.
        note = _build_handoff_note(record, UrgencyTier.NEEDS_MORE_INFO, "Awaiting more information from patient.", rule_match.reason)
        return TriageResult(
            tier=UrgencyTier.NEEDS_MORE_INFO,
            rationale="Awaiting more information from patient.",
            rule_reason=rule_match.reason,
            symptom_record=record,
            handoff_note=note,
            needs_follow_up=True,
        )

    llm_tier, rationale = _llm_tier_classification(record, client)
    final_tier = _clamp_up_only(rule_match.tier, llm_tier)

    note = _build_handoff_note(record, final_tier, rationale, rule_match.reason)
    return TriageResult(
        tier=final_tier,
        rationale=rationale,
        rule_reason=rule_match.reason,
        symptom_record=record,
        handoff_note=note,
        needs_follow_up=bool(record.follow_up_question),
    )
