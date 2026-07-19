"""
red_flags.py

Deterministic, rule-based red-flag detector. This is the safety net that
runs INDEPENDENTLY of the LLM: even if Gemma mis-extracts something or
hallucinates, a hard keyword/pattern match on danger signs still forces
an "EMERGENCY" escalation. This separation (rules decide urgency ceiling,
LLM handles language/structure/conversation) is the core design decision
to defend in the write-up and demo -- it's what keeps a generative model
from being the single point of failure in a health-triage tool.

IMPORTANT / HONEST LIMITATIONS (put this in the write-up, don't hide it):
  - These categories follow widely-taught public triage concepts (e.g. the
    general "danger signs" framing used in WHO Integrated Management of
    Childhood Illness materials, and common "call emergency services now"
    stroke/cardiac/breathing guidance used in public first-aid education).
  - The keyword lists and the exact "which symptoms => which tier" mapping
    below are ENGINEERING CHOICES for this hackathon prototype, not
    clinically validated thresholds. They have not been reviewed by a
    licensed clinician. Say this explicitly in the demo -- judges trust
    submissions that are honest about what's a real result vs a
    reasonable placeholder more than ones that oversell.
  - This is a triage AID, not a diagnostic tool, and should never be
    the sole basis for a real clinical decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class UrgencyTier(str, Enum):
    EMERGENCY = "EMERGENCY"        # call emergency services / go to ER now
    URGENT = "URGENT"              # see a doctor within hours, same day
    ROUTINE = "ROUTINE"            # see a doctor within a few days
    SELF_CARE = "SELF_CARE"        # home care / monitor, no visit needed yet
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"  # not enough signal, ask follow-up


@dataclass
class RedFlagMatch:
    tier: UrgencyTier
    reason: str
    matched_terms: List[str] = field(default_factory=list)


# Multilingual keyword sets (English + Hindi + Romanized Hindi + Chhattisgarhi
# common terms). This list is intentionally small and readable -- extend it
# per-language as you test with real speakers rather than machine-translating
# the whole thing blind, which is how you get false negatives on real slang.
EMERGENCY_PATTERNS = {
    "chest_pain_breathing": [
        r"chest pain", r"can'?t breathe", r"cannot breathe", r"saans nahi",
        r"seene mein dard", r"suicide", r"khud ko", r"unconscious",
        r"behosh", r"not breathing", r"saans ruk",
    ],
    "stroke_signs": [
        r"face droop", r"slurred speech", r"one side weak", r"ek taraf",
        r"chehra tedha", r"sudden confusion", r"achanak bolne mein",
    ],
    "severe_bleeding_trauma": [
        r"heavy bleeding", r"bahut khoon", r"severe bleeding",
        r"deep wound", r"gehra ghaav",
    ],
    "infant_danger_signs": [
        r"baby.*not (feeding|waking)", r"bachcha.*doodh nahi",
        r"newborn.*blue", r"seizure", r"daura pad", r"convulsion",
    ],
}

URGENT_PATTERNS = {
    "high_fever": [
        r"high fever", r"tez bukhar", r"fever.*(3|four|4|5) days",
        r"bukhar.*din se",
    ],
    "persistent_vomiting_dehydration": [
        r"can'?t keep (water|food) down", r"baar baar ulti",
        r"severe diarrhea", r"pani jaisa dast",
    ],
    "worsening_symptoms": [
        r"getting worse", r"badh raha hai", r"zyada ho raha",
    ],
}


def _search(patterns: List[str], text: str) -> List[str]:
    hits = []
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            hits.append(p)
    return hits


def check_red_flags(free_text: str, structured_symptoms: List[str] | None = None) -> RedFlagMatch:
    """
    Run the raw transcript (and, if available, Gemma's structured symptom
    list) through the rule-based patterns. Free text is checked because
    the LLM extraction step can miss things; structured symptoms are
    checked too because they may be phrased differently (e.g. translated).
    """
    combined = free_text or ""
    if structured_symptoms:
        combined += " " + " ".join(structured_symptoms)

    for category, patterns in EMERGENCY_PATTERNS.items():
        hits = _search(patterns, combined)
        if hits:
            return RedFlagMatch(
                tier=UrgencyTier.EMERGENCY,
                reason=f"Danger sign detected ({category.replace('_', ' ')})",
                matched_terms=hits,
            )

    for category, patterns in URGENT_PATTERNS.items():
        hits = _search(patterns, combined)
        if hits:
            return RedFlagMatch(
                tier=UrgencyTier.URGENT,
                reason=f"Warning sign detected ({category.replace('_', ' ')})",
                matched_terms=hits,
            )

    return RedFlagMatch(
        tier=UrgencyTier.NEEDS_MORE_INFO,
        reason="No hard-coded danger/warning sign matched; deferring to LLM classification.",
        matched_terms=[],
    )
