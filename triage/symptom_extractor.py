"""
symptom_extractor.py

Turns free-form, possibly multilingual patient text into a structured
symptom record using Gemma 4. This is the "agentic" part: if information
is missing, the model returns a follow_up_question instead of guessing,
and the app loop asks the user before proceeding -- rather than a
one-shot classifier that silently fills gaps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .gemma_client import GemmaClient

SYSTEM_PROMPT = """You are a multilingual medical symptom-intake assistant.
You do NOT diagnose. Your only job is to extract structured information
and, if something critical is missing, ask ONE short clarifying question.

The patient may write in English, Hindi, Romanized Hindi, or Chhattisgarhi.
Detect the language and respond conversationally in the SAME language for
the "follow_up_question" field, but always return the JSON fields below in
English.

Return ONLY valid JSON, no prose, no markdown fences, in this exact shape:
{
  "detected_language": "<language name>",
  "symptoms": ["<symptom 1>", "<symptom 2>", ...],
  "duration": "<e.g. '3 days', 'since this morning', or null>",
  "severity_self_reported": "<mild|moderate|severe|unknown>",
  "age_group": "<infant|child|adult|elderly|unknown>",
  "follow_up_question": "<one short question in the patient's language, or null if enough info>",
  "patient_summary": "<one sentence, plain language, English>"
}
"""


@dataclass
class SymptomRecord:
    detected_language: str = "unknown"
    symptoms: List[str] = field(default_factory=list)
    duration: Optional[str] = None
    severity_self_reported: str = "unknown"
    age_group: str = "unknown"
    follow_up_question: Optional[str] = None
    patient_summary: str = ""
    raw_backend: str = ""
    raw_model: str = ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_symptoms(patient_text: str, conversation_history: Optional[List[str]] = None,
                      client: Optional[GemmaClient] = None) -> SymptomRecord:
    client = client or GemmaClient()

    history_block = ""
    if conversation_history:
        history_block = "\nPrior conversation:\n" + "\n".join(conversation_history) + "\n"

    user_prompt = f"{history_block}\nPatient says: \"{patient_text}\"\n\nReturn the JSON now."

    resp = client.generate(SYSTEM_PROMPT, user_prompt, temperature=0.1, max_tokens=400)

    try:
        parsed = json.loads(_strip_fences(resp.text))
    except json.JSONDecodeError:
        # Model didn't return clean JSON (happens more on smaller edge
        # variants) -- fail soft into NEEDS_MORE_INFO rather than crashing,
        # the red_flags rule layer still runs on the raw text either way.
        parsed = {
            "detected_language": "unknown",
            "symptoms": [],
            "duration": None,
            "severity_self_reported": "unknown",
            "age_group": "unknown",
            "follow_up_question": "Could you describe your main symptom in a few words?",
            "patient_summary": patient_text[:200],
        }

    return SymptomRecord(
        detected_language=parsed.get("detected_language", "unknown"),
        symptoms=parsed.get("symptoms", []) or [],
        duration=parsed.get("duration"),
        severity_self_reported=parsed.get("severity_self_reported", "unknown"),
        age_group=parsed.get("age_group", "unknown"),
        follow_up_question=parsed.get("follow_up_question"),
        patient_summary=parsed.get("patient_summary", ""),
        raw_backend=resp.backend,
        raw_model=resp.model,
    )
