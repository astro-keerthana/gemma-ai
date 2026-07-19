"""
gemma_client.py

Thin wrapper around Gemma 4 so the rest of the app doesn't care whether
we're talking to a local Ollama server, the hosted Gemini API (which
serves Gemma 4 as "gemma-4-*-it"), or -- if neither is configured --
a deterministic mock so the demo still runs end-to-end offline.

Backends (auto-selected, override with GEMMA_BACKEND env var):
    "ollama"  - local, offline, free. Requires `ollama run gemma4:e4b` once.
    "gemini"  - hosted, needs GEMMA_API_KEY. Good for judges without local GPU.
    "mock"    - no network / no model. Returns rule-based stand-in output.

This matters for the pitch: Sanjeevani is meant to work in low-connectivity
settings, so the default path is local Ollama with a small edge variant
(E2B/E4B), not a cloud call.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional


DEFAULT_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_OLLAMA_MODEL = os.environ.get("GEMMA_OLLAMA_MODEL", "gemma4:e4b")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMMA_GEMINI_MODEL", "gemma-4-31b-it")


@dataclass
class GemmaResponse:
    text: str
    backend: str
    model: str


class GemmaClient:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or os.environ.get("GEMMA_BACKEND") or self._autodetect()

    def _autodetect(self) -> str:
        # Prefer local Ollama (aligns with the "works offline" pitch).
        if self._ollama_alive():
            return "ollama"
        if os.environ.get("GEMMA_API_KEY"):
            return "gemini"
        return "mock"

    @staticmethod
    def _ollama_alive() -> bool:
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            urllib.request.urlopen(req, timeout=0.5)
            return True
        except Exception:
            return False

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 512) -> GemmaResponse:
        if self.backend == "ollama":
            return self._generate_ollama(system_prompt, user_prompt, temperature, max_tokens)
        if self.backend == "gemini":
            return self._generate_gemini(system_prompt, user_prompt, temperature, max_tokens)
        return self._generate_mock(system_prompt, user_prompt)

    # -- backends -----------------------------------------------------

    def _generate_ollama(self, system_prompt, user_prompt, temperature, max_tokens) -> GemmaResponse:
        payload = {
            "model": DEFAULT_OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            DEFAULT_OLLAMA_URL, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
        return GemmaResponse(text=text, backend="ollama", model=DEFAULT_OLLAMA_MODEL)

    def _generate_gemini(self, system_prompt, user_prompt, temperature, max_tokens) -> GemmaResponse:
        api_key = os.environ.get("GEMMA_API_KEY")
        if not api_key:
            raise RuntimeError("GEMMA_API_KEY not set for gemini backend")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{DEFAULT_GEMINI_MODEL}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini API error: {e.read().decode()}") from e
        return GemmaResponse(text=text, backend="gemini", model=DEFAULT_GEMINI_MODEL)

    def _generate_mock(self, system_prompt, user_prompt) -> GemmaResponse:
        # Deterministic stand-in so `python app.py` works with zero setup.
        # Real submission MUST run against ollama or gemini -- see README.
        text = json.dumps({
            "note": "MOCK BACKEND ACTIVE - no Gemma model reachable. "
                    "Install Ollama + `ollama run gemma4:e4b`, or set GEMMA_API_KEY. "
                    "This mock only echoes structure so the UI/pipeline can be tested offline.",
            "symptoms": [],
            "follow_up_question": None,
        })
        return GemmaResponse(text=text, backend="mock", model="mock")
