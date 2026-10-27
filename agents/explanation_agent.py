"""Optional local-LLM explanation agent (Ollama).  Never contacts a cloud API.

If no local model is reachable, or if the model's text fails validation against the structured evidence,
the deterministic explanation is used instead.  MRI data never reaches the LLM – only the JSON summary of
computed statistics does, and only to a server on localhost.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Optional, Tuple

from src.xai.explanation import validate_llm_text
from .prompts import SYSTEM_PROMPT, USER_TEMPLATE

log = logging.getLogger("neuroscanx")
OLLAMA_HOST = os.environ.get("NEUROSCANX_OLLAMA_HOST", "http://127.0.0.1:11434")


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, model: Optional[str] = None, timeout: float = 90.0):
        self.host, self.model, self.timeout = host.rstrip("/"), model or os.environ.get("NEUROSCANX_LLM_MODEL"), timeout

    def _req(self, path, payload=None, timeout=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.host + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode())

    def available(self) -> bool:
        # local loopback only – refuse to talk to remote hosts
        if not any(h in self.host for h in ("127.0.0.1", "localhost", "[::1]")):
            return False
        try:
            tags = self._req("/api/tags", timeout=0.6)
            models = [m["name"] for m in tags.get("models", [])]
            if not models:
                return False
            if not self.model:
                self.model = models[0]
            return True
        except Exception:
            return False

    def generate(self, system: str, prompt: str) -> str:
        out = self._req("/api/generate", {"model": self.model, "system": system, "prompt": prompt, "stream": False,
                                          "options": {"temperature": 0.1}})
        return out.get("response", "").strip()


class ExplanationAgent:
    def __init__(self, use_llm: str = "auto", model: Optional[str] = None):
        self.enabled = use_llm != "off"
        self.client = OllamaClient(model=model) if self.enabled else None

    def explain(self, summary: dict, deterministic_text: str) -> Tuple[str, str]:
        """Returns (text, source) where source is 'deterministic' or 'llm:<model>'."""
        if not (self.enabled and self.client and self.client.available()):
            return deterministic_text, "deterministic"
        try:
            ev = {k: v for k, v in summary.items() if k not in ("cluster_selection", "clustering", "disclaimer")}
            prompt = USER_TEMPLATE.format(evidence_json=json.dumps(ev, indent=1, default=float))
            text = self.client.generate(SYSTEM_PROMPT.format(disclaimer=summary["disclaimer"]), prompt)
            ok, why = validate_llm_text(text, summary)
            if not ok:
                log.warning("LLM explanation rejected (%s) → deterministic fallback", why)
                return deterministic_text, "deterministic"
            if summary["disclaimer"] not in text:
                text += "\n\n" + summary["disclaimer"]
            return text, f"llm:{self.client.model}"
        except Exception as e:
            log.warning("Local LLM failed (%s) → deterministic fallback", e)
            return deterministic_text, "deterministic"
