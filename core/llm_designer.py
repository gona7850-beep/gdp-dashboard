"""Claude (Anthropic) integration for the composition design platform.

Two responsibilities:

1. ``LLMDesigner`` — a thin wrapper around the Anthropic SDK that calls
   Claude with the prompt templates defined in :mod:`core.composition_prompts`.
   When ``anthropic`` is not installed or no API key is configured, the
   class falls back to a deterministic, rule-based stand-in so the rest of
   the platform keeps working (very useful for CI and offline demos).

2. ``parse_target_spec`` — a small helper that takes a free-text request
   from the user, asks Claude (or the heuristic fallback) for a JSON
   target-property dict, validates it against the model's property columns,
   and returns a clean ``dict[str, float]``.

Configuration:

* Set ``ANTHROPIC_API_KEY`` in the environment to enable real Claude calls.
* Set ``CLAUDE_COMPOSITION_MODEL`` to pick a different model (default
  ``claude-sonnet-4-6``). Use Claude 4.X family identifiers; do not embed
  the model name in user-visible artifacts.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .composition_prompts import (
    SYSTEM_PROMPT,
    explain_candidates_prompt,
    feasibility_prompt,
    improvement_prompt,
    parse_target_prompt,
)


DEFAULT_MODEL = os.environ.get("CLAUDE_COMPOSITION_MODEL", "claude-sonnet-4-6")


@dataclass
class LLMResponse:
    text: str
    used_llm: bool          # False if the heuristic fallback produced it
    model: str | None       # which model answered, if any
    raw: dict[str, Any] | None = None


class LLMDesigner:
    """Talk to Claude about compositions. Falls back to heuristics offline."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------ raw call
    def _call(self, prompt: str, max_tokens: int = 800) -> LLMResponse:
        if self._client is None:
            return LLMResponse(text="", used_llm=False, model=None)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            getattr(block, "text", "") for block in msg.content
        ).strip()
        return LLMResponse(
            text=text,
            used_llm=True,
            model=self.model,
            raw={"id": getattr(msg, "id", None),
                 "stop_reason": getattr(msg, "stop_reason", None)},
        )

    # ------------------------------------------------------------------ tasks
    def parse_target(
        self,
        user_request: str,
        property_columns: list[str],
    ) -> tuple[dict[str, float], LLMResponse]:
        """Return ``({property: target}, llm_response)``.

        With an API key, asks Claude for a JSON object and validates it.
        Without one, uses a regex sweep that catches patterns like
        ``"yield strength of 650 MPa"`` or ``"hardness 200 HB"``.
        """
        prompt = parse_target_prompt(user_request, property_columns)
        resp = self._call(prompt, max_tokens=300)
        if resp.used_llm:
            parsed = _extract_first_json_object(resp.text)
            if parsed:
                clean = _clean_target(parsed, property_columns)
                if clean:
                    return clean, resp
        # heuristic fallback
        return _heuristic_parse_target(user_request, property_columns), resp

    def explain_candidates(
        self,
        target: dict[str, float],
        candidates: list[dict[str, Any]],
        model_r2: dict[str, float] | None = None,
    ) -> LLMResponse:
        prompt = explain_candidates_prompt(target, candidates, model_r2)
        resp = self._call(prompt, max_tokens=600)
        if resp.used_llm and resp.text:
            return resp
        return LLMResponse(
            text=_heuristic_explain(candidates, model_r2),
            used_llm=False,
            model=None,
        )

    def explain_feasibility(
        self,
        composition: dict[str, float],
        analysis: dict[str, Any],
    ) -> LLMResponse:
        prompt = feasibility_prompt(composition, analysis)
        resp = self._call(prompt, max_tokens=500)
        if resp.used_llm and resp.text:
            return resp
        return LLMResponse(
            text=_heuristic_feasibility(composition, analysis),
            used_llm=False,
            model=None,
        )

    def suggest_improvements(
        self,
        composition: dict[str, float],
        target: dict[str, float],
        predicted: dict[str, float],
        feature_columns: list[str],
    ) -> LLMResponse:
        prompt = improvement_prompt(composition, target, predicted, feature_columns)
        resp = self._call(prompt, max_tokens=400)
        if resp.used_llm and resp.text:
            return resp
        return LLMResponse(
            text=_heuristic_improve(composition, target, predicted),
            used_llm=False,
            model=None,
        )


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Tolerant JSON sniffer — Claude sometimes wraps JSON in prose or fences."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        candidate = m.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clean_target(
    obj: dict[str, Any], property_columns: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    lowered = {p.lower(): p for p in property_columns}
    for k, v in obj.items():
        key = lowered.get(str(k).lower())
        if key is None:
            continue
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Heuristic fallbacks (no API key required)
# ---------------------------------------------------------------------------

_NUMBER = r"([-+]?\d+(?:\.\d+)?)"


def _heuristic_parse_target(
    request: str, property_columns: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    lowered = request.lower()
    for prop in property_columns:
        prop_aliases = {prop.lower(), prop.replace("_", " ").lower()}
        for alias in prop_aliases:
            pattern = re.escape(alias) + r"\s*(?:of|around|near|to|=|:)?\s*" + _NUMBER
            m = re.search(pattern, lowered)
            if m:
                out[prop] = float(m.group(1))
                break
    return out


def _heuristic_explain(
    candidates: list[dict[str, Any]],
    model_r2: dict[str, float] | None,
) -> str:
    if not candidates:
        return "No candidates supplied to summarise."
    best = min(range(len(candidates)),
               key=lambda i: candidates[i].get("score", float("inf")))
    lines = ["(Heuristic summary — no LLM available)"]
    for i, c in enumerate(candidates):
        lines.append(
            f"Candidate #{i + 1}: score={c.get('score'):.4g}, "
            f"composition={c.get('composition')}"
        )
    weak = [p for p, r in (model_r2 or {}).items() if r < 0.7]
    if weak:
        lines.append(f"Warning: validation R^2 < 0.7 for: {weak}")
    lines.append(f"RECOMMENDED: candidate #{best + 1}")
    return "\n".join(lines)


def _heuristic_feasibility(
    composition: dict[str, float], analysis: dict[str, Any]
) -> str:
    rel = analysis.get("relative_errors") or {}
    if not rel:
        return ("(Heuristic) Predicted properties: "
                f"{analysis.get('predicted')}. No target supplied.")
    worst = max(rel.items(), key=lambda kv: kv[1])
    feasible = analysis.get("overall_feasible")
    head = "Within tolerance." if feasible else "Out of tolerance."
    return (
        f"(Heuristic) {head} Predicted: {analysis.get('predicted')}. "
        f"Largest relative error: {worst[0]}={worst[1]:.1%}. "
        f"{analysis.get('recommendation', '')}"
    )


def _heuristic_improve(
    composition: dict[str, float],
    target: dict[str, float],
    predicted: dict[str, float],
) -> str:
    lines = ["(Heuristic) Suggested adjustments:"]
    for prop, t in target.items():
        if prop not in predicted:
            continue
        diff = predicted[prop] - t
        if abs(diff) < 1e-6:
            continue
        direction = "decrease" if diff > 0 else "increase"
        lines.append(
            f"- {prop} predicted={predicted[prop]:.3g}, target={t:.3g} → "
            f"{direction} this property by ~{abs(diff):.3g} via composition tweaks."
        )
    if len(lines) == 1:
        lines.append("All target properties already match predictions.")
    return "\n".join(lines)


__all__ = ["LLMDesigner", "LLMResponse", "DEFAULT_MODEL"]
