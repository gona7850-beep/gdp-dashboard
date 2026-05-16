"""Prompt templates for using an LLM (Claude) with the composition platform.

Keeping prompts here — not hard-coded in the UI — lets us version them,
test them, and reuse the exact same strings between the Streamlit page,
the FastAPI router, and offline notebooks. Each builder returns a string
ready to send to ``client.messages.create(messages=[{"role":"user",...}])``.
"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a materials-design research assistant. You help users interact "
    "with a composition-property prediction and inverse-design platform. "
    "Always be precise: when you suggest a composition, give the element "
    "fractions explicitly (summing to 1.0) and explain the trade-offs. "
    "When you have numerical model output, ground your reasoning in those "
    "numbers and call out where the model's R^2 is low or uncertainty is "
    "high so the user knows where to trust the prediction less."
)


def parse_target_prompt(user_request: str, property_columns: list[str]) -> str:
    """Build a prompt asking Claude to extract a target-property JSON object
    from a free-text user request.

    The expected reply is a JSON dict mapping each ``property_columns`` entry
    to a numeric target. The caller can then ``json.loads`` it.
    """
    return (
        "Extract a target property specification from the following user "
        "request. The model knows about exactly these properties:\n"
        f"  {property_columns}\n\n"
        "Reply with ONLY a JSON object of the form "
        '{"property_name": numeric_target, ...} — no prose, no code fences. '
        "If the user did not mention a property, omit it (do not invent a "
        "default).\n\n"
        f"User request:\n\"\"\"\n{user_request}\n\"\"\"\n"
    )


def explain_candidates_prompt(
    target: dict[str, float],
    candidates: list[dict[str, Any]],
    model_r2: dict[str, float] | None = None,
) -> str:
    """Ask Claude to compare and explain a list of inverse-design candidates."""
    payload = {
        "target_properties": target,
        "candidates": candidates,
        "model_validation_r2": model_r2 or {},
    }
    return (
        "Below are inverse-design candidates produced by our random-forest "
        "based platform, scored by weighted MSE against the target "
        "properties (lower = better). Compare them, recommend the best "
        "one, and explain why. Where the validation R^2 of a property is "
        "below 0.7, explicitly warn the user that predictions for that "
        "property are unreliable.\n\n"
        "Be concise (5-8 sentences). End with a one-line recommendation "
        "of the form: `RECOMMENDED: candidate #N`.\n\n"
        f"Data (JSON):\n{json.dumps(payload, indent=2)}\n"
    )


def feasibility_prompt(
    composition: dict[str, float],
    analysis: dict[str, Any],
) -> str:
    """Ask Claude to turn a feasibility analysis into a short research memo."""
    payload = {"composition": composition, "analysis": analysis}
    return (
        "Turn the following feasibility analysis into a short research memo "
        "(under 200 words) suitable for a materials scientist to read. "
        "Cover: (1) what the model predicts, (2) how it compares to the "
        "targets, (3) which property is the bottleneck, (4) one concrete "
        "next experimental step. Do not invent numbers — only use what is "
        "in the payload.\n\n"
        f"Payload:\n{json.dumps(payload, indent=2)}\n"
    )


def improvement_prompt(
    composition: dict[str, float],
    target: dict[str, float],
    predicted: dict[str, float],
    feature_columns: list[str],
) -> str:
    """Ask Claude to suggest specific element adjustments to better hit targets."""
    payload = {
        "current_composition": composition,
        "target_properties": target,
        "predicted_properties": predicted,
        "available_elements": feature_columns,
    }
    return (
        "Given the current composition, predicted properties, and target "
        "properties below, suggest one or two specific composition "
        "adjustments (e.g. 'increase Cr by 0.05 and decrease Ni by 0.05') "
        "that you expect will move the predicted properties closer to the "
        "targets. Briefly justify each adjustment with metallurgical "
        "reasoning. Respect the constraint that fractions must sum to 1.0.\n\n"
        f"Data:\n{json.dumps(payload, indent=2)}\n"
    )


__all__ = [
    "SYSTEM_PROMPT",
    "parse_target_prompt",
    "explain_candidates_prompt",
    "feasibility_prompt",
    "improvement_prompt",
]
