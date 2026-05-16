"""
LLM assistant: structured prompts to Claude for reasoning over ML outputs.

Why this module?  ML outputs alone are insufficient for engineering judgment.
A SHAP plot tells you *which feature* contributes; it does not tell you
*whether the contribution makes sense given the metallurgy*. We send carefully
structured prompts to Claude with:
    1. The ML prediction + uncertainty.
    2. The SHAP attributions in feature space.
    3. The compositional context (which elements moved, by how much).
    4. Constraint-violation list.

We use the Anthropic SDK directly. The default model is configurable; for cost
control during long active-learning loops, route routine calls to Haiku 4.5 and
escalate to Opus for the design-review step.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from anthropic import Anthropic
    _HAS_ANTHROPIC = True
except ImportError:  # pragma: no cover
    _HAS_ANTHROPIC = False


SYSTEM_PROMPT_INTERPRETER = """You are a materials science research assistant embedded in
an alloy design platform. Users give you ML model outputs (predictions, uncertainties,
SHAP attributions) and ask you to interpret them through the lens of physical
metallurgy.

Rules:
1. Be specific. Reference concrete metallurgical mechanisms (solid-solution
   strengthening, precipitation, grain boundary phenomena, phase stability,
   processability) rather than generic statements.
2. Distinguish between what the ML *predicts* and what is *known* from
   literature. Flag disagreements explicitly.
3. If the prediction has high uncertainty (σ/μ > 0.15) or sits outside the
   training domain, say so plainly and recommend validation experiments.
4. Never invent literature citations. If you don't know a number, say so.
5. Use short paragraphs and structured lists. No fluff.
"""

SYSTEM_PROMPT_DESIGN_REVIEWER = """You are a senior research scientist reviewing
candidate alloy compositions proposed by an ML inverse-design routine. For each
candidate, assess:
  (a) Whether the composition is metallurgically plausible.
  (b) Whether the predicted property combination is physically consistent
      (e.g., strength-ductility trade-off shouldn't be unrealistic).
  (c) Processability concerns for the indicated manufacturing route.
  (d) Specific risks for the next experimental round.

End each review with a recommendation: STRONG_RECOMMEND / RECOMMEND / NEEDS_REVISION / REJECT,
plus a one-line rationale.
"""


@dataclass
class LLMAssistant:
    api_key: Optional[str] = None
    model: str = "claude-opus-4-7"   # routine: claude-haiku-4-5-20251001
    max_tokens: int = 1500
    client_: Optional[object] = field(default=None, repr=False)

    def __post_init__(self):
        if not _HAS_ANTHROPIC:
            return
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            self.client_ = Anthropic(api_key=key)

    @property
    def available(self) -> bool:
        return self.client_ is not None

    # -----------------------------------------------------------------------
    def interpret_prediction(self,
                              composition: Dict[str, float],
                              prediction: Dict[str, float],
                              shap_top: List[Dict],
                              feasibility: Optional[Dict] = None,
                              extrapolation_score: Optional[float] = None,
                              ) -> str:
        if not self.available:
            return self._offline_interpret_prediction(
                composition, prediction, shap_top, feasibility, extrapolation_score,
            )
        user = self._format_prediction_block(
            composition, prediction, shap_top, feasibility, extrapolation_score
        )
        return self._chat(SYSTEM_PROMPT_INTERPRETER, user)

    def review_candidates(self, candidates: List[Dict], context: str = "") -> str:
        if not self.available:
            return self._offline_review(candidates, context)
        block = self._format_candidates_block(candidates, context)
        return self._chat(SYSTEM_PROMPT_DESIGN_REVIEWER, block)

    # -----------------------------------------------------------------------
    def _chat(self, system: str, user: str) -> str:
        resp = self.client_.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Collect all text blocks
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts)

    # ---- Prompt formatters --------------------------------------------------
    def _format_prediction_block(self, composition, prediction, shap_top,
                                  feasibility, extrapolation) -> str:
        comp_str = ", ".join(f"{k}={v:.3f}" for k, v in composition.items() if v > 1e-4)
        pred_lines = []
        for tgt, val in prediction.items():
            if tgt.endswith("_mean"):
                key = tgt[:-5]
                mean = val
                std = prediction.get(f"{key}_std", 0)
                pred_lines.append(f"  {key}: {mean:.3f} ± {std:.3f}")
        shap_lines = [f"  {s['feature']}: SHAP={s['shap']:+.3f} (value={s['value']:.3f})"
                      for s in shap_top]
        out = [
            "## Composition (atomic fractions)",
            comp_str,
            "",
            "## Predicted properties (mean ± std)",
            *pred_lines,
            "",
            "## Top SHAP contributions (XGBoost head)",
            *shap_lines,
        ]
        if feasibility:
            out += [
                "",
                "## Feasibility check",
                f"feasible: {feasibility.get('feasible')}",
                f"hard violations: {feasibility.get('hard_violations', [])}",
                f"soft violations: {feasibility.get('soft_violations', [])}",
            ]
        if extrapolation is not None:
            out += [
                "",
                f"## Domain of applicability score: {extrapolation:.2f}",
                "(>1 means the query sits beyond the 95th-percentile NN distance "
                "of the training set in feature space)",
            ]
        out += [
            "",
            "Please interpret these results. Comment on:",
            "- Likely metallurgical mechanisms behind the predictions.",
            "- Whether the σ values look trustworthy.",
            "- Specific risks or follow-ups before committing to experiments.",
        ]
        return "\n".join(out)

    def _format_candidates_block(self, candidates, context) -> str:
        out = ["## Context", context, "", "## Candidates"]
        for i, c in enumerate(candidates, 1):
            out.append(f"### Candidate {i}")
            out.append(json.dumps(c, indent=2, default=str))
        out.append("\nReview each candidate per the rubric.")
        return "\n".join(out)

    # ---- Offline fallbacks (deterministic, no API key needed) --------------
    def _offline_interpret_prediction(self, composition, prediction, shap_top,
                                       feasibility, extrapolation) -> str:
        lines = ["# Prediction interpretation (offline mode — no LLM)",
                 "API key not configured; emitting heuristic summary.\n"]
        top = sorted(shap_top, key=lambda s: abs(s["shap"]), reverse=True)[:3]
        for s in top:
            sign = "positive" if s["shap"] > 0 else "negative"
            lines.append(f"- `{s['feature']}` has a {sign} contribution "
                         f"({s['shap']:+.3f}) at value {s['value']:.3f}.")
        if extrapolation is not None and extrapolation > 1.0:
            lines.append(f"\n⚠ DoA score {extrapolation:.2f} > 1.0 — extrapolation risk.")
        if feasibility and not feasibility.get("feasible", True):
            lines.append(f"\n⚠ Hard constraint violations: "
                         f"{feasibility.get('hard_violations')}")
        return "\n".join(lines)

    def _offline_review(self, candidates, context) -> str:
        return ("# Candidate review (offline mode)\n"
                f"{len(candidates)} candidates received. "
                "Connect an API key to receive a metallurgical review.")
