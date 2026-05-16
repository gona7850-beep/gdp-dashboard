"""
Explainability layer.

SHAP gives quantitative attributions; that's necessary but not sufficient. A
materials engineer wants to know whether the attribution is *physically sensible*
(e.g. "higher Mo correlates with higher hardness via solid-solution + carbide
formation") — this is where the LLM assistant in ``llm_assistant.py`` plugs in.

This module produces:
    - Per-prediction local attributions (SHAP values for XGBoost head + GP local
      sensitivities for the residual head).
    - Global feature importance averaged across predictions.
    - "Counterfactual" exploration: what minimal composition change would shift
      predicted UTS by Δ?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap

from .forward_model import ForwardModel


@dataclass
class Explainer:
    model: ForwardModel
    background_size: int = 50

    def explain(self, composition_df: pd.DataFrame,
                target: str,
                background_df: pd.DataFrame) -> pd.DataFrame:
        """Local SHAP attributions for the XGBoost head, one row per query point.

        Returns long-format DataFrame: rows × (sample_id, feature, value, shap_value).
        """
        if target not in self.model.models_:
            raise KeyError(f"Target {target} not in fitted model")
        m = self.model.models_[target]

        X_bg = self.model.featurizer.transform(background_df)
        X_q = self.model.featurizer.transform(composition_df)

        X_bg_s = m.preproc.transform(X_bg[m.feature_names])
        X_q_s = m.preproc.transform(X_q[m.feature_names])

        bg_idx = np.random.default_rng(0).choice(
            len(X_bg_s), size=min(self.background_size, len(X_bg_s)), replace=False
        )
        explainer = shap.TreeExplainer(m.xgb, data=X_bg_s[bg_idx])
        sv = explainer.shap_values(X_q_s)
        if sv.ndim == 1:
            sv = sv[None, :]

        rows = []
        for i in range(sv.shape[0]):
            for j, fname in enumerate(m.feature_names):
                rows.append({
                    "sample_id": i,
                    "feature": fname,
                    "value": float(X_q.iloc[i][fname]),
                    "shap": float(sv[i, j]),
                })
        return pd.DataFrame(rows)

    def global_importance(self, target: str,
                            sample_df: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature over a representative sample."""
        m = self.model.models_[target]
        X = self.model.featurizer.transform(sample_df)
        X_s = m.preproc.transform(X[m.feature_names])
        explainer = shap.TreeExplainer(m.xgb)
        sv = explainer.shap_values(X_s)
        if sv.ndim == 1:
            sv = sv[None, :]
        imp = np.mean(np.abs(sv), axis=0)
        return pd.DataFrame({"feature": m.feature_names, "importance": imp}) \
            .sort_values("importance", ascending=False) \
            .reset_index(drop=True)

    def counterfactual(self,
                        composition: pd.Series,
                        target: str,
                        delta: float,
                        element_columns: Sequence[str],
                        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                        step: float = 0.005,
                        max_iter: int = 200) -> Dict:
        """Greedy counterfactual search: find the smallest composition perturbation
        that shifts the predicted target by ``delta`` (in property units).

        Strategy: at each step, pick the (element, direction) pair whose SHAP-implied
        marginal effect best matches the desired Δ, take a step of size ``step``,
        renormalize, repeat. Stop when target met or budget exhausted.
        """
        comp = composition.copy().astype(float)
        path: List[Dict] = []
        bounds = bounds or {}

        for it in range(max_iter):
            df = pd.DataFrame([comp])
            preds = self.model.predict(df[list(element_columns)])
            current = float(preds[f"{target}_mean"].iloc[0])
            path.append({"iter": it, "value": current, "comp": comp.copy().to_dict()})

            if abs(current - (composition.get(target, current) + delta)) < 1e-3:
                break

            # Need to shift current → current + delta
            target_value = current + delta if it == 0 else path[0]["value"] + delta
            need = target_value - current
            if abs(need) < 1e-4:
                break

            # Try each (element, ±) and pick the one that moves prediction most
            best_score = 0.0
            best_move = None
            for el in element_columns:
                for sign in (+1, -1):
                    new_comp = comp.copy()
                    new_comp[el] = max(0.0, new_comp[el] + sign * step)
                    lo, hi = bounds.get(el, (0.0, 1.0))
                    if new_comp[el] < lo or new_comp[el] > hi:
                        continue
                    s = new_comp[list(element_columns)].sum()
                    if s <= 0:
                        continue
                    new_comp[list(element_columns)] = new_comp[list(element_columns)] / s
                    df2 = pd.DataFrame([new_comp])
                    p2 = self.model.predict(df2[list(element_columns)])
                    delta_pred = float(p2[f"{target}_mean"].iloc[0]) - current
                    score = -abs(delta_pred - need)
                    if score > best_score or best_move is None:
                        best_score = score
                        best_move = (el, sign, new_comp)
            if best_move is None:
                break
            comp = best_move[2]

        return {
            "achieved": float(self.model.predict(
                pd.DataFrame([comp])[list(element_columns)]
            )[f"{target}_mean"].iloc[0]),
            "final_composition": comp[list(element_columns)].to_dict(),
            "path": path,
        }
