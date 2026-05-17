"""HTS compound-ranking scores as forward-model features.

For each composition we want to predict properties for, find the best-
scoring matching compound in the bundled HTS database (or any
user-provided pool) and use that compound's three descriptors as
additional features. Intuition: alloys whose chemistry coincides with
a stable, coherent precipitate phase should hit higher hardness.

Output features (5 per row):
  hts_max_tie_line           best tie-line score among matching compounds
  hts_max_stability          best stability score (−ΔH normalised)
  hts_max_coherency          best coherency score with the host matrix
  hts_max_total              best weighted total
  hts_n_matching_compounds   how many compounds share the element set
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from .hts_screening import (
    HOSTS,
    KnownCompound,
    NB_HOST_COMPOUNDS,
    ScoreWeights,
    score_compound,
)


@dataclass
class HTSScoreFeaturizer:
    """Per-row HTS descriptor features.

    Parameters
    ----------
    host_symbol
        Host matrix to score against (default ``"Nb"``).
    compounds
        Compound pool to match against. Defaults to the bundled
        Nb-host DB.
    weights
        Scoring weights for the weighted total.
    require_host_in_compound
        If True, only consider compounds that include the host element.
        Default False so a row containing e.g. {Ti, Si} can still score
        against (Nb,Ti)5Si3-type compounds.
    """

    host_symbol: str = "Nb"
    compounds: Sequence[KnownCompound] = None
    weights: ScoreWeights = None
    require_host_in_compound: bool = False

    def __post_init__(self):
        if self.compounds is None:
            self.compounds = list(NB_HOST_COMPOUNDS)
        if self.weights is None:
            self.weights = ScoreWeights()
        if self.host_symbol not in HOSTS:
            raise KeyError(f"Unknown host {self.host_symbol!r}")

    @property
    def feature_names(self) -> List[str]:
        return [
            "hts_max_tie_line",
            "hts_max_stability",
            "hts_max_coherency",
            "hts_max_total",
            "hts_n_matching_compounds",
        ]

    def transform(self, comp_df: pd.DataFrame) -> pd.DataFrame:
        """Return one row per input with 5 HTS-derived features.

        A row's element set is the set of element columns whose value
        exceeds 1e-4 (i.e. the alloying elements present). Compounds
        whose constituents are a *subset* of the row's element set
        count as "matching". For each row we report the best-scoring
        match across the four descriptors.
        """
        host = HOSTS[self.host_symbol]
        n = len(comp_df)
        out = np.zeros((n, 5), dtype=float)
        el_cols = list(comp_df.columns)

        for i, (_, row) in enumerate(comp_df.iterrows()):
            present_elements = {
                el for el in el_cols if float(row[el]) > 1e-4
            }
            best_tie = best_sta = best_coh = best_total = 0.0
            n_match = 0
            for c in self.compounds:
                if self.require_host_in_compound and \
                        self.host_symbol not in c.elements:
                    continue
                # subset match: every element in the compound must be in
                # the row's element set
                if not set(c.elements).issubset(present_elements):
                    continue
                n_match += 1
                s = score_compound(c, host, self.weights)
                best_tie = max(best_tie, s.tie_line_score)
                best_sta = max(best_sta, s.stability_score)
                best_coh = max(best_coh, s.coherency_score)
                best_total = max(best_total, s.total)
            out[i, 0] = best_tie
            out[i, 1] = best_sta
            out[i, 2] = best_coh
            out[i, 3] = best_total
            out[i, 4] = float(n_match)

        return pd.DataFrame(out, columns=self.feature_names,
                            index=comp_df.index)


__all__ = ["HTSScoreFeaturizer"]
