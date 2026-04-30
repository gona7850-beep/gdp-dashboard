import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

DEFAULT_WEIGHTS: Dict[str, float] = {
    "stability": 0.25,
    "coherency": 0.20,
    "strength": 0.20,
    "brittle": 0.15,
    "cost": 0.10,
    "process": 0.10,
}

REQUIRED_COLUMNS = [
    "candidate_id",
    "composition_atpct",
    "e_above_hull",
    "delta_hf",
    "brittle_risk",
    "processability",
    "cost_index",
    "strength_proxy",
    "interface_surrogate",
    "lattice_a_precip",
    "lattice_a_nb",
]


def minmax_inverse(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([1.0] * len(s), index=s.index)
    return 1 - (s - mn) / (mx - mn)


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([1.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    positive = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(positive.values())
    if s == 0:
        raise ValueError("All objective weights are zero.")
    return {k: v / s for k, v in positive.items()}


def compute_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["misfit_pct"] = (out["lattice_a_precip"] - out["lattice_a_nb"]).abs() / out["lattice_a_nb"] * 100

    out["stability_score"] = 0.6 * minmax_inverse(out["e_above_hull"]) + 0.4 * minmax_inverse(out["delta_hf"])
    out["coherency_score"] = 0.7 * minmax_inverse(out["misfit_pct"]) + 0.3 * minmax_inverse(out["interface_surrogate"])
    out["strength_score"] = minmax(out["strength_proxy"])
    out["process_score"] = minmax(out["processability"])
    out["cost_score"] = minmax_inverse(out["cost_index"])
    out["brittle_penalty"] = minmax(out["brittle_risk"])

    out["total_score"] = (
        weights["stability"] * out["stability_score"]
        + weights["coherency"] * out["coherency_score"]
        + weights["strength"] * out["strength_score"]
        + weights["process"] * out["process_score"]
        + weights["cost"] * out["cost_score"]
        - weights["brittle"] * out["brittle_penalty"]
    )
    return out


def hard_filter(df: pd.DataFrame, hull_th=0.10, hf_th=0.0, brittle_th=0.60) -> pd.DataFrame:
    return df[(df["e_above_hull"] <= hull_th) & (df["delta_hf"] <= hf_th) & (df["brittle_risk"] <= brittle_th)].copy()


def select_diverse_top_n(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    df_sorted = df.sort_values("total_score", ascending=False).copy()
    if "system_class" not in df_sorted.columns:
        return df_sorted.head(top_n)

    selected = []
    used_classes = set()
    unique_classes = list(df_sorted["system_class"].dropna().unique())

    for _, row in df_sorted.iterrows():
        cls = row.get("system_class", "unknown")
        remaining_slots = top_n - len(selected)
        unseen = len([c for c in unique_classes if c not in used_classes])
        if cls not in used_classes or unseen >= remaining_slots:
            selected.append(row)
            used_classes.add(cls)
        if len(selected) >= top_n:
            break
    return pd.DataFrame(selected)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--output_dir", default="materials_design/out")
    parser.add_argument("--hull_th", type=float, default=0.10)
    parser.add_argument("--hf_th", type=float, default=0.0)
    parser.add_argument("--brittle_th", type=float, default=0.60)
    parser.add_argument("--w_stability", type=float, default=DEFAULT_WEIGHTS["stability"])
    parser.add_argument("--w_coherency", type=float, default=DEFAULT_WEIGHTS["coherency"])
    parser.add_argument("--w_strength", type=float, default=DEFAULT_WEIGHTS["strength"])
    parser.add_argument("--w_brittle", type=float, default=DEFAULT_WEIGHTS["brittle"])
    parser.add_argument("--w_cost", type=float, default=DEFAULT_WEIGHTS["cost"])
    parser.add_argument("--w_process", type=float, default=DEFAULT_WEIGHTS["process"])
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    validate_input(df)

    weights = normalize_weights(
        {
            "stability": args.w_stability,
            "coherency": args.w_coherency,
            "strength": args.w_strength,
            "brittle": args.w_brittle,
            "cost": args.w_cost,
            "process": args.w_process,
        }
    )

    filtered = hard_filter(df, hull_th=args.hull_th, hf_th=args.hf_th, brittle_th=args.brittle_th)
    scored = compute_scores(filtered, weights=weights)
    ranked = scored.sort_values("total_score", ascending=False)
    top = select_diverse_top_n(ranked, args.top_n)

    ranked.to_csv(out_dir / "ranked_candidates.csv", index=False)
    top.to_csv(out_dir / "top_candidates.csv", index=False)

    next_batch = top[["candidate_id", "composition_atpct", "total_score"]].copy()
    next_batch["suggested_action"] = "experiment_doe"
    next_batch.to_csv(out_dir / "next_batch.csv", index=False)

    print(f"Filtered candidates: {len(filtered)}")
    print(f"Saved: {out_dir / 'ranked_candidates.csv'}")
    print(f"Saved: {out_dir / 'top_candidates.csv'}")
    print(f"Saved: {out_dir / 'next_batch.csv'}")


if __name__ == "__main__":
    main()
