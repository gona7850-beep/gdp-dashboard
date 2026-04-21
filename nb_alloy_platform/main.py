"""Command-line entry point for the Nb alloy platform."""

from __future__ import annotations

import argparse
import joblib

from .data_preprocessing import compute_correlations, load_data, select_top_features
from .model_training import train_models
from .optimization import random_search_optimization


def parse_param_ranges(range_str: str):
    """Parse ``NAME:lo:hi`` items into a dict of tuples."""
    ranges = {}
    for item in range_str.split():
        name, lo, hi = item.split(":")
        ranges[name] = (float(lo), float(hi))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Nb alloy composition platform CLI")
    subparsers = parser.add_subparsers(dest="task", required=True)

    p_corr = subparsers.add_parser("corr", help="Compute correlations")
    p_corr.add_argument("--input", required=True)
    p_corr.add_argument("--target", required=True)
    p_corr.add_argument("--features", nargs="+")
    p_corr.add_argument("--top", type=int, default=10)
    p_corr.add_argument("--method", choices=["mic", "pcc", "union"], default="mic")

    p_train = subparsers.add_parser("train", help="Train models")
    p_train.add_argument("--input", required=True)
    p_train.add_argument("--target", required=True)
    p_train.add_argument("--features", nargs="+", required=True)
    p_train.add_argument("--group")
    p_train.add_argument("--algorithms", nargs="+", default=["ElasticNet", "RandomForest", "SVR", "BayesianRidge"])
    p_train.add_argument("--n_splits", type=int, default=5)
    p_train.add_argument("--tol_abs", type=float)
    p_train.add_argument("--tol_rel", type=float)
    p_train.add_argument("--output")

    p_opt = subparsers.add_parser("optimise", help="Optimise compositions")
    p_opt.add_argument("--model_path", required=True)
    p_opt.add_argument("--param_ranges", required=True)
    p_opt.add_argument("--samples", type=int, default=1000)
    p_opt.add_argument("--objective", choices=["max", "min"], default="max")
    p_opt.add_argument("--out")

    args = parser.parse_args()

    if args.task == "corr":
        df = load_data(args.input)
        corr_df = compute_correlations(df, target_col=args.target, feature_cols=args.features)
        selected = select_top_features(corr_df, top_n=args.top, method=args.method)
        print(corr_df.loc[selected])
        return

    if args.task == "train":
        df = load_data(args.input)
        models, results_df = train_models(
            df,
            target_col=args.target,
            feature_cols=args.features,
            group_col=args.group,
            algorithms=args.algorithms,
            n_splits=args.n_splits,
            tol_abs=args.tol_abs,
            tol_rel=args.tol_rel,
        )
        if args.output:
            results_df.to_csv(args.output, index=False)
        else:
            print(results_df)

        for algo, model in models.items():
            joblib.dump(model, f"{args.target}_{algo}.joblib")
        return

    model = joblib.load(args.model_path)
    ranges = parse_param_ranges(args.param_ranges)
    df = random_search_optimization(model, param_ranges=ranges, n_samples=args.samples, objective=args.objective)
    if args.out:
        df.to_csv(args.out, index=False)
    else:
        print(df.head())


if __name__ == "__main__":
    main()
