"""One-shot end-to-end verification of the Composition Design Platform.

Run from the repo root:

    python scripts/verify.py            # full check
    python scripts/verify.py --fast     # skip pytest, skip Optuna trials

Exits 0 if everything works, non-zero on the first failure. Prints a
status table at the end so you can paste it into a bug report.
"""

from __future__ import annotations

import argparse
import importlib
import io
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
# Make `import backend / core` work whether invoked from repo root or from
# scripts/ or via `make verify`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Pretty output (no external deps; works on any terminal)
# ---------------------------------------------------------------------------

class C:
    OK = "\033[32m"
    FAIL = "\033[31m"
    WARN = "\033[33m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    END = "\033[0m"


def _is_tty() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not _is_tty():
        return text
    return f"{code}{text}{C.END}"


@dataclass
class StepResult:
    name: str
    passed: bool
    duration: float
    detail: str = ""


RESULTS: list[StepResult] = []


def step(name: str, fn: Callable[[], str | None]) -> None:
    """Run ``fn`` and capture pass/fail + duration. ``fn`` may return a
    short detail string (e.g. metric values) to print alongside the status.
    """
    t0 = time.perf_counter()
    icon_run = _c("…", C.DIM)
    print(f"  {icon_run} {name}", end="", flush=True)
    try:
        detail = fn() or ""
        passed = True
        err = ""
    except SystemExit:
        raise
    except Exception as exc:
        passed = False
        err = f"{type(exc).__name__}: {exc}"
        detail = ""
        traceback.print_exc()
    dt = time.perf_counter() - t0
    RESULTS.append(StepResult(name=name, passed=passed, duration=dt, detail=detail or err))
    icon = _c("✓", C.OK) if passed else _c("✗", C.FAIL)
    detail_str = f"  {_c(detail, C.DIM)}" if detail else ""
    err_str = f"  {_c(err, C.FAIL)}" if not passed and err else ""
    # rewrite line: clear and reprint
    print(f"\r  {icon} {name}{detail_str}{err_str}  {_c(f'[{dt:.2f}s]', C.DIM)}")


def section(title: str) -> None:
    print()
    print(_c(f"━━━ {title}", C.BOLD))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def check_deps() -> str:
    required = [
        ("numpy", None),
        ("pandas", None),
        ("scipy", None),
        ("sklearn", "scikit-learn"),
        ("fastapi", None),
        ("uvicorn", None),
        ("pydantic", None),
        ("httpx", None),
        ("streamlit", None),
        ("xgboost", None),
        ("optuna", None),
        ("pymoo", None),
        ("shap", None),
        ("joblib", None),
    ]
    missing: list[str] = []
    versions: dict[str, str] = {}
    for mod, pkg in required:
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except ImportError:
            missing.append(pkg or mod)
    if missing:
        raise RuntimeError(
            f"Missing packages: {', '.join(missing)}. "
            f"Install with: pip install -r requirements.txt"
        )
    return f"{len(required)} packages OK"


def check_pytest() -> str:
    import pytest
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        code = pytest.main(["-q", "--no-header", str(ROOT / "tests")])
    if code != 0:
        # Print captured output so the user can see which test failed
        sys.stdout.write(buf.getvalue())
        raise RuntimeError(f"pytest exit code {code}")
    last_line = [line for line in buf.getvalue().splitlines() if line.strip()][-1]
    return last_line


def check_fastapi_boot() -> str:
    from backend.main import app
    from fastapi.testclient import TestClient

    c = TestClient(app)
    expected = {
        "/": 200,
        "/health": 200,
        "/api": 200,
        "/docs": 200,
        "/openapi.json": 200,
        "/api/v1/composition/status": 200,
        "/api/v1/alloyforge/status": 200,
    }
    failed = []
    for path, want in expected.items():
        r = c.get(path)
        if r.status_code != want:
            failed.append(f"{path} -> {r.status_code} (want {want})")
    if failed:
        raise RuntimeError("; ".join(failed))
    return f"{len(expected)} routes OK"


def check_web_ui_serves() -> str:
    from backend.main import app
    from fastapi.testclient import TestClient

    c = TestClient(app)
    r = c.get("/")
    if r.status_code != 200:
        raise RuntimeError(f"/ returned {r.status_code}")
    if "Composition Design" not in r.text:
        raise RuntimeError("branding text missing from index.html")
    if "/api/v1/alloyforge/" not in r.text:
        raise RuntimeError("UI is not wired to /api/v1/alloyforge/")
    return f"{len(r.content)} bytes html, branded & wired"


def check_streamlit_pages() -> str:
    import py_compile

    pages = [
        ROOT / "app" / "streamlit_app.py",
        ROOT / "app" / "pages" / "7_조성설계_플랫폼.py",
        ROOT / "app" / "pages" / "8_AlloyForge_고급플랫폼.py",
    ]
    for p in pages:
        if not p.exists():
            raise FileNotFoundError(p)
        py_compile.compile(str(p), doraise=True)
    return f"{len(pages)} pages compile"


def check_lite_end_to_end() -> str:
    from core.composition_platform import CompositionDesigner, PropertyPredictor
    from core.synthetic_alloy_data import generate_synthetic_dataset

    df = generate_synthetic_dataset(n_samples=200, random_state=0)
    p = PropertyPredictor(estimator="rf", random_state=0)
    rep = p.train(df, cv_folds=3)
    designer = CompositionDesigner(p)
    target = {prop: float(df[prop].quantile(0.8)) for prop in p.property_columns}
    cands = designer.design_inverse(target, num_candidates=300, top_k=3, random_state=0)
    if len(cands) != 3:
        raise RuntimeError(f"expected 3 candidates, got {len(cands)}")
    mean_r2 = sum(rep.val_r2.values()) / len(rep.val_r2)
    best = cands[0]
    return f"R²̄={mean_r2:.2f}, best score={best.score:.4g}"


def check_advanced_end_to_end(n_trials: int) -> str:
    import numpy as np
    import pandas as pd
    from core.alloyforge import (
        CompositionFeaturizer, ConformalCalibrator, Dataset,
        DesignSpec, DomainOfApplicability, Explainer, ForwardModel,
        InverseDesigner, default_checker,
    )

    rng = np.random.default_rng(0)
    comp = rng.dirichlet([3, 1, 1, 0.5, 0.3], size=80)
    df = pd.DataFrame(comp, columns=["Fe", "Ni", "Cr", "Mo", "Ti"])
    df["hv"] = 200 + 800 * df["Mo"] + 600 * df["Cr"] + 1100 * df["Ti"] + rng.normal(0, 10, 80)
    df["uts"] = 400 + 1100 * df["Mo"] + 700 * df["Cr"] + 200 * df["Ni"] + rng.normal(0, 20, 80)

    ds = Dataset(compositions=df[["Fe", "Ni", "Cr", "Mo", "Ti"]],
                 properties=df[["hv", "uts"]])
    fm = ForwardModel(
        featurizer=CompositionFeaturizer(element_columns=["Fe", "Ni", "Cr", "Mo", "Ti"]),
        targets=["hv", "uts"], n_cv_splits=3,
    )
    fm.fit(ds, n_trials=n_trials)
    cal = ConformalCalibrator(alpha=0.1).calibrate(fm, ds)
    doa = DomainOfApplicability().fit(fm, ds)

    spec = DesignSpec(
        objectives=[("hv", "max")],
        element_bounds={"Fe": (0.5, 0.85), "Ni": (0.05, 0.25),
                        "Cr": (0.05, 0.25), "Mo": (0, 0.1), "Ti": (0, 0.05)},
        risk_lambda=0.5,
        feasibility=default_checker(["Fe", "Ni", "Cr", "Mo", "Ti"]),
    )
    designer = InverseDesigner(model=fm, spec=spec,
                               element_columns=["Fe", "Ni", "Cr", "Mo", "Ti"])
    front = designer.run_nsga2(pop_size=16, n_gen=8, seed=0)
    if len(front) == 0:
        raise RuntimeError("NSGA-II returned empty front")

    top = front.sort_values("agg_score").head(1)[["Fe", "Ni", "Cr", "Mo", "Ti"]]
    expl = Explainer(model=fm)
    sv = expl.explain(top, target="hv",
                      background_df=df[["Fe", "Ni", "Cr", "Mo", "Ti"]])
    if sv.empty:
        raise RuntimeError("SHAP returned no rows")

    r2 = fm.metrics_["hv"]["cv_r2"]
    return f"CV R²(hv)={r2:.2f}, front={len(front)}, SHAP rows={len(sv)}"


def check_fastapi_pipeline(n_trials: int) -> str:
    import numpy as np
    import pandas as pd
    from backend.main import app
    from fastapi.testclient import TestClient

    c = TestClient(app)
    rng = np.random.default_rng(0)
    comp = rng.dirichlet([3, 1, 1, 0.5], size=60)
    df = pd.DataFrame(comp, columns=["Fe", "Ni", "Cr", "Mo"])
    df["hv"] = 200 + 800 * df["Mo"] + 600 * df["Cr"] + rng.normal(0, 10, 60)

    # /fit
    r = c.post("/api/v1/alloyforge/fit", json={
        "element_columns": ["Fe", "Ni", "Cr", "Mo"],
        "target_columns": ["hv"],
        "n_trials": max(3, n_trials), "n_cv_splits": 3,
        "data": df.to_dict(orient="records"),
    })
    if r.status_code != 200:
        raise RuntimeError(f"/fit -> {r.status_code} {r.text[:200]}")
    sid = r.json()["session_id"]

    # /predict
    r = c.post("/api/v1/alloyforge/predict", json={
        "session_id": sid,
        "compositions": [{"Fe": 0.6, "Ni": 0.2, "Cr": 0.15, "Mo": 0.05}],
    })
    if r.status_code != 200 or "doa_scores" not in r.json():
        raise RuntimeError(f"/predict failed")

    # /inverse-design
    r = c.post("/api/v1/alloyforge/inverse-design", json={
        "session_id": sid,
        "objectives": [{"target": "hv", "direction": "max"}],
        "element_bounds": {"Fe": [0.5, 0.9], "Ni": [0, 0.3],
                           "Cr": [0, 0.3], "Mo": [0, 0.1]},
        "pop_size": 16, "n_gen": 5, "top_k": 3,
    })
    if r.status_code != 200:
        raise RuntimeError(f"/inverse-design -> {r.status_code}")
    n_cands = len(r.json()["candidates"])

    # /explain
    r = c.post("/api/v1/alloyforge/explain", json={
        "session_id": sid,
        "composition": {"Fe": 0.6, "Ni": 0.2, "Cr": 0.15, "Mo": 0.05},
        "target": "hv",
    })
    if r.status_code != 200 or len(r.json()["shap"]) == 0:
        raise RuntimeError(f"/explain failed")

    return f"fit→predict→design({n_cands})→explain OK"


def check_composition_pipeline() -> str:
    from backend.main import app
    from fastapi.testclient import TestClient

    c = TestClient(app)
    # demo dataset endpoint
    r = c.post("/api/v1/composition/demo-dataset", json={"n_samples": 100})
    if r.status_code != 200:
        raise RuntimeError(f"/demo-dataset -> {r.status_code}")
    rows = r.json()["rows"]
    # train
    r = c.post("/api/v1/composition/train", json={
        "rows": rows, "estimator": "rf", "cv_folds": 2,
    })
    if r.status_code != 200:
        raise RuntimeError(f"/train -> {r.status_code} {r.text[:200]}")
    # design
    r = c.post("/api/v1/composition/design", json={
        "target_properties": {
            "yield_strength": 350, "hardness": 200,
            "elongation": 20, "density": 7.0,
        },
        "num_candidates": 300, "top_k": 3, "strategy": "ga",
    })
    if r.status_code != 200:
        raise RuntimeError(f"/design -> {r.status_code} {r.text[:200]}")
    n = len(r.json()["candidates"])
    return f"demo→train→design({n}) OK"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="Skip pytest (covered by CI) and use 2 Optuna trials")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip pytest only")
    args = parser.parse_args()

    print(_c(f"\nComposition Design Platform — verification suite", C.BOLD))
    print(_c(f"Repository: {ROOT}", C.DIM))

    section("1. Dependencies")
    step("import all required packages", check_deps)

    if not (args.fast or args.skip_tests):
        section("2. Unit tests")
        step("pytest tests/ -q", check_pytest)

    section("3. FastAPI app")
    step("boot & ping every route", check_fastapi_boot)
    step("web UI is wired & branded", check_web_ui_serves)

    section("4. Streamlit pages")
    step("compile landing + pages 7 & 8", check_streamlit_pages)

    section("5. End-to-end ML pipelines")
    n_trials = 2 if args.fast else 5
    step("lite (RF + Dirichlet MC + GA inverse design)",
         check_lite_end_to_end)
    step(f"advanced (XGB+GP + Optuna×{n_trials} + NSGA-II + SHAP)",
         lambda: check_advanced_end_to_end(n_trials))

    section("6. REST API pipelines")
    step("/api/v1/composition: demo→train→design",
         check_composition_pipeline)
    step("/api/v1/alloyforge: fit→predict→design→explain",
         lambda: check_fastapi_pipeline(n_trials))

    # Summary
    print()
    print(_c("━━━ Summary", C.BOLD))
    n_pass = sum(1 for r in RESULTS if r.passed)
    n_fail = sum(1 for r in RESULTS if not r.passed)
    total_dt = sum(r.duration for r in RESULTS)
    for r in RESULTS:
        icon = _c("✓", C.OK) if r.passed else _c("✗", C.FAIL)
        print(f"  {icon} {r.name}  {_c(f'[{r.duration:.2f}s]', C.DIM)}")
    print()
    if n_fail == 0:
        msg = f"All {n_pass} checks passed in {total_dt:.1f}s"
        print(_c(f"✓ {msg}", C.OK + C.BOLD))
        print()
        print("Next steps:")
        print("  • Web UI:        uvicorn backend.main:app --reload  →  http://localhost:8000")
        print("  • Streamlit:     streamlit run app/streamlit_app.py  →  http://localhost:8501")
        print("  • Docker:        docker compose up")
        print("  • CLI demo:      python examples/alloyforge_demo.py")
        return 0
    else:
        print(_c(f"✗ {n_fail} of {n_pass + n_fail} checks failed", C.FAIL + C.BOLD))
        return 1


if __name__ == "__main__":
    sys.exit(main())
