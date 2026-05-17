# Nb-Si alloy benchmark dataset

Three CSV files of Nb-silicide composite alloys with mechanical properties,
provided as a real-data benchmark for the AlloyForge ML pipeline.

| File | Rows | Elements (wt%) | Targets |
|---|---|---|---|
| `nb_silicide_hardness.csv` | 184 | **Nb explicit** + Si, Ti, Cr, Al, Hf, Mo, W, Ta, Zr, Y, B, Fe, Ga, Ge, V, Mg, Sn, Re | Vickers hardness (HV) |
| `nb_silicide_with_compressive.csv` | 51 | Nb = balance + 13 alloying elements | HV + compressive σ_max (MPa) at RT |
| `nb_silicide_temp_dependent.csv` | 94 | Nb = balance + 14 alloying elements | HV + compressive σ_max with **test temperature** (25 / 1150-1400 °C) as a process variable |

All compositions are in **weight percent**. In the `_hardness.csv` and `_with_compressive.csv` files there are trailing spaces around values — pandas handles this automatically.

For `_with_compressive.csv` and `_temp_dependent.csv`, the Nb fraction is the
balance: ``Nb_wt = 100 − Σ_other_wt``. This is performed automatically by
``examples/benchmark_real_nb_si.py`` before atomic-fraction conversion.

The dataset originates from a literature survey of Nb-Si in-situ composites
(Nb solid solution + Nb₅Si₃ silicide) reported between roughly 2010 and 2024
for ultra-high-temperature structural applications. Multiple processing
routes (arc melting, directional solidification, powder metallurgy) are
mixed; user code should treat alloy-family / processing-route mismatch as
a source of noise and use group-aware CV.

These files are committed for **reproducibility of the benchmark report**
in `docs/benchmark_nb_si_results.md`. Always cite the original sources
when re-publishing.
