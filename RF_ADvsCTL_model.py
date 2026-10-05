"""
Alzheimer's Disease vs Control classification from blood expression - Random Forest
====================================================================================
Features: memory genes from ThreeOrMoreLocally.txt measured in both GSE63060 and GSE63061.

Evaluation (shared with XGB and MLP, see ../ml_common.py):
1. Cross-cohort validation: train on GSE63060 -> test on GSE63061, and the reverse.
   Tuning and threshold selection happen inside the training cohort only; no ComBat,
   each cohort standardised on its own.
2. Repeated nested CV (5x5) within each cohort, with feature-stability analysis.
3. Brain validation against Braak stage and AD vs CTL (GSE1297 independent; GSE48350 not).
Metrics: AUC (DeLong + bootstrap CI), PR-AUC, Brier, calibration intercept/slope,
accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, MCC.

Run from anywhere:  python MLmicroarray/RF/RF_ADvsCTL_model.py
Outputs:            MLmicroarray/RF/crosscohort_results/
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ml_common as mc  # noqa: E402

PARAM_GRID = {
    "clf__n_estimators": [300, 500],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_leaf": [1, 4],
    "clf__max_features": ["sqrt", None],
}


def build_pipeline():
    return mc.make_pipeline(RandomForestClassifier(
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        random_state=mc.SEED,
        n_jobs=1,
    ))


def native_importance(pipe):
    return pd.Series(pipe.named_steps["clf"].feature_importances_, index=pipe.feature_names_in_).sort_values(
        ascending=False)


def extra_report(pipe, tag, outdir):
    rf = pipe.named_steps["clf"]
    print(f"  RF configuration: {rf.n_estimators} trees, max_depth={rf.max_depth}, "
          f"min_samples_leaf={rf.min_samples_leaf}, max_features={rf.max_features}, "
          f"OOB accuracy (training cohort)={rf.oob_score_:.4f}")


SPEC = mc.ModelSpec(
    name="RF",
    title="Random Forest",
    build=build_pipeline,
    param_grid=PARAM_GRID,
    native_importance=native_importance,
    extra_report=extra_report,
)

if __name__ == "__main__":
    mc.run_all(SPEC, Path(__file__).resolve().parent / "crosscohort_results")
