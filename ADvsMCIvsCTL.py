"""
Alzheimer's Disease vs Control classification from blood expression - XGBoost
==============================================================================
Features: memory genes from ThreeOrMoreLocally.txt measured in both GSE63060 and GSE63061.
The model is trained on CTL vs AD; MCI samples are scored on the CTL -> AD continuum.

Evaluation (shared with RF and MLP, see ../ml_common.py):
1. Cross-cohort validation: train on GSE63060 -> test on GSE63061, and the reverse.
   Tuning and threshold selection happen inside the training cohort only; no ComBat,
   each cohort standardised on its own.
2. Repeated nested CV (5x5) within each cohort, with feature-stability analysis.
3. Brain validation against Braak stage and AD vs CTL (GSE1297 independent; GSE48350 not).
Metrics: AUC (DeLong + bootstrap CI), PR-AUC, Brier, calibration intercept/slope,
accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, MCC.

Run from anywhere:  python MLmicroarray/XGB/ADvsMCIvsCTL.py
Outputs:            MLmicroarray/XGB/crosscohort_results/
"""

import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ml_common as mc  # noqa: E402

PARAM_GRID = {
    "clf__n_estimators": [200, 400],
    "clf__learning_rate": [0.03, 0.1],
    "clf__max_depth": [2, 3, 4],
    "clf__colsample_bytree": [0.8, 1.0],
    "clf__reg_lambda": [1.0, 5.0],
}


def build_pipeline():
    return mc.make_pipeline(xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        subsample=0.8,
        random_state=mc.SEED,
        n_jobs=1,
        verbosity=0,
    ))


def native_importance(pipe):
    return pd.Series(pipe.named_steps["clf"].feature_importances_, index=pipe.feature_names_in_).sort_values(
        ascending=False)


def extra_report(pipe, tag, outdir):
    model = pipe.named_steps["clf"]
    print(f"  XGBoost configuration: n_estimators={model.n_estimators}, learning_rate={model.learning_rate}, "
          f"max_depth={model.max_depth}, subsample={model.subsample}, "
          f"colsample_bytree={model.colsample_bytree}, reg_lambda={model.reg_lambda}")


SPEC = mc.ModelSpec(
    name="XGB",
    title="XGBoost",
    build=build_pipeline,
    param_grid=PARAM_GRID,
    native_importance=native_importance,
    extra_report=extra_report,
)

if __name__ == "__main__":
    mc.run_all(SPEC, Path(__file__).resolve().parent / "crosscohort_results")
