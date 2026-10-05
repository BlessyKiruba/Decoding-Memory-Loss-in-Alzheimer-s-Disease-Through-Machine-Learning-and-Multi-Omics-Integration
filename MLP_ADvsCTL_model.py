"""
Alzheimer's Disease vs Control classification from blood expression - Multilayer Perceptron
============================================================================================
Features: memory genes from ThreeOrMoreLocally.txt measured in both GSE63060 and GSE63061.

Evaluation (shared with RF and XGB, see ../ml_common.py):
1. Cross-cohort validation: train on GSE63060 -> test on GSE63061, and the reverse.
   Tuning and threshold selection happen inside the training cohort only; no ComBat,
   each cohort standardised on its own. This replaces the earlier grouped CV and
   leave-one-batch-out check.
2. Repeated nested CV (5x5) within each cohort, with feature-stability analysis.
3. Brain validation against Braak stage and AD vs CTL (GSE1297 independent; GSE48350 not).
Metrics: AUC (DeLong + bootstrap CI), PR-AUC, Brier, calibration intercept/slope,
accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, MCC.

The network is sized for ~10 input genes; wider layers only add parameters to overfit.

Run from anywhere:  python MLmicroarray/MLP/MLP_ADvsCTL_model.py
Outputs:            MLmicroarray/MLP/crosscohort_results/
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ml_common as mc  # noqa: E402

PARAM_GRID = {
    "clf__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
    "clf__alpha": [1e-3, 1e-2, 1e-1],
    "clf__learning_rate_init": [1e-3, 5e-4],
}


def build_pipeline():
    return mc.make_pipeline(MLPClassifier(
        activation="relu",
        solver="adam",
        batch_size=32,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=mc.SEED,
    ))


def extra_report(pipe, tag, outdir):
    mlp = pipe.named_steps["clf"]
    n_in = len(pipe.feature_names_in_)
    layers = [n_in, *mlp.hidden_layer_sizes, 1]
    n_params = sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))
    print(f"  MLP architecture: {' -> '.join(map(str, layers))} ({n_params:,} parameters), "
          f"alpha={mlp.alpha}, learning_rate_init={mlp.learning_rate_init}, epochs={mlp.n_iter_}")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(mlp.loss_curve_, lw=2, color="#2E86AB", label="Training loss")
    ax1.set(xlabel="Epoch", ylabel="Training loss", title=f"MLP training history ({tag})")
    ax2 = ax1.twinx()
    ax2.plot(mlp.validation_scores_, lw=2, color="#A23B72", label="Validation accuracy (early stopping)")
    ax2.set_ylabel("Validation accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    ax1.grid(alpha=0.3)
    mc.save(fig, outdir / f"{tag}_training_history.tif")


SPEC = mc.ModelSpec(
    name="MLP",
    title="MLP",
    build=build_pipeline,
    param_grid=PARAM_GRID,
    extra_report=extra_report,
)

if __name__ == "__main__":
    mc.run_all(SPEC, Path(__file__).resolve().parent / "crosscohort_results")
