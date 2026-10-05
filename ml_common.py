"""
Shared data loading, evaluation and plotting for the AD vs CTL blood classifiers
(RF/RF_ADvsCTL_model.py, XGB/ADvsMCIvsCTL.py, MLP/MLP_ADvsCTL_model.py).

All three models go through exactly the same procedure, so their results can be
compared directly (see compare_models.py).

Evaluation design
-----------------
Features
    Genes from ThreeOrMoreLocally.txt (memory DEGs in >= 3 brain regions, written by
    R_scripts/MicroModel.R from Consolidated_memory_genes.txt) that are measured in BOTH blood cohorts
    (GSE63060, GSE63061). Genes missing from either cohort are reported and dropped.

No joint preprocessing
    The expression matrix is the per-cohort quantile-normalised data WITHOUT ComBat.
    Each cohort is standardised (per-gene z-score) using only its own samples, all
    diagnostic groups included and no labels used. Nothing computed on one cohort is
    ever applied to the other.

Primary estimate: cross-cohort validation
    Train on GSE63060 and test on GSE63061, then the reverse. Hyperparameters
    (GridSearchCV, stratified 5-fold) and the decision threshold (Youden's J on
    out-of-fold training predictions) are chosen inside the training cohort only.
    The test cohort is used once, for evaluation.

Secondary estimate: repeated nested cross-validation within each cohort
    5 repeats x 5 outer folds. Inside every outer training fold: scaling, grid search
    (5-fold) and threshold selection. Outer folds are identical for all models
    (same seeds), so fold-level results are paired across models.

Uncertainty
    Stratified bootstrap 95% CIs (2000 resamples) for every external metric,
    DeLong 95% CI for AUC, Mann-Whitney p-value for AUC > 0.5.

Brain validation
    A model trained on both blood cohorts is applied to GSE1297 (hippocampus, not used
    anywhere else, so independent) and GSE48350 (used for DEG discovery, therefore NOT
    independent and reported separately). Brain data are z-scored within each
    cohort x region.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kruskal, mannwhitneyu, norm, spearmanr
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report, confusion_matrix,
    f1_score, log_loss, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, LeaveOneGroupOut, StratifiedKFold, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "MLmicroarray"

BLOOD_EXPR = ROOT / "R_scripts" / "ML" / "GSE63060_GSE63061_mergedExpression_data.csv"  # per-cohort QN, no ComBat
BLOOD_META = ROOT / "R_scripts" / "ML" / "GSE63060_GSE63061_mergedMetadata.csv"
GENE_LIST = ML_DIR / "ThreeOrMoreLocally.txt"  # written by R_scripts/MicroModel.R
# Memory-gene candidate list used by R_scripts/MicroModel.R to filter the DEGs; the universe for all enrichment tests
MEMORY_LIST = ML_DIR / "Consolidated_memory_genes.txt"
BLOOD_SOFT = {c: ML_DIR / "MLP" / "data" / f"{c}_family.soft.gz" for c in ("GSE63060", "GSE63061")}
BRAIN_EXPR = ROOT / "R_scripts" / "GSE1297_48350_110226_expr.csv"
BRAIN_META = ROOT / "R_scripts" / "GSE1297_48350_110226_metadata.csv"
COMPARISON_DIR = ML_DIR / "model_comparison"

BLOOD_COHORTS = ("GSE63060", "GSE63061")
# GSE110226 is excluded: its column block in BRAIN_EXPR is a duplicate of GSE1297 (untar bug)
# and it is choroid plexus on a custom Rosetta/Merck array.
INDEPENDENT_BRAIN = ("GSE1297",)
DISCOVERY_BRAIN = ("GSE48350",)  # used to derive the memory DEGs -> not independent
BRAAK_MAP = {"i-ii": 1.5, "iii": 3.0, "iii-iv": 3.5, "iv": 4.0, "v-vi": 5.5}

GROUPS = ("CTL", "MCI", "AD")
GROUP_COLORS = {"CTL": "lightgreen", "MCI": "gold", "AD": "lightcoral"}

SEED = 42
INNER_FOLDS = 5
OUTER_FOLDS = 5
N_REPEATS = 5
N_BOOT = 2000
N_PERM_REPEATS = 30
TOP_K = 8
STABILITY_TOP_K = 5
DPI = 600


@dataclass
class ModelSpec:
    name: str                      # short label used in file names, e.g. "RF"
    title: str                     # label used in plot titles
    build: Callable[[], Pipeline]  # returns an unfitted Pipeline(scaler -> clf)
    param_grid: dict               # keys prefixed with "clf__"
    native_importance: Callable[[Pipeline], pd.Series] | None = None
    extra_report: Callable[[Pipeline, str, Path], None] | None = None


def make_pipeline(clf) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# =============================================================================
# DATA
# =============================================================================
def read_gene_list(path: Path) -> list[str]:
    """Gene symbols from a text file; tolerates numbering ("1.\tADD3"), commas and CR line endings."""
    text = Path(path).read_text()
    tokens = re.split(r"[\s,;]+", text)
    genes = [t.strip().upper() for t in tokens if t.strip() and not re.fullmatch(r"\d+\.?", t.strip())]
    return list(dict.fromkeys(genes))


def zscore_by(X: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Per-gene z-score computed separately within each group (label-free)."""
    return X.groupby(groups.loc[X.index], group_keys=False).transform(
        lambda col: (col - col.mean()) / col.std(ddof=0)
    )


def load_blood(gene_file: Path = GENE_LIST):
    expr = pd.read_csv(BLOOD_EXPR, index_col=0)
    expr.index = expr.index.astype(str).str.upper().str.strip()

    meta = pd.read_csv(BLOOD_META).set_index("sample").rename(columns={"batch": "cohort"})
    meta["disease"] = meta["disease"].astype(str).str.upper().str.strip()
    meta = meta[meta["disease"].isin(GROUPS)]

    samples = expr.columns.intersection(meta.index)
    requested = read_gene_list(gene_file)
    genes = [g for g in requested if g in expr.index]
    missing = [g for g in requested if g not in expr.index]
    if not genes:
        raise ValueError(f"None of the genes in {gene_file.name} are in {BLOOD_EXPR.name}")

    X = expr.loc[genes, samples].T.astype(float)
    if X.isna().any().any():
        raise ValueError("Missing values in the blood expression matrix for the selected genes")
    meta = meta.loc[samples, ["cohort", "disease"]]
    return X, meta, missing


def load_brain(genes: list[str]):
    expr = pd.read_csv(BRAIN_EXPR, index_col=0)
    expr.index = expr.index.astype(str).str.upper().str.strip()
    # pandas renames repeated headers (GSM21203 -> GSM21203.1); keep only the first copy
    expr = expr.loc[:, expr.columns.str.fullmatch(r"GSM\d+")]

    meta = pd.read_csv(BRAIN_META).set_index("sample")
    meta = meta[meta["batch"].isin(INDEPENDENT_BRAIN + DISCOVERY_BRAIN)]
    samples = expr.columns.intersection(meta.index)
    meta = meta.loc[samples].copy()
    meta["disease"] = meta["disease"].astype(str).str.upper().replace({"CONTROL": "CTL"})
    meta["braak_numeric"] = meta["braak"].astype(str).str.lower().str.strip().map(BRAAK_MAP)
    meta["region"] = meta["region"].astype(str).str.lower().str.strip()

    genes_b = [g for g in genes if g in expr.index]
    X = expr.loc[genes_b, samples].T.astype(float)
    Xz = zscore_by(X, meta["batch"] + "|" + meta["region"])
    return Xz, meta, [g for g in genes if g not in expr.index]


def read_soft_characteristics(path: Path) -> pd.DataFrame:
    """Per-sample title and "key: value" characteristics from a GEO family SOFT file."""
    import gzip
    rows, cur = [], None
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if line.startswith("^SAMPLE"):
                cur = {"sample": line.split("=", 1)[1].strip()}
                rows.append(cur)
            elif cur is not None and line.startswith("!Sample_title"):
                cur["title"] = line.split("=", 1)[1].strip()
            elif cur is not None and line.startswith("!Sample_characteristics_ch1"):
                value = line.split("=", 1)[1].strip()
                if ":" in value:
                    k, v = value.split(":", 1)
                    cur[k.strip().lower()] = v.strip()
    return pd.DataFrame(rows).set_index("sample")


def load_blood_covariates() -> pd.DataFrame:
    """Status, age and sex for every GEO sample in both blood cohorts (before any filtering)."""
    frames = []
    for cohort, path in BLOOD_SOFT.items():
        d = read_soft_characteristics(path)
        frames.append(pd.DataFrame({
            "cohort": cohort,
            "geo_status": d["status"],
            "age": pd.to_numeric(d["age"], errors="coerce"),
            "sex": d["gender"].str.lower().str.strip(),
        }, index=d.index))
    return pd.concat(frames)


ENRICHR_CACHE = ML_DIR / "enrichr_libraries"


def load_enrichr_library(name: str) -> dict[str, set[str]]:
    """Enrichr gene-set library (e.g. GO_Biological_Process_2025), cached as a GMT file."""
    import gseapy
    path = ENRICHR_CACHE / f"{name}.gmt"
    if not path.exists():
        ENRICHR_CACHE.mkdir(parents=True, exist_ok=True)
        lib = gseapy.get_library(name=name, organism="Human")
        path.write_text("".join(f"{term}\t\t" + "\t".join(genes) + "\n" for term, genes in lib.items()))
    sets = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        sets[parts[0]] = {g.upper() for g in parts[2:] if g}
    return sets


def binary_target(disease: pd.Series) -> pd.Series:
    return disease.map({"CTL": 0, "AD": 1}).astype(int)


# =============================================================================
# METRICS
# =============================================================================
def _ratio(a, b):
    return a / b if b else np.nan


def binary_metrics(y, p, threshold) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return {
        "AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
        "Accuracy": (tp + tn) / len(y),
        "Balanced_accuracy": np.nanmean([sens, spec]),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": _ratio(tp, tp + fp),
        "NPV": _ratio(tn, tn + fn),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def calibration_intercept_slope(y, p) -> tuple[float, float]:
    """Calibration-in-the-large (ideal 0) and calibration slope (ideal 1)."""
    y = np.asarray(y).astype(int)
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        return np.nan, np.nan
    return float(citl), float(slope)


def bootstrap_ci(y, p, threshold, n_boot=N_BOOT, seed=SEED) -> pd.DataFrame:
    """Stratified bootstrap percentile 95% CI for every metric in binary_metrics."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        rows.append(binary_metrics(y[idx], p[idx], threshold))
    return pd.DataFrame(rows).quantile([0.025, 0.975]).T.set_axis(["CI_low", "CI_high"], axis=1)


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_auc_cov(y, preds):
    """AUCs and their DeLong covariance for k score vectors on the same samples (Sun & Xu, 2014)."""
    y = np.asarray(y).astype(int)
    preds = np.atleast_2d(np.asarray(preds, dtype=float))
    order = np.argsort(-y, kind="stable")  # positives first
    preds = preds[:, order]
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_ci(y, p, alpha=0.05):
    aucs, cov = delong_auc_cov(y, p)
    se = np.sqrt(cov[0, 0])
    z = norm.ppf(1 - alpha / 2)
    return float(aucs[0]), float(max(0, aucs[0] - z * se)), float(min(1, aucs[0] + z * se))


def delong_test(y, p1, p2):
    """Two-sided DeLong test for paired AUCs. Returns auc1, auc2, z, p."""
    aucs, cov = delong_auc_cov(y, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))


def external_metrics_table(y, p, threshold) -> pd.DataFrame:
    """Point estimates + bootstrap CIs at the training-cohort threshold and at 0.5."""
    y = np.asarray(y).astype(int)
    tables = []
    for rule, thr in [("Youden (training cohort)", threshold), ("Fixed 0.5", 0.5)]:
        est = pd.Series(binary_metrics(y, p, thr), name="Estimate")
        tab = pd.concat([est, bootstrap_ci(y, p, thr)], axis=1)
        tab.insert(0, "Threshold", thr)
        tab.insert(0, "Threshold_rule", rule)
        tables.append(tab)
    out = pd.concat(tables).rename_axis("Metric").reset_index()

    auc, lo, hi = delong_ci(y, p)
    citl, slope = calibration_intercept_slope(y, p)
    extra = pd.DataFrame([
        ["AUC (DeLong)", auc, lo, hi],
        ["AUC>0.5 Mann-Whitney p", mannwhitneyu(p[y == 1], p[y == 0], alternative="greater").pvalue, np.nan, np.nan],
        ["Calibration intercept (CITL)", citl, np.nan, np.nan],
        ["Calibration slope", slope, np.nan, np.nan],
        ["N_CTL", int((y == 0).sum()), np.nan, np.nan],
        ["N_AD", int((y == 1).sum()), np.nan, np.nan],
        ["AD prevalence", y.mean(), np.nan, np.nan],
    ], columns=["Metric", "Estimate", "CI_low", "CI_high"])
    extra.insert(1, "Threshold_rule", "threshold-free")
    extra.insert(2, "Threshold", np.nan)
    return pd.concat([out, extra], ignore_index=True)


# =============================================================================
# MODEL FITTING
# =============================================================================
def tune(spec: ModelSpec, X, y, cv=None) -> GridSearchCV:
    cv = cv if cv is not None else StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(spec.build(), spec.param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    return search.fit(X, y)


def training_threshold(estimator, X, y) -> tuple[float, np.ndarray]:
    """Youden threshold from out-of-fold predictions inside the training data only."""
    cv = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return youden_threshold(y, oof), oof


def importance_frame(result, genes) -> pd.DataFrame:
    return pd.DataFrame({
        "gene": genes,
        "importance_mean": result.importances_mean,
        "importance_sd": result.importances_std,
    })


# =============================================================================
# PLOTS
# =============================================================================
def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_binary_diagnostics(y, p, threshold, title, prefix: Path):
    """ROC, PR, calibration, confusion matrix and true-vs-predicted plots."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    auc, lo, hi = delong_ci(y, p)

    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"{title}\nROC (CTL vs AD)")
    ax.legend(loc="lower right")
    save(fig, prefix.with_name(prefix.name + "_roc.tif"))

    fig, ax = plt.subplots(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y, p)
    ax.plot(rec, prec, lw=2, label=f"PR-AUC = {average_precision_score(y, p):.3f}")
    ax.axhline(y.mean(), ls="--", color="gray", label=f"Prevalence = {y.mean():.2f}")
    ax.set(xlabel="Recall (Sensitivity)", ylabel="Precision (PPV)", title=f"{title}\nPrecision-Recall")
    ax.legend(loc="lower left")
    save(fig, prefix.with_name(prefix.name + "_pr.tif"))

    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    citl, slope = calibration_intercept_slope(y, p)
    ax.plot(mean_pred, frac_pos, marker="o", lw=2,
            label=f"Model (intercept={citl:.2f}, slope={slope:.2f}, Brier={brier_score_loss(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives", title=f"{title}\nCalibration")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, prefix.with_name(prefix.name + "_calibration.tif"))

    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues", alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=20, fontweight="bold")
    ax.set_xticks([0, 1], ["Pred: CTL", "Pred: AD"])
    ax.set_yticks([0, 1], ["True: CTL", "True: AD"])
    m = binary_metrics(y, p, threshold)
    ax.set_title(f"{title}\nThreshold {threshold:.2f} | Acc {m['Accuracy']:.3f} | "
                 f"Sens {m['Sensitivity']:.3f} | Spec {m['Specificity']:.3f}", fontsize=10)
    fig.colorbar(im, ax=ax)
    save(fig, prefix.with_name(prefix.name + "_confusion.tif"))

    fig, ax = plt.subplots(figsize=(6, 5))
    rng = np.random.default_rng(SEED)
    ax.scatter(y + rng.normal(0, 0.03, len(y)), p, alpha=0.6, s=30)
    ax.axhline(threshold, ls="--", color="gray", label=f"Threshold {threshold:.2f}")
    ax.set_xticks([0, 1], ["CTL (0)", "AD (1)"])
    ax.set(ylabel="Predicted P(AD)", xlabel="True label", title=f"{title}\nTrue vs predicted")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, prefix.with_name(prefix.name + "_true_vs_pred.tif"))


def add_pvalue_bracket(ax, x1, x2, y_vals_1, y_vals_2, label, y_offset_frac=0.07):
    y_top = max(np.nanmax(y_vals_1), np.nanmax(y_vals_2))
    y_min = min(np.nanmin(y_vals_1), np.nanmin(y_vals_2))
    y_range = (y_top - y_min) or 1.0
    h, tip = y_range * y_offset_frac, y_range * 0.02
    y = y_top + h
    ax.plot([x1, x1, x2, x2], [y - tip, y, y, y - tip], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + tip * 0.5, label, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(top=y + y_range * 0.18)


def group_boxplot(ax, scores: pd.Series, disease: pd.Series, threshold=None):
    present = [g for g in GROUPS if (disease == g).any()]
    data = [scores[disease == g].values for g in present]
    bp = ax.boxplot(data, tick_labels=present, patch_artist=True, showfliers=True)
    for patch, g in zip(bp["boxes"], present):
        patch.set_facecolor(GROUP_COLORS[g])
    for i, d in enumerate(data):
        ax.text(i + 1, -0.08, f"n={len(d)}", ha="center", fontsize=9)
    if threshold is not None:
        ax.axhline(threshold, ls="--", color="gray", alpha=0.7)
    ax.set_ylim(-0.12, 1.05)
    ax.grid(axis="y", alpha=0.3)


# =============================================================================
# ANALYSIS STEPS
# =============================================================================
def progression_stats(scores: pd.Series, disease: pd.Series) -> dict:
    groups = {g: scores[disease == g].values for g in GROUPS if (disease == g).any()}
    out = {"Kruskal_H": np.nan, "Kruskal_p": np.nan}
    if len(groups) > 1:
        h, p = kruskal(*groups.values())
        out.update(Kruskal_H=h, Kruskal_p=p)
    rho, p = spearmanr(disease.map({"CTL": 0, "MCI": 1, "AD": 2}), scores)
    out.update(Spearman_rho_CTL_MCI_AD=rho, Spearman_p=p)
    pairs = [(a, b) for a, b in [("CTL", "MCI"), ("MCI", "AD"), ("CTL", "AD")] if a in groups and b in groups]
    raw = [mannwhitneyu(groups[a], groups[b], alternative="two-sided").pvalue for a, b in pairs]
    if raw:
        adj = multipletests(raw, method="fdr_bh")[1]
        for (a, b), pr, pa in zip(pairs, raw, adj):
            out[f"MWU_{a}_vs_{b}_p"] = pr
            out[f"MWU_{a}_vs_{b}_q"] = pa
    return out


def cross_cohort(spec: ModelSpec, Xz, meta, outdir: Path):
    """Train on one blood cohort, test on the other, both directions."""
    results = {}
    for train_c, test_c in [BLOOD_COHORTS, BLOOD_COHORTS[::-1]]:
        tag = f"{spec.name}_train{train_c}_test{test_c}"
        print(f"\n--- {tag} ---")
        tr = meta.index[(meta["cohort"] == train_c) & meta["disease"].isin(["CTL", "AD"])]
        te_all = meta.index[meta["cohort"] == test_c]
        te = te_all[meta.loc[te_all, "disease"].isin(["CTL", "AD"])]
        y_tr, y_te = binary_target(meta.loc[tr, "disease"]), binary_target(meta.loc[te, "disease"])

        search = tune(spec, Xz.loc[tr], y_tr)
        threshold, oof_tr = training_threshold(search.best_estimator_, Xz.loc[tr], y_tr)
        model = search.best_estimator_
        print(f"  Best params (5-fold CV in {train_c}): {search.best_params_}")
        print(f"  Inner-CV AUC in {train_c}: {search.best_score_:.4f} | "
              f"training OOF AUC: {roc_auc_score(y_tr, oof_tr):.4f} | Youden threshold: {threshold:.3f}")

        proba_all = pd.Series(model.predict_proba(Xz.loc[te_all])[:, 1], index=te_all)
        p_te = proba_all.loc[te].values
        metrics = external_metrics_table(y_te.values, p_te, threshold)
        metrics.insert(0, "Direction", f"train {train_c} -> test {test_c}")
        m = binary_metrics(y_te, p_te, threshold)
        print(f"  External {test_c}: AUC {m['AUC']:.4f} | Acc {m['Accuracy']:.4f} | "
              f"Sens {m['Sensitivity']:.4f} | Spec {m['Specificity']:.4f} | "
              f"PPV {m['PPV']:.4f} | NPV {m['NPV']:.4f} | MCC {m['MCC']:.4f}")
        print(classification_report(y_te, (p_te >= threshold).astype(int), target_names=["CTL", "AD"], digits=4))
        plot_binary_diagnostics(y_te, p_te, threshold, f"{spec.title}: train {train_c}, test {test_c}",
                                outdir / f"{tag}_external")

        perm = permutation_importance(model, Xz.loc[te], y_te, scoring="roc_auc",
                                      n_repeats=N_PERM_REPEATS, random_state=SEED, n_jobs=-1)
        imp = importance_frame(perm, list(Xz.columns))
        imp["significant"] = imp["importance_mean"] - 2 * imp["importance_sd"] > 0

        if spec.native_importance is not None:
            spec.native_importance(model).rename("importance").to_csv(outdir / f"{tag}_native_importance.csv")
        if spec.extra_report is not None:
            spec.extra_report(model, tag, outdir)

        results[(train_c, test_c)] = {
            "search": search, "model": model, "threshold": threshold, "metrics": metrics,
            "proba_all": proba_all, "importance": imp,
        }
    return results


def repeated_nested_cv(spec: ModelSpec, X, meta):
    """Repeated stratified nested CV inside each blood cohort (raw features, scaler inside the pipeline)."""
    fold_rows, imp_rows, oof_rows = [], [], []
    for cohort in BLOOD_COHORTS:
        idx = meta.index[(meta["cohort"] == cohort) & meta["disease"].isin(["CTL", "AD"])]
        Xc, yc = X.loc[idx], binary_target(meta.loc[idx, "disease"])
        for r in range(N_REPEATS):
            outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + r)
            for f, (tr, te) in enumerate(outer.split(Xc, yc)):
                search = tune(spec, Xc.iloc[tr], yc.iloc[tr])
                threshold, _ = training_threshold(search.best_estimator_, Xc.iloc[tr], yc.iloc[tr])
                p = search.predict_proba(Xc.iloc[te])[:, 1]
                row = binary_metrics(yc.iloc[te], p, threshold)
                row.update(cohort=cohort, repeat=r, fold=f, threshold=threshold,
                           n_train=len(tr), n_test=len(te), best_params=str(search.best_params_))
                fold_rows.append(row)

                perm = permutation_importance(search.best_estimator_, Xc.iloc[te], yc.iloc[te], scoring="roc_auc",
                                              n_repeats=10, random_state=SEED, n_jobs=-1)
                imp = importance_frame(perm, list(Xc.columns)).assign(cohort=cohort, repeat=r, fold=f)
                imp["rank"] = imp["importance_mean"].rank(ascending=False, method="min")
                imp_rows.append(imp)

                oof_rows.append(pd.DataFrame({
                    "sample": Xc.index[te], "cohort": cohort, "repeat": r, "fold": f,
                    "y": yc.iloc[te].values, "proba": p, "threshold": threshold,
                }))
            done = pd.DataFrame(fold_rows)
            done = done[(done["cohort"] == cohort) & (done["repeat"] == r)]
            print(f"  {cohort} repeat {r + 1}/{N_REPEATS}: mean outer AUC {done['AUC'].mean():.4f}")
    return pd.DataFrame(fold_rows), pd.concat(imp_rows, ignore_index=True), pd.concat(oof_rows, ignore_index=True)


def summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["AUC", "PR_AUC", "Brier", "Accuracy", "Balanced_accuracy", "Sensitivity",
               "Specificity", "PPV", "NPV", "F1", "MCC"]
    rows = []
    for cohort, g in folds.groupby("cohort"):
        for mname in metrics:
            v = g[mname].dropna()
            rows.append({"cohort": cohort, "Metric": mname, "mean": v.mean(), "sd": v.std(),
                         "p2.5": v.quantile(0.025), "p97.5": v.quantile(0.975), "n_folds": len(v)})
    return pd.DataFrame(rows)


def feature_stability(imp: pd.DataFrame) -> pd.DataFrame:
    n_folds = imp.groupby(["cohort", "repeat", "fold"]).ngroups
    return (imp.groupby("gene")
            .agg(mean_importance=("importance_mean", "mean"),
                 mean_rank=("rank", "mean"),
                 sd_rank=("rank", "std"),
                 freq_top_k=("rank", lambda r: (r <= STABILITY_TOP_K).sum() / n_folds),
                 freq_positive=("importance_mean", lambda v: (v > 0).sum() / n_folds))
            .sort_values("mean_rank")
            .rename(columns={"freq_top_k": f"freq_top{STABILITY_TOP_K}"}))


def brain_validation(spec: ModelSpec, Xz_blood, meta_blood, outdir: Path):
    Xb, mb, missing = load_brain(list(Xz_blood.columns))
    genes = list(Xb.columns)
    print(f"  Genes available in brain data: {len(genes)}/{Xz_blood.shape[1]} (missing: {missing or 'none'})")

    ext = meta_blood.index[meta_blood["disease"].isin(["CTL", "AD"])]
    y = binary_target(meta_blood.loc[ext, "disease"])
    cohorts = meta_blood.loc[ext, "cohort"]
    # Tune for cross-cohort generalisation: each fold holds out one blood cohort.
    cv = list(LeaveOneGroupOut().split(Xz_blood.loc[ext, genes], y, cohorts))
    search = tune(spec, Xz_blood.loc[ext, genes], y, cv=cv)
    print(f"  Brain model params (leave-one-cohort-out CV): {search.best_params_}, CV AUC {search.best_score_:.4f}")

    mb["predicted_P_AD"] = search.best_estimator_.predict_proba(Xb)[:, 1]
    rows = []
    for cohort, g in mb.groupby("batch"):
        independent = cohort in INDEPENDENT_BRAIN
        row = {"cohort": cohort, "independent": independent, "n": len(g)}
        braak = g.dropna(subset=["braak_numeric"])
        if braak["braak_numeric"].nunique() > 1:
            rho, p = spearmanr(braak["braak_numeric"], braak["predicted_P_AD"])
            stages = [s["predicted_P_AD"].values for _, s in braak.groupby("braak_numeric")]
            row.update(n_braak=len(braak), Spearman_rho=rho, Spearman_p=p, Kruskal_p=kruskal(*stages).pvalue)
        cc = g[g["disease"].isin(["CTL", "AD"])]
        if cc["disease"].nunique() == 2:
            yy = binary_target(cc["disease"]).values
            auc, lo, hi = delong_ci(yy, cc["predicted_P_AD"].values)
            row.update(n_CTL=int((yy == 0).sum()), n_AD=int(yy.sum()), AD_vs_CTL_AUC=auc, AUC_CI_low=lo, AUC_CI_high=hi,
                       MWU_p=mannwhitneyu(cc.loc[cc.disease == "AD", "predicted_P_AD"],
                                          cc.loc[cc.disease == "CTL", "predicted_P_AD"], alternative="greater").pvalue)
        rows.append(row)
        print(f"  {cohort} ({'independent' if independent else 'NOT independent: used in DEG discovery'}): "
              + ", ".join(f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float)))
        if cohort in DISCOVERY_BRAIN:
            for region, gr in braak.groupby("region"):
                if gr["braak_numeric"].nunique() > 1:
                    rho, p = spearmanr(gr["braak_numeric"], gr["predicted_P_AD"])
                    rows.append({"cohort": f"{cohort} | {region}", "independent": False,
                                 "n_braak": len(gr), "Spearman_rho": rho, "Spearman_p": p})

        if len(braak):
            stages = sorted(braak["braak_numeric"].unique())
            fig, ax = plt.subplots(figsize=(8, 6))
            data = [braak.loc[braak["braak_numeric"] == s, "predicted_P_AD"].values for s in stages]
            bp = ax.boxplot(data, positions=stages, patch_artist=True, widths=0.4)
            for patch, color in zip(bp["boxes"], plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(stages)))):
                patch.set_facecolor(color)
            rng = np.random.default_rng(SEED)
            for s, d in zip(stages, data):
                ax.scatter(rng.normal(s, 0.05, len(d)), d, alpha=0.6, color="black", s=20)
            reverse = {v: k.upper() for k, v in BRAAK_MAP.items()}
            ax.set_xticks(stages, [reverse[s] for s in stages])
            label = "independent" if independent else "discovery cohort, not independent"
            rho_txt = f"Spearman r = {row.get('Spearman_rho', np.nan):.3f}, p = {row.get('Spearman_p', np.nan):.3e}"
            ax.set(title=f"{spec.title}: P(AD) vs Braak stage, {cohort} ({label})\n{rho_txt}",
                   xlabel=f"Braak stage ({cohort})", ylabel="Predicted P(AD)", ylim=(-0.05, 1.05))
            ax.grid(axis="y", alpha=0.3)
            save(fig, outdir / f"{spec.name}_Braak_validation_{cohort}.tif")

    pd.DataFrame(rows).to_csv(outdir / f"{spec.name}_brain_validation_stats.csv", index=False)
    mb[["batch", "region", "disease", "braak", "braak_numeric", "predicted_P_AD"]].to_csv(
        outdir / f"{spec.name}_brain_predictions.csv")


def top_gene_boxplots(spec: ModelSpec, Xz, meta, top_genes, outdir: Path):
    ext = meta.index[meta["disease"].isin(["CTL", "AD"])]
    stats = []
    for gene in top_genes:
        row = {"gene": gene}
        for cohort in BLOOD_COHORTS + ("pooled",):
            idx = ext if cohort == "pooled" else ext[meta.loc[ext, "cohort"] == cohort]
            ctl = Xz.loc[idx[meta.loc[idx, "disease"] == "CTL"], gene]
            ad = Xz.loc[idx[meta.loc[idx, "disease"] == "AD"], gene]
            row[f"{cohort}_AD_minus_CTL_z"] = ad.mean() - ctl.mean()
            row[f"{cohort}_MWU_p"] = mannwhitneyu(ctl, ad, alternative="two-sided").pvalue
        stats.append(row)
    stats = pd.DataFrame(stats)
    for col in [c for c in stats.columns if c.endswith("_MWU_p")]:
        stats[col.replace("_p", "_q")] = multipletests(stats[col], method="fdr_bh")[1]
    stats["direction_consistent"] = (np.sign(stats[f"{BLOOD_COHORTS[0]}_AD_minus_CTL_z"])
                                     == np.sign(stats[f"{BLOOD_COHORTS[1]}_AD_minus_CTL_z"]))
    stats.to_csv(outdir / f"{spec.name}_top{len(top_genes)}_genes_AD_vs_CTL_stats.csv", index=False)

    ncols = 4
    nrows = int(np.ceil(len(top_genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, gene, (_, s) in zip(axes, top_genes, stats.iterrows()):
        ctl = Xz.loc[ext[meta.loc[ext, "disease"] == "CTL"], gene].values
        ad = Xz.loc[ext[meta.loc[ext, "disease"] == "AD"], gene].values
        bp = ax.boxplot([ctl, ad], tick_labels=["CTL", "AD"], showfliers=False, patch_artist=True)
        for patch, g in zip(bp["boxes"], ["CTL", "AD"]):
            patch.set_facecolor(GROUP_COLORS[g])
        ax.set_title(gene, fontsize=11)
        ax.set_ylabel("Expression (per-cohort z-score)")
        add_pvalue_bracket(ax, 1, 2, ctl, ad, f"q={s['pooled_MWU_q']:.3e}\n(p={s['pooled_MWU_p']:.3e})")
    for ax in axes[len(top_genes):]:
        ax.axis("off")
    fig.suptitle(f"Top {len(top_genes)} permutation-important genes ({spec.title}), both blood cohorts", fontsize=14)
    save(fig, outdir / f"{spec.name}_top{len(top_genes)}_genes_AD_vs_CTL_boxplots.tif")


# =============================================================================
# ORCHESTRATION
# =============================================================================
def run_all(spec: ModelSpec, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"{spec.title}: cross-cohort AD vs CTL classification")
    print("=" * 80)

    # ---- Data --------------------------------------------------------------
    X, meta, missing = load_blood()
    print(f"Genes used ({X.shape[1]}): {', '.join(X.columns)}")
    print(f"Genes in {GENE_LIST.name} not measured in both blood cohorts ({len(missing)}): {', '.join(missing)}")
    balance = pd.crosstab(meta["cohort"], meta["disease"]).reindex(columns=list(GROUPS))
    balance["AD_fraction_of_CTL+AD"] = balance["AD"] / (balance["AD"] + balance["CTL"])
    print("\nClass balance:\n" + balance.to_string())
    balance.to_csv(outdir / "class_balance.csv")
    Xz = zscore_by(X, meta["cohort"])

    # ---- 1. Cross-cohort validation (primary) -------------------------------
    print("\n" + "=" * 80 + "\n1. CROSS-COHORT VALIDATION (primary estimate)\n" + "=" * 80)
    cc = cross_cohort(spec, Xz, meta, outdir)
    metrics = pd.concat([r["metrics"] for r in cc.values()], ignore_index=True)
    metrics.to_csv(outdir / f"{spec.name}_external_metrics.csv", index=False)
    metrics.to_csv(COMPARISON_DIR / f"{spec.name}_external_metrics.csv", index=False)

    # Every blood sample gets a prediction from the model trained on the OTHER cohort.
    preds = []
    for (train_c, test_c), r in cc.items():
        d = meta.loc[r["proba_all"].index].copy()
        d["train_cohort"], d["predicted_proba"], d["threshold"] = train_c, r["proba_all"], r["threshold"]
        preds.append(d)
    preds = pd.concat(preds)
    preds["predicted_class"] = (preds["predicted_proba"] >= preds["threshold"]).astype(int)
    preds["predicted_label"] = preds["predicted_class"].map({0: "CTL", 1: "AD"})
    preds["true_binary"] = preds["disease"].map({"CTL": 0, "AD": 1})
    preds.rename_axis("sample").to_csv(outdir / f"all_samples_predictions_{spec.name}.csv")
    preds.rename_axis("sample").to_csv(COMPARISON_DIR / f"{spec.name}_external_predictions.csv")

    ext_mask = preds["disease"].isin(["CTL", "AD"])
    pooled = binary_metrics(preds.loc[ext_mask, "true_binary"], preds.loc[ext_mask, "predicted_proba"], 0.5)
    pooled_acc = (preds.loc[ext_mask, "predicted_class"] == preds.loc[ext_mask, "true_binary"]).mean()
    print(f"\nAll external predictions (both directions, CTL+AD, own thresholds): accuracy {pooled_acc:.4f}; "
          f"pooled AUC {pooled['AUC']:.4f} (pooling mixes two models; per-direction values are primary)")

    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.default_rng(SEED)
    for disease, marker, color, x in [("CTL", "o", "green", 0), ("MCI", "s", "gold", 0.5), ("AD", "^", "red", 1)]:
        d = preds[preds["disease"] == disease]
        ax.scatter(x + rng.normal(0, 0.02, len(d)), d["predicted_proba"], label=f"{disease} (n={len(d)})",
                   marker=marker, s=50, alpha=0.6, color=color, edgecolors="black", linewidth=0.5)
    ax.set(xlabel="True label (0=CTL, 0.5=MCI, 1=AD)", ylabel="Predicted P(AD)", xlim=(-0.2, 1.2), ylim=(-0.05, 1.05),
           title=f"{spec.title}: out-of-cohort predictions, all samples")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    save(fig, outdir / f"all_samples_prediction_scatter_{spec.name}.tif")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    prog_rows = []
    for ax, ((train_c, test_c), r) in zip(axes, cc.items()):
        d = preds[preds["train_cohort"] == train_c]
        group_boxplot(ax, d["predicted_proba"], d["disease"], r["threshold"])
        st = progression_stats(d["predicted_proba"], d["disease"])
        prog_rows.append({"test_cohort": test_c, "train_cohort": train_c, **st})
        ax.set(title=f"Test {test_c} (trained on {train_c})\nKW p={st['Kruskal_p']:.2e}, "
                     f"Spearman rho={st['Spearman_rho_CTL_MCI_AD']:.3f}", xlabel="Disease group")
    axes[0].set_ylabel("Predicted P(AD) (AD-like score)")
    fig.suptitle(f"{spec.title}: progression scores CTL -> MCI -> AD (external)")
    save(fig, outdir / f"prediction_by_category_boxplot_{spec.name}.tif")
    pd.DataFrame(prog_rows).to_csv(outdir / f"{spec.name}_progression_stats.csv", index=False)

    cm = confusion_matrix(preds.loc[ext_mask, "true_binary"], preds.loc[ext_mask, "predicted_class"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", alpha=0.7)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=20, fontweight="bold")
    ax.set_xticks([0, 1], ["Pred: CTL", "Pred: AD"])
    ax.set_yticks([0, 1], ["True: CTL", "True: AD"])
    ax.set_title(f"Confusion matrix, both external cohorts (CTL+AD)\nAccuracy: {pooled_acc:.3f}", fontweight="bold")
    fig.colorbar(im, ax=ax)
    save(fig, outdir / f"confusion_matrix_extremes_{spec.name}.tif")

    # ---- 2. Gene importance (external) -------------------------------------
    imp = pd.concat([r["importance"].assign(direction=f"train {a} -> test {b}") for (a, b), r in cc.items()])
    wide = imp.pivot(index="gene", columns="direction", values=["importance_mean", "importance_sd", "significant"])
    wide.columns = [f"{m} [{d}]" for m, d in wide.columns]
    sig_cols = [c for c in wide.columns if c.startswith("significant")]
    wide["significant_both_directions"] = wide[sig_cols].astype(bool).all(axis=1)
    wide["mean_importance"] = imp.groupby("gene")["importance_mean"].mean()
    wide = wide.sort_values("mean_importance", ascending=False)
    wide.to_csv(outdir / f"{spec.name}_permutation_importance_external.csv")
    wide.reset_index()[["gene", "mean_importance", "significant_both_directions"]].to_csv(
        COMPARISON_DIR / f"{spec.name}_significant_genes.csv", index=False)
    top_genes = wide.index[:TOP_K].tolist()
    pd.Series(top_genes, name="gene").to_csv(outdir / f"{spec.name}_top{TOP_K}_genes_ml_list.csv", index=False)
    print(f"\nTop {TOP_K} genes (mean external permutation importance): {', '.join(top_genes)}")
    print(f"Significant in both directions (mean - 2 SD > 0): "
          f"{', '.join(wide.index[wide['significant_both_directions']]) or 'none'}")

    fig, ax = plt.subplots(figsize=(9, 5))
    directions = imp["direction"].unique()
    width = 0.8 / len(directions)
    xs = np.arange(len(wide))
    for k, d in enumerate(directions):
        sub = imp[imp["direction"] == d].set_index("gene").loc[wide.index]
        ax.bar(xs + k * width, sub["importance_mean"], width, yerr=sub["importance_sd"], capsize=2, label=d)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs + width * (len(directions) - 1) / 2, wide.index, rotation=45, ha="right")
    ax.set(ylabel="Decrease in AUC when permuted", title=f"{spec.title}: permutation importance on the external cohort")
    ax.legend()
    save(fig, outdir / f"{spec.name}_permutation_importance_external.tif")

    top_gene_boxplots(spec, Xz, meta, top_genes, outdir)

    # ---- 3. Repeated nested CV (secondary) ---------------------------------
    print("\n" + "=" * 80 + f"\n2. REPEATED NESTED CV WITHIN EACH COHORT ({N_REPEATS}x{OUTER_FOLDS})\n" + "=" * 80)
    folds, fold_imp, oof = repeated_nested_cv(spec, X, meta)
    folds.to_csv(outdir / f"{spec.name}_nested_cv_folds.csv", index=False)
    folds.to_csv(COMPARISON_DIR / f"{spec.name}_nested_cv_folds.csv", index=False)
    summary = summarise_folds(folds)
    summary.to_csv(outdir / f"{spec.name}_nested_cv_summary.csv", index=False)
    print(summary[summary["Metric"].isin(["AUC", "Accuracy", "Sensitivity", "Specificity"])]
          .to_string(index=False, float_format="%.4f"))

    stability = feature_stability(fold_imp)
    stability.to_csv(outdir / f"{spec.name}_feature_stability.csv")
    print("\nFeature stability across outer folds:\n" + stability.to_string(float_format="%.3f"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(stability.index, stability[f"freq_top{STABILITY_TOP_K}"], color="steelblue")
    ax.set(ylabel=f"Fraction of outer folds in top {STABILITY_TOP_K}", ylim=(0, 1.05),
           title=f"{spec.title}: feature selection stability ({len(folds)} outer folds)")
    ax.tick_params(axis="x", rotation=45)
    save(fig, outdir / f"{spec.name}_feature_stability.tif")

    fig, ax = plt.subplots(figsize=(6, 5))
    data = [folds.loc[folds["cohort"] == c, "AUC"].values for c in BLOOD_COHORTS]
    ax.boxplot(data, tick_labels=BLOOD_COHORTS)
    for k, d in enumerate(data, 1):
        ax.scatter(np.random.default_rng(SEED).normal(k, 0.04, len(d)), d, alpha=0.6, s=18)
    ax.set(ylabel="Outer-fold AUC", title=f"{spec.title}: repeated nested CV AUC")
    ax.grid(axis="y", alpha=0.3)
    save(fig, outdir / f"{spec.name}_nested_cv_auc.tif")

    oof_first = oof[oof["repeat"] == 0]
    for cohort, g in oof_first.groupby("cohort"):
        thr = g["threshold"].median()  # thresholds differ by fold; median is used only for the plots
        plot_binary_diagnostics(g["y"].values, g["proba"].values, thr,
                                f"{spec.title}: nested-CV OOF in {cohort} (repeat 1)",
                                outdir / f"{spec.name}_{cohort}_oof")

    # ---- 4. Brain validation -----------------------------------------------
    print("\n" + "=" * 80 + "\n3. BRAIN VALIDATION (Braak stage, AD vs CTL)\n" + "=" * 80)
    brain_validation(spec, Xz, meta, outdir)

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 80 + f"\nSUMMARY: {spec.title}\n" + "=" * 80)
    key = metrics[(metrics["Threshold_rule"] == "Youden (training cohort)")
                  & metrics["Metric"].isin(["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "MCC"])]
    print(key[["Direction", "Metric", "Estimate", "CI_low", "CI_high"]].to_string(index=False, float_format="%.4f"))
    auc_cv = summary[summary["Metric"] == "AUC"]
    for _, r in auc_cv.iterrows():
        print(f"Nested CV AUC {r['cohort']}: {r['mean']:.4f} +/- {r['sd']:.4f}")
    print(f"\nOutputs: {outdir}\nComparison inputs: {COMPARISON_DIR}")
