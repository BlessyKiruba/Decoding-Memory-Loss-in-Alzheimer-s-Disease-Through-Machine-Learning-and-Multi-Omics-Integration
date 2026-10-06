"""
Independent network-based validation with AlzGenPred
=====================================================
AlzGenPred (Shukla & Singh, Sci Rep 2024;14:30294; github.com/shuklarohit815/AlzGenPred)
is a CatBoost classifier that labels genes as AD / non-AD from four topological
features of a STRING protein-protein interaction network:
AverageShortestPathLength, ClosenessCentrality, NeighborhoodConnectivity,
TopologicalCoefficient (Cytoscape NetworkAnalyzer, undirected). Its positive
training genes come from DisGeNET.

What this script does
---------------------
1. Rebuilds the published AlzGenPred model from the authors' training table
   (selected_4_features_final.csv) with the authors' hyperparameters, split and
   threshold (0.57), instead of unpickling catboost_model.pkl.
2. Reimplements the NetworkAnalyzer features in networkx and checks them against
   the authors' example (topological_features.csv) on a STRING network of the same genes.
   With STRING v11.5 the four features are reproduced exactly (r = 1.000).
3. Transferability check: recomputes the features of AlzGenPred's OWN training genes
   with the documented procedure (one network of all training genes, and separate
   networks for AD and non-AD genes) and measures how well the model recovers its own
   labels. The AlzGenPred probability depends on which genes are queried together,
   so this check shows how much weight the per-gene scores below can carry.
4. Builds a STRING v11.5 network (combined score >= 0.4, STRING's default) for the
   query context and scores every gene:
     - primary context: the memory-gene universe (Consolidated_memory_genes.txt) plus every signature gene,
       so the signature is scored in a realistic network and there is a background to test against;
     - sensitivity context: the signature genes alone (the literal AlzGenPred manual workflow).
5. Reports, per gene: AlzGenPred probability and class, and whether the gene is in
   AlzGenPred's own training set (the "source database"; such predictions are not independent).
6. Tests whether each signature is enriched for AlzGenPred AD genes relative to the
   memory-gene universe:
     - Fisher's exact test on AD calls (all genes, and excluding AlzGenPred training genes)
     - Fisher's exact test on AlzGenPred training AD genes (DisGeNET-derived)
     - Mann-Whitney test of probabilities (signature vs rest of universe)
     - permutation test of the mean probability (random and degree-matched gene sets)
   All p-values are BH-adjusted together.

Signatures tested
    MR_DEGs_26          FourOrMoreRegions.txt (the 26 MR-DEGs of the submitted manuscript)
    Candidates_5        BCL6, MDH1, GNA12, PDGFRB, VCAN
    ThreeOrMore_Updated ml_common.GENE_LIST
    ML_features         genes used by the ML models (GENE_LIST genes measured in both blood cohorts)
    ML_consensus        genes significant in all three models
                        (model_comparison_updated/ML_significant_genes_intersection.csv, from compare_models.py)

Requirements: pip install catboost networkx requests
Network access on first run (GitHub raw files, STRING API). Responses are cached in
AlzGenPred_validation/cache/ so re-runs are offline and reproducible.

Run:  python MLmicroarray/AlzGenPred_validation.py
Outputs: MLmicroarray/AlzGenPred_validation_updated/
"""

import hashlib
import io
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_common as mc  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTDIR = mc.ML_DIR / f"AlzGenPred_validation{mc.OUTPUT_SUFFIX}"
# Downloads and STRING responses are shared with the original run; cache keys include the query itself
CACHE = mc.ML_DIR / "AlzGenPred_validation" / "cache"

ALZGENPRED_COMMIT = "6ebe03317afcd41684b1c4cb53b8ea9d92069fb6"
ALZGENPRED_RAW = f"https://raw.githubusercontent.com/shuklarohit815/AlzGenPred/{ALZGENPRED_COMMIT}"
FEATURES = ["AverageShortestPathLength", "ClosenessCentrality", "NeighborhoodConnectivity", "TopologicalCoefficient"]
ALZGENPRED_THRESHOLD = 0.57  # from AlzGenPred.py

STRING_VERSION = "11.5"  # reproduces the authors' example features exactly; v12 also lacks MDH1
STRING_API = f"https://version-{STRING_VERSION.replace('.', '-')}.string-db.org/api"
STRING_SPECIES = 9606
STRING_REQUIRED_SCORE = 400
STRING_CALLER = "bioinfoProjectVIT_AlzGenPred_validation"
STRING_MAX_IDS = 2000

UNIVERSE_FILE = mc.MEMORY_LIST
MR_DEG_FILE = mc.ML_DIR / "FourOrMoreRegions.txt"
CANDIDATES = ["BCL6", "MDH1", "GNA12", "PDGFRB", "VCAN"]
ML_CONSENSUS_FILE = mc.COMPARISON_DIR / "ML_significant_genes_intersection.csv"

N_PERM = 10000
N_DEGREE_BINS = 5


# =============================================================================
# DOWNLOADS (cached)
# =============================================================================
def cached_get(url: str, name: str) -> str:
    path = CACHE / name
    if not path.exists():
        print(f"  Downloading {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_text(r.text)
    return path.read_text()


def string_post(method: str, identifiers: list[str], cache_name: str, **params) -> pd.DataFrame:
    # The cache key includes the query itself, so a changed gene list never reuses an old response
    key = hashlib.md5(("\n".join(sorted(identifiers)) + repr(sorted(params.items()))).encode()).hexdigest()[:10]
    path = CACHE / f"{Path(cache_name).stem}_{key}{Path(cache_name).suffix}"
    if not path.exists():
        data = {"identifiers": "\r".join(identifiers), "species": STRING_SPECIES,
                "caller_identity": STRING_CALLER, **params}
        r = requests.post(f"{STRING_API}/tsv/{method}", data=data, timeout=300)
        r.raise_for_status()
        path.write_text(r.text)
        time.sleep(1)  # STRING asks for one second between calls
    text = path.read_text()
    return pd.read_csv(io.StringIO(text), sep="\t") if text.strip() else pd.DataFrame()


# =============================================================================
# ALZGENPRED MODEL
# =============================================================================
def build_alzgenpred():
    """Re-create catboost_model.pkl exactly as Generate_CB_model.py does."""
    train = pd.read_csv(io.StringIO(cached_get(f"{ALZGENPRED_RAW}/selected_4_features_final.csv",
                                               "selected_4_features_final.csv")))
    x, y = train[FEATURES], train["Label"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=1)
    model = CatBoostClassifier(depth=8, iterations=250, l2_leaf_reg=3, learning_rate=0.3,
                               random_seed=0, verbose=0)
    model.fit(x_tr, y_tr)
    p = model.predict_proba(x_te)[:, 1]
    print(f"  Training table: {len(train)} genes ({int(y.sum())} AD, {int((y == 0).sum())} non-AD)")
    print(f"  Reproduced AlzGenPred on its held-out 20%: accuracy {accuracy_score(y_te, model.predict(x_te)):.4f}, "
          f"AUC {roc_auc_score(y_te, p):.4f}")
    labels = train.assign(name=train["name"].astype(str).str.upper()).set_index("name")["Label"]
    return model, labels[~labels.index.duplicated()]


# =============================================================================
# NETWORK FEATURES (Cytoscape NetworkAnalyzer, undirected)
# =============================================================================
def topological_coefficient(G: nx.Graph, n) -> float:
    """T(n) = mean over nodes m sharing >=1 neighbour with n of J(n,m), divided by k_n,
    where J(n,m) = shared neighbours (+1 if n and m are linked). 0 for k_n < 2."""
    nbrs = set(G[n])
    k = len(nbrs)
    if k < 2:
        return 0.0
    partners = {m for v in nbrs for m in G[v]} - {n}
    if not partners:
        return 0.0
    j = [len(nbrs & set(G[m])) + (1 if m in nbrs else 0) for m in partners]
    return float(np.mean(j) / k)


def network_features(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for n in G.nodes:
        dist = nx.single_source_shortest_path_length(G, n)
        d = [v for m, v in dist.items() if m != n]
        aspl = float(np.mean(d)) if d else 0.0
        k = G.degree(n)
        rows.append({
            "gene": n,
            "degree": k,
            "component_size": len(dist),
            "AverageShortestPathLength": aspl,
            "ClosenessCentrality": 1.0 / aspl if aspl > 0 else 0.0,
            "NeighborhoodConnectivity": float(np.mean([G.degree(m) for m in G[n]])) if k else 0.0,
            "TopologicalCoefficient": topological_coefficient(G, n),
        })
    return pd.DataFrame(rows).set_index("gene")


def string_edges(string_ids: list[str], tag: str) -> pd.DataFrame:
    """Edges among string_ids. STRING's network method refuses > 2000 identifiers, so larger
    sets are split into blocks and every pair of blocks is queried."""
    half = STRING_MAX_IDS // 2
    blocks = [string_ids[i:i + half] for i in range(0, len(string_ids), half)]
    if len(blocks) == 1:
        return string_post("network", string_ids, f"{tag}_string_network.tsv", required_score=STRING_REQUIRED_SCORE)
    parts = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            parts.append(string_post("network", blocks[i] + blocks[j], f"{tag}_string_network_{i}_{j}.tsv",
                                     required_score=STRING_REQUIRED_SCORE))
    return pd.concat(parts, ignore_index=True)


def string_network(genes: list[str], tag: str):
    """Map genes to STRING and return (graph on preferred names, query->preferred mapping)."""
    ids = string_post("get_string_ids", genes, f"{tag}_string_ids.tsv", limit=1, echo_query=1)
    ids["queryItem"] = ids["queryItem"].astype(str).str.upper()
    mapping = ids.drop_duplicates("queryItem").set_index("queryItem")
    edges = string_edges(mapping["stringId"].tolist(), tag)
    G = nx.Graph()
    G.add_nodes_from(mapping["preferredName"].str.upper())
    if len(edges):
        G.add_edges_from(zip(edges["preferredName_A"].str.upper(), edges["preferredName_B"].str.upper()))
    G.remove_edges_from(nx.selfloop_edges(G))
    query_to_node = mapping["preferredName"].str.upper()
    print(f"  STRING v{STRING_VERSION} ({tag}): {len(mapping)}/{len(genes)} genes mapped, "
          f"{G.number_of_edges()} edges at score >= {STRING_REQUIRED_SCORE / 1000:.1f}, "
          f"{sum(1 for n in G if G.degree(n) == 0)} isolated")
    return G, query_to_node


def check_feature_implementation():
    """Compare our features to the authors' example file on a STRING network of the same genes."""
    ref = pd.read_csv(io.StringIO(cached_get(f"{ALZGENPRED_RAW}/topological_features.csv",
                                             "topological_features.csv")))
    ref["name"] = ref["name"].astype(str).str.upper()
    ref = ref.drop_duplicates("name").set_index("name")
    G, q2n = string_network(ref.index.tolist(), "reference_example")
    ours = network_features(G)
    paired = ref.join(q2n.rename("node")).dropna(subset=["node"])
    paired = paired[paired["node"].map(ours["degree"]).fillna(0) > 0]
    mine = ours.loc[paired["node"]]
    rows = []
    for f in FEATURES:
        rows.append({"feature": f, "n_genes": len(paired),
                     "pearson_r": pearsonr(paired[f], mine[f])[0],
                     "spearman_rho": spearmanr(paired[f], mine[f])[0]})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "feature_implementation_check.csv", index=False)
    print(f"  Agreement with the authors' example features (STRING v{STRING_VERSION}):\n"
          + out.to_string(index=False, float_format="%.3f"))


def transferability_check(model, training_labels: pd.Series):
    """Can AlzGenPred recover its own training labels from features rebuilt with the documented procedure?"""
    published = pd.read_csv(CACHE / "selected_4_features_final.csv")
    published["name"] = published["name"].astype(str).str.upper()
    published = published.drop_duplicates("name").set_index("name")

    def rebuilt(genes, tag):
        G, q2n = string_network(genes, tag)
        t = pd.DataFrame({"node": q2n}).join(training_labels.rename("label")).join(network_features(G), on="node")
        return t[t["degree"] > 0]

    contexts = {
        "one network of all training genes": rebuilt(training_labels.index.tolist(), "training_joint"),
        "separate networks for AD and non-AD training genes": pd.concat(
            [rebuilt(training_labels.index[training_labels == lab].tolist(), f"training_label{lab}") for lab in (1, 0)]),
    }
    rows = []
    for name, t in contexts.items():
        p = model.predict_proba(t[FEATURES])[:, 1]
        common = t.index.intersection(published.index)
        rows.append({
            "context": name, "n_genes": len(t),
            "AUC_vs_own_labels": roc_auc_score(t["label"], p),
            "AD_call_rate_AD_genes": (p[t["label"] == 1] >= ALZGENPRED_THRESHOLD).mean(),
            "AD_call_rate_nonAD_genes": (p[t["label"] == 0] >= ALZGENPRED_THRESHOLD).mean(),
            **{f"spearman_vs_published_{f}": spearmanr(t.loc[common, f], published.loc[common, f])[0] for f in FEATURES},
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "AlzGenPred_transferability_check.csv", index=False)
    print(out.set_index("context").round(3).T.to_string())
    print("  An AUC near 0.5 means that, with features rebuilt as the AlzGenPred manual describes, the model\n"
          "  cannot separate its own AD and non-AD training genes: its calls track the query network, not the gene.")


# =============================================================================
# SCORING AND ENRICHMENT
# =============================================================================
def score_context(model, genes: list[str], tag: str) -> pd.DataFrame:
    G, q2n = string_network(genes, tag)
    feats = network_features(G)
    table = pd.DataFrame(index=pd.Index(genes, name="gene"))
    table["string_name"] = q2n.reindex(table.index)
    table = table.join(feats, on="string_name")
    table["scorable"] = table["degree"].fillna(0) > 0  # isolated nodes have no defined topology
    s = table["scorable"]
    table.loc[s, "AlzGenPred_probability"] = model.predict_proba(table.loc[s, FEATURES])[:, 1]
    table["AlzGenPred_class"] = np.where(
        ~s, "not scorable", np.where(table["AlzGenPred_probability"] >= ALZGENPRED_THRESHOLD, "AD", "non-AD"))
    return table


def permutation_p(values: pd.Series, members: pd.Index, degree: pd.Series, rng, matched: bool):
    obs = values.loc[members].mean()
    pool = values.index
    if matched:
        bins = pd.qcut(degree.loc[pool].rank(method="first"), N_DEGREE_BINS, labels=False)
        need = bins.loc[members].value_counts()
        by_bin = {b: pool[bins == b].to_numpy() for b in need.index}
        null = np.array([
            np.mean(np.concatenate([values.loc[rng.choice(by_bin[b], n, replace=False)].to_numpy()
                                    for b, n in need.items()]))
            for _ in range(N_PERM)])
    else:
        arr = values.to_numpy()
        null = np.array([arr[rng.choice(len(arr), len(members), replace=False)].mean() for _ in range(N_PERM)])
    return obs, null, (1 + (null >= obs).sum()) / (1 + N_PERM)


def fisher_row(test, name, in_set: pd.Series, positive: pd.Series):
    a = int((in_set & positive).sum())
    b = int((in_set & ~positive).sum())
    c = int((~in_set & positive).sum())
    d = int((~in_set & ~positive).sum())
    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {"signature": name, "test": test, "n_signature": a + b, "n_positive_in_signature": a,
            "n_background": c + d, "n_positive_in_background": c,
            "fraction_signature": a / (a + b) if a + b else np.nan,
            "fraction_background": c / (c + d) if c + d else np.nan,
            "odds_ratio": odds, "p": p}


def enrichment(table: pd.DataFrame, signatures: dict[str, list[str]], training_labels: pd.Series):
    rng = np.random.default_rng(mc.SEED)
    universe = table[table["in_universe"]]
    scorable = universe[universe["scorable"]]
    independent = scorable[scorable["AlzGenPred_training_label"].isna()]
    rows, nulls = [], {}
    for name, genes in signatures.items():
        members = [g for g in genes if g in universe.index]
        if not members:
            continue
        rows.append(fisher_row("AD call (all scorable genes)", name,
                               scorable.index.isin(members), scorable["AlzGenPred_class"].eq("AD").to_numpy()))
        rows.append(fisher_row("AD call (excluding AlzGenPred training genes)", name,
                               independent.index.isin(members), independent["AlzGenPred_class"].eq("AD").to_numpy()))
        rows.append(fisher_row("AlzGenPred training AD gene (DisGeNET-derived)", name,
                               universe.index.isin(members), universe["AlzGenPred_training_label"].eq(1).to_numpy()))

        sc_members = scorable.index.intersection(members)
        if len(sc_members) >= 2:
            probs = scorable["AlzGenPred_probability"]
            rest = probs.drop(sc_members)
            rows.append({"signature": name, "test": "Mann-Whitney probability (signature > rest)",
                         "n_signature": len(sc_members), "n_background": len(rest),
                         "median_signature": probs.loc[sc_members].median(), "median_background": rest.median(),
                         "p": mannwhitneyu(probs.loc[sc_members], rest, alternative="greater").pvalue})
            for matched in (False, True):
                label = "degree-matched" if matched else "random"
                obs, null, p = permutation_p(probs, sc_members, scorable["degree"], rng, matched)
                nulls[(name, label)] = (obs, null)
                rows.append({"signature": name, "test": f"Permutation mean probability ({label} sets)",
                             "n_signature": len(sc_members), "n_background": len(scorable),
                             "observed_mean": obs, "null_mean": null.mean(), "p": p})
    out = pd.DataFrame(rows)
    out["q_BH"] = multipletests(out["p"], method="fdr_bh")[1]
    return out, nulls


# =============================================================================
# PLOTS
# =============================================================================
def plot_probabilities(table, signatures):
    universe = table[table["in_universe"] & table["scorable"]]
    groups = {"Memory-gene universe": universe["AlzGenPred_probability"]}
    for name, genes in signatures.items():
        groups[name] = table.loc[[g for g in genes if g in table.index and table.at[g, "scorable"]],
                                 "AlzGenPred_probability"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot([v.values for v in groups.values()], tick_labels=list(groups), showfliers=False)
    rng = np.random.default_rng(mc.SEED)
    for k, v in enumerate(groups.values(), 1):
        ax.scatter(rng.normal(k, 0.05, len(v)), v, s=12 if k == 1 else 30, alpha=0.3 if k == 1 else 0.8)
    ax.axhline(ALZGENPRED_THRESHOLD, ls="--", color="red", label=f"AlzGenPred threshold {ALZGENPRED_THRESHOLD}")
    ax.set(ylabel="AlzGenPred P(AD gene)", title="AlzGenPred probabilities: signatures vs memory-gene universe")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    mc.save(fig, OUTDIR / "AlzGenPred_probability_by_signature.tif")


def plot_genes(table, genes, name):
    d = table.loc[[g for g in genes if g in table.index]].sort_values("AlzGenPred_probability", na_position="first")
    colors = ["lightgray" if not s else ("firebrick" if c == "AD" else "steelblue")
              for s, c in zip(d["scorable"], d["AlzGenPred_class"])]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(d) + 1)))
    ax.barh(d.index, d["AlzGenPred_probability"].fillna(0), color=colors)
    for y, (g, r) in enumerate(d.iterrows()):
        mark = {1: " [training: AD]", 0: " [training: non-AD]"}.get(r["AlzGenPred_training_label"], "")
        txt = "not scorable (no STRING interactions)" if not r["scorable"] else f"{r['AlzGenPred_probability']:.2f}"
        ax.text(r["AlzGenPred_probability"] if r["scorable"] else 0.01, y, f" {txt}{mark}", va="center", fontsize=7)
    ax.axvline(ALZGENPRED_THRESHOLD, ls="--", color="red")
    ax.set(xlim=(0, 1.35), xlabel="AlzGenPred P(AD gene)", title=f"AlzGenPred per-gene results: {name}")
    mc.save(fig, OUTDIR / f"AlzGenPred_genes_{name}.tif")


def plot_nulls(nulls):
    if not nulls:
        return
    fig, axes = plt.subplots(len(nulls) // 2 + len(nulls) % 2, 2, figsize=(11, 3.2 * (len(nulls) // 2 + 1)))
    axes = np.atleast_1d(axes).flatten()
    for ax, ((name, label), (obs, null)) in zip(axes, nulls.items()):
        ax.hist(null, bins=50, color="lightgray")
        ax.axvline(obs, color="red", lw=2)
        ax.set_title(f"{name}: {label} null (observed in red)", fontsize=9)
        ax.set_xlabel("Mean AlzGenPred probability")
    for ax in axes[len(nulls):]:
        ax.axis("off")
    mc.save(fig, OUTDIR / "AlzGenPred_permutation_nulls.tif")


# =============================================================================
# MAIN
# =============================================================================
def load_signatures() -> dict[str, list[str]]:
    sigs = {"MR_DEGs_26": mc.read_gene_list(MR_DEG_FILE), "Candidates_5": CANDIDATES,
            mc.GENE_LIST.stem: mc.read_gene_list(mc.GENE_LIST)}
    X, _, _ = mc.load_blood()
    sigs["ML_features"] = list(X.columns)
    if ML_CONSENSUS_FILE.exists():
        cons = pd.read_csv(ML_CONSENSUS_FILE)
        genes = cons.loc[cons["n_models"] == cons["n_models"].max(), "gene"].tolist()
        if len(genes) and cons["n_models"].max() == 3:
            sigs["ML_consensus"] = genes
        else:
            print(f"  No gene is significant in all three models; ML_consensus uses genes significant in "
                  f"{cons['n_models'].max()} model(s): {', '.join(genes)}")
            sigs[f"ML_consensus_{cons['n_models'].max()}models"] = genes
    else:
        print(f"  {ML_CONSENSUS_FILE.name} not found - run the three ML scripts and compare_models.py "
              f"to include the ML consensus genes")
    return sigs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    print("=" * 80 + "\n1. REBUILD ALZGENPRED\n" + "=" * 80)
    model, training_labels = build_alzgenpred()

    print("\n" + "=" * 80 + "\n2. CHECK NETWORK FEATURE IMPLEMENTATION\n" + "=" * 80)
    check_feature_implementation()

    print("\n" + "=" * 80 + "\n3. TRANSFERABILITY: RECOVER ALZGENPRED'S OWN TRAINING LABELS\n" + "=" * 80)
    transferability_check(model, training_labels)

    print("\n" + "=" * 80 + "\n4. SCORE GENES\n" + "=" * 80)
    signatures = load_signatures()
    for name, genes in signatures.items():
        print(f"  {name} ({len(genes)}): {', '.join(genes)}")
    universe = mc.read_gene_list(UNIVERSE_FILE)
    signature_genes = list(dict.fromkeys(g for genes in signatures.values() for g in genes))
    print(f"  Memory-gene universe ({UNIVERSE_FILE.name}): {len(universe)} genes")

    primary = score_context(model, list(dict.fromkeys(universe + signature_genes)), "universe_context")
    primary["in_universe"] = primary.index.isin(universe)
    signature_only = score_context(model, signature_genes, "signature_only_context")

    table = primary.copy()
    table["AlzGenPred_training_label"] = table.index.map(training_labels)
    for name, genes in signatures.items():
        table[f"in_{name}"] = table.index.isin(genes)
    table["signature_only_probability"] = signature_only["AlzGenPred_probability"].reindex(table.index)
    table["signature_only_class"] = signature_only["AlzGenPred_class"].reindex(table.index)
    table.to_csv(OUTDIR / "AlzGenPred_all_scored_genes.csv")

    report_cols = (["string_name", "degree", *FEATURES, "AlzGenPred_probability", "AlzGenPred_class",
                    "signature_only_probability", "signature_only_class", "AlzGenPred_training_label", "in_universe"]
                   + [f"in_{n}" for n in signatures])
    sig_table = table.loc[signature_genes, report_cols]
    sig_table.to_csv(OUTDIR / "AlzGenPred_signature_genes.csv")
    print("\n  Signature genes (primary context = memory-gene network; training label: 1 = AlzGenPred AD "
          "training gene, 0 = non-AD training gene, blank = not in AlzGenPred training data):")
    print(sig_table[["degree", "AlzGenPred_probability", "AlzGenPred_class", "signature_only_probability",
                     "AlzGenPred_training_label"]].to_string(float_format="%.3f"))
    for name, genes in signatures.items():
        plot_genes(table, genes, name)

    agreement = table.dropna(subset=["AlzGenPred_probability", "signature_only_probability"])
    if len(agreement) > 2:
        rho = spearmanr(agreement["AlzGenPred_probability"], agreement["signature_only_probability"])[0]
        same = (agreement["AlzGenPred_class"] == agreement["signature_only_class"]).mean()
        print(f"\n  Context sensitivity (signature genes, universe network vs signature-only network): "
              f"Spearman rho {rho:.3f}, same class for {same:.0%} of {len(agreement)} genes")

    print("\n" + "=" * 80 + "\n5. ENRICHMENT AGAINST THE MEMORY-GENE UNIVERSE\n" + "=" * 80)
    stats, nulls = enrichment(table, signatures, training_labels)
    stats.to_csv(OUTDIR / "AlzGenPred_enrichment.csv", index=False)
    print(stats[["signature", "test", "n_signature", "n_positive_in_signature", "fraction_signature",
                 "fraction_background", "odds_ratio", "p", "q_BH"]].to_string(index=False, float_format="%.4g"))
    excluded = {n: [g for g in genes if g not in universe] for n, genes in signatures.items()}
    for n, genes in excluded.items():
        if genes:
            print(f"  Not in the universe, excluded from {n} enrichment tests: {', '.join(genes)}")

    plot_probabilities(table, signatures)
    plot_nulls(nulls)
    print(f"\nOutputs: {OUTDIR}")


if __name__ == "__main__":
    main()
