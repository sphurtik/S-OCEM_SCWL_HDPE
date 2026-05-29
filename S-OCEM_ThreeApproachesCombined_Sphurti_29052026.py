# COMPLETE THREE-APPROACH S-OCEM CODE
# To run in Google Colab
# A = equal-weight balanced candidate selection
# B = strict sensitivity pruning
# C = chemistry-constrained balanced selection

try:
    import xlsxwriter
except ImportError:
    !pip install XlsxWriter -q
    import xlsxwriter

import re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from google.colab import files
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# Basic Values 
SHEET = "Sheet1"
C_MIN, C_MAX, CUT_C = 5, 45, 20
T_REF = 25.0
FIG_DPI = 500
FAST_MODE = False

DATA_DIR = Path("relative_severity_reference_HDPE_SCWL_LKM_files")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("three_approaches_OCEM_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "plots"; PLOT_DIR.mkdir(exist_ok=True)
BAR_DIR = OUTPUT_DIR / "bar_plots"; BAR_DIR.mkdir(exist_ok=True)
PIONA_DIR = OUTPUT_DIR / "PIONA_bar_plots"; PIONA_DIR.mkdir(exist_ok=True)
PARITY_DIR = OUTPUT_DIR / "parity_plots"; PARITY_DIR.mkdir(exist_ok=True)

# Separate approach-specific plot folders
A_PLOT_DIR = OUTPUT_DIR / "Approach_A_separate_plots"; A_PLOT_DIR.mkdir(exist_ok=True)
B_PLOT_DIR = OUTPUT_DIR / "Approach_B_separate_plots"; B_PLOT_DIR.mkdir(exist_ok=True)
C_PLOT_DIR = OUTPUT_DIR / "Approach_C_separate_plots"; C_PLOT_DIR.mkdir(exist_ok=True)
COMBINED_PLOT_DIR = OUTPUT_DIR / "Combined_comparison_plots"; COMBINED_PLOT_DIR.mkdir(exist_ok=True)

if FAST_MODE:
    FULL_STARTS, CANDIDATE_STARTS, FINAL_STARTS = 4, 3, 6
    FULL_MAX_NFEV, CANDIDATE_MAX_NFEV, FINAL_MAX_NFEV = 600, 500, 900
    N_RANDOM_MODELS = 30
else:
    FULL_STARTS, CANDIDATE_STARTS, FINAL_STARTS = 8, 5, 15
    FULL_MAX_NFEV, CANDIDATE_MAX_NFEV, FINAL_MAX_NFEV = 1000, 700, 1800
    N_RANDOM_MODELS = 120

RANDOM_SIZES = [8, 10, 12, 14, 16, 18, 20]
NEAR_RMSE_RELAXATION = 1.10
COMPLEXITY_WEIGHT_A = 1.0
COMPLEXITY_WEIGHT_C = 1.0
CHEMISTRY_WEIGHT_C = 1.0
MIN_K_KEEP = 1e-2
MIN_RELATIVE_DELTA_LOSS_KEEP = 0.05
TARGET_PATHWAY = ("LNAP", "aroma")

POOLS = ["HP", "HIP", "LP", "LIP", "HO", "HNAP", "LO", "LNAP", "aroma"]
PIONA_SHORT = ["P", "iP", "O", "N", "A"]
POOL_INDEX = {p: i for i, p in enumerate(POOLS)}
FAMILY_MAP = {"Paraffins": "P", "Isoparaffins": "iP", "Olefins": "O", "Naphthenes": "N", "Aromatics": "A"}
PATTERN = re.compile(r"^(?P<T>\d+)_+(?P<heat>\d+)_+(?P<hold>\d+)_+(?P<label>.+)\.xlsx$", re.IGNORECASE)

ESSENTIAL_PATHWAYS_FOR_SCORE_C = [
    ("HP", "LP"), ("HIP", "LIP"),
    ("HP", "HO"), ("LP", "LO"),
    ("HO", "HNAP"), ("LO", "LNAP"),
    ("HNAP", "aroma"), ("LNAP", "aroma"),
]
ESSENTIAL_PATHWAYS_FORCE_C = [("HNAP", "aroma"), ("LNAP", "aroma")]

#  Defining  DATA FUNCTIONS 
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def parse_filename(name):
    m = PATTERN.match(name)
    if not m:
        raise ValueError(f"Bad filename: {name}; expected T_heat_hold_label.xlsx")
    d = m.groupdict()
    return float(d["T"]), float(d["heat"]), float(d["hold"]), d["label"]

def read_piona(fp):
    df = pd.read_excel(fp, sheet_name=SHEET)
    col_norm = {str(c).strip().lower(): c for c in df.columns}
    if "carbonnumber" in col_norm:
        cn = col_norm["carbonnumber"]
    elif "carbon number" in col_norm:
        cn = col_norm["carbon number"]
    else:
        raise ValueError(f"{fp.name}: Carbon number column not found")
    missing = [k for k in FAMILY_MAP if k not in df.columns]
    if missing: raise ValueError(f"{fp.name}: missing columns {missing}")
    df = df[[cn] + list(FAMILY_MAP.keys())].copy()
    df[cn] = df[cn].astype(str).str.replace(r"[^\d\.]", "", regex=True)
    df[cn] = pd.to_numeric(df[cn], errors="coerce")
    df = df.dropna(subset=[cn]).copy()
    df[cn] = df[cn].astype(int)
    df = df[(df[cn] >= C_MIN) & (df[cn] <= C_MAX)].copy().rename(columns={cn: "C"})
    long = df.melt(id_vars=["C"], var_name="fam_long", value_name="val")
    long["family"] = long["fam_long"].map(FAMILY_MAP)
    long["val"] = pd.to_numeric(long["val"], errors="coerce").fillna(0.0)
    long.loc[long["val"] < 0, "val"] = 0.0
    return long[["family", "C", "val"]]

def get_carbon_distribution(long):
    cd = long.groupby("C", as_index=False)["val"].sum()
    cd["w"] = cd["val"] / max(cd["val"].sum(), 1e-12)
    return cd[["C", "w"]]

def get_piona_distribution(long):
    fam = long.groupby("family")["val"].sum().reindex(PIONA_SHORT).fillna(0.0)
    return fam / max(fam.sum(), 1e-12)

def piona_to_pools(long, cut_c=CUT_C):
    total = long["val"].sum()
    if total <= 0: return np.ones(len(POOLS)) / len(POOLS)
    long = long.copy(); long["w"] = long["val"] / total
    def sum_family(fam, cmin=None, cmax=None):
        sub = long[long["family"] == fam]
        if cmin is not None: sub = sub[sub["C"] >= cmin]
        if cmax is not None: sub = sub[sub["C"] <= cmax]
        return float(sub["w"].sum())
    v = np.array([
        sum_family("P", cmin=cut_c), sum_family("iP", cmin=cut_c),
        sum_family("P", cmax=cut_c-1), sum_family("iP", cmax=cut_c-1),
        sum_family("O", cmin=cut_c), sum_family("N", cmin=cut_c),
        sum_family("O", cmax=cut_c-1), sum_family("N", cmax=cut_c-1),
        sum_family("A")], dtype=float)
    v = np.maximum(v, 0.0)
    return v / max(v.sum(), 1e-12)

# All  Pathways
PATHWAYS_LARGE = [
    ("HP","LP"),("HIP","LIP"),("HO","LO"),("HNAP","LNAP"),
    ("HP","HIP"),("HIP","HP"),("LP","LIP"),("LIP","LP"),
    ("HP","HO"),("HO","HP"),("LP","LO"),("LO","LP"),
    ("HO","HNAP"),("HNAP","HO"),("LO","LNAP"),("LNAP","LO"),
    ("HNAP","HP"),("LNAP","LP"),("HP","HNAP"),("LP","LNAP"),
    ("HIP","HO"),("LIP","LO"),("HIP","HNAP"),("LIP","LNAP"),
    ("HO","aroma"),("LO","aroma"),("HNAP","aroma"),("LNAP","aroma"),
    ("HP","aroma"),("HIP","aroma"),("LP","aroma"),("LIP","aroma"),
    ("HP","LIP"),("HIP","LP"),("HO","LNAP"),("HNAP","LO"),
]
if len(PATHWAYS_LARGE) != len(set(PATHWAYS_LARGE)): raise ValueError("Duplicate pathways")

def pathway_category(src, dst):
    p = (src, dst)
    if p in [("HP","LP"),("HIP","LIP"),("HO","LO"),("HNAP","LNAP")]: return "heavy_to_light_cracking"
    if p in [("HP","HIP"),("HIP","HP"),("LP","LIP"),("LIP","LP")]: return "paraffin_iso_interconversion"
    if p in [("HP","HO"),("HO","HP"),("LP","LO"),("LO","LP")]: return "paraffin_olefin_interconversion"
    if p in [("HO","HNAP"),("HNAP","HO"),("LO","LNAP"),("LNAP","LO")]: return "olefin_naphthene_interconversion"
    if p in [("HNAP","HP"),("LNAP","LP")]: return "naphthene_to_paraffin"
    if p in [("HP","HNAP"),("LP","LNAP")]: return "paraffin_to_naphthene"
    if p in [("HIP","HO"),("LIP","LO")]: return "iso_to_olefin"
    if p in [("HIP","HNAP"),("LIP","LNAP")]: return "iso_to_naphthene"
    if p in [("HO","aroma"),("LO","aroma")]: return "aromatization_from_olefin"
    if p in [("HNAP","aroma"),("LNAP","aroma")]: return "aromatization_from_naphthene"
    if p in [("HP","aroma"),("HIP","aroma"),("LP","aroma"),("LIP","aroma")]: return "aromatization_from_paraffin_iso"
    if p in [("HP","LIP"),("HIP","LP"),("HO","LNAP"),("HNAP","LO")]: return "cross_pool_redistribution"
    return "uncategorized"

#  Data input and reading files
print("Upload Excel files named like: 430_60_120_SCW-04.xlsx")
if IN_COLAB:
    uploaded = files.upload()
    for fn in uploaded.keys():
        src = Path(fn); dst = DATA_DIR / fn
        if dst.exists(): dst.unlink()
        src.rename(dst)

meta_rows, carbon_rows, piona_rows, lump_rows = [], [], [], []
for fp in sorted(DATA_DIR.glob("*.xlsx")):
    Tset, theat, thold, label = parse_filename(fp.name)
    long = read_piona(fp); run_id = fp.stem
    meta_rows.append({"run_id": run_id, "file": fp.name, "label": label, "T_set_C": Tset, "t_heat_min": theat, "t_hold_min": thold})
    cd = get_carbon_distribution(long); cd["run_id"] = run_id; carbon_rows.append(cd)
    fam = get_piona_distribution(long).reset_index(); fam.columns = ["family", "fraction"]; fam["run_id"] = run_id; piona_rows.append(fam)
    lump_rows.append(pd.DataFrame({"pool": POOLS, "fraction": piona_to_pools(long), "run_id": run_id}))

meta = pd.DataFrame(meta_rows)
if len(meta) == 0: raise RuntimeError("No valid files processed")
carbon_df = pd.concat(carbon_rows, ignore_index=True)
piona_df = pd.concat(piona_rows, ignore_index=True)
lump_df = pd.concat(lump_rows, ignore_index=True)
lump_wide = lump_df.pivot(index="run_id", columns="pool", values="fraction").reset_index()
lkm_df = meta.merge(lump_wide, on="run_id", how="left")
Y_exp = lkm_df[POOLS].values.astype(float)
Y_exp = np.maximum(Y_exp, 0.0)
Y_exp = Y_exp / np.maximum(Y_exp.sum(axis=1, keepdims=True), 1e-12)
POOL_SCALE = Y_exp.mean(axis=0) + 0.01
T_data = lkm_df["T_set_C"].values.astype(float)
t_heat_data = lkm_df["t_heat_min"].values.astype(float)
t_hold_data = lkm_df["t_hold_min"].values.astype(float)
scale_table = pd.DataFrame({"pool": POOLS, "mean_exp_fraction": Y_exp.mean(axis=0), "pool_scale_used": POOL_SCALE})
display(meta); display(scale_table)

#  Defining model functions 
def build_K(k, pathways):
    K = np.zeros((len(POOLS), len(POOLS)))
    for rate, (src, dst) in zip(k, pathways):
        i, j = POOL_INDEX[src], POOL_INDEX[dst]
        K[j, i] += rate; K[i, i] -= rate
    return K

def calculate_severity(alpha, beta):
    S_raw = (t_hold_data + alpha * t_heat_data) * np.exp(beta * (T_data - T_REF))
    S_ref = np.min(S_raw)
    return S_raw / max(S_ref, 1e-12), S_raw, S_ref

def predict_lkm(theta, pathways):
    logit_alpha, beta = theta[0], theta[1]
    k = theta[2:]
    alpha = sigmoid(logit_alpha)
    S_rel, S_raw, S_ref = calculate_severity(alpha, beta)
    order = np.argsort(S_rel)
    y0 = Y_exp[order[0]].copy(); y0 = y0 / max(y0.sum(), 1e-12)
    K = build_K(k, pathways)
    Y_pred = np.zeros_like(Y_exp)
    for idx in order:
        dS = S_rel[idx] - 1.0
        y = expm(K * dS) @ y0
        y = np.maximum(y, 0.0); y = y / max(y.sum(), 1e-12)
        Y_pred[idx, :] = y
    return Y_pred, S_rel, S_raw, S_ref, alpha, beta

def residual_lkm(theta, pathways, rate_reg=1e-3, beta_reg=1e-2):
    Y_pred, S_rel, S_raw, S_ref, alpha, beta = predict_lkm(theta, pathways)
    comp_res = ((Y_pred - Y_exp) / POOL_SCALE).ravel()
    k = theta[2:]
    return np.concatenate([comp_res, rate_reg * k, np.array([beta_reg * beta])])

def objective_lkm(theta, pathways): return np.sum(residual_lkm(theta, pathways) ** 2)

def fit_lkm(pathways, n_starts=5, seed=42, max_nfev=700, verbose=True):
    rng = np.random.default_rng(seed); n_k = len(pathways)
    lower = np.array([-6.0, 0.0001] + [0.0] * n_k)
    upper = np.array([6.0, 0.1000] + [20.0] * n_k)
    starts = []
    for beta0 in [0.005, 0.01, 0.02, 0.04]:
        x0 = np.zeros(2 + n_k); x0[1] = beta0; x0[2:] = 0.5; starts.append(x0)
    for _ in range(n_starts):
        x0 = lower + rng.random(2 + n_k) * (upper - lower)
        x0[2:] = np.clip(rng.lognormal(mean=-1.0, sigma=1.0, size=n_k), 0.0, 20.0)
        starts.append(x0)
    best = None
    if verbose: print(f"Fitting {len(pathways)} pathways with {len(starts)} starts...")
    for ii, x0 in enumerate(starts):
        res = least_squares(lambda theta: residual_lkm(theta, pathways), x0, bounds=(lower, upper), max_nfev=max_nfev, xtol=1e-5, ftol=1e-5, gtol=1e-5)
        loss = objective_lkm(res.x, pathways)
        if best is None or loss < best["loss"]:
            Y_pred, S_rel, S_raw, S_ref, alpha_fit, beta_fit = predict_lkm(res.x, pathways)
            best = {"theta": res.x, "Y_pred": Y_pred, "S_rel": S_rel, "S_raw": S_raw, "S_ref": S_ref, "alpha": alpha_fit, "beta": beta_fit, "k": res.x[2:], "pathways": pathways, "loss": loss, "ls_result": res}
        if verbose and (ii + 1) % 5 == 0: print(f"  completed {ii+1}/{len(starts)} starts | best loss = {best['loss']:.4e}")
    return best

#  Model performance metrics
def make_metrics(Y_true, Y_model):
    rows = []
    for i, pool in enumerate(POOLS):
        rmse = np.sqrt(mean_squared_error(Y_true[:, i], Y_model[:, i]))
        rows.append({"pool": pool, "R2": r2_score(Y_true[:, i], Y_model[:, i]), "MAE": mean_absolute_error(Y_true[:, i], Y_model[:, i]), "RMSE": rmse, "Max_abs_error": np.max(np.abs(Y_true[:, i] - Y_model[:, i])), "Bias_exp_minus_pred": np.mean(Y_true[:, i] - Y_model[:, i]), "Normalized_RMSE_by_pool_scale": rmse / POOL_SCALE[i]})
    metrics_df = pd.DataFrame(rows)
    overall_df = pd.DataFrame({"metric": ["overall_R2", "overall_MAE", "overall_RMSE", "overall_Max_abs_error", "overall_scaled_RMSE"], "value": [r2_score(Y_true.ravel(), Y_model.ravel()), mean_absolute_error(Y_true.ravel(), Y_model.ravel()), np.sqrt(mean_squared_error(Y_true.ravel(), Y_model.ravel())), np.max(np.abs(Y_true.ravel() - Y_model.ravel())), np.sqrt(np.mean(((Y_model - Y_true) / POOL_SCALE) ** 2))]})
    return metrics_df, overall_df

def overall_value(overall_df, metric_name): return float(overall_df.loc[overall_df["metric"] == metric_name, "value"].iloc[0])

def model_information_criteria(Y_true, Y_pred, n_params):
    n_obs = Y_true.size; rss = np.sum((Y_true - Y_pred) ** 2)
    return rss, n_obs * np.log(rss / n_obs + 1e-12) + 2 * n_params, n_obs * np.log(rss / n_obs + 1e-12) + n_params * np.log(n_obs)

def make_rate_table(pathways, k, model_name):
    return pd.DataFrame({"model": model_name, "source": [p[0] for p in pathways], "target": [p[1] for p in pathways], "pathway": [f"{p[0]}->{p[1]}" for p in pathways], "category": [pathway_category(p[0], p[1]) for p in pathways], "k_fit": k})

def pathway_flux_table(fit_result, pathways, model_name):
    rows = []
    for r in range(len(fit_result["Y_pred"])):
        for k, (src, dst) in zip(fit_result["k"], pathways):
            src_i = POOL_INDEX[src]
            rows.append({"model": model_name, "run_id": lkm_df.loc[r, "run_id"], "label": lkm_df.loc[r, "label"], "S_rel": fit_result["S_rel"][r], "source": src, "target": dst, "pathway": f"{src}->{dst}", "category": pathway_category(src, dst), "k_fit": k, "source_fraction": fit_result["Y_pred"][r, src_i], "flux": k * fit_result["Y_pred"][r, src_i]})
    return pd.DataFrame(rows)

def flux_summary(flux_df):
    return flux_df.groupby(["model", "pathway", "source", "target", "category"], as_index=False).agg(mean_flux=("flux", "mean"), max_flux=("flux", "max"), total_flux=("flux", "sum"), mean_source_fraction=("source_fraction", "mean"), k_fit=("k_fit", "first")).sort_values(["model", "total_flux"], ascending=[True, False]).reset_index(drop=True)

def sensitivity_analysis(fit_result, pathways):
    theta_fit = fit_result["theta"]; base_loss = objective_lkm(theta_fit, pathways); rows = []
    for i, (src, dst) in enumerate(pathways):
        theta_test = theta_fit.copy(); theta_test[2 + i] = 0.0
        new_loss = objective_lkm(theta_test, pathways); delta_loss = new_loss - base_loss
        rows.append({"pathway": f"{src}->{dst}", "source": src, "target": dst, "category": pathway_category(src, dst), "k_fit": theta_fit[2 + i], "loss_without_pathway": new_loss, "delta_loss": delta_loss, "relative_delta_loss": delta_loss / max(base_loss, 1e-12)})
    return pd.DataFrame(rows).sort_values(["relative_delta_loss", "k_fit"], ascending=[False, False]).reset_index(drop=True)

def make_fit_table(fit_result, model_name):
    tab = lkm_df[["run_id", "label", "T_set_C", "t_heat_min", "t_hold_min"]].copy()
    tab["model"] = model_name; tab["severity_raw"] = fit_result["S_raw"]; tab["severity_reference_raw"] = fit_result["S_ref"]; tab["relative_severity_S_over_Sref"] = fit_result["S_rel"]
    for i, pool in enumerate(POOLS):
        tab[f"{pool}_exp"] = Y_exp[:, i]; tab[f"{pool}_pred"] = fit_result["Y_pred"][:, i]; tab[f"{pool}_error_pred_minus_exp"] = fit_result["Y_pred"][:, i] - Y_exp[:, i]
    return tab

def make_diagnostics_long(fit_result, model_name):
    rows = []
    for r in range(len(Y_exp)):
        for i, pool in enumerate(POOLS):
            rows.append({"model": model_name, "run_id": lkm_df.loc[r, "run_id"], "label": lkm_df.loc[r, "label"], "T_set_C": lkm_df.loc[r, "T_set_C"], "t_heat_min": lkm_df.loc[r, "t_heat_min"], "t_hold_min": lkm_df.loc[r, "t_hold_min"], "severity_raw": fit_result["S_raw"][r], "severity_reference_raw": fit_result["S_ref"], "relative_severity_S_over_Sref": fit_result["S_rel"][r], "pool": pool, "experimental": Y_exp[r, i], "predicted": fit_result["Y_pred"][r, i], "residual_exp_minus_pred": Y_exp[r, i] - fit_result["Y_pred"][r, i], "scaled_residual_pred_minus_exp": (fit_result["Y_pred"][r, i] - Y_exp[r, i]) / POOL_SCALE[i], "absolute_error": abs(Y_exp[r, i] - fit_result["Y_pred"][r, i])})
    return pd.DataFrame(rows)

# Global pathway full model 
print("\nSTEP 1: FIT FULL 36-PATHWAY MODEL")
full_fit = fit_lkm(PATHWAYS_LARGE, n_starts=FULL_STARTS, seed=100, max_nfev=FULL_MAX_NFEV)
full_metrics_df, full_overall_df = make_metrics(Y_exp, full_fit["Y_pred"])
full_rate_df = make_rate_table(PATHWAYS_LARGE, full_fit["k"], "full_36")
full_flux_df = pathway_flux_table(full_fit, PATHWAYS_LARGE, "full_36")
full_flux_summary_df = flux_summary(full_flux_df)
full_sensitivity_df = sensitivity_analysis(full_fit, PATHWAYS_LARGE)
full_rss, full_AIC, full_BIC = model_information_criteria(Y_exp, full_fit["Y_pred"], 2 + len(PATHWAYS_LARGE))

# Generation of candidate models
rng = np.random.default_rng(123); candidate_networks = {}
rank_k = full_rate_df.sort_values("k_fit", ascending=False)["pathway"].tolist()
rank_flux = full_flux_summary_df.sort_values("total_flux", ascending=False)["pathway"].tolist()
def pathway_from_name(name): src, dst = name.split("->"); return (src, dst)
for size in RANDOM_SIZES:
    top_k = [pathway_from_name(x) for x in rank_k[:size]]; top_flux = [pathway_from_name(x) for x in rank_flux[:size]]
    candidate_networks[f"A_topK_{size}"] = [p for p in PATHWAYS_LARGE if p in top_k]
    candidate_networks[f"A_topFlux_{size}"] = [p for p in PATHWAYS_LARGE if p in top_flux]
for size in RANDOM_SIZES:
    half = max(1, size // 2); names = list(dict.fromkeys(rank_k[:half] + rank_flux[:size-half])); paths = [pathway_from_name(x) for x in names]
    candidate_networks[f"A_mixedKF_{size}"] = [p for p in PATHWAYS_LARGE if p in paths]
for i in range(N_RANDOM_MODELS):
    size = int(rng.choice(RANDOM_SIZES)); idxs = rng.choice(len(PATHWAYS_LARGE), size=size, replace=False); selected = [PATHWAYS_LARGE[j] for j in idxs]
    candidate_networks[f"A_random_{i+1:03d}_{size}"] = [p for p in PATHWAYS_LARGE if p in selected]
family_like = [("HP","LP"),("HIP","LIP"),("HNAP","LNAP"),("HP","HO"),("LP","LO"),("HO","HNAP"),("LO","LNAP"),("HNAP","HP"),("LNAP","LP"),("LNAP","aroma"),("LO","aroma"),("LP","aroma")]
candidate_networks["A_family_like_LNAP_aroma"] = [p for p in PATHWAYS_LARGE if p in family_like]
unique_networks, seen = {}, set()
for name, paths in candidate_networks.items():
    sig = tuple(paths)
    if sig not in seen and len(paths) > 0: seen.add(sig); unique_networks[name] = paths
candidate_networks = unique_networks
print(f"\nTotal candidate networks: {len(candidate_networks)}")

candidate_results, candidate_fits = {}, {}
for idx, (model_name, pathways) in enumerate(candidate_networks.items(), start=1):
    print(f"\nFitting candidate {idx}/{len(candidate_networks)}: {model_name} | n={len(pathways)}")
    fit = fit_lkm(pathways, n_starts=CANDIDATE_STARTS, seed=2000+idx, max_nfev=CANDIDATE_MAX_NFEV, verbose=False)
    metrics_df, overall_df = make_metrics(Y_exp, fit["Y_pred"])
    rss, AIC, BIC = model_information_criteria(Y_exp, fit["Y_pred"], 2+len(pathways)); aroma_row = metrics_df[metrics_df["pool"] == "aroma"].iloc[0]
    candidate_results[model_name] = {"model": model_name, "n_pathways": len(pathways), "n_params": 2+len(pathways), "contains_LNAP_to_aroma": TARGET_PATHWAY in pathways, "loss_scaled_objective": fit["loss"], "RSS": rss, "AIC": AIC, "BIC": BIC, "overall_R2": overall_value(overall_df,"overall_R2"), "overall_RMSE": overall_value(overall_df,"overall_RMSE"), "aroma_R2": aroma_row["R2"], "aroma_RMSE": aroma_row["RMSE"], "alpha": fit["alpha"], "beta": fit["beta"], "pathways": "; ".join([f"{p[0]}->{p[1]}" for p in pathways])}
    candidate_fits[model_name] = fit
candidate_df = pd.DataFrame(candidate_results.values())
candidate_df["overall_RMSE_norm"] = candidate_df["overall_RMSE"] / candidate_df["overall_RMSE"].min()
candidate_df["aroma_RMSE_norm"] = candidate_df["aroma_RMSE"] / candidate_df["aroma_RMSE"].min()
candidate_df["complexity_norm"] = candidate_df["n_pathways"] / candidate_df["n_pathways"].min()
best_rmse = candidate_df["overall_RMSE"].min()
candidate_near_df = candidate_df[candidate_df["overall_RMSE"] <= NEAR_RMSE_RELAXATION * best_rmse].copy()

#  Approach A:Combined Score
A_rank_df = candidate_near_df.copy()
A_rank_df["A_balanced_score"] = A_rank_df["overall_RMSE_norm"] + A_rank_df["aroma_RMSE_norm"] + COMPLEXITY_WEIGHT_A * A_rank_df["complexity_norm"]
A_rank_df = A_rank_df.sort_values(["A_balanced_score","overall_RMSE","aroma_RMSE","n_pathways"], ascending=[True,True,True,True]).reset_index(drop=True)
selected_A_name = A_rank_df.iloc[0]["model"]; PATHWAYS_A = candidate_networks[selected_A_name]
print("\nAPPROACH A SELECTED:", selected_A_name, "n =", len(PATHWAYS_A))
final_A_fit = fit_lkm(PATHWAYS_A, n_starts=FINAL_STARTS, seed=9000, max_nfev=FINAL_MAX_NFEV)
final_A_metrics, final_A_overall = make_metrics(Y_exp, final_A_fit["Y_pred"])
final_A_rates = make_rate_table(PATHWAYS_A, final_A_fit["k"], "A_balanced_final")
final_A_sensitivity = sensitivity_analysis(final_A_fit, PATHWAYS_A)
final_A_flux = pathway_flux_table(final_A_fit, PATHWAYS_A, "A_balanced_final")
final_A_flux_summary = flux_summary(final_A_flux)
final_A_rss, final_A_AIC, final_A_BIC = model_information_criteria(Y_exp, final_A_fit["Y_pred"], 2+len(PATHWAYS_A))

# Approach B: Global sensitivity reduction (krate and loss function)
prune_table_B = full_sensitivity_df.copy()
prune_table_B["keep_by_k"] = prune_table_B["k_fit"] > MIN_K_KEEP
prune_table_B["keep_by_sensitivity"] = prune_table_B["relative_delta_loss"] > MIN_RELATIVE_DELTA_LOSS_KEEP
prune_table_B["keep_final"] = prune_table_B["keep_by_k"] & prune_table_B["keep_by_sensitivity"]
selected_B_names = prune_table_B.loc[prune_table_B["keep_final"], "pathway"].tolist()
PATHWAYS_B = [p for p in PATHWAYS_LARGE if f"{p[0]}->{p[1]}" in selected_B_names]
if len(PATHWAYS_B) == 0:
    top_names = prune_table_B.head(8)["pathway"].tolist(); PATHWAYS_B = [p for p in PATHWAYS_LARGE if f"{p[0]}->{p[1]}" in top_names]
print("\nAPPROACH B SELECTED n =", len(PATHWAYS_B))
final_B_fit = fit_lkm(PATHWAYS_B, n_starts=FINAL_STARTS, seed=8100, max_nfev=FINAL_MAX_NFEV)
final_B_metrics, final_B_overall = make_metrics(Y_exp, final_B_fit["Y_pred"])
final_B_rates = make_rate_table(PATHWAYS_B, final_B_fit["k"], "B_sensitivity_pruned_final")
final_B_sensitivity = sensitivity_analysis(final_B_fit, PATHWAYS_B)
final_B_flux = pathway_flux_table(final_B_fit, PATHWAYS_B, "B_sensitivity_pruned_final")
final_B_flux_summary = flux_summary(final_B_flux)
final_B_rss, final_B_AIC, final_B_BIC = model_information_criteria(Y_exp, final_B_fit["Y_pred"], 2+len(PATHWAYS_B))

#  Approach C : Combined score + chemically essential pathways
def chemistry_penalty(pathways): return sum(1 for p in ESSENTIAL_PATHWAYS_FOR_SCORE_C if p not in pathways) / len(ESSENTIAL_PATHWAYS_FOR_SCORE_C)
C_rank_df = candidate_near_df.copy()
C_rank_df["chemistry_penalty"] = C_rank_df["model"].apply(lambda m: chemistry_penalty(candidate_networks[m]))
C_rank_df["C_chemistry_constrained_score"] = C_rank_df["overall_RMSE_norm"] + C_rank_df["aroma_RMSE_norm"] + COMPLEXITY_WEIGHT_C*C_rank_df["complexity_norm"] + CHEMISTRY_WEIGHT_C*C_rank_df["chemistry_penalty"]
C_rank_df = C_rank_df.sort_values(["C_chemistry_constrained_score","overall_RMSE","aroma_RMSE","n_pathways"], ascending=[True,True,True,True]).reset_index(drop=True)
selected_C_name = C_rank_df.iloc[0]["model"]; PATHWAYS_C = candidate_networks[selected_C_name].copy()
for p in ESSENTIAL_PATHWAYS_FORCE_C:
    if p in PATHWAYS_LARGE and p not in PATHWAYS_C: PATHWAYS_C.append(p)
PATHWAYS_C = [p for p in PATHWAYS_LARGE if p in PATHWAYS_C]
print("\nAPPROACH C SELECTED:", selected_C_name, "n =", len(PATHWAYS_C))
final_C_fit = fit_lkm(PATHWAYS_C, n_starts=FINAL_STARTS, seed=9900, max_nfev=FINAL_MAX_NFEV)
final_C_metrics, final_C_overall = make_metrics(Y_exp, final_C_fit["Y_pred"])
final_C_rates = make_rate_table(PATHWAYS_C, final_C_fit["k"], "C_chemistry_constrained_final")
final_C_sensitivity = sensitivity_analysis(final_C_fit, PATHWAYS_C)
final_C_flux = pathway_flux_table(final_C_fit, PATHWAYS_C, "C_chemistry_constrained_final")
final_C_flux_summary = flux_summary(final_C_flux)
final_C_rss, final_C_AIC, final_C_BIC = model_information_criteria(Y_exp, final_C_fit["Y_pred"], 2+len(PATHWAYS_C))

# Comparing three approach outputs 
comparison_rows = []
def add_summary(model, approach, fit, pathways, metrics_df, overall_df, rss, AIC, BIC):
    aroma_row = metrics_df[metrics_df["pool"] == "aroma"].iloc[0]
    comparison_rows.append({"model": model, "approach": approach, "n_pathways": len(pathways), "n_params": 2+len(pathways), "loss_scaled_objective": fit["loss"], "RSS": rss, "AIC": AIC, "BIC": BIC, "alpha": fit["alpha"], "beta": fit["beta"], "S_ref_raw": fit["S_ref"], "overall_R2": overall_value(overall_df,"overall_R2"), "overall_MAE": overall_value(overall_df,"overall_MAE"), "overall_RMSE": overall_value(overall_df,"overall_RMSE"), "overall_scaled_RMSE": overall_value(overall_df,"overall_scaled_RMSE"), "aroma_R2": aroma_row["R2"], "aroma_RMSE": aroma_row["RMSE"], "contains_HNAP_to_aroma": ("HNAP","aroma") in pathways, "contains_LNAP_to_aroma": ("LNAP","aroma") in pathways, "pathways": "; ".join([f"{p[0]}->{p[1]}" for p in pathways])})
add_summary("full_36", "reference_full_model", full_fit, PATHWAYS_LARGE, full_metrics_df, full_overall_df, full_rss, full_AIC, full_BIC)
add_summary("A_balanced_final", "equal_weight_balanced_selection", final_A_fit, PATHWAYS_A, final_A_metrics, final_A_overall, final_A_rss, final_A_AIC, final_A_BIC)
add_summary("B_sensitivity_pruned_final", "strict_sensitivity_pruning", final_B_fit, PATHWAYS_B, final_B_metrics, final_B_overall, final_B_rss, final_B_AIC, final_B_BIC)
add_summary("C_chemistry_constrained_final", "chemistry_constrained_balanced_selection", final_C_fit, PATHWAYS_C, final_C_metrics, final_C_overall, final_C_rss, final_C_AIC, final_C_BIC)
model_comparison = pd.DataFrame(comparison_rows).sort_values("overall_RMSE").reset_index(drop=True)
display(model_comparison)

combined_metrics = pd.concat([full_metrics_df.assign(model="full_36"), final_A_metrics.assign(model="A_balanced_final"), final_B_metrics.assign(model="B_sensitivity_pruned_final"), final_C_metrics.assign(model="C_chemistry_constrained_final")], ignore_index=True)
combined_overall = pd.concat([full_overall_df.assign(model="full_36"), final_A_overall.assign(model="A_balanced_final"), final_B_overall.assign(model="B_sensitivity_pruned_final"), final_C_overall.assign(model="C_chemistry_constrained_final")], ignore_index=True)
combined_rates = pd.concat([full_rate_df, final_A_rates, final_B_rates, final_C_rates], ignore_index=True)
combined_sensitivity = pd.concat([full_sensitivity_df.assign(model="full_36"), final_A_sensitivity.assign(model="A_balanced_final"), final_B_sensitivity.assign(model="B_sensitivity_pruned_final"), final_C_sensitivity.assign(model="C_chemistry_constrained_final")], ignore_index=True)
combined_flux = pd.concat([full_flux_df, final_A_flux, final_B_flux, final_C_flux], ignore_index=True)
combined_flux_summary = flux_summary(combined_flux)
combined_fit_wide = pd.concat([make_fit_table(full_fit,"full_36"), make_fit_table(final_A_fit,"A_balanced_final"), make_fit_table(final_B_fit,"B_sensitivity_pruned_final"), make_fit_table(final_C_fit,"C_chemistry_constrained_final")], ignore_index=True)
combined_diagnostics = pd.concat([make_diagnostics_long(full_fit,"full_36"), make_diagnostics_long(final_A_fit,"A_balanced_final"), make_diagnostics_long(final_B_fit,"B_sensitivity_pruned_final"), make_diagnostics_long(final_C_fit,"C_chemistry_constrained_final")], ignore_index=True)

# Plotting all output
def save_plot(path): plt.tight_layout(); plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight"); plt.show()
models_dict = {"A_balanced_final": final_A_fit, "B_sensitivity_pruned_final": final_B_fit, "C_chemistry_constrained_final": final_C_fit}
markers = {"A_balanced_final": "o", "B_sensitivity_pruned_final": "x", "C_chemistry_constrained_final": "s"}
S_plot = final_C_fit["S_rel"]; order = np.argsort(S_plot)

plt.figure(figsize=(15,9))
for i,pool in enumerate(POOLS):
    plt.scatter(S_plot[order], Y_exp[order,i], s=45, alpha=0.75)
    plt.plot(final_A_fit["S_rel"][order], final_A_fit["Y_pred"][order,i], linewidth=2, linestyle="-", label="A" if i==0 else None)
    plt.plot(final_B_fit["S_rel"][order], final_B_fit["Y_pred"][order,i], linewidth=2, linestyle="--", label="B" if i==0 else None)
    plt.plot(final_C_fit["S_rel"][order], final_C_fit["Y_pred"][order,i], linewidth=2, linestyle=":", label="C" if i==0 else None)
plt.xlabel("Relative severity, S/Sref", fontweight="bold"); plt.ylabel("Oil-normalized lump fraction", fontweight="bold"); plt.title("Experimental vs three reduced-network approaches", fontweight="bold"); plt.grid(True, alpha=0.3); plt.legend(); save_plot(PLOT_DIR/"01_all_lumps_three_approaches.png")

plt.figure(figsize=(8,8))
for model_name, fit in models_dict.items(): plt.scatter(Y_exp.ravel(), fit["Y_pred"].ravel(), s=70, marker=markers[model_name], label=model_name)
minv = min([Y_exp.min()] + [fit["Y_pred"].min() for fit in models_dict.values()]); maxv = max([Y_exp.max()] + [fit["Y_pred"].max() for fit in models_dict.values()])
plt.plot([minv,maxv],[minv,maxv], linewidth=2); plt.xlabel("Experimental fraction", fontweight="bold"); plt.ylabel("Predicted fraction", fontweight="bold"); plt.title("Overall parity: three approaches", fontweight="bold"); plt.grid(True, alpha=0.3); plt.legend(); save_plot(PLOT_DIR/"02_overall_parity_three_approaches.png")

pool_r2_rows = []
for i,pool in enumerate(POOLS):
    y_true = Y_exp[:,i]; plt.figure(figsize=(7,7))
    for model_name, fit in models_dict.items():
        y_pred = fit["Y_pred"][:,i]; r2 = r2_score(y_true, y_pred); pool_r2_rows.append({"pool": pool, "model": model_name, "R2": r2})
        plt.scatter(y_true, y_pred, s=80, marker=markers[model_name], label=f"{model_name}, R²={r2:.3f}")
    minv = min([y_true.min()] + [fit["Y_pred"][:,i].min() for fit in models_dict.values()]); maxv = max([y_true.max()] + [fit["Y_pred"][:,i].max() for fit in models_dict.values()])
    plt.plot([minv,maxv],[minv,maxv], linewidth=2); plt.xlabel(f"Experimental {pool}", fontweight="bold"); plt.ylabel(f"Predicted {pool}", fontweight="bold"); plt.title(f"Parity plot: {pool}", fontweight="bold"); plt.grid(True, alpha=0.3); plt.legend(fontsize=8); save_plot(PARITY_DIR/f"parity_{pool}_three_approaches.png")
pool_r2_df = pd.DataFrame(pool_r2_rows)

CASE_NAMES = lkm_df["run_id"].astype(str).tolist() if "run_id" in lkm_df.columns else [f"case_{i+1}" for i in range(len(Y_exp))]
bar_width = 0.20
for exp_i in range(len(Y_exp)):
    exp_name = CASE_NAMES[exp_i].replace("/","_").replace("\\","_"); x = np.arange(len(POOLS)); plt.figure(figsize=(13,6))
    plt.bar(x-1.5*bar_width, Y_exp[exp_i], width=bar_width, label="Experimental"); plt.bar(x-0.5*bar_width, final_A_fit["Y_pred"][exp_i], width=bar_width, label="A"); plt.bar(x+0.5*bar_width, final_B_fit["Y_pred"][exp_i], width=bar_width, label="B"); plt.bar(x+1.5*bar_width, final_C_fit["Y_pred"][exp_i], width=bar_width, label="C")
    plt.xticks(x, POOLS, rotation=45); plt.ylabel("Oil-normalized lump fraction"); plt.title(f"Experimental vs A/B/C: {exp_name}"); plt.grid(axis="y", alpha=0.3); plt.legend(); save_plot(BAR_DIR/f"bar_lumps_{exp_name}.png")

PIONA_NAMES = ["P","iP","O","N","A"]
def lump_to_total_PIONA(Y): return np.column_stack([Y[:,0]+Y[:,2], Y[:,1]+Y[:,3], Y[:,4]+Y[:,6], Y[:,5]+Y[:,7], Y[:,8]])
PIONA_exp, PIONA_A, PIONA_B, PIONA_C = lump_to_total_PIONA(Y_exp), lump_to_total_PIONA(final_A_fit["Y_pred"]), lump_to_total_PIONA(final_B_fit["Y_pred"]), lump_to_total_PIONA(final_C_fit["Y_pred"])
for exp_i in range(len(Y_exp)):
    exp_name = CASE_NAMES[exp_i].replace("/","_").replace("\\","_"); x = np.arange(len(PIONA_NAMES)); plt.figure(figsize=(10,6))
    plt.bar(x-1.5*bar_width, PIONA_exp[exp_i], width=bar_width, label="Experimental"); plt.bar(x-0.5*bar_width, PIONA_A[exp_i], width=bar_width, label="A"); plt.bar(x+0.5*bar_width, PIONA_B[exp_i], width=bar_width, label="B"); plt.bar(x+1.5*bar_width, PIONA_C[exp_i], width=bar_width, label="C")
    plt.xticks(x, PIONA_NAMES); plt.ylabel("Total PIONA fraction"); plt.title(f"Total PIONA comparison: {exp_name}"); plt.grid(axis="y", alpha=0.3); plt.legend(); save_plot(PIONA_DIR/f"PIONA_bar_{exp_name}.png")

for model_name, sens_df in [("A_balanced_final", final_A_sensitivity), ("B_sensitivity_pruned_final", final_B_sensitivity), ("C_chemistry_constrained_final", final_C_sensitivity)]:
    plt.figure(figsize=(10, max(6, 0.35*len(sens_df)))); plot_df = sens_df.sort_values("relative_delta_loss", ascending=True)
    plt.barh(plot_df["pathway"], plot_df["relative_delta_loss"]); plt.xlabel("Relative loss increase when pathway removed", fontweight="bold"); plt.title(f"Sensitivity: {model_name}", fontweight="bold"); plt.grid(axis="x", alpha=0.3); save_plot(PLOT_DIR/f"sensitivity_{model_name}.png")
for model_name, flux_sum in [("A_balanced_final", final_A_flux_summary), ("B_sensitivity_pruned_final", final_B_flux_summary), ("C_chemistry_constrained_final", final_C_flux_summary)]:
    sub = flux_sum[flux_sum["model"] == model_name]; plt.figure(figsize=(10, max(6, 0.35*len(sub)))); plot_df = sub.sort_values("total_flux", ascending=True)
    plt.barh(plot_df["pathway"], plot_df["total_flux"]); plt.xlabel("Total flux across experiments", fontweight="bold"); plt.title(f"Flux contribution: {model_name}", fontweight="bold"); plt.grid(axis="x", alpha=0.3); save_plot(PLOT_DIR/f"flux_{model_name}.png")



# EXTRA SEPARATE + COMBINED PLOTS 
# 1) separate detailed plots for each approach in its own folder
# 2) combined comparison plots in a dedicated combined folder

APPROACH_INFO = {
    "A_balanced_final": {
        "fit": final_A_fit,
        "pathways": PATHWAYS_A,
        "metrics": final_A_metrics,
        "overall": final_A_overall,
        "rates": final_A_rates,
        "sensitivity": final_A_sensitivity,
        "flux_summary": final_A_flux_summary,
        "piona": PIONA_A,
        "plot_dir": A_PLOT_DIR,
        "label": "Approach A: balanced selection",
        "linestyle": "-",
        "marker": "o",
    },
    "B_sensitivity_pruned_final": {
        "fit": final_B_fit,
        "pathways": PATHWAYS_B,
        "metrics": final_B_metrics,
        "overall": final_B_overall,
        "rates": final_B_rates,
        "sensitivity": final_B_sensitivity,
        "flux_summary": final_B_flux_summary,
        "piona": PIONA_B,
        "plot_dir": B_PLOT_DIR,
        "label": "Approach B: sensitivity pruning",
        "linestyle": "--",
        "marker": "x",
    },
    "C_chemistry_constrained_final": {
        "fit": final_C_fit,
        "pathways": PATHWAYS_C,
        "metrics": final_C_metrics,
        "overall": final_C_overall,
        "rates": final_C_rates,
        "sensitivity": final_C_sensitivity,
        "flux_summary": final_C_flux_summary,
        "piona": PIONA_C,
        "plot_dir": C_PLOT_DIR,
        "label": "Approach C: chemistry constrained",
        "linestyle": ":",
        "marker": "s",
    },
}

def safe_name(x):
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")

def savefig_to(path):
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()

#  Separate detailed plots for each approach 
for model_name, info in APPROACH_INFO.items():
    fit = info["fit"]
    out_dir = info["plot_dir"]
    out_dir.mkdir(exist_ok=True)
    order = np.argsort(fit["S_rel"])

    # 1. All lump evolution vs severity for one approach
    plt.figure(figsize=(15, 9))
    for i, pool in enumerate(POOLS):
        plt.scatter(fit["S_rel"][order], Y_exp[order, i], s=55, alpha=0.75, label=f"{pool} exp" if i < 1 else None)
        plt.plot(fit["S_rel"][order], fit["Y_pred"][order, i], linewidth=2, label=pool)
    plt.xlabel("Relative severity, S/Sref", fontweight="bold")
    plt.ylabel("Oil-normalized lump fraction", fontweight="bold")
    plt.title(f"Lump evolution: {info['label']}", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8)
    savefig_to(out_dir / f"01_lump_evolution_{model_name}.png")

    # 2. Overall parity for one approach
    plt.figure(figsize=(8, 8))
    plt.scatter(Y_exp.ravel(), fit["Y_pred"].ravel(), s=75, marker=info["marker"])
    minv = min(Y_exp.min(), fit["Y_pred"].min())
    maxv = max(Y_exp.max(), fit["Y_pred"].max())
    plt.plot([minv, maxv], [minv, maxv], linewidth=2)
    plt.xlabel("Experimental fraction", fontweight="bold")
    plt.ylabel("Predicted fraction", fontweight="bold")
    plt.title(f"Overall parity: {info['label']}", fontweight="bold")
    plt.grid(True, alpha=0.3)
    savefig_to(out_dir / f"02_overall_parity_{model_name}.png")

    # 3. Pool-wise parity plots for one approach
    for i, pool in enumerate(POOLS):
        y_true = Y_exp[:, i]
        y_pred = fit["Y_pred"][:, i]
        r2 = r2_score(y_true, y_pred)
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, s=80, marker=info["marker"], label=f"R² = {r2:.3f}")
        minv = min(y_true.min(), y_pred.min())
        maxv = max(y_true.max(), y_pred.max())
        plt.plot([minv, maxv], [minv, maxv], linewidth=2)
        plt.xlabel(f"Experimental {pool}", fontweight="bold")
        plt.ylabel(f"Predicted {pool}", fontweight="bold")
        plt.title(f"Parity plot for {pool}: {info['label']}", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig_to(out_dir / f"03_parity_{pool}_{model_name}.png")

    # 4. Case-wise lump bar plots for one approach
    for exp_i in range(len(Y_exp)):
        exp_name = safe_name(CASE_NAMES[exp_i])
        x = np.arange(len(POOLS))
        bw = 0.35
        plt.figure(figsize=(13, 6))
        plt.bar(x - bw/2, Y_exp[exp_i], width=bw, label="Experimental")
        plt.bar(x + bw/2, fit["Y_pred"][exp_i], width=bw, label="Predicted")
        plt.xticks(x, POOLS, rotation=45)
        plt.ylabel("Oil-normalized lump fraction")
        plt.title(f"Lump comparison: {exp_name} | {info['label']}")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        savefig_to(out_dir / f"04_lump_bar_{exp_name}_{model_name}.png")

    # 5. Case-wise PIONA bar plots for one approach
    for exp_i in range(len(Y_exp)):
        exp_name = safe_name(CASE_NAMES[exp_i])
        x = np.arange(len(PIONA_NAMES))
        bw = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - bw/2, PIONA_exp[exp_i], width=bw, label="Experimental")
        plt.bar(x + bw/2, info["piona"][exp_i], width=bw, label="Predicted")
        plt.xticks(x, PIONA_NAMES)
        plt.ylabel("Total PIONA fraction")
        plt.title(f"PIONA comparison: {exp_name} | {info['label']}")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        savefig_to(out_dir / f"05_PIONA_bar_{exp_name}_{model_name}.png")

    # 6. Sensitivity plot
    sens_df = info["sensitivity"].copy()
    plt.figure(figsize=(10, max(6, 0.35 * len(sens_df))))
    plot_df = sens_df.sort_values("relative_delta_loss", ascending=True)
    plt.barh(plot_df["pathway"], plot_df["relative_delta_loss"])
    plt.xlabel("Relative loss increase when pathway is removed", fontweight="bold")
    plt.title(f"Pathway sensitivity: {info['label']}", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    savefig_to(out_dir / f"06_sensitivity_{model_name}.png")

    # 7. Flux contribution plot
    flux_sum = info["flux_summary"].copy()
    sub = flux_sum[flux_sum["model"] == model_name]
    plt.figure(figsize=(10, max(6, 0.35 * len(sub))))
    plot_df = sub.sort_values("total_flux", ascending=True)
    plt.barh(plot_df["pathway"], plot_df["total_flux"])
    plt.xlabel("Total flux across experiments", fontweight="bold")
    plt.title(f"Pathway flux contribution: {info['label']}", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    savefig_to(out_dir / f"07_flux_{model_name}.png")

    # 8. Fitted pathway-rate plot
    rate_df = info["rates"].sort_values("k_fit", ascending=True)
    plt.figure(figsize=(10, max(6, 0.35 * len(rate_df))))
    plt.barh(rate_df["pathway"], rate_df["k_fit"])
    plt.xlabel("Fitted severity-based pathway coefficient, k", fontweight="bold")
    plt.title(f"Fitted pathway coefficients: {info['label']}", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    savefig_to(out_dir / f"08_rate_coefficients_{model_name}.png")

# Combined comparison plots 
COMBINED_PLOT_DIR.mkdir(exist_ok=True)

# 1. Combined model performance: R2/RMSE/complexity
plt.figure(figsize=(10, 6))
summary_plot_df = model_comparison[model_comparison["model"].isin(APPROACH_INFO.keys())].copy()
x = np.arange(len(summary_plot_df))
plt.bar(x, summary_plot_df["overall_R2"])
plt.xticks(x, summary_plot_df["model"], rotation=30, ha="right")
plt.ylabel("Overall R²")
plt.title("Combined comparison: overall R²", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
savefig_to(COMBINED_PLOT_DIR / "01_combined_overall_R2.png")

plt.figure(figsize=(10, 6))
plt.bar(x, summary_plot_df["overall_RMSE"])
plt.xticks(x, summary_plot_df["model"], rotation=30, ha="right")
plt.ylabel("Overall RMSE")
plt.title("Combined comparison: overall RMSE", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
savefig_to(COMBINED_PLOT_DIR / "02_combined_overall_RMSE.png")

plt.figure(figsize=(10, 6))
plt.bar(x, summary_plot_df["n_pathways"])
plt.xticks(x, summary_plot_df["model"], rotation=30, ha="right")
plt.ylabel("Number of retained pathways")
plt.title("Combined comparison: model complexity", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
savefig_to(COMBINED_PLOT_DIR / "03_combined_number_of_pathways.png")

# 2. Pool-wise R2 grouped comparison
r2_pivot = pool_r2_df.pivot(index="pool", columns="model", values="R2").reindex(POOLS)
plt.figure(figsize=(14, 6))
x = np.arange(len(POOLS))
bw = 0.25
for j, model_name in enumerate(APPROACH_INFO.keys()):
    plt.bar(x + (j - 1) * bw, r2_pivot[model_name].values, width=bw, label=model_name)
plt.xticks(x, POOLS, rotation=45)
plt.ylabel("Pool-wise R²")
plt.title("Combined comparison: pool-wise R²", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
plt.legend()
savefig_to(COMBINED_PLOT_DIR / "04_combined_poolwise_R2.png")

# 3. Combined severity evolution for every pool separately
for i, pool in enumerate(POOLS):
    plt.figure(figsize=(9, 6))
    plt.scatter(final_C_fit["S_rel"][order], Y_exp[order, i], s=70, label="Experimental")
    for model_name, info in APPROACH_INFO.items():
        fit = info["fit"]
        ord_i = np.argsort(fit["S_rel"])
        plt.plot(fit["S_rel"][ord_i], fit["Y_pred"][ord_i, i], linewidth=2, linestyle=info["linestyle"], label=model_name)
    plt.xlabel("Relative severity, S/Sref", fontweight="bold")
    plt.ylabel(f"{pool} fraction", fontweight="bold")
    plt.title(f"Combined severity evolution: {pool}", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    savefig_to(COMBINED_PLOT_DIR / f"05_combined_severity_{pool}.png")

# 4. Combined PIONA parity plots
for j, fam in enumerate(PIONA_NAMES):
    plt.figure(figsize=(7, 7))
    for model_name, info in APPROACH_INFO.items():
        plt.scatter(PIONA_exp[:, j], info["piona"][:, j], s=80, marker=info["marker"], label=model_name)
    minv = min(PIONA_exp[:, j].min(), PIONA_A[:, j].min(), PIONA_B[:, j].min(), PIONA_C[:, j].min())
    maxv = max(PIONA_exp[:, j].max(), PIONA_A[:, j].max(), PIONA_B[:, j].max(), PIONA_C[:, j].max())
    plt.plot([minv, maxv], [minv, maxv], linewidth=2)
    plt.xlabel(f"Experimental {fam}", fontweight="bold")
    plt.ylabel(f"Predicted {fam}", fontweight="bold")
    plt.title(f"Combined PIONA parity: {fam}", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    savefig_to(COMBINED_PLOT_DIR / f"06_combined_PIONA_parity_{fam}.png")


#  Additional approach specific excel output
# This block creates three separate Excel files, one for each final approach.
# Existing combined Excel and ZIP exports below are kept unchanged.

def export_single_approach_excel(
    out_path,
    approach_label,
    fit_result,
    pathways,
    metrics_df,
    overall_df,
    rates_df,
    sensitivity_df,
    flux_df,
    flux_summary_df,
    fit_wide_df,
    diagnostics_long_df,
    piona_pred_matrix,
    selection_table=None
):
    """
    Export all relevant outputs for one final approach into a separate Excel file.
    """
    summary_df = pd.DataFrame([{
        "approach": approach_label,
        "n_pathways": len(pathways),
        "n_params": 2 + len(pathways),
        "loss_scaled_objective": fit_result["loss"],
        "alpha": fit_result["alpha"],
        "beta": fit_result["beta"],
        "S_ref_raw": fit_result["S_ref"],
        "contains_HNAP_to_aroma": ("HNAP", "aroma") in pathways,
        "contains_LNAP_to_aroma": ("LNAP", "aroma") in pathways,
        "pathways": "; ".join([f"{p[0]}->{p[1]}" for p in pathways])
    }])

    selected_pathways_df = pd.DataFrame({
        "source": [p[0] for p in pathways],
        "target": [p[1] for p in pathways],
        "pathway": [f"{p[0]}->{p[1]}" for p in pathways],
        "category": [pathway_category(p[0], p[1]) for p in pathways]
    })

    piona_pred_df = pd.DataFrame(piona_pred_matrix, columns=PIONA_NAMES)
    piona_pred_df.insert(0, "run_id", lkm_df["run_id"].values)
    piona_pred_df.insert(1, "label", lkm_df["label"].values)

    piona_exp_df = pd.DataFrame(PIONA_exp, columns=PIONA_NAMES)
    piona_exp_df.insert(0, "run_id", lkm_df["run_id"].values)
    piona_exp_df.insert(1, "label", lkm_df["label"].values)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="approach_summary", index=False)
        selected_pathways_df.to_excel(writer, sheet_name="selected_pathways", index=False)
        metrics_df.to_excel(writer, sheet_name="pool_metrics", index=False)
        overall_df.to_excel(writer, sheet_name="overall_metrics", index=False)
        rates_df.to_excel(writer, sheet_name="rates", index=False)
        sensitivity_df.to_excel(writer, sheet_name="sensitivity", index=False)
        flux_summary_df.to_excel(writer, sheet_name="flux_summary", index=False)
        flux_df.to_excel(writer, sheet_name="flux_long", index=False)
        fit_wide_df.to_excel(writer, sheet_name="fit_wide", index=False)
        diagnostics_long_df.to_excel(writer, sheet_name="diagnostics_long", index=False)
        piona_exp_df.to_excel(writer, sheet_name="PIONA_exp", index=False)
        piona_pred_df.to_excel(writer, sheet_name="PIONA_pred", index=False)

        if selection_table is not None:
            selection_table.to_excel(writer, sheet_name="selection_decision", index=False)

# Individual approach Excel files
out_A_xlsx = OUTPUT_DIR / "Approach_A_balanced_final_outputs.xlsx"
out_B_xlsx = OUTPUT_DIR / "Approach_B_sensitivity_pruned_final_outputs.xlsx"
out_C_xlsx = OUTPUT_DIR / "Approach_C_chemistry_constrained_final_outputs.xlsx"

export_single_approach_excel(
    out_A_xlsx,
    "A_balanced_final",
    final_A_fit,
    PATHWAYS_A,
    final_A_metrics,
    final_A_overall,
    final_A_rates,
    final_A_sensitivity,
    final_A_flux,
    final_A_flux_summary,
    make_fit_table(final_A_fit, "A_balanced_final"),
    make_diagnostics_long(final_A_fit, "A_balanced_final"),
    PIONA_A,
    selection_table=A_rank_df
)

export_single_approach_excel(
    out_B_xlsx,
    "B_sensitivity_pruned_final",
    final_B_fit,
    PATHWAYS_B,
    final_B_metrics,
    final_B_overall,
    final_B_rates,
    final_B_sensitivity,
    final_B_flux,
    final_B_flux_summary,
    make_fit_table(final_B_fit, "B_sensitivity_pruned_final"),
    make_diagnostics_long(final_B_fit, "B_sensitivity_pruned_final"),
    PIONA_B,
    selection_table=prune_table_B
)

export_single_approach_excel(
    out_C_xlsx,
    "C_chemistry_constrained_final",
    final_C_fit,
    PATHWAYS_C,
    final_C_metrics,
    final_C_overall,
    final_C_rates,
    final_C_sensitivity,
    final_C_flux,
    final_C_flux_summary,
    make_fit_table(final_C_fit, "C_chemistry_constrained_final"),
    make_diagnostics_long(final_C_fit, "C_chemistry_constrained_final"),
    PIONA_C,
    selection_table=C_rank_df
)

print("Additional approach-specific Excel files:")
print("A:", out_A_xlsx)
print("B:", out_B_xlsx)
print("C:", out_C_xlsx)


# All excel exports 
out_xlsx = OUTPUT_DIR / "three_approaches_OCEM_outputs.xlsx"
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    meta.to_excel(writer, sheet_name="meta", index=False); scale_table.to_excel(writer, sheet_name="pool_scale", index=False); carbon_df.to_excel(writer, sheet_name="carbon_vs_run", index=False); piona_df.to_excel(writer, sheet_name="PIONA_vs_run", index=False); lump_df.to_excel(writer, sheet_name="lumps_vs_run", index=False)
    candidate_df.to_excel(writer, sheet_name="candidate_models_all", index=False); candidate_near_df.to_excel(writer, sheet_name="candidate_models_near", index=False); A_rank_df.to_excel(writer, sheet_name="A_balanced_ranking", index=False); prune_table_B.to_excel(writer, sheet_name="B_pruning_decision", index=False); C_rank_df.to_excel(writer, sheet_name="C_chemistry_ranking", index=False)
    model_comparison.to_excel(writer, sheet_name="model_comparison", index=False); combined_metrics.to_excel(writer, sheet_name="combined_pool_metrics", index=False); combined_overall.to_excel(writer, sheet_name="combined_overall_metrics", index=False); pool_r2_df.to_excel(writer, sheet_name="poolwise_R2", index=False)
    combined_rates.to_excel(writer, sheet_name="combined_rates", index=False); combined_sensitivity.to_excel(writer, sheet_name="combined_sensitivity", index=False); combined_flux_summary.to_excel(writer, sheet_name="combined_flux_summary", index=False); combined_flux.to_excel(writer, sheet_name="combined_flux_long", index=False); combined_fit_wide.to_excel(writer, sheet_name="combined_fit_wide", index=False); combined_diagnostics.to_excel(writer, sheet_name="combined_diagnostics_long", index=False)
    full_rate_df.to_excel(writer, sheet_name="full_36_rates", index=False); final_A_rates.to_excel(writer, sheet_name="A_final_rates", index=False); final_B_rates.to_excel(writer, sheet_name="B_final_rates", index=False); final_C_rates.to_excel(writer, sheet_name="C_final_rates", index=False)
    full_sensitivity_df.to_excel(writer, sheet_name="full_36_sensitivity", index=False); final_A_sensitivity.to_excel(writer, sheet_name="A_final_sensitivity", index=False); final_B_sensitivity.to_excel(writer, sheet_name="B_final_sensitivity", index=False); final_C_sensitivity.to_excel(writer, sheet_name="C_final_sensitivity", index=False)
    pd.DataFrame(PIONA_exp, columns=PIONA_NAMES).to_excel(writer, sheet_name="PIONA_exp", index=False); pd.DataFrame(PIONA_A, columns=PIONA_NAMES).to_excel(writer, sheet_name="PIONA_A", index=False); pd.DataFrame(PIONA_B, columns=PIONA_NAMES).to_excel(writer, sheet_name="PIONA_B", index=False); pd.DataFrame(PIONA_C, columns=PIONA_NAMES).to_excel(writer, sheet_name="PIONA_C", index=False)


#  Detailed outputs for paper 
# These exports mirror the detailed output structure of:
#   Pareto_36-14_final.xlsx
#   relative_severity_reference_LKM_results_36path_datadriven.xlsx
# They are generated for Approach A, Approach B, Approach C, and a combined workbook.

DETAILED_OUTPUT_DIR = OUTPUT_DIR / "publication_style_detailed_outputs"
DETAILED_OUTPUT_DIR.mkdir(exist_ok=True)

def make_final_LKM_fit_reference_style(fit_result):
    """
    Reference-style wide fit table:
    run/meta/severity + exp/pred/error/scaled_error for each pool.
    Here error = pred - exp, and scaled_error = (pred - exp)/POOL_SCALE.
    """
    tab = lkm_df[["run_id", "label", "T_set_C", "t_heat_min", "t_hold_min"]].copy()
    tab["severity_raw"] = fit_result["S_raw"]
    tab["severity_reference_raw"] = fit_result["S_ref"]
    tab["relative_severity_S_over_Sref"] = fit_result["S_rel"]

    for i, pool in enumerate(POOLS):
        err = fit_result["Y_pred"][:, i] - Y_exp[:, i]
        tab[f"{pool}_exp"] = Y_exp[:, i]
        tab[f"{pool}_pred"] = fit_result["Y_pred"][:, i]
        tab[f"{pool}_error"] = err
        tab[f"{pool}_scaled_error"] = err / POOL_SCALE[i]
    return tab

def make_meta_severity_reference_style(fit_result):
    tab = meta.copy()
    tab["severity_raw"] = fit_result["S_raw"]
    tab["severity_reference_raw"] = fit_result["S_ref"]
    tab["relative_severity_S_over_Sref"] = fit_result["S_rel"]
    return tab

def add_severity_to_long_table(base_df, fit_result):
    sev = pd.DataFrame({
        "run_id": lkm_df["run_id"].values,
        "label": lkm_df["label"].values,
        "relative_severity_S_over_Sref": fit_result["S_rel"]
    })
    return base_df.merge(sev, on="run_id", how="left")

def make_final_severity_params(fit_result, pathways, approach_name):
    return pd.DataFrame({
        "parameter": [
            "approach",
            "alpha",
            "beta",
            "S_ref_raw",
            "n_pathways",
            "n_params",
            "loss_scaled_objective",
            "contains_HNAP_to_aroma",
            "contains_LNAP_to_aroma",
            "pathways"
        ],
        "value": [
            approach_name,
            fit_result["alpha"],
            fit_result["beta"],
            fit_result["S_ref"],
            len(pathways),
            2 + len(pathways),
            fit_result["loss"],
            ("HNAP", "aroma") in pathways,
            ("LNAP", "aroma") in pathways,
            "; ".join([f"{p[0]}->{p[1]}" for p in pathways])
        ]
    })

def complexity_weight_sweep_A(weights=None):
    if weights is None:
        weights = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    rows = []
    base = candidate_near_df.copy()
    for w in weights:
        tmp = base.copy()
        tmp["balanced_score"] = (
            tmp["overall_RMSE_norm"]
            + tmp["aroma_RMSE_norm"]
            + w * tmp["complexity_norm"]
        )
        best = tmp.sort_values(
            ["balanced_score", "overall_RMSE", "aroma_RMSE", "n_pathways"],
            ascending=[True, True, True, True]
        ).iloc[0]
        rows.append({
            "complexity_weight": w,
            "selected_model": best["model"],
            "n_pathways": best["n_pathways"],
            "contains_LNAP_to_aroma": best["contains_LNAP_to_aroma"],
            "overall_R2": best["overall_R2"],
            "overall_RMSE": best["overall_RMSE"],
            "aroma_R2": best["aroma_R2"],
            "aroma_RMSE": best["aroma_RMSE"],
            "BIC": best["BIC"],
            "balanced_score": best["balanced_score"],
            "pathways": best["pathways"]
        })
    return pd.DataFrame(rows)

def complexity_weight_sweep_C(weights=None):
    if weights is None:
        weights = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    rows = []
    base = candidate_near_df.copy()
    base["chemistry_penalty"] = base["model"].apply(lambda m: chemistry_penalty(candidate_networks[m]))
    for w in weights:
        tmp = base.copy()
        tmp["chemistry_constrained_score"] = (
            tmp["overall_RMSE_norm"]
            + tmp["aroma_RMSE_norm"]
            + w * tmp["complexity_norm"]
            + CHEMISTRY_WEIGHT_C * tmp["chemistry_penalty"]
        )
        best = tmp.sort_values(
            ["chemistry_constrained_score", "overall_RMSE", "aroma_RMSE", "n_pathways"],
            ascending=[True, True, True, True]
        ).iloc[0]
        rows.append({
            "complexity_weight": w,
            "selected_model": best["model"],
            "n_pathways": best["n_pathways"],
            "contains_LNAP_to_aroma": best["contains_LNAP_to_aroma"],
            "chemistry_penalty": best["chemistry_penalty"],
            "overall_R2": best["overall_R2"],
            "overall_RMSE": best["overall_RMSE"],
            "aroma_R2": best["aroma_R2"],
            "aroma_RMSE": best["aroma_RMSE"],
            "BIC": best["BIC"],
            "chemistry_constrained_score": best["chemistry_constrained_score"],
            "pathways": best["pathways"]
        })
    return pd.DataFrame(rows)

A_complexity_sweep = complexity_weight_sweep_A()
C_complexity_sweep = complexity_weight_sweep_C()

def make_approach_model_comparison_row(model, approach, fit, pathways, metrics_df, overall_df):
    rss, AIC, BIC = model_information_criteria(Y_exp, fit["Y_pred"], 2 + len(pathways))
    aroma_row = metrics_df[metrics_df["pool"] == "aroma"].iloc[0]
    return {
        "model": model,
        "approach": approach,
        "n_pathways": len(pathways),
        "n_params": 2 + len(pathways),
        "loss": fit["loss"],
        "RSS": rss,
        "AIC": AIC,
        "BIC": BIC,
        "alpha": fit["alpha"],
        "beta": fit["beta"],
        "S_ref_raw": fit["S_ref"],
        "overall_R2": overall_value(overall_df, "overall_R2"),
        "overall_MAE": overall_value(overall_df, "overall_MAE"),
        "overall_RMSE": overall_value(overall_df, "overall_RMSE"),
        "overall_scaled_RMSE": overall_value(overall_df, "overall_scaled_RMSE"),
        "aroma_R2": aroma_row["R2"],
        "aroma_RMSE": aroma_row["RMSE"],
        "contains_HNAP_to_aroma": ("HNAP", "aroma") in pathways,
        "contains_LNAP_to_aroma": ("LNAP", "aroma") in pathways,
        "pathways": "; ".join([f"{p[0]}->{p[1]}" for p in pathways])
    }

def export_publication_style_approach_excel(
    out_path,
    model_name,
    approach_description,
    fit_result,
    pathways,
    metrics_df,
    overall_df,
    rates_df,
    sensitivity_df,
    flux_df,
    flux_summary_df,
    selection_table,
    ranked_table_name,
    complexity_sweep_df=None
):
    """
    Export one approach with sheet names and content arranged like the user's reference Excel files.
    """
    final_fit = make_final_LKM_fit_reference_style(fit_result)
    meta_severity = make_meta_severity_reference_style(fit_result)
    carbon_sev = add_severity_to_long_table(carbon_df, fit_result)
    piona_sev = add_severity_to_long_table(piona_df, fit_result)
    lumps_sev = add_severity_to_long_table(lump_df, fit_result)
    diagnostics = make_diagnostics_long(fit_result, model_name)
    final_params = make_final_severity_params(fit_result, pathways, model_name)
    model_comp_single = pd.DataFrame([
        make_approach_model_comparison_row(model_name, approach_description, fit_result, pathways, metrics_df, overall_df)
    ])

    # Keep rate table names exactly similar to reference output.
    rate_out = rates_df.drop(columns=["model"], errors="ignore")
    sens_out = sensitivity_df.drop(columns=["model"], errors="ignore")
    flux_summary_out = flux_summary_df.copy()
    if "model" in flux_summary_out.columns:
        flux_summary_out = flux_summary_out[flux_summary_out["model"] == model_name].drop(columns=["model"], errors="ignore")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Main reference-style sheets
        final_fit.to_excel(writer, sheet_name="final_fit", index=False)
        metrics_df.to_excel(writer, sheet_name="pool_metrics", index=False)
        overall_df.to_excel(writer, sheet_name="overall_metrics", index=False)
        rate_out.to_excel(writer, sheet_name="rate_constants", index=False)
        sens_out.to_excel(writer, sheet_name="pathway_sensitivity", index=False)
        flux_summary_out.to_excel(writer, sheet_name="flux_summary", index=False)
        flux_df.to_excel(writer, sheet_name="flux_long", index=False)
        diagnostics.to_excel(writer, sheet_name="diagnostics_long", index=False)

        # Selection and reduction details
        candidate_df.to_excel(writer, sheet_name="all_candidate_models", index=False)
        selection_table.to_excel(writer, sheet_name=ranked_table_name, index=False)
        if complexity_sweep_df is not None:
            complexity_sweep_df.to_excel(writer, sheet_name="complexity_weight_sweep", index=False)

        # Full 36-pathway reference details
        full_rate_df.drop(columns=["model"], errors="ignore").to_excel(writer, sheet_name="full36_rate_constants", index=False)
        full_flux_summary_df.drop(columns=["model"], errors="ignore").to_excel(writer, sheet_name="full36_flux_summary", index=False)
        full_sensitivity_df.to_excel(writer, sheet_name="full36_sensitivity", index=False)

        # Severity/data transformation details
        final_params.to_excel(writer, sheet_name="final_severity_params", index=False)
        scale_table.to_excel(writer, sheet_name="pool_scale", index=False)
        scale_table.to_excel(writer, sheet_name="pool_scale_used", index=False)
        model_comp_single.to_excel(writer, sheet_name="model_comparison", index=False)
        meta.to_excel(writer, sheet_name="meta", index=False)
        meta_severity.to_excel(writer, sheet_name="meta_severity", index=False)
        carbon_sev.to_excel(writer, sheet_name="carbon_vs_severity", index=False)
        piona_sev.to_excel(writer, sheet_name="PIONA_vs_severity", index=False)
        lumps_sev.to_excel(writer, sheet_name="lumps_vs_severity", index=False)

        # Extra raw composition tables for convenience
        carbon_df.to_excel(writer, sheet_name="carbon_vs_run", index=False)
        piona_df.to_excel(writer, sheet_name="PIONA_vs_run", index=False)
        lump_df.to_excel(writer, sheet_name="lumps_vs_run", index=False)

def export_combined_publication_style_excel(out_path):
    """
    Combined workbook containing all detailed sheets for full 36 + A/B/C comparison.
    """
    full_final_fit = make_final_LKM_fit_reference_style(full_fit)
    A_final_fit = make_final_LKM_fit_reference_style(final_A_fit)
    B_final_fit = make_final_LKM_fit_reference_style(final_B_fit)
    C_final_fit = make_final_LKM_fit_reference_style(final_C_fit)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Overall comparison
        model_comparison.to_excel(writer, sheet_name="model_comparison", index=False)
        combined_metrics.to_excel(writer, sheet_name="combined_pool_metrics", index=False)
        combined_overall.to_excel(writer, sheet_name="combined_overall_metrics", index=False)
        combined_rates.to_excel(writer, sheet_name="combined_rates", index=False)
        combined_sensitivity.to_excel(writer, sheet_name="combined_sensitivity", index=False)
        combined_flux_summary.to_excel(writer, sheet_name="combined_flux_summary", index=False)
        combined_flux.to_excel(writer, sheet_name="combined_flux_long", index=False)
        combined_fit_wide.to_excel(writer, sheet_name="combined_fit_wide", index=False)
        combined_diagnostics.to_excel(writer, sheet_name="combined_diagnostics_long", index=False)
        pool_r2_df.to_excel(writer, sheet_name="poolwise_R2", index=False)

        # Final fit tables in reference style for each model
        full_final_fit.to_excel(writer, sheet_name="full36_final_fit", index=False)
        A_final_fit.to_excel(writer, sheet_name="A_final_fit", index=False)
        B_final_fit.to_excel(writer, sheet_name="B_final_fit", index=False)
        C_final_fit.to_excel(writer, sheet_name="C_final_fit", index=False)

        # Metrics
        full_metrics_df.to_excel(writer, sheet_name="full36_pool_metrics", index=False)
        final_A_metrics.to_excel(writer, sheet_name="A_pool_metrics", index=False)
        final_B_metrics.to_excel(writer, sheet_name="B_pool_metrics", index=False)
        final_C_metrics.to_excel(writer, sheet_name="C_pool_metrics", index=False)
        full_overall_df.to_excel(writer, sheet_name="full36_overall_metrics", index=False)
        final_A_overall.to_excel(writer, sheet_name="A_overall_metrics", index=False)
        final_B_overall.to_excel(writer, sheet_name="B_overall_metrics", index=False)
        final_C_overall.to_excel(writer, sheet_name="C_overall_metrics", index=False)

        # Pathways and sensitivity
        full_rate_df.to_excel(writer, sheet_name="full36_rate_constants", index=False)
        final_A_rates.to_excel(writer, sheet_name="A_rate_constants", index=False)
        final_B_rates.to_excel(writer, sheet_name="B_rate_constants", index=False)
        final_C_rates.to_excel(writer, sheet_name="C_rate_constants", index=False)
        full_sensitivity_df.to_excel(writer, sheet_name="full36_sensitivity", index=False)
        final_A_sensitivity.to_excel(writer, sheet_name="A_pathway_sensitivity", index=False)
        final_B_sensitivity.to_excel(writer, sheet_name="B_pathway_sensitivity", index=False)
        final_C_sensitivity.to_excel(writer, sheet_name="C_pathway_sensitivity", index=False)

        # Flux
        full_flux_summary_df.to_excel(writer, sheet_name="full36_flux_summary", index=False)
        final_A_flux_summary.to_excel(writer, sheet_name="A_flux_summary", index=False)
        final_B_flux_summary.to_excel(writer, sheet_name="B_flux_summary", index=False)
        final_C_flux_summary.to_excel(writer, sheet_name="C_flux_summary", index=False)
        full_flux_df.to_excel(writer, sheet_name="full36_flux_long", index=False)
        final_A_flux.to_excel(writer, sheet_name="A_flux_long", index=False)
        final_B_flux.to_excel(writer, sheet_name="B_flux_long", index=False)
        final_C_flux.to_excel(writer, sheet_name="C_flux_long", index=False)

        # Selection process
        candidate_df.to_excel(writer, sheet_name="all_candidate_models", index=False)
        candidate_near_df.to_excel(writer, sheet_name="candidate_models_near", index=False)
        A_rank_df.to_excel(writer, sheet_name="A_balanced_ranked_models", index=False)
        prune_table_B.to_excel(writer, sheet_name="B_pruning_decision", index=False)
        C_rank_df.to_excel(writer, sheet_name="C_chemistry_ranked_models", index=False)
        A_complexity_sweep.to_excel(writer, sheet_name="A_complexity_weight_sweep", index=False)
        C_complexity_sweep.to_excel(writer, sheet_name="C_complexity_weight_sweep", index=False)

        # Severity/data transformation
        scale_table.to_excel(writer, sheet_name="pool_scale_used", index=False)
        meta.to_excel(writer, sheet_name="meta", index=False)
        make_meta_severity_reference_style(final_A_fit).to_excel(writer, sheet_name="A_meta_severity", index=False)
        make_meta_severity_reference_style(final_B_fit).to_excel(writer, sheet_name="B_meta_severity", index=False)
        make_meta_severity_reference_style(final_C_fit).to_excel(writer, sheet_name="C_meta_severity", index=False)
        add_severity_to_long_table(carbon_df, final_A_fit).to_excel(writer, sheet_name="A_carbon_vs_severity", index=False)
        add_severity_to_long_table(carbon_df, final_B_fit).to_excel(writer, sheet_name="B_carbon_vs_severity", index=False)
        add_severity_to_long_table(carbon_df, final_C_fit).to_excel(writer, sheet_name="C_carbon_vs_severity", index=False)
        add_severity_to_long_table(piona_df, final_A_fit).to_excel(writer, sheet_name="A_PIONA_vs_severity", index=False)
        add_severity_to_long_table(piona_df, final_B_fit).to_excel(writer, sheet_name="B_PIONA_vs_severity", index=False)
        add_severity_to_long_table(piona_df, final_C_fit).to_excel(writer, sheet_name="C_PIONA_vs_severity", index=False)
        add_severity_to_long_table(lump_df, final_A_fit).to_excel(writer, sheet_name="A_lumps_vs_severity", index=False)
        add_severity_to_long_table(lump_df, final_B_fit).to_excel(writer, sheet_name="B_lumps_vs_severity", index=False)
        add_severity_to_long_table(lump_df, final_C_fit).to_excel(writer, sheet_name="C_lumps_vs_severity", index=False)

        # PIONA matrices
        pd.DataFrame(PIONA_exp, columns=PIONA_NAMES).to_excel(writer, sheet_name="PIONA_exp", index=False)
        pd.DataFrame(PIONA_A, columns=PIONA_NAMES).to_excel(writer, sheet_name="PIONA_A", index=False)
        pd.DataFrame(PIONA_B, columns=PIONA_NAMES).to_excel(writer, sheet_name="PIONA_B", index=False)
        pd.DataFrame(PIONA_C, columns=PIONA_NAMES).to_excel(writer, sheet_name="PIONA_C", index=False)

# Detailed individual workbooks
detailed_A_xlsx = DETAILED_OUTPUT_DIR / "Approach_A_outputs_like_reference.xlsx"
detailed_B_xlsx = DETAILED_OUTPUT_DIR / "Approach_B_outputs_like_reference.xlsx"
detailed_C_xlsx = DETAILED_OUTPUT_DIR / "Approach_C_outputs_like_reference.xlsx"
detailed_combined_xlsx = DETAILED_OUTPUT_DIR / "Combined_A_B_C_outputs_like_reference.xlsx"

export_publication_style_approach_excel(
    detailed_A_xlsx,
    "A_balanced_final",
    "equal_weight_balanced_selection",
    final_A_fit,
    PATHWAYS_A,
    final_A_metrics,
    final_A_overall,
    final_A_rates,
    final_A_sensitivity,
    final_A_flux,
    final_A_flux_summary,
    A_rank_df,
    "balanced_ranked_models",
    complexity_sweep_df=A_complexity_sweep
)

export_publication_style_approach_excel(
    detailed_B_xlsx,
    "B_sensitivity_pruned_final",
    "strict_sensitivity_pruning",
    final_B_fit,
    PATHWAYS_B,
    final_B_metrics,
    final_B_overall,
    final_B_rates,
    final_B_sensitivity,
    final_B_flux,
    final_B_flux_summary,
    prune_table_B,
    "pruning_decision",
    complexity_sweep_df=None
)

export_publication_style_approach_excel(
    detailed_C_xlsx,
    "C_chemistry_constrained_final",
    "chemistry_constrained_balanced_selection",
    final_C_fit,
    PATHWAYS_C,
    final_C_metrics,
    final_C_overall,
    final_C_rates,
    final_C_sensitivity,
    final_C_flux,
    final_C_flux_summary,
    C_rank_df,
    "chemistry_ranked_models",
    complexity_sweep_df=C_complexity_sweep
)

export_combined_publication_style_excel(detailed_combined_xlsx)

print("\nPublication-style detailed Excel files:")
print("A detailed:", detailed_A_xlsx)
print("B detailed:", detailed_B_xlsx)
print("C detailed:", detailed_C_xlsx)
print("Combined detailed:", detailed_combined_xlsx)


#  Exporting the outputs in ZIP format
zip_path = Path("three_approaches_OCEM_outputs.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_xlsx, arcname=out_xlsx.name)

    # Add three separate approach-specific Excel files to the ZIP
    z.write(out_A_xlsx, arcname=out_A_xlsx.name)
    z.write(out_B_xlsx, arcname=out_B_xlsx.name)
    z.write(out_C_xlsx, arcname=out_C_xlsx.name)

    # Add paper-style detailed Excel files that mirror the reference workbooks
    z.write(detailed_A_xlsx, arcname=f"publication_style_detailed_outputs/{detailed_A_xlsx.name}")
    z.write(detailed_B_xlsx, arcname=f"publication_style_detailed_outputs/{detailed_B_xlsx.name}")
    z.write(detailed_C_xlsx, arcname=f"publication_style_detailed_outputs/{detailed_C_xlsx.name}")
    z.write(detailed_combined_xlsx, arcname=f"publication_style_detailed_outputs/{detailed_combined_xlsx.name}")

    for folder_name, folder_path in [
        ("plots", PLOT_DIR),
        ("bar_plots", BAR_DIR),
        ("PIONA_bar_plots", PIONA_DIR),
        ("parity_plots", PARITY_DIR),
        ("Approach_A_separate_plots", A_PLOT_DIR),
        ("Approach_B_separate_plots", B_PLOT_DIR),
        ("Approach_C_separate_plots", C_PLOT_DIR),
        ("Combined_comparison_plots", COMBINED_PLOT_DIR),
    ]:
        for fp in folder_path.glob("*.png"):
            z.write(fp, arcname=f"{folder_name}/{fp.name}")

print("\nDONE.")
print("Combined Excel:", out_xlsx)
print("Approach A Excel:", out_A_xlsx)
print("Approach B Excel:", out_B_xlsx)
print("Approach C Excel:", out_C_xlsx)
print("ZIP:", zip_path)

if IN_COLAB:
    files.download(str(out_xlsx))
    files.download(str(out_A_xlsx))
    files.download(str(out_B_xlsx))
    files.download(str(out_C_xlsx))
    files.download(str(detailed_A_xlsx))
    files.download(str(detailed_B_xlsx))
    files.download(str(detailed_C_xlsx))
    files.download(str(detailed_combined_xlsx))
    files.download(str(zip_path))


