# ============================================================
# CLEAN END-TO-END PIPELINE
# Code-A strategy + carbon-derived Mw + BP + transport + carbon validation
#
# Inputs:
#   1) LKM prediction workbook (e.g. Combined_A_B_C_outputs_like_reference.xlsx)
#      containing A_final_fit / B_final_fit / C_final_fit / full36_final_fit
#      or combined_fit_wide / final_LKM_fit
#   2) Raw detailed C5-C45 PIONA Excel files
#
# Outputs:
#   - reconstructed pseudo-C5-C45 distributions from LKM predicted lumps
#   - full C5-C45 distributions from PIONA
#   - BP curves and distillation points
#   - Mw, density, API from carbon-family distribution
#   - transport properties: mu, D_eff, L_eff, Da_mob, Da_ext, Da_heat
#   - carbon-number validation: integer C profiles, C10/C50/C90, HTI/LTI, moments
#   - plots and Excel workbook
# ============================================================

!pip install openpyxl xlsxwriter scipy -q

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

# ============================================================
# 1. USER INPUT
# ============================================================

print("Upload:")
print("1) LKM prediction workbook, e.g. Combined_A_B_C_outputs_like_reference.xlsx")
print("2) Raw detailed C5-C45 PIONA Excel files")
uploaded = files.upload()

uploaded_files = list(uploaded.keys())

lkm_file = None
piona_files = []

# Prefer a workbook with Combined/LKM/output in name as LKM file.
for f in uploaded_files:
    fl = f.lower()
    if f.endswith(".xlsx") and lkm_file is None and (
        "combined" in fl or "lkm" in fl or "relative_severity" in fl or "outputs" in fl
    ):
        lkm_file = f
    else:
        piona_files.append(f)

if lkm_file is None:
    raise ValueError("Could not identify LKM workbook. Rename it with Combined/LKM/outputs, or set lkm_file manually.")

if len(piona_files) == 0:
    raise ValueError("No raw C5-C45 PIONA files uploaded.")

OUTDIR = Path("FINAL_CodeA_CarbonValidation_BP_Transport")
OUTDIR.mkdir(exist_ok=True)
PLOT_DIR = OUTDIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)
FIG_DPI = 500

# ============================================================
# 2. SETTINGS
# ============================================================

FAMILY_COLS = ["Paraffins", "Isoparaffins", "Olefins", "Naphthenes", "Aromatics"]
LUMPS = ["HP", "HIP", "LP", "LIP", "HO", "HNAP", "LO", "LNAP", "aroma"]

LUMP_TO_FAMILY = {
    "LP": "Paraffins", "HP": "Paraffins",
    "LIP": "Isoparaffins", "HIP": "Isoparaffins",
    "LO": "Olefins", "HO": "Olefins",
    "LNAP": "Naphthenes", "HNAP": "Naphthenes",
    "aroma": "Aromatics",
}

ZONE_DEFINITIONS = {
    "LP": [("front_light", 5, 8), ("mid_light", 9, 13), ("upper_light", 14, 19)],
    "LIP": [("front_light", 5, 8), ("mid_light", 9, 13), ("upper_light", 14, 19)],
    "LO": [("front_light", 5, 8), ("mid_light", 9, 13), ("upper_light", 14, 19)],
    "LNAP": [("front_light", 5, 8), ("mid_light", 9, 13), ("upper_light", 14, 19)],
    "HP": [("lower_heavy", 20, 26), ("mid_heavy", 27, 34), ("upper_heavy", 35, 45)],
    "HIP": [("lower_heavy", 20, 26), ("mid_heavy", 27, 34), ("upper_heavy", 35, 45)],
    "HO": [("lower_heavy", 20, 26), ("mid_heavy", 27, 34), ("upper_heavy", 35, 45)],
    "HNAP": [("lower_heavy", 20, 26), ("mid_heavy", 27, 34), ("upper_heavy", 35, 45)],
    "aroma": [("light_aroma", 6, 10), ("mid_aroma", 11, 18), ("heavy_aroma", 19, 35)],
}

C_MIN = 5
C_MAX = 45
LIGHT_C_MAX = 20
CUT_POINTS = [1, 5, 10, 30, 50, 70, 90, 95, 99]
CARBON_QUANTILES = [10, 50, 90]
MW_C = 12.011
MW_H = 1.008

# BP and molar volume anchors from the previous Code B/C style.
ANCHORS = {
    "Paraffins": {"family_head": "Methane", "C_ref": 1, "Tb_ref_K": 111.15, "Vm20_ref_cm3mol": 47.9},
    "Isoparaffins": {"family_head": "2,2-dimethylbutane", "C_ref": 6, "Tb_ref_K": 322.15, "Vm20_ref_cm3mol": 90.1},
    "Olefins": {"family_head": "Ethylene", "C_ref": 2, "Tb_ref_K": 169.45, "Vm20_ref_cm3mol": 48.0},
    "Naphthenes": {"family_head": "Cyclohexane", "C_ref": 6, "Tb_ref_K": 354.00, "Vm20_ref_cm3mol": 113.3},
    "Aromatics": {"family_head": "Benzene", "C_ref": 6, "Tb_ref_K": 353.00, "Vm20_ref_cm3mol": 92.8},
}

# Transport constants / priors
R = 8.314
kB = 1.380649e-23
NA = 6.02214076e23
T_ref_kin_K = 425.0 + 273.15
k_ref = 1.0e-2
Ea_app = 200e3
T_visc_ref_K = 298.15
mu_SCW = 4.0e-5
rho_SCW = 200.0
k_SCW = 0.14
Cp_SCW = 6500.0
D_SCW = 2.0e-7
N_stir = 5.0
d_hydro = 0.05
L_wax = 8e-4
L_oil = 1e-4
L_min = 5e-6
L_max = 1e-3
mu_ref = 1e-4
p_mu = 0.20
eps_min = 0.05
eps_max = 0.85
eps_alpha = 1.0
tortuosity_m = 1.5
shape_factor_Rh = 1.5
phi_HC_low = 0.85
phi_HC_high = 0.35
THERMAL = {
    "wax": {"k": 0.25, "rho": 840.0, "Cp": 2200.0},
    "oil": {"k": 0.15, "rho": 720.0, "Cp": 2500.0},
}
FAMILY_VISC = {
    "Paraffins": {"mu0": 0.010, "E": 25000},
    "Isoparaffins": {"mu0": 0.008, "E": 23000},
    "Olefins": {"mu0": 0.006, "E": 22000},
    "Naphthenes": {"mu0": 0.015, "E": 28000},
    "Aromatics": {"mu0": 0.020, "E": 28000},
}

# ============================================================
# 3. HELPER / PROPERTY FUNCTIONS
# ============================================================

def clean_name(path):
    name = Path(path).stem
    name = re.sub(r"\(\d+\)$", "", name).strip()
    return name


def parse_filename(path):
    name = clean_name(path)
    m = re.match(
        r"(?P<T>\d+(?:\.\d+)?)_"
        r"(?P<heat>\d+(?:\.\d+)?)_"
        r"(?P<hold>\d+(?:\.\d+)?)_"
        r"(?P<label>.+)$",
        name,
    )
    if m is None:
        return {"run_id": name, "T_set_C": np.nan, "t_heat_min": np.nan, "t_hold_min": np.nan, "label": name}
    return {
        "run_id": name,
        "T_set_C": float(m.group("T")),
        "t_heat_min": float(m.group("heat")),
        "t_hold_min": float(m.group("hold")),
        "label": m.group("label"),
    }


def find_col(df, names):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for n in names:
        key = n.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def assign_lump(family, C):
    C = int(C)
    if family == "Aromatics":
        return "aroma"
    if family == "Paraffins":
        return "LP" if C < 20 else "HP"
    if family == "Isoparaffins":
        return "LIP" if C < 20 else "HIP"
    if family == "Olefins":
        return "LO" if C < 20 else "HO"
    if family == "Naphthenes":
        return "LNAP" if C < 20 else "HNAP"
    return None


def assign_zone(lump, C):
    if lump not in ZONE_DEFINITIONS:
        return None
    for zone_name, cmin, cmax in ZONE_DEFINITIONS[lump]:
        if int(C) >= cmin and int(C) <= cmax:
            return zone_name
    return None


def tb_from_family_anchor(C, family):
    a = ANCHORS[family]
    val = np.exp(a["Tb_ref_K"] / 307.63) + 0.31012 * (float(C) - a["C_ref"])
    if val <= 0:
        return np.nan
    return 307.63 * np.log(val) - 273.15


def molecular_weight(C, family):
    C = float(C)
    if family in ["Paraffins", "Isoparaffins"]:
        H = 2 * C + 2
    elif family in ["Olefins", "Naphthenes"]:
        H = 2 * C
    elif family == "Aromatics":
        H = max(2 * C - 6, 0)
    else:
        raise ValueError(f"Unknown family: {family}")
    return C * MW_C + H * MW_H


def molar_volume(C, family):
    a = ANCHORS[family]
    return max(a["Vm20_ref_cm3mol"] + 16.0 * (float(C) - a["C_ref"]), 1e-12)


def density_from_C_family(C, family):
    return molecular_weight(C, family) / molar_volume(C, family)


def mixture_density_specific_volume(weights, densities):
    denom = 0.0
    for w, rho in zip(weights, densities):
        if pd.notna(w) and pd.notna(rho) and rho > 0:
            denom += float(w) / float(rho)
    return 1.0 / denom if denom > 0 else np.nan


def api_gravity(rho_gml):
    return 141.5 / rho_gml - 131.5


def minmax(x, invert=False):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.isclose(np.nanmax(x) - np.nanmin(x), 0):
        y = np.zeros_like(x)
    else:
        y = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    return 1 - y if invert else y


def arrhenius_rate(T_K):
    return k_ref * np.exp((-Ea_app / R) * ((1 / T_K) - (1 / T_ref_kin_K)))


def hydrodynamic_radius(Mw_gmol, rho_gml):
    Mw_kgmol = Mw_gmol / 1000.0
    rho_kgm3 = rho_gml * 1000.0
    mol_volume = Mw_kgmol / (rho_kgm3 * NA)
    R_sphere = (3 * mol_volume / (4 * np.pi)) ** (1 / 3)
    return shape_factor_Rh * R_sphere


def interpolate_crossing(x, y, threshold):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    for i in range(len(x) - 1):
        y1 = y[i] - threshold
        y2 = y[i + 1] - threshold
        if y1 == 0:
            return x[i]
        if y1 * y2 < 0:
            return x[i] + (threshold - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return np.nan


def carbon_quantile(g, q):
    d = g.groupby("C", as_index=False)["mass_fraction"].sum().sort_values("C")
    if d.empty or d["mass_fraction"].sum() <= 0:
        return np.nan
    d["w"] = d["mass_fraction"] / d["mass_fraction"].sum()
    d["cum"] = d["w"].cumsum() * 100.0
    idx = (d["cum"] - q).abs().idxmin()
    return float(d.loc[idx, "C"])


def carbon_moments(g):
    d = g.groupby("C", as_index=False)["mass_fraction"].sum().sort_values("C")
    if d.empty or d["mass_fraction"].sum() <= 0:
        return pd.Series({})
    w = d["mass_fraction"].values.astype(float)
    w = w / w.sum()
    C = d["C"].values.astype(float)
    Cbar = np.sum(w * C)
    var = np.sum(w * (C - Cbar) ** 2)
    std = np.sqrt(var)
    skew = np.sum(w * (C - Cbar) ** 3) / (std ** 3 + 1e-30)
    lti = w[C < 15].sum()
    hti = w[C > 30].sum()
    return pd.Series({
        "Cbar_integer": Cbar,
        "C_std": std,
        "C_skew": skew,
        "C10": carbon_quantile(g, 10),
        "C50": carbon_quantile(g, 50),
        "C90": carbon_quantile(g, 90),
        "LTI_C_lt_15": lti,
        "HTI_C_gt_30": hti,
        "HTI_over_LTI": hti / max(lti, 1e-12),
    })

# ============================================================
# 4. READ LKM PREDICTION WORKBOOK
# ============================================================

xls = pd.ExcelFile(lkm_file)
preferred_sheets = ["A_final_fit", "B_final_fit", "C_final_fit", "full36_final_fit"]
available_model_sheets = [s for s in preferred_sheets if s in xls.sheet_names]

if len(available_model_sheets) == 0:
    if "combined_fit_wide" in xls.sheet_names:
        available_model_sheets = ["combined_fit_wide"]
    elif "final_LKM_fit" in xls.sheet_names:
        available_model_sheets = ["final_LKM_fit"]
    else:
        raise ValueError("No usable LKM prediction sheet found.")

lkm_frames = []
for sh in available_model_sheets:
    df = pd.read_excel(lkm_file, sheet_name=sh)
    case_col = find_col(df, ["run_id", "label", "Case_ID", "case_id", "file"])
    severity_col = find_col(df, ["relative_severity_S_over_Sref", "relative_severity", "severity_normalized", "S_rel", "S/Sref"])
    if case_col is None:
        df["case_id_auto"] = [f"case_{i+1}" for i in range(len(df))]
        case_col = "case_id_auto"
    if severity_col is None:
        raise ValueError(f"Severity column missing in sheet {sh}")
    if "T_set_C" not in df.columns:
        raise ValueError(f"T_set_C missing in sheet {sh}")
    if "label" not in df.columns:
        df["label"] = df[case_col].astype(str)
    df["case_id"] = df[case_col].astype(str).apply(lambda x: re.sub(r"\(\d+\)$", "", x).strip())
    df["relative_severity_S_over_Sref"] = pd.to_numeric(df[severity_col], errors="coerce")
    df["dS_rel_from_reference"] = df["relative_severity_S_over_Sref"] - 1.0
    if "model" not in df.columns:
        df["model"] = sh.replace("_final_fit", "")
    for lump in LUMPS:
        col = find_col(df, [f"{lump}_pred", f"{lump} pred", f"{lump}pred", lump])
        df[lump] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) if col is not None else 0.0
    total = df[LUMPS].sum(axis=1).replace(0, np.nan)
    for lump in LUMPS:
        df[lump] = df[lump] / total
    df[LUMPS] = df[LUMPS].fillna(0.0)
    keep_cols = ["model", "case_id", "label", "T_set_C", "t_heat_min", "t_hold_min", "relative_severity_S_over_Sref", "dS_rel_from_reference"] + LUMPS
    keep_cols = [c for c in keep_cols if c in df.columns]
    lkm_frames.append(df[keep_cols].copy())

lkm_pred = pd.concat(lkm_frames, ignore_index=True)
severity_ref = lkm_pred[["case_id", "label", "relative_severity_S_over_Sref", "dS_rel_from_reference"]].drop_duplicates(subset=["case_id"])
print("Loaded LKM prediction rows:", len(lkm_pred))
display(lkm_pred.head())

# ============================================================
# 5. READ RAW C5-C45 PIONA FILES
# ============================================================

def read_piona_file(path):
    meta = parse_filename(path)
    xl = pd.ExcelFile(path)
    selected = None
    selected_sheet = None
    for sh in xl.sheet_names:
        temp = pd.read_excel(path, sheet_name=sh)
        temp = temp.loc[:, ~temp.columns.astype(str).str.startswith("Unnamed")]
        temp.columns = [str(c).strip() for c in temp.columns]
        lower = {str(c).lower().strip(): c for c in temp.columns}
        cn_col = lower.get("carbonnumber", lower.get("carbon number", None))
        has_fams = all(f.lower() in lower for f in FAMILY_COLS)
        if cn_col is not None and has_fams:
            selected = temp
            selected_sheet = sh
            break
    if selected is None:
        raise ValueError(f"No valid PIONA sheet found in {path}")
    lower = {str(c).lower().strip(): c for c in selected.columns}
    rename = {}
    if "carbonnumber" in lower:
        rename[lower["carbonnumber"]] = "CarbonNumber"
    elif "carbon number" in lower:
        rename[lower["carbon number"]] = "CarbonNumber"
    for fam in FAMILY_COLS:
        rename[lower[fam.lower()]] = fam
    df = selected.rename(columns=rename)
    df = df[["CarbonNumber"] + FAMILY_COLS].copy()
    df["CarbonNumber"] = pd.to_numeric(df["CarbonNumber"], errors="coerce")
    for fam in FAMILY_COLS:
        df[fam] = pd.to_numeric(df[fam], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["CarbonNumber"])
    df["CarbonNumber"] = df["CarbonNumber"].astype(int)
    df = df[(df["CarbonNumber"] >= C_MIN) & (df["CarbonNumber"] <= C_MAX)].copy()
    long = df.melt(id_vars=["CarbonNumber"], value_vars=FAMILY_COLS, var_name="family", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce").fillna(0.0)
    long.loc[long["value"] < 0, "value"] = 0.0
    total = long["value"].sum()
    if total <= 0:
        raise ValueError(f"No positive PIONA signal in {path}")
    long["mass_fraction"] = long["value"] / total
    long["case_id"] = meta["run_id"]
    long["label"] = meta["label"]
    long["T_set_C"] = meta["T_set_C"]
    long["t_heat_min"] = meta["t_heat_min"]
    long["t_hold_min"] = meta["t_hold_min"]
    long["sheet_used"] = selected_sheet
    return long

piona_long = pd.concat([read_piona_file(f) for f in piona_files], ignore_index=True)
piona_long = piona_long.merge(severity_ref[["case_id", "relative_severity_S_over_Sref", "dS_rel_from_reference"]], on="case_id", how="left")
if piona_long["relative_severity_S_over_Sref"].isna().sum() > 0:
    piona_long = piona_long.drop(columns=["relative_severity_S_over_Sref", "dS_rel_from_reference"])
    piona_long = piona_long.merge(severity_ref[["label", "relative_severity_S_over_Sref", "dS_rel_from_reference"]], on="label", how="left")

piona_long["lump"] = piona_long.apply(lambda r: assign_lump(r["family"], r["CarbonNumber"]), axis=1)
piona_long["zone"] = piona_long.apply(lambda r: assign_zone(r["lump"], r["CarbonNumber"]), axis=1)
piona_long = piona_long.dropna(subset=["lump", "zone"]).copy()
print("Loaded PIONA rows:", len(piona_long))
display(piona_long.head())

# ============================================================
# 6. CODE-A STYLE SEVERITY-CONDITIONED ZONE + CARBON FINGERPRINTS
# ============================================================

zone_case_rows = []
for (case_id, lump), g in piona_long.groupby(["case_id", "lump"]):
    total_lump = g["value"].sum()
    if total_lump <= 0:
        continue
    S_rel = g["relative_severity_S_over_Sref"].iloc[0]
    dS_rel = g["dS_rel_from_reference"].iloc[0]
    for zone_name, cmin, cmax in ZONE_DEFINITIONS[lump]:
        z = g[g["zone"] == zone_name]
        zone_case_rows.append({
            "case_id": case_id, "lump": lump, "family": LUMP_TO_FAMILY[lump], "zone": zone_name,
            "C_min": cmin, "C_max": cmax, "relative_severity_S_over_Sref": S_rel,
            "dS_rel_from_reference": dS_rel, "zone_weight_exp": z["value"].sum() / total_lump,
        })
zone_case_df = pd.DataFrame(zone_case_rows)

zone_model_rows = []
for (lump, zone), g in zone_case_df.groupby(["lump", "zone"]):
    x = g["dS_rel_from_reference"].values.astype(float)
    y = g["zone_weight_exp"].values.astype(float)
    if len(g) >= 2 and np.nanstd(x) > 1e-12:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        intercept = np.nanmean(y); slope = 0.0
    ref = g.iloc[0]
    zone_model_rows.append({
        "lump": lump, "family": ref["family"], "zone": zone, "C_min": ref["C_min"], "C_max": ref["C_max"],
        "intercept": intercept, "slope": slope, "n_cases": len(g), "mean_exp_weight": np.nanmean(y),
    })
zone_model = pd.DataFrame(zone_model_rows)

carbon_fp_rows = []
for (lump, zone, C), g in piona_long.groupby(["lump", "zone", "CarbonNumber"]):
    rows = []
    for case_id, cg in g.groupby("case_id"):
        parent = piona_long[(piona_long["case_id"] == case_id) & (piona_long["lump"] == lump) & (piona_long["zone"] == zone)]
        zone_total = parent["value"].sum()
        if zone_total > 0:
            rows.append({
                "case_id": case_id, "lump": lump, "zone": zone, "CarbonNumber": C,
                "family": cg["family"].iloc[0], "dS_rel_from_reference": cg["dS_rel_from_reference"].iloc[0],
                "carbon_weight_exp_within_zone": cg["value"].sum() / zone_total,
            })
    if not rows:
        continue
    tmp = pd.DataFrame(rows)
    x = tmp["dS_rel_from_reference"].values.astype(float)
    y = tmp["carbon_weight_exp_within_zone"].values.astype(float)
    if len(tmp) >= 2 and np.nanstd(x) > 1e-12:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        intercept = np.nanmean(y); slope = 0.0
    carbon_fp_rows.append({
        "lump": lump, "zone": zone, "CarbonNumber": C, "family": tmp["family"].iloc[0],
        "intercept": intercept, "slope": slope, "mean_exp_weight": np.nanmean(y), "n_cases": len(tmp),
    })
carbon_fingerprint_model = pd.DataFrame(carbon_fp_rows)

# Add fallback integer carbon numbers in every zone so no zone is empty.
fallback_rows = []
for _, zr in zone_model.iterrows():
    lump, zone, fam = zr["lump"], zr["zone"], zr["family"]
    cmin, cmax = int(zr["C_min"]), int(zr["C_max"])
    existing = carbon_fingerprint_model[(carbon_fingerprint_model["lump"] == lump) & (carbon_fingerprint_model["zone"] == zone)]["CarbonNumber"].tolist()
    missing = [C for C in range(cmin, cmax + 1) if C not in existing]
    if missing:
        n = cmax - cmin + 1
        for C in missing:
            fallback_rows.append({
                "lump": lump, "zone": zone, "CarbonNumber": C, "family": fam,
                "intercept": 1.0 / n, "slope": 0.0, "mean_exp_weight": 1.0 / n, "n_cases": 0,
            })
if fallback_rows:
    carbon_fingerprint_model = pd.concat([carbon_fingerprint_model, pd.DataFrame(fallback_rows)], ignore_index=True)

# ============================================================
# 7. RECONSTRUCT PSEUDO C5-C45 DISTRIBUTION FROM LKM LUMPS
# ============================================================

def predict_zone_weights(lump, dS_rel):
    g = zone_model[zone_model["lump"] == lump].copy()
    if g.empty:
        raise ValueError(f"No zone mapping for lump {lump}")
    w = g["intercept"].values + g["slope"].values * float(dS_rel)
    w = np.maximum(w, 0.0)
    w = np.ones_like(w) / len(w) if w.sum() <= 0 else w / w.sum()
    g["zone_weight_pred"] = w
    return g


def predict_carbon_weights(lump, zone, dS_rel):
    g = carbon_fingerprint_model[(carbon_fingerprint_model["lump"] == lump) & (carbon_fingerprint_model["zone"] == zone)].copy()
    if g.empty:
        raise ValueError(f"No carbon fingerprint for {lump}/{zone}")
    w = g["intercept"].values + g["slope"].values * float(dS_rel)
    w = np.maximum(w, 0.0)
    w = np.ones_like(w) / len(w) if w.sum() <= 0 else w / w.sum()
    g["carbon_weight_pred"] = w
    return g

recon_rows = []
for _, row in lkm_pred.iterrows():
    for lump in LUMPS:
        lump_frac = float(row[lump])
        if lump_frac <= 0:
            continue
        zmap = predict_zone_weights(lump, row["dS_rel_from_reference"])
        for _, zr in zmap.iterrows():
            cmap = predict_carbon_weights(lump, zr["zone"], row["dS_rel_from_reference"])
            for _, cr in cmap.iterrows():
                recon_rows.append({
                    "source": "LKM_reconstructed", "model": row["model"], "case_id": row["case_id"], "label": row["label"],
                    "T_set_C": row["T_set_C"], "relative_severity_S_over_Sref": row["relative_severity_S_over_Sref"],
                    "dS_rel_from_reference": row["dS_rel_from_reference"], "lump": lump, "family": cr["family"],
                    "zone": zr["zone"], "CarbonNumber": int(cr["CarbonNumber"]),
                    "mass_fraction": lump_frac * zr["zone_weight_pred"] * cr["carbon_weight_pred"],
                })
recon_long = pd.DataFrame(recon_rows)
recon_long["mass_fraction"] = recon_long["mass_fraction"] / recon_long.groupby(["model", "case_id"])["mass_fraction"].transform("sum")

full_long = piona_long[["case_id", "label", "T_set_C", "relative_severity_S_over_Sref", "dS_rel_from_reference", "family", "CarbonNumber", "mass_fraction", "lump", "zone"]].copy()
full_long["model"] = "Full_C5C45_PIONA"
full_long["source"] = "Full_C5C45_PIONA"
full_long["mass_fraction"] = full_long["mass_fraction"] / full_long.groupby("case_id")["mass_fraction"].transform("sum")
combined_long = pd.concat([recon_long, full_long], ignore_index=True, sort=False)
combined_long["C"] = combined_long["CarbonNumber"].round().astype(int)

# ============================================================
# 8. INTEGER CARBON DISTRIBUTION + CARBON METRICS
# ============================================================

integer_carbon = (
    combined_long
    .groupby(["source", "model", "case_id", "label", "relative_severity_S_over_Sref", "dS_rel_from_reference", "C"], as_index=False)["mass_fraction"]
    .sum()
)
integer_carbon["mass_fraction"] = integer_carbon["mass_fraction"] / integer_carbon.groupby(["source", "model", "case_id"])["mass_fraction"].transform("sum")
integer_carbon["wt_percent"] = 100 * integer_carbon["mass_fraction"]

carbon_metrics = (
    integer_carbon
    .groupby(["source", "model", "case_id", "label", "relative_severity_S_over_Sref", "dS_rel_from_reference"])
    .apply(carbon_moments)
    .reset_index()
)

# Carbon profile comparison: LKM vs full for each integer C
full_int_ref = integer_carbon[integer_carbon["source"] == "Full_C5C45_PIONA"][["case_id", "C", "wt_percent"]].rename(columns={"wt_percent": "wt_percent_C5C45"})
lkm_int = integer_carbon[integer_carbon["source"] == "LKM_reconstructed"].copy()
carbon_profile_comparison = lkm_int.merge(full_int_ref, on=["case_id", "C"], how="left")
carbon_profile_comparison["wt_percent_error"] = carbon_profile_comparison["wt_percent"] - carbon_profile_comparison["wt_percent_C5C45"]

carbon_profile_metrics_rows = []
for model, g in carbon_profile_comparison.groupby("model"):
    temp = g[["case_id", "C", "wt_percent", "wt_percent_C5C45"]].dropna()
    if len(temp) == 0:
        continue
    carbon_profile_metrics_rows.append({
        "model": model,
        "n_points": len(temp),
        "carbon_profile_MAE_wt_percent": np.mean(np.abs(temp["wt_percent"] - temp["wt_percent_C5C45"])),
        "carbon_profile_RMSE_wt_percent": np.sqrt(np.mean((temp["wt_percent"] - temp["wt_percent_C5C45"]) ** 2)),
        "carbon_profile_bias_wt_percent": np.mean(temp["wt_percent"] - temp["wt_percent_C5C45"]),
        "carbon_profile_correlation": np.corrcoef(temp["wt_percent"], temp["wt_percent_C5C45"])[0, 1] if len(temp) >= 3 else np.nan,
    })
carbon_profile_metrics = pd.DataFrame(carbon_profile_metrics_rows)

# ============================================================
# 9. BP CURVES AND PROPERTY SUMMARY FROM CARBON-FAMILY DISTRIBUTION
# ============================================================

def summarize_distribution(g):
    weights = g["mass_fraction"].values.astype(float)
    weights = weights / max(weights.sum(), 1e-30)
    Cs = g["CarbonNumber"].values
    fams = g["family"].values
    MWs = np.array([molecular_weight(C, fam) for C, fam in zip(Cs, fams)])
    densities = np.array([density_from_C_family(C, fam) for C, fam in zip(Cs, fams)])
    Mw_app = np.sum(weights * MWs)
    rho_mix = mixture_density_specific_volume(weights, densities)
    API = api_gravity(rho_mix)
    Cbar = np.sum(weights * Cs)
    light = g.loc[g["CarbonNumber"] <= LIGHT_C_MAX, "mass_fraction"].sum()
    heavy = g.loc[g["CarbonNumber"] > LIGHT_C_MAX, "mass_fraction"].sum()
    arom = g.loc[g["family"] == "Aromatics", "mass_fraction"].sum()
    fam_sums = g.groupby("family")["mass_fraction"].sum().to_dict()
    return pd.Series({
        "Mw_app_gmol": Mw_app, "density_app_gml": rho_mix, "API_app": API, "Cbar": Cbar,
        "Light_frac": light, "Heavy_frac": heavy, "Aromatic_frac": arom,
        "P_frac": fam_sums.get("Paraffins", 0.0), "iP_frac": fam_sums.get("Isoparaffins", 0.0),
        "O_frac": fam_sums.get("Olefins", 0.0), "N_frac": fam_sums.get("Naphthenes", 0.0),
        "A_frac": fam_sums.get("Aromatics", 0.0), "Light_Heavy_ratio": light / max(heavy, 1e-12),
    })

summary = (
    combined_long
    .groupby(["source", "model", "case_id", "label", "T_set_C", "relative_severity_S_over_Sref", "dS_rel_from_reference"])
    .apply(summarize_distribution)
    .reset_index()
)
summary = summary.merge(
    carbon_metrics.drop(columns=["source", "label", "relative_severity_S_over_Sref", "dS_rel_from_reference"]),
    on=["model", "case_id"], how="left"
)

# BP curves and distillation points
bp_rows, dist_rows = [], []
for (source, model, case_id), g in combined_long.groupby(["source", "model", "case_id"]):
    meta = g.iloc[0]
    curve = g.copy()
    curve["boiling_point_C"] = curve.apply(lambda r: tb_from_family_anchor(r["CarbonNumber"], r["family"]), axis=1)
    curve = curve.groupby("boiling_point_C", as_index=False)["mass_fraction"].sum().sort_values("boiling_point_C")
    curve["mass_fraction"] = curve["mass_fraction"] / curve["mass_fraction"].sum()
    curve["cumulative_wt_percent"] = curve["mass_fraction"].cumsum() * 100.0
    curve["source"] = source; curve["model"] = model; curve["case_id"] = case_id; curve["label"] = meta["label"]
    curve["relative_severity_S_over_Sref"] = meta["relative_severity_S_over_Sref"]
    curve["dS_rel_from_reference"] = meta["dS_rel_from_reference"]
    bp_rows.append(curve)
    for cut in CUT_POINTS:
        idx = (curve["cumulative_wt_percent"] - cut).abs().idxmin()
        dist_rows.append({
            "source": source, "model": model, "case_id": case_id, "label": meta["label"],
            "relative_severity_S_over_Sref": meta["relative_severity_S_over_Sref"],
            "dS_rel_from_reference": meta["dS_rel_from_reference"],
            "distilled_wt_percent": cut, "pseudo_BP_C": curve.loc[idx, "boiling_point_C"],
        })
bp_curves = pd.concat(bp_rows, ignore_index=True)
dist_points = pd.DataFrame(dist_rows)

# ============================================================
# 10. TRANSPORT ANALYSIS
# ============================================================

def add_X_comp(df):
    df = df.copy()
    df["I_light"] = df.groupby("model")["Light_frac"].transform(lambda x: minmax(x))
    df["I_aroma"] = df.groupby("model")["Aromatic_frac"].transform(lambda x: minmax(x))
    df["I_heavy_inverse"] = df.groupby("model")["Heavy_frac"].transform(lambda x: minmax(x, invert=True))
    df["I_Cshift"] = df.groupby("model")["Cbar"].transform(lambda x: minmax(x, invert=True))
    df["X_comp"] = (df["I_light"] + df["I_aroma"] + df["I_heavy_inverse"] + df["I_Cshift"]) / 4.0
    return df

summary = add_X_comp(summary)

def calculate_viscosity(row):
    T_K = row["T_set_C"] + 273.15
    fam_fracs = {"Paraffins": row["P_frac"], "Isoparaffins": row["iP_frac"], "Olefins": row["O_frac"], "Naphthenes": row["N_frac"], "Aromatics": row["A_frac"]}
    ln_mu = 0.0
    for fam, frac in fam_fracs.items():
        mu0, E = FAMILY_VISC[fam]["mu0"], FAMILY_VISC[fam]["E"]
        mu_i = mu0 * np.exp((E / R) * ((1 / T_K) - (1 / T_visc_ref_K)))
        ln_mu += frac * np.log(max(mu_i, 1e-30))
    mu_HC = np.exp(ln_mu)
    phi_assoc = np.clip(row["Heavy_frac"] + 0.5 * row["Aromatic_frac"], 0.0, 0.63)
    phi_max, intrinsic_eta = 0.64, 2.5
    structure_factor = (1 - phi_assoc / phi_max) ** (-(intrinsic_eta * phi_max))
    mu_struct = mu_HC * structure_factor
    phi_HC = phi_HC_low - (phi_HC_low - phi_HC_high) * row["X_comp"]
    phi_HC = np.clip(phi_HC, 0.05, 0.98)
    mu_local = np.exp(phi_HC * np.log(max(mu_struct, 1e-30)) + (1 - phi_HC) * np.log(mu_SCW))
    return pd.Series({"mu_HC_Pas": mu_HC, "phi_assoc": phi_assoc, "structure_factor": structure_factor, "mu_struct_Pas": mu_struct, "phi_HC_local": phi_HC, "mu_local_Pas": mu_local})


def calculate_mobility(row):
    T_K = row["T_set_C"] + 273.15
    Rh = hydrodynamic_radius(row["Mw_app_gmol"], row["density_app_gml"])
    D_SE = kB * T_K / (6 * np.pi * row["mu_local_Pas"] * Rh)
    eps_mobile = eps_min + (eps_max - eps_min) * row["X_comp"]
    eps_mobile = np.clip(eps_mobile, 1e-4, 0.99)
    tau = eps_mobile ** (-tortuosity_m)
    D_eff = D_SE * eps_mobile ** eps_alpha / tau
    L_base = L_wax - (L_wax - L_oil) * row["X_comp"]
    L_eff = L_base * (row["mu_local_Pas"] / mu_ref) ** p_mu
    L_eff = np.clip(L_eff, L_min, L_max)
    k_rxn = arrhenius_rate(T_K)
    t_rxn = 1 / max(k_rxn, 1e-30)
    t_mob = L_eff ** 2 / max(D_eff, 1e-30)
    Da_mob = t_mob / t_rxn
    return pd.Series({"Rh_m": Rh, "D_SE_m2s": D_SE, "eps_mobile": eps_mobile, "tau": tau, "D_eff_m2s": D_eff, "L_eff_m": L_eff, "L_eff_um": L_eff * 1e6, "k_rxn_1s": k_rxn, "t_rxn_s": t_rxn, "t_mob_s": t_mob, "Da_mob": Da_mob})


def calculate_external(row):
    Re = rho_SCW * N_stir * d_hydro ** 2 / mu_SCW
    Sc = mu_SCW / (rho_SCW * D_SCW)
    Sh = 2 + 0.6 * np.sqrt(Re) * Sc ** (1 / 3)
    kL = Sh * D_SCW / max(row["L_eff_m"], 1e-30)
    t_ext = row["L_eff_m"] / max(kL, 1e-30)
    Da_ext = t_ext / max(row["t_rxn_s"], 1e-30)
    Bi_m = kL * (row["L_eff_m"] / 2) / max(row["D_eff_m2s"], 1e-30)
    return pd.Series({"Re": Re, "Sc": Sc, "Sh": Sh, "kL_ms": kL, "t_ext_s": t_ext, "Da_ext": Da_ext, "Bi_m": Bi_m})


def calculate_heat(row):
    X = row["X_comp"]
    k_HC = THERMAL["wax"]["k"] * (1 - X) + THERMAL["oil"]["k"] * X
    rho_HC = THERMAL["wax"]["rho"] * (1 - X) + THERMAL["oil"]["rho"] * X
    Cp_HC = THERMAL["wax"]["Cp"] * (1 - X) + THERMAL["oil"]["Cp"] * X
    phi = row["phi_HC_local"]
    k_local = phi * k_HC + (1 - phi) * k_SCW
    rhoCp_local = phi * rho_HC * Cp_HC + (1 - phi) * rho_SCW * Cp_SCW
    alpha_local = k_local / max(rhoCp_local, 1e-30)
    t_heat = row["L_eff_m"] ** 2 / max(alpha_local, 1e-30)
    Da_heat = t_heat / max(row["t_rxn_s"], 1e-30)
    return pd.Series({"k_HC_WmK": k_HC, "rho_HC_kgm3": rho_HC, "Cp_HC_JkgK": Cp_HC, "k_local_WmK": k_local, "rhoCp_local_Jm3K": rhoCp_local, "alpha_local_m2s": alpha_local, "t_heat_s": t_heat, "Da_heat": Da_heat})


def classify_regime(row):
    mob, ext, heat = row["Da_mob"], row["Da_ext"], row["Da_heat"]
    if mob < 0.1 and ext < 0.1 and heat < 0.1:
        return "reaction/product-evolution controlled"
    if mob >= 1 and ext < 1 and heat < 1:
        return "product-mobility influenced"
    if ext >= 1 and mob < 1 and heat < 1:
        return "external mass-transfer influenced"
    if heat >= 1 and mob < 1 and ext < 1:
        return "heat-transfer influenced"
    if mob >= 1 or ext >= 1 or heat >= 1:
        return "mixed transport influenced"
    return "transition regime"

summary = pd.concat([summary, summary.apply(calculate_viscosity, axis=1)], axis=1)
summary = pd.concat([summary, summary.apply(calculate_mobility, axis=1)], axis=1)
summary = pd.concat([summary, summary.apply(calculate_external, axis=1)], axis=1)
summary = pd.concat([summary, summary.apply(calculate_heat, axis=1)], axis=1)
summary["transport_regime"] = summary.apply(classify_regime, axis=1)
summary["I_mu"] = summary.groupby("model")["mu_local_Pas"].transform(lambda x: minmax(np.log10(x), invert=True))
summary["I_Deff"] = summary.groupby("model")["D_eff_m2s"].transform(lambda x: minmax(np.log10(x)))
summary["I_Damob"] = summary.groupby("model")["Da_mob"].transform(lambda x: minmax(np.log10(x), invert=True))
summary["PPQ_product_phase_quality"] = 0.40 * summary["X_comp"] + 0.20 * summary["I_mu"] + 0.20 * summary["I_Deff"] + 0.20 * summary["I_Damob"]

# ============================================================
# 11. COMPARISON AND METRICS
# ============================================================

compare_cols = [
    "Mw_app_gmol", "density_app_gml", "API_app", "Cbar", "Cbar_integer", "C_std", "C_skew", "C10", "C50", "C90", "LTI_C_lt_15", "HTI_C_gt_30", "HTI_over_LTI",
    "Light_frac", "Heavy_frac", "Aromatic_frac", "P_frac", "iP_frac", "O_frac", "N_frac", "A_frac",
    "mu_local_Pas", "D_eff_m2s", "L_eff_um", "Da_mob", "Da_ext", "Da_heat", "PPQ_product_phase_quality", "X_comp",
]
full_ref = summary[summary["source"] == "Full_C5C45_PIONA"][["case_id", "label"] + compare_cols + ["transport_regime"]].drop_duplicates(subset=["case_id"])
full_ref = full_ref.rename(columns={c: f"{c}_C5C45" for c in compare_cols})
full_ref = full_ref.rename(columns={"transport_regime": "transport_regime_C5C45"})
lkm_summary = summary[summary["source"] == "LKM_reconstructed"].copy()
comparison = lkm_summary.merge(full_ref, on=["case_id", "label"], how="left", validate="many_to_one")
if comparison["Mw_app_gmol_C5C45"].isna().sum() > 0:
    full_ref_label = full_ref.drop(columns=["case_id"]).drop_duplicates(subset=["label"])
    comparison = lkm_summary.merge(full_ref_label, on="label", how="left", validate="many_to_one")
for c in compare_cols:
    comparison[f"{c}_error_LKM_minus_C5C45"] = comparison[c] - comparison[f"{c}_C5C45"]
    comparison[f"{c}_ratio_LKM_over_C5C45"] = comparison[c] / comparison[f"{c}_C5C45"].replace(0, np.nan)
comparison["same_transport_regime_as_C5C45"] = comparison["transport_regime"] == comparison["transport_regime_C5C45"]

# BP point comparison
full_bp_ref = dist_points[dist_points["source"] == "Full_C5C45_PIONA"][["case_id", "distilled_wt_percent", "pseudo_BP_C"]].rename(columns={"pseudo_BP_C": "pseudo_BP_C_C5C45"})
lkm_bp = dist_points[dist_points["source"] == "LKM_reconstructed"].copy()
bp_comparison = lkm_bp.merge(full_bp_ref, on=["case_id", "distilled_wt_percent"], how="left")
bp_comparison["BP_error_LKM_minus_C5C45"] = bp_comparison["pseudo_BP_C"] - bp_comparison["pseudo_BP_C_C5C45"]

# Scalar metrics
metric_rows = []
for model, g in comparison.groupby("model"):
    for c in compare_cols:
        c2 = f"{c}_C5C45"
        temp = g[["case_id", c, c2]].dropna().drop_duplicates(subset=["case_id"])
        if len(temp) < 2:
            continue
        y, ref = temp[c].values, temp[c2].values
        metric_rows.append({"model": model, "quantity": c, "n_cases": len(temp), "MAE": np.mean(np.abs(y - ref)), "RMSE": np.sqrt(np.mean((y - ref) ** 2)), "Bias_LKM_minus_C5C45": np.mean(y - ref), "Correlation": np.corrcoef(y, ref)[0, 1] if len(temp) >= 3 else np.nan})
comparison_metrics = pd.DataFrame(metric_rows)

bp_metrics = []
for model, g in bp_comparison.groupby("model"):
    temp = g[["case_id", "distilled_wt_percent", "pseudo_BP_C", "pseudo_BP_C_C5C45"]].dropna()
    bp_metrics.append({"model": model, "n_points": len(temp), "BP_MAE_C": np.mean(np.abs(temp["pseudo_BP_C"] - temp["pseudo_BP_C_C5C45"])), "BP_RMSE_C": np.sqrt(np.mean((temp["pseudo_BP_C"] - temp["pseudo_BP_C_C5C45"]) ** 2)), "BP_Bias_C": np.mean(temp["pseudo_BP_C"] - temp["pseudo_BP_C_C5C45"]), "BP_Correlation": np.corrcoef(temp["pseudo_BP_C"], temp["pseudo_BP_C_C5C45"])[0, 1] if len(temp) >= 3 else np.nan})
bp_metrics = pd.DataFrame(bp_metrics)

regime_agreement = (
    comparison.drop_duplicates(subset=["model", "case_id"])
    .groupby("model")
    .agg(n_cases=("same_transport_regime_as_C5C45", "size"), n_same=("same_transport_regime_as_C5C45", "sum"), agreement_fraction=("same_transport_regime_as_C5C45", "mean"))
    .reset_index()
)

# Transition severities
transition_rows = []
for model, g in summary.groupby("model"):
    g = g.sort_values("relative_severity_S_over_Sref")
    ppq_max = g["PPQ_product_phase_quality"].max()
    ppq_s = g.loc[g["PPQ_product_phase_quality"] >= 0.90 * ppq_max, "relative_severity_S_over_Sref"]
    heavy_min = g["Heavy_frac"].min()
    heavy_s = g.loc[g["Heavy_frac"] <= 1.10 * heavy_min, "relative_severity_S_over_Sref"]
    A0, Amax = g["Aromatic_frac"].iloc[0], g["Aromatic_frac"].max()
    A_s = g.loc[g["Aromatic_frac"] >= A0 + 0.20 * (Amax - A0), "relative_severity_S_over_Sref"]
    transition_rows.append({
        "model": model, "source": g["source"].iloc[0],
        "S_Da_mob_equals_1": interpolate_crossing(g["relative_severity_S_over_Sref"], g["Da_mob"], 1.0),
        "S_Da_mob_equals_0p1": interpolate_crossing(g["relative_severity_S_over_Sref"], g["Da_mob"], 0.1),
        "S_Da_ext_equals_1": interpolate_crossing(g["relative_severity_S_over_Sref"], g["Da_ext"], 1.0),
        "S_Da_ext_equals_0p1": interpolate_crossing(g["relative_severity_S_over_Sref"], g["Da_ext"], 0.1),
        "S_Da_heat_equals_1": interpolate_crossing(g["relative_severity_S_over_Sref"], g["Da_heat"], 1.0),
        "S_Da_heat_equals_0p1": interpolate_crossing(g["relative_severity_S_over_Sref"], g["Da_heat"], 0.1),
        "S_PPQ_90percent_plateau": ppq_s.min() if len(ppq_s) else np.nan,
        "S_heavy_pool_depletion": heavy_s.min() if len(heavy_s) else np.nan,
        "S_aromatic_onset": A_s.min() if len(A_s) else np.nan,
        "PPQ_max": ppq_max, "Heavy_min": heavy_min, "Aromatic_max": Amax,
    })
transition_df = pd.DataFrame(transition_rows)

# ============================================================
# 12. PLOTS
# ============================================================

def savefig(name):
    path = PLOT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    return path

# Integer carbon distribution plots: each model/case as Code D style
for (source, model, case_id), sub in integer_carbon.groupby(["source", "model", "case_id"]):
    sub = sub.sort_values("C")
    S_rel = sub["relative_severity_S_over_Sref"].iloc[0]
    plt.figure(figsize=(8, 5))
    plt.bar(sub["C"], sub["wt_percent"], width=0.8)
    plt.xlabel("Carbon number")
    plt.ylabel("Total wt% per carbon number")
    plt.title(f"Integer carbon distribution: {model} | {case_id}\nS/Sref = {S_rel:.3f}")
    plt.xlim(4.5, 45.5)
    plt.grid(axis="y", alpha=0.3)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{model}_{case_id}")
    savefig(f"integer_carbon_distribution_{safe}.png")

# Overlay all cases for full C5-C45 and first LKM models
for model in list(lkm_summary["model"].unique())[:3] + ["Full_C5C45_PIONA"]:
    source = "Full_C5C45_PIONA" if model == "Full_C5C45_PIONA" else "LKM_reconstructed"
    sub_all = integer_carbon[(integer_carbon["model"] == model) & (integer_carbon["source"] == source)]
    if sub_all.empty:
        continue
    plt.figure(figsize=(10, 6))
    for case_id, sub in sub_all.groupby("case_id"):
        sub = sub.sort_values("C")
        plt.plot(sub["C"], sub["wt_percent"], marker="o", linewidth=1.5, label=case_id)
    plt.xlabel("Carbon number")
    plt.ylabel("Total wt% per carbon number")
    plt.title(f"Integer carbon distribution comparison: {model}")
    plt.xlim(4.5, 45.5)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    savefig(f"integer_carbon_overlay_{model}.png")

# Carbon profile parity
plt.figure(figsize=(7, 7))
for model, g in carbon_profile_comparison.groupby("model"):
    plt.scatter(g["wt_percent_C5C45"], g["wt_percent"], s=18, alpha=0.7, label=model)
vals = pd.concat([carbon_profile_comparison["wt_percent"], carbon_profile_comparison["wt_percent_C5C45"]]).dropna()
if len(vals):
    plt.plot([vals.min(), vals.max()], [vals.min(), vals.max()], "k--")
plt.xlabel("Full C5-C45 wt% per carbon")
plt.ylabel("LKM reconstructed wt% per carbon")
plt.title("Integer carbon profile parity")
plt.grid(True, linestyle=":")
plt.legend(fontsize=8)
savefig("integer_carbon_profile_parity.png")

# Da_mob vs C50 map
plt.figure(figsize=(8, 6))
for model, g in lkm_summary.groupby("model"):
    plt.loglog(g["C50"], g["Da_mob"], "o", label=f"{model} LKM")
fg = summary[summary["source"] == "Full_C5C45_PIONA"]
plt.loglog(fg["C50"], fg["Da_mob"], "ks", label="Full C5-C45")
plt.axhline(1, color="black", linestyle="--")
plt.axhline(0.1, color="black", linestyle=":")
plt.xlabel("C50 carbon number")
plt.ylabel("Da_mob")
plt.title("Mobility regime vs median carbon number")
plt.grid(True, which="both", linestyle=":")
plt.legend(fontsize=8)
savefig("Damob_vs_C50.png")

# BP parity
plt.figure(figsize=(7, 7))
for model, g in bp_comparison.groupby("model"):
    plt.scatter(g["pseudo_BP_C_C5C45"], g["pseudo_BP_C"], s=45, label=model)
vals = pd.concat([bp_comparison["pseudo_BP_C"], bp_comparison["pseudo_BP_C_C5C45"]]).dropna()
if len(vals):
    plt.plot([vals.min(), vals.max()], [vals.min(), vals.max()], "k--")
plt.xlabel("Full C5-C45 pseudo BP [°C]")
plt.ylabel("LKM reconstructed pseudo BP [°C]")
plt.title("BP curve point parity")
plt.grid(True, linestyle=":")
plt.legend(fontsize=8)
savefig("BP_point_parity.png")

# Trend plots
trend_cols = ["Mw_app_gmol", "Cbar", "C50", "HTI_C_gt_30", "mu_local_Pas", "D_eff_m2s", "L_eff_um", "Da_mob", "Da_ext", "Da_heat", "PPQ_product_phase_quality"]
for c in trend_cols:
    plt.figure(figsize=(9, 6))
    for model, g in lkm_summary.groupby("model"):
        g = g.sort_values("relative_severity_S_over_Sref")
        if c in ["mu_local_Pas", "D_eff_m2s", "Da_mob", "Da_ext", "Da_heat"]:
            plt.semilogy(g["relative_severity_S_over_Sref"], g[c], "o-", label=f"{model} LKM")
        else:
            plt.plot(g["relative_severity_S_over_Sref"], g[c], "o-", label=f"{model} LKM")
    fg = summary[summary["source"] == "Full_C5C45_PIONA"].sort_values("relative_severity_S_over_Sref")
    if c in ["mu_local_Pas", "D_eff_m2s", "Da_mob", "Da_ext", "Da_heat"]:
        plt.semilogy(fg["relative_severity_S_over_Sref"], fg[c], "k--o", linewidth=2.5, label="Full C5-C45")
    else:
        plt.plot(fg["relative_severity_S_over_Sref"], fg[c], "k--o", linewidth=2.5, label="Full C5-C45")
    if c.startswith("Da"):
        plt.axhline(1, color="black", linestyle="--")
        plt.axhline(0.1, color="black", linestyle=":")
    plt.xlabel("Relative severity, S/Sref")
    plt.ylabel(c)
    plt.title(f"{c}: LKM reconstruction vs full C5-C45")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=8)
    savefig(f"trend_{c}.png")

# Regime map
plt.figure(figsize=(8, 6))
for model, g in lkm_summary.groupby("model"):
    plt.loglog(g["Da_mob"], g["Da_heat"], "o", label=f"{model} LKM")
plt.loglog(fg["Da_mob"], fg["Da_heat"], "ks", label="Full C5-C45")
plt.axvline(1, color="black", linestyle="--")
plt.axhline(1, color="black", linestyle="--")
plt.axvline(0.1, color="black", linestyle=":")
plt.axhline(0.1, color="black", linestyle=":")
plt.xlabel("Da_mob")
plt.ylabel("Da_heat")
plt.title("Transport regime map")
plt.grid(True, which="both", linestyle=":")
plt.legend(fontsize=8)
savefig("regime_map_Damob_Daheat.png")

# ============================================================
# 13. EXPORT
# ============================================================

formulae = pd.DataFrame({
    "Quantity": ["Zone model", "Carbon fingerprint", "Reconstructed mass", "Integer carbon wt%", "C10/C50/C90", "HTI/LTI", "Mw", "Density", "BP", "Da_mob", "Da_ext", "Da_heat"],
    "Formula / definition": [
        "w_zone = intercept + slope*(S_rel - 1), clipped and normalized",
        "w_C|zone = intercept + slope*(S_rel - 1), clipped and normalized",
        "m_lump,zone,C = y_lump * w_zone * w_C|zone",
        "wt%_C = 100 * sum_family(m_C,f) / sum_all(m)",
        "Carbon numbers at 10, 50, 90 cumulative wt%",
        "HTI=sum(C>30); LTI=sum(C<15); HTI/LTI",
        "Mw = sum(w_C,f * MW_C,f), from reconstructed/full carbon-family distribution",
        "rho = 1 / sum(w_i/rho_i), specific-volume mixing",
        "Pseudo BP from family-anchor carbon-number relation",
        "Da_mob = (L_eff^2/D_eff)/t_rxn",
        "Da_ext = (L_eff/kL)/t_rxn",
        "Da_heat = (L_eff^2/alpha)/t_rxn",
    ],
})
interpretation = pd.DataFrame({
    "Item": ["Purpose", "Main addition from Code D/E", "Main improvement", "Best use", "Limitation"],
    "Explanation": [
        "Expand LKM lumps into pseudo-C5-C45, calculate BP/properties/transport, and validate against full C5-C45.",
        "Integer carbon distributions, carbon profile parity, C10/C50/C90, heavy-tail/light-tail indices, and Da_mob vs C50.",
        "Mw and transport are calculated from reconstructed carbon-family distributions, not fixed lump MW.",
        "Use carbon metrics to diagnose whether reconstruction delays/over-retains heavy tail and whether mobility transition is preserved.",
        "Reconstruction remains calibrated within the experimental severity range and should not be used as universal scale-up prediction.",
    ],
})

out_excel = OUTDIR / "FINAL_CodeA_CarbonValidation_BP_Transport_results.xlsx"
with pd.ExcelWriter(out_excel, engine="xlsxwriter") as writer:
    lkm_pred.to_excel(writer, sheet_name="input_LKM_predictions", index=False)
    piona_long.to_excel(writer, sheet_name="input_full_PIONA_long", index=False)
    zone_case_df.to_excel(writer, sheet_name="zone_weights_by_case", index=False)
    zone_model.to_excel(writer, sheet_name="CodeA_zone_model", index=False)
    carbon_fingerprint_model.to_excel(writer, sheet_name="carbon_fingerprint_model", index=False)
    recon_long.to_excel(writer, sheet_name="LKM_reconstructed_long", index=False)
    full_long.to_excel(writer, sheet_name="Full_C5C45_long", index=False)
    combined_long.to_excel(writer, sheet_name="combined_long", index=False)
    integer_carbon.to_excel(writer, sheet_name="integer_carbon_wt_percent", index=False)
    carbon_metrics.to_excel(writer, sheet_name="carbon_metrics", index=False)
    carbon_profile_comparison.to_excel(writer, sheet_name="carbon_profile_comparison", index=False)
    carbon_profile_metrics.to_excel(writer, sheet_name="carbon_profile_metrics", index=False)
    bp_curves.to_excel(writer, sheet_name="BP_curves", index=False)
    dist_points.to_excel(writer, sheet_name="distillation_points", index=False)
    bp_comparison.to_excel(writer, sheet_name="BP_point_comparison", index=False)
    bp_metrics.to_excel(writer, sheet_name="BP_curve_metrics", index=False)
    summary.to_excel(writer, sheet_name="property_transport_summary", index=False)
    comparison.to_excel(writer, sheet_name="LKM_vs_C5C45_comparison", index=False)
    comparison_metrics.to_excel(writer, sheet_name="property_metrics", index=False)
    regime_agreement.to_excel(writer, sheet_name="regime_agreement", index=False)
    transition_df.to_excel(writer, sheet_name="transition_severities", index=False)
    formulae.to_excel(writer, sheet_name="formulae", index=False)
    interpretation.to_excel(writer, sheet_name="interpretation", index=False)

zip_path = OUTDIR / "FINAL_CodeA_CarbonValidation_BP_Transport_plots.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in PLOT_DIR.glob("*.png"):
        z.write(p, arcname=p.name)

print("Saved Excel:", out_excel)
print("Saved plots:", zip_path)
files.download(str(out_excel))
files.download(str(zip_path))
