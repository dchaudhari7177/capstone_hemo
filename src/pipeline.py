"""Shared pipeline for PW26_DN_03 - FVIII inhibitor risk in haemophilia A (EAHAD/HADB F8 data).

Cleaning, biology reference tables and feature sets are copied verbatim from notebook 02
(02_model_improvement_and_literature_comparison.ipynb), so notebook 03 trains and explains exactly
the model that notebook 02 evaluated. Every feature is computed row-wise from one report and uses no
labels, so nothing here can leak across folds.
"""
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

N_JOBS = 8
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_raw(kind):
    """Locate mmc2 / mmc3: data/raw/ of this project first, then the working directory and ~/Downloads."""
    names = [f"{kind}.csv", f"BVTH_VTH-2024-000215-{kind}.csv"]
    roots = [PROJECT_ROOT, Path.cwd(), *Path.cwd().parents, Path.home() / "Downloads"]
    for root in roots:
        for sub in ["data/raw", "data/raw/hadb", "."]:
            for n in names:
                p = root / sub / n
                if p.exists():
                    return p
    raise FileNotFoundError(f"{kind}.csv not found - put it in data/raw/")


MMC2, MMC3 = find_raw("mmc2"), find_raw("mmc3")

# ---------------------------------------------------------------- cleaning
MISSING = {"", "-", "?", "na", "nan", "none", "not", "not r", "not report", "not reported", "null", "n/a"}

def clean_str(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in MISSING else s

def variant_key(m2):
    """Union-find: mut_ids sharing an HGVS cDNA or protein name are one biological variant."""
    parent = {m: m for m in m2["mut_id"]}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for col in ("mut_syn", "aa_syn"):
        first = {}
        for mid, name in zip(m2["mut_id"], m2[col].map(lambda s: clean_str(s).lower())):
            if not name:
                continue
            if name in first:
                a, b = find(mid), find(first[name])
                if a != b:
                    parent[a] = b
            else:
                first[name] = mid
    return {m: "v" + find(m) for m in parent}

VARIANT_COLS = ["mut_id", "mut_type", "mut_effect", "d_id", "location", "e_i_numb", "locnumb", "aa_numb", "n_bp",
                "nuc_numb", "ntchange", "codon_first", "codon_last", "aa_first", "aa_last", "mut_syn", "aa_syn"]

def load_reports(keep_unknown=False):
    m2 = pd.read_csv(MMC2, dtype=str, keep_default_na=False)
    m3 = pd.read_csv(MMC3, dtype=str, keep_default_na=False)
    flow = [("raw MMC3 reports", len(m3), np.nan)]
    lab = m3["Inhibitors"].str.strip().str.lower()
    m3["y"] = lab.map({"yes": 1.0, "no": 0.0})
    def step(name, mask):
        nonlocal m3
        m3 = m3[mask]; flow.append((name, len(m3), int(m3["y"].sum())))
    step("shifted-column rows removed", ~lab.loc[m3.index].isin({"mild", "severe", "moderate"}))
    step("gnomAD population rows removed", m3["reference"].str.strip().str.lower() != "gnomad v2.1.1")
    step("mut_id missing from MMC2 removed", m3["mut_id"].isin(set(m2["mut_id"])))
    f8 = m2[m2["g_name"].str.strip().str.upper() == "F8"]
    step("F9 (haemophilia B) variants removed", m3["mut_id"].isin(set(f8["mut_id"])))
    if not keep_unknown:
        step("unknown outcome removed (never set to 0)", m3["y"].notna())
    df = m3.merge(m2[VARIANT_COLS], on="mut_id", how="left", validate="many_to_one").reset_index(drop=True)
    df["variant_key"] = df["mut_id"].map(variant_key(f8))
    return df, pd.DataFrame(flow, columns=["step", "rows remaining", "positives"])

# ---------------------------------------------------------------- parsing
def parse_level(v):
    """FVIII level -> (value, how reported). '<1' -> 0.5 (half the bound); '23 to 40' -> midpoint."""
    s = clean_str(v).replace("%", "").replace(",", ".")
    if not s:
        return np.nan, "missing"
    s = s.split("|")[0].strip()
    m = re.fullmatch(r"([<>])=?\s*([\d.]+)\??", s)
    if m:
        val = float(m.group(2))
        return (val * 0.5, "upper_bound") if m.group(1) == "<" else (val, "lower_bound")
    m = re.fullmatch(r"([\d.]+)\s*(?:to|-|–|and)\s*([\d.]+)\??", s, flags=re.I)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0, "range_mid"
    m = re.match(r"^([\d.]+)", s)
    if m:
        try:
            return float(m.group(1)), "exact"
        except ValueError:
            pass
    return np.nan, "missing"

def severity(v):
    s = clean_str(v).lower().replace(" ", "")
    if not s or s in {"unclassified", "notreported", "notaffected"}:
        return "unknown"
    if "mild" in s and "mod" in s:
        return "mild_moderate"
    if s in {"non-severe", "nonsevere"}:
        return "moderate"
    if re.search(r"sev|svere", s):
        return "severe"
    if re.search(r"mod|moderare|mpderate", s):
        return "moderate"
    return "mild" if "mild" in s else "unknown"

def effect(v):
    s = clean_str(v).lower()
    if not s:
        return "unknown"
    if "large" in s and "del" in s:
        return "large_deletion"
    if "large" in s and ("dup" in s or "ins" in s):
        return "large_duplication"
    for k, lab in (("nonsense", "nonsense"), ("frameshift", "frameshift"), ("missense", "missense"), ("splic", "splice")):
        if k in s:
            return lab
    if "silent" in s or "synonym" in s:
        return "silent"
    return "in_frame" if ("in-frame" in s or "inframe" in s) else "other"

DOMAIN_MAP = {"5utr": "utr", "3utr": "utr", "5-utr": "utr", "3-utr": "utr", "a1-a2": "multiple", "a1-a1-a2": "multiple",
              "a3-c1": "multiple", "c1-c2": "multiple", "multiple domains": "multiple", "undefined": "unknown"}
domain = lambda v: DOMAIN_MAP.get(clean_str(v).lower(), clean_str(v).lower() or "unknown")

def exon_number(v):
    s = clean_str(v)
    if not re.fullmatch(r"\d+", s):
        return np.nan
    n = int(s)
    return float(n) if 1 <= n <= 26 else np.nan      # values > 26 are spreadsheet date serials

AA3 = {"ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C", "gln": "Q", "glu": "E", "gly": "G", "his": "H",
       "ile": "I", "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P", "ser": "S", "thr": "T", "trp": "W",
       "tyr": "Y", "val": "V", "ter": "*", "*": "*", "stop": "*"}
def aa1(v):
    s = clean_str(v).lower()
    if not s:
        return ""
    if s in AA3:
        return AA3[s]
    if len(s) == 1 and s.upper() in "ARNDCQEGHILKMFPSTWYV*":
        return s.upper()
    return "fs" if "fs" in s else "other"

# ---------------------------------------------------------------- biology reference tables (UniProt P00451, literature)
SIG, MATURE, HEAVY_LIGHT = 19, 2332, 1648
INHIBITOR_EPITOPES = [(484, 508), (1804, 1819), (2091, 2115), (2181, 2243), (2248, 2312)]   # A2, A3, C1, C2 x2
FUNCTIONAL_SITES = [(1670, 1684), (1811, 1818), (337, 372), (2199, 2332), (372, 372), (740, 740), (1689, 1689)]
HYDROPATHY = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9,
                  F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOLUME = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7,
              K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
POLARITY = dict(A=8.1, R=10.5, N=11.6, D=13.0, C=5.5, Q=10.5, E=12.3, G=9.0, H=10.4, I=5.2, L=4.9, K=11.3, M=5.7,
                F=5.2, P=8.0, S=9.2, T=8.6, W=5.4, Y=6.2, V=5.9)
CHARGE = {**{a: 0.0 for a in HYDROPATHY}, "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0}
COMPOSITION = dict(S=1.42, R=0.65, L=0.0, P=0.39, T=0.71, A=0.0, V=0.0, G=0.74, I=0.0, F=0.0, Y=0.20, C=2.75,
                   H=0.58, Q=0.89, N=1.33, K=0.33, D=1.38, E=0.92, M=0.0, W=0.13)
AROMATIC = set("FWY")

def grantham(a, b):
    """Grantham (1974) distance from its published composition/polarity/volume formula."""
    if a not in POLARITY or b not in POLARITY:
        return np.nan
    d = 1.833 * (COMPOSITION[a] - COMPOSITION[b]) ** 2 + 0.1018 * (POLARITY[a] - POLARITY[b]) ** 2 \
        + 0.000399 * (VOLUME[a] - VOLUME[b]) ** 2
    return 50.723 * d ** 0.5

_B62 = "ARNDCQEGHILKMFPSTWYV"
_B62_ROWS = """4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0|-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3|
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3|-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3|
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1|-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2|
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2|0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3|
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3|-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3|
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1|-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2|
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1|-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1|
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2|1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2|
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0|-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3|
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1|0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""
BLOSUM62 = {(_B62[i], _B62[j]): int(v) for i, row in enumerate(_B62_ROWS.replace("\n", "").split("|"))
            for j, v in enumerate(row.split())}

def dist_to_span(p, lo, hi):
    return np.nan if p != p else (lo - p if p < lo else (p - hi if p > hi else 0.0))

# ---------------------------------------------------------------- feature sets
NULL_EFFECTS = {"large_deletion", "large_duplication", "nonsense", "frameshift"}

def base_columns(df):
    """Row-wise parsing; uses no labels and no other rows, so it cannot leak across folds."""
    b = pd.DataFrame(index=df.index)
    b["effect"] = df["mut_effect"].map(effect)
    b["mut_type"] = df["mut_type"].map(lambda v: clean_str(v).lower() or "unknown").replace({"duptication": "duplication"})
    b["domain"] = df["d_id"].map(domain)
    b["location"] = df["location"].map(lambda v: clean_str(v).lower().split(" ")[0].strip("/") or "unknown") \
                                  .replace({"promote": "promoter", "5utr": "utr"})
    b["severity"] = df["cli_phe"].map(severity)
    b["nt_change"] = df["ntchange"].map(lambda v: clean_str(v).upper().replace(" ", "") or "unknown")
    b["aa_ref"] = df["aa_first"].map(aa1).replace("", "none")
    b["aa_alt"] = df["aa_last"].map(aa1).replace("", "none")
    b["crm_type"] = df["type"].map(lambda v: clean_str(v).lower()).where(lambda s: s.isin(["i", "ii"]), "unknown")
    for col, src in (("fviii_activity", "clotting"), ("fviii_antigen", "antigen")):
        parsed = df[src].map(parse_level)
        b[col] = [v for v, _ in parsed]; b[col + "_kind"] = [k for _, k in parsed]
    b["aa_position"] = pd.to_numeric(df["aa_numb"], errors="coerce").replace(0, np.nan)
    b["cdna_position"] = pd.to_numeric(df["nuc_numb"], errors="coerce")
    b["exon"] = df["e_i_numb"].map(exon_number)
    b["n_bp"] = pd.to_numeric(df["n_bp"], errors="coerce")
    b["codon_ref"] = df["codon_first"].map(lambda v: clean_str(v).lower())
    return b

def features_v1(b):
    """The report's 'combined' set: 13 numeric + 11 categorical inputs (115 encoded columns in notebook 01)."""
    f = pd.DataFrame(index=b.index)
    f["cdna_position"], f["aa_position"], f["exon_intron_number"] = b["cdna_position"], b["aa_position"], b["exon"]
    f["log_n_bp"] = np.log10(b["n_bp"].clip(lower=1))
    sub = b["aa_ref"].isin(list(HYDROPATHY)) & b["aa_alt"].isin(list(HYDROPATHY))
    f["hydropathy_change"] = np.where(sub, b["aa_alt"].map(HYDROPATHY) - b["aa_ref"].map(HYDROPATHY), np.nan)
    f["charge_change"] = np.where(sub, b["aa_alt"].map(CHARGE) - b["aa_ref"].map(CHARGE), np.nan)
    f["cys_gained"] = np.where(sub, (b["aa_alt"] == "C").astype(float), np.nan)
    f["cys_lost"] = np.where(sub, (b["aa_ref"] == "C").astype(float), np.nan)
    f["arg_lost"] = np.where(b["aa_ref"] != "none", ((b["aa_ref"] == "R") & (b["aa_alt"] != "R")).astype(float), np.nan)
    f["ref_codon_has_cpg"] = np.where(b["codon_ref"] != "", b["codon_ref"].str.contains("cg").astype(float), np.nan)
    snv = b["nt_change"].str.fullmatch(r"[ACGT]>[ACGT]")
    f["is_transition"] = np.where(snv, b["nt_change"].isin(["A>G", "G>A", "C>T", "T>C"]).astype(float), np.nan)
    f["log_fviii_activity"] = np.log10(b["fviii_activity"] + 0.1)
    f["log_fviii_antigen"] = np.log10(b["fviii_antigen"] + 0.1)
    cat = b[["mut_type", "effect", "domain", "location", "nt_change", "aa_ref", "severity", "crm_type",
             "fviii_activity_kind", "fviii_antigen_kind"]].copy()
    cat["aa_alt_class"] = np.select([b["aa_alt"] == "*", b["aa_alt"] == "fs", b["aa_alt"].isin(list(HYDROPATHY)),
                                     b["aa_alt"] == "none"], ["stop", "frameshift", "amino_acid", "missing"], "other")
    return f, cat

def features_v2(b):
    """v1 + mechanistic biology and literature-named interactions (all row-wise, label-free)."""
    num, cat = features_v1(b)
    e = b["effect"]
    mature = (b["aa_position"] - SIG).where(lambda s: s > 0)
    sub = b["aa_ref"].isin(list(HYDROPATHY)) & b["aa_alt"].isin(list(HYDROPATHY))
    ref, alt = b["aa_ref"].where(sub, ""), b["aa_alt"].where(sub, "")
    new = {
        # genotype class: null variants abolish endogenous FVIII -> no central tolerance (Gouw 2012)
        "is_null_variant": e.isin(NULL_EFFECTS).astype(float),
        "is_large_deletion": (e == "large_deletion").astype(float),
        "cpg_signature": b["nt_change"].isin(["C>T", "G>A"]).astype(float),
        # where in the protein
        "relative_pos": mature / MATURE,
        "is_light_chain": np.where(mature.notna(), (mature > HEAVY_LIGHT).astype(float), np.nan),
        "epitope_distance": mature.map(lambda p: min(dist_to_span(p, lo, hi) for lo, hi in INHIBITOR_EPITOPES)),
        "functional_site_distance": mature.map(lambda p: min(dist_to_span(p, lo, hi) for lo, hi in FUNCTIONAL_SITES)),
        "truncation_fraction": np.where(e.isin(["nonsense", "frameshift"]), 1 - (mature / MATURE).clip(0, 1), np.nan),
        "nmd_escape": ((b["exon"] == 26) & e.isin(["nonsense", "frameshift"])).astype(float),
        "is_exon14_B_domain": (b["exon"] == 14).astype(float),
        # residue chemistry of a substitution
        "grantham": [grantham(a, c) if a else np.nan for a, c in zip(ref, alt)],
        "blosum62": [BLOSUM62.get((a, c), np.nan) if a else np.nan for a, c in zip(ref, alt)],
        "delta_volume": np.where(sub, alt.map(VOLUME) - ref.map(VOLUME), np.nan),
        "delta_polarity": np.where(sub, alt.map(POLARITY) - ref.map(POLARITY), np.nan),
        "aromatic_change": np.where(sub, alt.isin(list(AROMATIC)).astype(float) - ref.isin(list(AROMATIC)).astype(float), np.nan),
        "pro_gly_involved": np.where(sub, (ref.isin(["P", "G"]) | alt.isin(["P", "G"])).astype(float), np.nan),
        # patient clinical state
        "severity_ordinal": b["severity"].map({"mild": 0.0, "mild_moderate": 0.5, "moderate": 1.0, "severe": 2.0}),
        "fviii_below_1": (b["fviii_activity"] < 1).astype(float),
        "fviii_measured": b["fviii_activity"].notna().astype(float),
        # interactions named in the literature (null x severe; missense hotspots in C1/C2 and A2, Eckhardt 2013)
        "null_x_severe": e.isin(NULL_EFFECTS).astype(float) * (b["severity"] == "severe"),
        "missense_x_C_domain": ((e == "missense") & b["domain"].isin(["c1", "c2"])).astype(float),
        "missense_x_A2": ((e == "missense") & (b["domain"] == "a2")).astype(float),
        "missense_x_mild": ((e == "missense") & b["severity"].isin(["mild", "mild_moderate"])).astype(float),
    }
    num = pd.concat([num, pd.DataFrame(new, index=b.index)], axis=1)
    num["log_epitope_distance"] = np.log1p(num["epitope_distance"])
    num["in_epitope"] = (num["epitope_distance"] == 0).astype(float)
    cat = cat.copy()
    cat["effect_x_severity"] = b["effect"] + "|" + b["severity"]
    cat["effect_x_domain"] = b["effect"] + "|" + b["domain"]
    return num, cat

# ---------------------------------------------------------------- models (identical settings to notebook 02)
def encoder(num, cat, impute):
    nump = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()) if impute else "passthrough"
    return ColumnTransformer([("num", nump, list(num.columns)),
                              ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                                    sparse_output=False), list(cat.columns))])


# (factory, needs imputation). Tree boosters handle missing values natively.
MODELS = {
    "Majority class": (lambda: DummyClassifier(strategy="prior"), True),
    "Logistic regression": (lambda: LogisticRegression(C=0.1, max_iter=3000), True),
    "Random forest": (lambda: RandomForestClassifier(500, max_depth=12, min_samples_leaf=3, max_features=0.3,
                                                     n_jobs=N_JOBS, random_state=0), True),
    "LightGBM": (lambda: LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=20,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                                        verbose=-1, n_jobs=N_JOBS, random_state=0), False),
    "XGBoost": (lambda: XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=3,
                                      subsample=0.8, colsample_bytree=0.5, reg_lambda=5, n_jobs=N_JOBS,
                                      random_state=0, verbosity=0), False),
    "CatBoost": (lambda: CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, verbose=0, random_seed=0,
                                            allow_writing_files=False, thread_count=N_JOBS), False),
}
ENSEMBLE = ["LightGBM", "XGBoost", "CatBoost", "Random forest"]


def build_features(df):
    """Raw MMC3+MMC2 rows -> (numeric frame, categorical frame) of the final v2 feature set (38 + 13 inputs)."""
    return features_v2(base_columns(df))


class InhibitorEnsemble:
    """The final model: mean predicted probability of LightGBM + XGBoost + CatBoost + random forest,
    each behind its own encoder. Takes the 51 raw v2 inputs as one DataFrame."""

    def __init__(self, names=ENSEMBLE):
        self.names = list(names)

    def fit(self, num, cat, y):
        self.num_cols, self.cat_cols = list(num.columns), list(cat.columns)
        X = pd.concat([num, cat], axis=1)
        self.pipes = {}
        for n in self.names:
            make, impute = MODELS[n]
            self.pipes[n] = make_pipeline(encoder(num, cat, impute), make()).fit(X, np.asarray(y))
        return self

    @property
    def feature_names(self):
        return self.num_cols + self.cat_cols

    def predict_each(self, X):
        X = X[self.feature_names]
        return {n: p.predict_proba(X)[:, 1] for n, p in self.pipes.items()}

    def predict_proba(self, X):
        return np.mean(list(self.predict_each(X).values()), axis=0)


def fit_predict(names, num, cat, y, tr, te):
    """Fit the named models on rows tr; mean predicted probability on rows te."""
    m = InhibitorEnsemble(names).fit(num.iloc[tr], cat.iloc[tr], np.asarray(y)[tr])
    return m.predict_proba(pd.concat([num.iloc[te], cat.iloc[te]], axis=1))


def best_threshold(y_true, p, metric="mcc"):
    # np.inf first = "predict nobody positive"; wins ties, so a no-skill model is never forced to flag everyone
    grid = np.concatenate([[np.inf], np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))])
    score = {"mcc": lambda t: matthews_corrcoef(y_true, p >= t), "acc": lambda t: accuracy_score(y_true, p >= t)}[metric]
    return grid[int(np.argmax([score(t) for t in grid]))]


def metric_panel(y_true, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"ROC-AUC": roc_auc_score(y_true, p), "PR-AUC": average_precision_score(y_true, p),
            "PR-AUC baseline (prevalence)": y_true.mean(), "Brier": brier_score_loss(y_true, p),
            "MCC": matthews_corrcoef(y_true, pred), "Balanced accuracy": balanced_accuracy_score(y_true, pred),
            "F1": f1_score(y_true, pred, zero_division=0), "Sensitivity": tp / (tp + fn),
            "Specificity": tn / (tn + fp), "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "Accuracy": accuracy_score(y_true, pred), "Majority-class accuracy": max(y_true.mean(), 1 - y_true.mean())}
