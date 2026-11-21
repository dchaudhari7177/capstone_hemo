"""SHAP and LIME for the final InhibitorEnsemble (PW26_DN_03).

SHAP  - exact interventional TreeSHAP for each of the four tree models, computed in *probability* space against
        one shared background sample, mapped from the encoded columns back to the 51 raw inputs and averaged.
        Because Shapley values are linear in the model, the average of the four is the exact SHAP explanation of
        the ensemble's averaged probability: base value + sum of contributions = the ensemble's prediction.
LIME  - LimeTabularExplainer in categorical mode. Every numeric input is binned into quartiles of the training
        data plus an explicit "missing" bin, so a perturbed patient is always built from values that real patients
        have (a missing value stays missing; it is never replaced by a sentinel number the trees would misread).
"""
import contextlib
import os
import sys

import numpy as np
import pandas as pd
import shap
import shap.explainers._tree as _shap_tree
from lime.lime_tabular import LimeTabularExplainer


def _patch_shap_for_xgboost3():
    """shap 0.47 cannot parse the bracketed base_score ('[2.8875E-1]') that xgboost >= 3 writes into its model
    file. Strip the brackets before shap reads it. Binary models only have one value, so nothing else changes."""
    if getattr(_shap_tree, "_pw26_patched", False):
        return
    original = _shap_tree.decode_ubjson_buffer

    def fixed(fd):
        model = original(fd)
        p = model["learner"]["learner_model_param"]
        p["base_score"] = str(p["base_score"]).strip("[]")
        return model

    _shap_tree.decode_ubjson_buffer = fixed
    _shap_tree._pw26_patched = True


_patch_shap_for_xgboost3()


@contextlib.contextmanager
def _quiet_native_stderr():
    """shap's compiled TreeSHAP prints a progress bar straight to file descriptor 2 for long runs; mute it."""
    try:
        sys.stderr.flush()
        saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    except (OSError, ValueError):
        yield
        return
    try:
        yield
    finally:
        os.dup2(saved, 2); os.close(saved); os.close(devnull)


# ---------------------------------------------------------------- SHAP
def source_feature(encoded_name, cat_cols):
    """Encoded column name -> the raw input it came from ('cat__effect_missense' -> 'effect')."""
    kind, rest = encoded_name.split("__", 1)
    if kind == "num":
        return rest[len("missingindicator_"):] if rest.startswith("missingindicator_") else rest
    for c in sorted(cat_cols, key=len, reverse=True):
        if rest.startswith(c + "_"):
            return c
    raise KeyError(encoded_name)


class EnsembleShap:
    """Exact SHAP values of InhibitorEnsemble.predict_proba, expressed over its 51 raw inputs."""

    def __init__(self, model, background):
        self.model, self.features = model, model.feature_names
        background = background[self.features]
        self.parts, self.base_each = {}, {}
        for name, pipe in model.pipes.items():
            enc, est = pipe[:-1], pipe[-1]
            enc_names = enc.get_feature_names_out()
            to_raw = np.zeros((len(enc_names), len(self.features)))
            for i, n in enumerate(enc_names):
                to_raw[i, self.features.index(source_feature(n, model.cat_cols))] = 1.0
            kw = dict(data=enc.transform(background), feature_perturbation="interventional")
            if name != "Random forest":                 # sklearn forests already output probabilities
                kw["model_output"] = "probability"
            ex = shap.TreeExplainer(est, **kw)
            self.parts[name] = (enc, ex, to_raw)
            self.base_each[name] = float(np.ravel(ex.expected_value)[-1])
        self.base_value = float(np.mean(list(self.base_each.values())))

    def values_each(self, X):
        out = {}
        for name, (enc, ex, to_raw) in self.parts.items():
            with _quiet_native_stderr():
                sv = ex.shap_values(enc.transform(X[self.features]), check_additivity=False)
            sv = sv[1] if isinstance(sv, list) else sv
            sv = sv[..., 1] if np.ndim(sv) == 3 else sv
            out[name] = np.asarray(sv) @ to_raw
        return out

    def __call__(self, X):
        """shap.Explanation for the rows of X (numeric data shown on the colour scale; categoricals grey)."""
        each = self.values_each(X)
        values = np.mean(list(each.values()), axis=0)
        data = X[self.features].copy()
        for c in self.model.cat_cols:
            data[c] = np.nan
        exp = shap.Explanation(values=values, base_values=np.full(len(X), self.base_value),
                               data=data.to_numpy(dtype=float), feature_names=self.features)
        exp.each = each
        exp.raw = X[self.features].reset_index(drop=True)
        return exp


def row_explanation(exp, i, display_names=None):
    """Single-row Explanation whose data are the readable raw values (for waterfall plots)."""
    raw = exp.raw.iloc[i]
    shown = [("missing" if pd.isna(v) else (f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)))
             for v in raw]
    return shap.Explanation(values=exp.values[i], base_values=exp.base_values[i], data=np.array(shown, dtype=object),
                            feature_names=display_names or exp.feature_names)


# ---------------------------------------------------------------- LIME
class LimeView:
    """Maps the 51 raw inputs to the integer-coded matrix LIME works on, and back.

    numeric input   -> bins from the training quartiles (+ 'missing'); low-cardinality inputs keep their values
    categorical     -> its categories
    Decoding a perturbed row: an input LIME left in the patient's own bin keeps the patient's exact value;
    an input moved to another bin takes a value drawn from real training patients in that bin.
    """

    def __init__(self, model, X_train, X_all=None, seed=0):
        self.model, self.features = model, model.feature_names
        X_all = X_train if X_all is None else X_all
        self.rng = np.random.default_rng(seed)
        self.labels, self.pools, self.edges, self.kind = {}, {}, {}, {}
        for c in model.num_cols:
            v = X_train[c].to_numpy(dtype=float)
            finite = v[~np.isnan(v)]
            uniq = np.unique(finite)
            if len(uniq) <= 5:
                self.kind[c] = "levels"
                self.edges[c] = uniq
                self.labels[c] = [f"{u:g}" for u in uniq] + ["missing"]
            else:
                self.kind[c] = "bins"
                e = np.unique(np.quantile(finite, [0.25, 0.5, 0.75]))
                self.edges[c] = e
                lab = [f"<= {e[0]:.3g}"] + [f"{a:.3g} to {b:.3g}" for a, b in zip(e[:-1], e[1:])] + [f"> {e[-1]:.3g}"]
                self.labels[c] = lab + ["missing"]
            codes = self._code_num(c, v)
            self.pools[c] = {k: v[codes == k] for k in range(len(self.labels[c]))}
        for c in model.cat_cols:
            self.kind[c] = "cat"
            self.labels[c] = sorted(X_all[c].astype(str).unique())
            v = X_train[c].astype(str).to_numpy()
            self.pools[c] = {k: np.array([lab]) for k, lab in enumerate(self.labels[c])}
        self.train_codes = self.encode(X_train)

    def _code_num(self, c, v):
        v = np.asarray(v, dtype=float)
        miss = len(self.labels[c]) - 1
        if self.kind[c] == "levels":
            idx = np.searchsorted(self.edges[c], v)
            idx = np.clip(idx, 0, len(self.edges[c]) - 1)
            # nearest level for values unseen in training
            near = np.abs(self.edges[c][:, None] - v[None, :]).argmin(0) if len(v) else idx
            codes = np.where(np.isin(v, self.edges[c]), idx, near)
        else:
            codes = np.searchsorted(self.edges[c], v, side="left")
        return np.where(np.isnan(v), miss, codes).astype(int)

    def encode(self, X):
        cols = []
        for c in self.features:
            if self.kind[c] == "cat":
                lookup = {lab: k for k, lab in enumerate(self.labels[c])}
                cols.append(X[c].astype(str).map(lookup).to_numpy())
            else:
                cols.append(self._code_num(c, X[c].to_numpy(dtype=float)))
        return np.column_stack(cols).astype(float)

    def decode(self, A, instance):
        """Integer matrix from LIME -> raw DataFrame the model can score."""
        A = np.asarray(A).astype(int)
        own = self.encode(instance)[0].astype(int)
        out = {}
        for j, c in enumerate(self.features):
            col = np.empty(len(A), dtype=object)
            same = A[:, j] == own[j]
            col[same] = instance[c].iloc[0]
            for k in np.unique(A[~same, j]):
                rows = np.where((~same) & (A[:, j] == k))[0]
                pool = self.pools[c][k]
                col[rows] = self.rng.choice(pool, len(rows)) if len(pool) else np.nan
            out[c] = col
        df = pd.DataFrame(out)
        for c in self.model.num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def explainer(self, seed=0, kernel_width=None):
        return LimeTabularExplainer(
            self.train_codes, mode="classification", feature_names=self.features,
            categorical_features=list(range(len(self.features))),
            categorical_names={j: self.labels[c] for j, c in enumerate(self.features)},
            class_names=["no inhibitor", "inhibitor"], discretize_continuous=False,
            kernel_width=kernel_width, random_state=seed)

    def explain(self, instance, num_features=10, num_samples=5000, seed=0):
        """LIME explanation of one patient (a one-row raw DataFrame)."""
        instance = instance[self.features].reset_index(drop=True)
        self.rng = np.random.default_rng(seed)

        def predict_fn(A):
            p = self.model.predict_proba(self.decode(A, instance))
            return np.column_stack([1 - p, p])

        return self.explainer(seed).explain_instance(self.encode(instance)[0], predict_fn, labels=(1,),
                                                     num_features=num_features, num_samples=num_samples)

    def weights(self, lime_exp):
        """LIME weights as a Series over all raw inputs (0 for inputs not selected)."""
        w = pd.Series(0.0, index=self.features)
        for j, val in lime_exp.as_map()[1]:
            w.iloc[j] = val
        return w
