"""Entrainement et evaluation du modele de risque de pluie a J+1.

Cible : pluie journaliere (cumul UTC) du jour D+1 >= seuil (1 mm par defaut), prevue a partir
des seules observations jusqu'au jour D. Modeles : climatologie mensuelle, persistance (Markov
d'ordre 1), regression logistique, gradient boosting. Selection sur la validation (Brier),
seuil de decision choisi sur la validation (F1), evaluation finale sur le test (jamais utilise
pour choisir quoi que ce soit). Modele deploye = modele retenu reentraine sur train+validation.

Usage : python -m src.train_model [--origin synthetic_demo|infoclimat] [--fast]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import database as db
from .config import Settings, get_settings
from .preprocessing import FEATURES, build_features, hourly_to_daily, make_splits

log = logging.getLogger("goutte.train")
RANDOM_STATE = 42


# ---------- modeles de reference ----------
def climatology_rates(feats_train: pd.DataFrame) -> dict[int, float]:
    """Frequence de jours pluvieux par mois du jour cible, calculee sur l'entrainement."""
    m = pd.to_datetime(feats_train["target_date"]).dt.month
    rates = feats_train.groupby(m)["target"].mean()
    overall = float(feats_train["target"].mean())
    return {mo: float(rates.get(mo, overall)) for mo in range(1, 13)}


def proba_climatology(feats: pd.DataFrame, rates: dict[int, float]) -> np.ndarray:
    return pd.to_datetime(feats["target_date"]).dt.month.map(rates).to_numpy(dtype=float)


def proba_persistence(feats: pd.DataFrame, feats_train: pd.DataFrame, thr: float) -> np.ndarray:
    """Chaine de Markov d'ordre 1 : P(pluie demain | pluie aujourd'hui) estimee sur l'entrainement."""
    wet = feats_train["rain_today_mm"] >= thr
    ok = feats_train["rain_today_mm"].notna()
    p_wet = float(feats_train.loc[ok & wet, "target"].mean())
    p_dry = float(feats_train.loc[ok & ~wet, "target"].mean())
    overall = float(feats_train["target"].mean())
    t = feats["rain_today_mm"]
    return np.where(t.isna(), overall, np.where(t >= thr, p_wet, p_dry))


def make_estimator(name: str, params: dict | None = None) -> Pipeline:
    params = params or {}
    if name == "logistic":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, C=params.get("C", 1.0)))])
    if name == "hist_gb":
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", HistGradientBoostingClassifier(
                             max_depth=params.get("max_depth", 3), learning_rate=params.get("learning_rate", 0.05),
                             max_iter=params.get("max_iter", 200), l2_regularization=1.0,
                             early_stopping=False, random_state=RANDOM_STATE))])
    raise ValueError(name)


GRIDS = {
    "logistic": [{"C": 0.1}, {"C": 1.0}],
    "hist_gb": [{"max_depth": 2, "max_iter": 100}, {"max_depth": 3, "max_iter": 150},
                {"max_depth": 3, "max_iter": 300}],
}


# ---------- metriques ----------
def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "positive_rate": round(float(y.mean()), 4), "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, p), 4) if both else None,
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.arange(0.05, 0.951, 0.01)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def bootstrap_ci(y, p, thr, n_boot: int = 300, seed: int = RANDOM_STATE) -> dict:
    """IC a 95 % par bootstrap simple (hypothese d'independance des jours : approximation)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y).astype(int), np.asarray(p)
    aucs, briers, f1s = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
        briers.append(brier_score_loss(y[i], p[i]))
        f1s.append(f1_score(y[i], (p[i] >= thr).astype(int), zero_division=0))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"roc_auc_ci95": q(aucs), "brier_ci95": q(briers), "f1_ci95": q(f1s)}


# ---------- entrainement ----------
def load_training_table(conn, settings: Settings, origin: str) -> pd.DataFrame:
    obs = db.load_observations(conn, origin=origin)
    if obs.empty:
        raise RuntimeError(f"Aucune observation d'origine '{origin}' en base : lancer la collecte d'abord")
    daily = hourly_to_daily(obs, settings.min_rain_hours)
    return build_features(daily, settings.rain_threshold_mm)


def _plots(figdir, results_test, y_test, p_test_by_model, sel, p_sel, imp_df, feats_all):
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in p_test_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={results_test[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="hasard")
    ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbes ROC - jeu de test")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figdir / "roc_test.png", dpi=150); plt.close(fig)

    frac, mean_p = calibration_curve(y_test, p_sel, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="calibration parfaite")
    ax.plot(mean_p, frac, "o-", label=sel)
    ax.set(xlabel="Probabilite predite (moyenne par classe)", ylabel="Frequence observee",
           title="Calibration - jeu de test"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "calibration_test.png", dpi=150); plt.close(fig)

    c = results_test[sel]["confusion"]
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=13, color="black" if v < cm.max() * 0.6 else "white")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["sec", "pluie"], yticklabels=["sec", "pluie"],
           xlabel="Predit", ylabel="Observe", title=f"Matrice de confusion - {sel}")
    fig.tight_layout(); fig.savefig(figdir / "confusion_test.png", dpi=150); plt.close(fig)

    imp = imp_df.sort_values("importance")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"])
    ax.set(xlabel="Baisse du Brier score apres permutation (test)", title="Importance des variables")
    fig.tight_layout(); fig.savefig(figdir / "importance_variables.png", dpi=150); plt.close(fig)

    lab = feats_all[feats_all["target"].notna()].copy()
    lab["month"] = pd.to_datetime(lab["target_date"]).dt.month
    m = lab.groupby("month")["target"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(m.index, m.values * 100)
    ax.set(xlabel="Mois", ylabel="% de jours pluvieux (>= seuil)", xticks=range(1, 13),
           title="Frequence mensuelle des jours pluvieux (donnees d'entrainement)")
    fig.tight_layout(); fig.savefig(figdir / "pluie_par_mois.png", dpi=150); plt.close(fig)


def train(conn, settings: Settings, origin: str, fast: bool = False, make_figures: bool = True) -> dict:
    thr_mm = settings.rain_threshold_mm
    feats = load_training_table(conn, settings, origin)
    sp = settings.region["split"]
    masks = make_splits(feats, sp["train"], sp["validation"], sp["purge_days"])
    tr, va, te = (feats[masks[k]] for k in ("train", "validation", "test"))
    if min(len(tr), len(va), len(te)) < 50:
        raise RuntimeError("Donnees insuffisantes pour un decoupage train/validation/test")
    y_tr, y_va, y_te = (d["target"].to_numpy() for d in (tr, va, te))
    trva = pd.concat([tr, va])
    y_trva = trva["target"].to_numpy()

    splits_info = {k: {"start": str(feats[masks[k]]["feature_date"].min().date()),
                       "end": str(feats[masks[k]]["feature_date"].max().date()),
                       "n": int(masks[k].sum()), "positive_rate": round(float(feats[masks[k]]["target"].mean()), 4)}
                   for k in masks}

    # --- references ---
    rates_tr, rates_trva = climatology_rates(tr), climatology_rates(trva)
    val_p, test_p, thresholds = {}, {}, {}
    val_p["climatologie"] = proba_climatology(va, rates_tr)
    test_p["climatologie"] = proba_climatology(te, rates_trva)
    val_p["persistance"] = proba_persistence(va, tr, thr_mm)
    test_p["persistance"] = proba_persistence(te, trva, thr_mm)

    # --- modeles ML : selection des hyperparametres sur validation (Brier) ---
    chosen_params, fitted_final = {}, {}
    grids = {k: (v[:1] if fast else v) for k, v in GRIDS.items()}
    for name, grid in grids.items():
        best = None
        for prm in grid:
            est = make_estimator(name, prm).fit(tr[FEATURES], y_tr)
            b = brier_score_loss(y_va, est.predict_proba(va[FEATURES])[:, 1])
            if best is None or b < best[0]:
                best = (b, prm, est)
        chosen_params[name] = best[1]
        val_p[name] = best[2].predict_proba(va[FEATURES])[:, 1]
        final = make_estimator(name, best[1]).fit(trva[FEATURES], y_trva)  # train + validation
        fitted_final[name] = final
        test_p[name] = final.predict_proba(te[FEATURES])[:, 1]

    results_val, results_test = {}, {}
    for name in val_p:
        thr = best_f1_threshold(y_va, val_p[name])
        thresholds[name] = thr
        results_val[name] = evaluate(y_va, val_p[name], thr)
        results_test[name] = evaluate(y_te, test_p[name], thr)

    ml = ["logistic", "hist_gb"]
    sel = min(ml, key=lambda n: results_val[n]["brier"])
    brier_clim = results_test["climatologie"]["brier"]
    for name in results_test:
        results_test[name]["brier_skill_score_vs_climatology"] = round(1 - results_test[name]["brier"] / brier_clim, 4)
    ci = bootstrap_ci(y_te, test_p[sel], thresholds[sel])

    # importance par permutation (sur le test, modele deploye)
    perm = permutation_importance(fitted_final[sel], te[FEATURES], y_te, scoring="neg_brier_score",
                                  n_repeats=5, random_state=RANDOM_STATE)
    imp_df = pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean, "std": perm.importances_std})

    frac, mean_p = calibration_curve(y_te, test_p[sel], n_bins=8, strategy="quantile")
    model_id = f"rain_d1_{sel}_{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    metrics = {
        "model_id": model_id, "selected_model": sel, "data_origin": origin,
        "target": f"pluie_journaliere_J+1 >= {thr_mm} mm (cumul UTC)", "features": FEATURES,
        "splits": splits_info, "chosen_hyperparameters": chosen_params,
        "validation": results_val, "test": results_test, "test_ci95_selected": ci,
        "calibration_test": {"mean_predicted": [round(float(x), 4) for x in mean_p],
                             "observed_frequency": [round(float(x), 4) for x in frac]},
        "permutation_importance": imp_df.round(5).to_dict(orient="records"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": ("DONNEES SYNTHETIQUES : scores non representatifs d'une performance reelle"
                    if origin == "synthetic_demo" else ""),
    }
    bundle = {"model_id": model_id, "pipeline": fitted_final[sel], "features": FEATURES,
              "threshold": thresholds[sel], "climatology": climatology_rates(trva),
              "data_origin": origin, "rain_threshold_mm": thr_mm, "algorithm": sel,
              "trained_at": metrics["trained_at"]}
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, settings.model_path)
    settings.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    db.log_model_run(conn, model_id, origin, sel, metrics["test"][sel], settings.model_path)
    if make_figures:
        _plots(settings.figures_dir, results_test, y_te, test_p, sel, test_p[sel], imp_df, feats)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--origin", choices=["synthetic_demo", "infoclimat"])
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    conn = db.get_connection(settings.db_path)
    origins = db.list_origins(conn)
    origin = a.origin or ("infoclimat" if origins.get("infoclimat", 0) > 0 else "synthetic_demo")
    log.info("Entrainement sur l'origine de donnees : %s", origin)
    m = train(conn, settings, origin, fast=a.fast)
    log.info("Modele retenu : %s | test : %s", m["selected_model"], m["test"][m["selected_model"]])


if __name__ == "__main__":
    main()
