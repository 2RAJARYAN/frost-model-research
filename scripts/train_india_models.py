#!/usr/bin/env python3
"""
Train and evaluate Tier 3 (India) models on data/processed/india_tier3.csv.gz.

    python scripts/train_india_models.py [--log] [--export]

Primary approach (the project's design decision): REGRESS next-day Tmin, then threshold.
A class-weighted logistic regression is kept as the documented classifier ablation.

Protocol (every stage reported whatever the outcome):
  * Split by winter season: train <= 2013, val 2014-2017, test 2018+. Models are fit on train;
    decision thresholds are chosen on val ONLY; test is touched once.
  * Baselines: persistence; climatology + persisted anomaly (climatology from train seasons);
    class-weighted logistic regression.
  * Models: Random Forest, XGBoost, LightGBM, CatBoost (default-ish parameters; tuning did not
    help on the Europe data).
  * Skill: POD, POFD, Peirce skill with 95% block-bootstrap CIs over (station, season) blocks.
    Models inside the best model's interval are NOT ranked.
  * Sensor ablation: core (temp+humidity) vs +pressure change vs +wind.
  * Leave-one-station-out (LightGBM): does the model transfer to an unseen station?
  * Negative controls (Nagpur, Patna, Kota, Gorakhpur): false alarms where cold nights barely occur.
  * --export: refit CatBoost on ALL seasons of the modelling stations and write models/india_v1_*.

Labels are rule-derived (Tmin <= 4 C), not observed frost or damage.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import log_run  # noqa: E402
from export_model import export_catboost_model, export_logreg_json  # noqa: E402
from india_dataset import CORE_FEATURES, PLUS_PRESSURE, PLUS_WIND, PRIMARY_THRESHOLD_C  # noqa: E402
from india_eval import (  # noqa: E402
    block_counts, bootstrap_ci, bootstrap_diff_ci, bootstrap_mean_diff_ci, choose_threshold,
    climatology_persistence_predict, climatology_table, confusion_counts, skill_from_counts,
)
from models import catboost_model, lightgbm_model, random_forest_model, xgboost_model  # noqa: E402

MODEL_ROLES = ("event_station", "low_event")
THRESHOLDS = np.arange(2.0, 8.01, 0.25)
PROBS = np.arange(0.02, 0.99, 0.02)
B = 1000
DATASET_TAG = "India_Tier3_pooled"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def evaluate(df, score, thr, below=True):
    y = df["label_le4"].to_numpy().astype(bool)
    pred = score <= thr if below else score >= thr
    s = skill_from_counts(*confusion_counts(y, pred))
    bc = block_counts(df["block"].to_numpy(), y, pred)
    lo, hi = bootstrap_ci(bc, B=B)
    return s, lo, hi, bc


def regression_errors(df, score):
    err = score - df["target_tmin_c"].to_numpy()
    cold = df["label_le4"].to_numpy().astype(bool)
    return {"mae": np.abs(err).mean(), "rmse": np.sqrt((err ** 2).mean()),
            "mae_cold_nights": np.abs(err[cold]).mean() if cold.any() else np.nan}, err


def row(method, rule, thr, df, score, below=True, regression=True):
    s, lo, hi, bc = evaluate(df, score, thr, below)
    e = regression_errors(df, score)[0] if regression else {"mae": np.nan, "rmse": np.nan, "mae_cold_nights": np.nan}
    return ({"method": method, "rule": rule, "threshold": r(thr, 2), "mae": r(e["mae"]), "rmse": r(e["rmse"]),
             "mae_cold": r(e["mae_cold_nights"]), "pod": r(s["pod"]), "pofd": r(s["pofd"], 4),
             "pss": r(s["pss"]), "pss_lo": r(lo), "pss_hi": r(hi), "precision": r(s["precision"]),
             "alerts": s["alerts"], "events": s["events"]}, bc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/india_tier3.csv.gz")
    ap.add_argument("--out-dir", default="experiments")
    ap.add_argument("--tables", default="reports/india_tier3_tables.md")
    ap.add_argument("--log", action="store_true", help="append test results to experiments/runs.csv")
    ap.add_argument("--export", action="store_true", help="refit CatBoost on all seasons and export")
    ap.add_argument("--skip-rf", action="store_true", help="skip Random Forest (slow on one core)")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(exist_ok=True)

    d = pd.read_csv(a.data, dtype={"STATION": str}, parse_dates=["DATE", "target_date"])
    d["block"] = d["STATION"] + "_" + d["season"].astype(str)
    m = d[d["role"].isin(MODEL_ROLES)]
    tr, va, te = (m[m["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    neg = d[d["role"] == "negative_control"].reset_index(drop=True)
    print(f"train {len(tr):,} (cold {tr.label_le4.sum()}) | val {len(va):,} ({va.label_le4.sum()}) | "
          f"test {len(te):,} ({te.label_le4.sum()}) | stations {m.STATION.nunique()} | "
          f"negative-control rows {len(neg):,}", flush=True)

    F = CORE_FEATURES
    y_tr = tr["target_tmin_c"]
    methods = {}          # name -> dict(val=, test=, below=, regression=)

    # ------------------------------------------------------------- baselines
    methods["Persistence"] = dict(val=va["tmin_c"].to_numpy(), test=te["tmin_c"].to_numpy(), below=True, regression=True)
    clim = climatology_table(tr)
    methods["Climatology+persisted anomaly"] = dict(
        val=climatology_persistence_predict(clim, va), test=climatology_persistence_predict(clim, te),
        below=True, regression=True)
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
    lr.fit(tr[F], tr["label_le4"])
    methods["Logistic regression (classifier)"] = dict(
        val=lr.predict_proba(va[F])[:, 1], test=lr.predict_proba(te[F])[:, 1], below=False, regression=False)

    # ---------------------------------------------------------------- models
    factories = {"XGBoost": xgboost_model, "LightGBM": lightgbm_model, "CatBoost": catboost_model}
    if not a.skip_rf:
        factories = {"RandomForest": lambda: random_forest_model(n_estimators=150, min_samples_leaf=5,
                                                                  max_features=0.5), **factories}
    fitted = {}
    for name, fac in factories.items():
        t = time.time()
        mdl = fac().fit(tr[F], y_tr)
        fitted[name] = mdl
        methods[name] = dict(val=mdl.predict(va[F]), test=mdl.predict(te[F]), below=True, regression=True)
        print(f"  fit {name}: {time.time() - t:.0f}s", flush=True)

    # ------------------------------------------------ comparison (val-tuned threshold)
    rows, bcs, tuned = [], {}, {}
    for name, mth in methods.items():
        reg = mth["regression"]
        nominal = PRIMARY_THRESHOLD_C if reg else 0.5
        cands = THRESHOLDS if reg else PROBS
        thr_star, val_pss = choose_threshold(va["label_le4"], mth["val"], cands, event_if_below=mth["below"])
        tuned[name] = thr_star
        for rule, thr in (("nominal", nominal), ("val-tuned", thr_star)):
            rr, bc = row(name, rule, thr, te, mth["test"], mth["below"], reg)
            rr["val_pss_at_thr"] = r(val_pss) if rule == "val-tuned" else None
            rows.append(rr)
            bcs[(name, rule)] = bc
    comp = pd.DataFrame(rows)
    best = comp[comp["rule"] == "val-tuned"].sort_values("pss", ascending=False).iloc[0]
    comp["inside_best_CI"] = [
        (best["pss_lo"] <= p <= best["pss_hi"]) if rl == "val-tuned" else None
        for p, rl in zip(comp["pss"], comp["rule"])]
    comp.to_csv(out_dir / "india_tier3_comparison.csv", index=False)
    print("\nTEST, val-tuned thresholds:")
    print(comp[comp["rule"] == "val-tuned"].drop(columns=["rule"]).to_string(index=False), flush=True)
    print("\nTEST, nominal thresholds (4 C / p=0.5):")
    print(comp[comp["rule"] == "nominal"].drop(columns=["rule", "val_pss_at_thr", "inside_best_CI"]).to_string(index=False), flush=True)

    # paired differences versus the strongest baseline, on the same blocks
    base_name = max(["Persistence", "Climatology+persisted anomaly"],
                    key=lambda n: comp[(comp.method == n) & (comp.rule == "val-tuned")]["pss"].iloc[0])
    diffs = []
    for name in methods:
        if name == base_name:
            continue
        lo, hi = bootstrap_diff_ci(bcs[(name, "val-tuned")], bcs[(base_name, "val-tuned")], B=B)
        pt = (comp[(comp.method == name) & (comp.rule == "val-tuned")]["pss"].iloc[0]
              - comp[(comp.method == base_name) & (comp.rule == "val-tuned")]["pss"].iloc[0])
        diffs.append({"method": name, f"PSS minus {base_name}": r(pt), "ci_lo": r(lo), "ci_hi": r(hi),
                      "significant": bool(lo > 0 or hi < 0)})
    diffs = pd.DataFrame(diffs)
    print(f"\nPaired PSS difference vs strongest baseline ({base_name}):\n{diffs.to_string(index=False)}", flush=True)

    # paired differences versus CatBoost (the model that ships): can the strong models be ranked?
    vs_cb = []
    for name in ("Logistic regression (classifier)", "RandomForest", "XGBoost", "LightGBM"):
        if name not in methods:
            continue
        lo, hi = bootstrap_diff_ci(bcs[(name, "val-tuned")], bcs[("CatBoost", "val-tuned")], B=B)
        pt = (comp[(comp.method == name) & (comp.rule == "val-tuned")]["pss"].iloc[0]
              - comp[(comp.method == "CatBoost") & (comp.rule == "val-tuned")]["pss"].iloc[0])
        vs_cb.append({"method": name, "PSS minus CatBoost": r(pt), "ci_lo": r(lo), "ci_hi": r(hi),
                      "distinguishable": bool(lo > 0 or hi < 0)})
    vs_cb = pd.DataFrame(vs_cb)
    print(f"\nPaired PSS difference vs CatBoost:\n{vs_cb.to_string(index=False)}", flush=True)

    # ------------------------------------------- per-station view (best-tuned CatBoost)
    cb_thr = tuned["CatBoost"]
    per = []
    for sid, g in te.groupby("name"):
        pred = fitted["CatBoost"].predict(g[F]) <= cb_thr
        s_ = skill_from_counts(*confusion_counts(g["label_le4"].astype(bool), pred))
        per.append({"station": sid, "test_rows": len(g), "cold_nights": int(g.label_le4.sum()),
                    "alerts": s_["alerts"], "pod": r(s_["pod"]), "pofd": r(s_["pofd"], 4), "pss": r(s_["pss"])})
    per = pd.DataFrame(per).sort_values("cold_nights", ascending=False)
    per.to_csv(out_dir / "india_tier3_per_station_test.csv", index=False)
    print(f"\nPer-station TEST (CatBoost @ {cb_thr}):\n{per.to_string(index=False)}", flush=True)

    # ---------------------------------------------------- sensor ablation (LightGBM)
    # Rows where pressure change AND wind both exist (SLP is missing for ~50-70% of Amritsar, Lucknow,
    # Jaipur, Nagpur days). All three feature sets are scored at ONE common threshold (the core model's
    # validation-tuned one): separately tuned thresholds on a 0.25 C grid would add threshold noise.
    ok = lambda x: x[x["wdsp_ms"].notna() & x["slp_change_1d"].notna() & x["slp_change_3d"].notna()].reset_index(drop=True)  # noqa: E731
    tr_c, va_c, te_c = ok(tr), ok(va), ok(te)
    print(f"\nablation rows (pressure+wind present): train {len(tr_c):,} val {len(va_c):,} test {len(te_c):,} "
          f"({te_c.STATION.nunique()} stations, {int(te_c.label_le4.sum())} cold nights in test)", flush=True)
    core_k = "core (temp+humidity)"
    sets = ((core_k, CORE_FEATURES), ("+ pressure change", PLUS_PRESSURE), ("+ pressure change + wind", PLUS_WIND))
    preds = {}
    for label, feats in sets:
        mdl = lightgbm_model().fit(tr_c[feats], tr_c["target_tmin_c"])
        preds[label] = (mdl.predict(va_c[feats]), mdl.predict(te_c[feats]))
    thr_common, _ = choose_threshold(va_c["label_le4"], preds[core_k][0], THRESHOLDS)
    abl, abl_bc, abl_err = [], {}, {}
    for label, _f in sets:
        rr, bc = row(label, "common-thr", thr_common, te_c, preds[label][1])
        abl.append(rr)
        abl_bc[label] = bc
        abl_err[label] = preds[label][1] - te_c["target_tmin_c"].to_numpy()
    ab = pd.DataFrame(abl)[["method", "threshold", "mae", "mae_cold", "pod", "pofd", "pss", "pss_lo", "pss_hi"]]
    ab["dPSS_vs_core"], ab["dPSS_ci"], ab["dMAE_vs_core"], ab["dMAE_ci"] = None, None, None, None
    for i, k in enumerate(ab["method"]):
        if k == core_k:
            continue
        lo, hi = bootstrap_diff_ci(abl_bc[k], abl_bc[core_k], B=B)
        ab.loc[i, "dPSS_vs_core"] = r(ab.loc[i, "pss"] - ab.loc[0, "pss"])
        ab.loc[i, "dPSS_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        lo2, hi2 = bootstrap_mean_diff_ci(abl_err[k], abl_err[core_k], te_c["block"].to_numpy(), B=B)
        ab.loc[i, "dMAE_vs_core"] = r(ab.loc[i, "mae"] - ab.loc[0, "mae"])
        ab.loc[i, "dMAE_ci"] = f"[{lo2:.3f}, {hi2:.3f}]"
    ab.to_csv(out_dir / "india_tier3_ablation.csv", index=False)
    print(f"Sensor ablation (LightGBM, TEST, common threshold {thr_common:g}):\n{ab.to_string(index=False)}", flush=True)

    # ------------------------------------- leave-one-station-out (LightGBM, core)
    loso = []
    for sid in sorted(m["STATION"].unique()):
        held = m[m["STATION"] == sid].reset_index(drop=True)
        if held["label_le4"].sum() < 10:
            continue
        rest = m[m["STATION"] != sid]
        mdl = lightgbm_model().fit(rest[F], rest["target_tmin_c"])
        p = mdl.predict(held[F])
        pers = held["tmin_c"].to_numpy()
        rec = {"station": held["name"].iloc[0], "rows": len(held), "cold_nights": int(held.label_le4.sum())}
        for tag, sc, thr in (("model@4", p, PRIMARY_THRESHOLD_C), (f"model@{tuned['LightGBM']:g}", p, tuned["LightGBM"]),
                             ("persist@4", pers, PRIMARY_THRESHOLD_C),
                             (f"persist@{tuned['Persistence']:g}", pers, tuned["Persistence"])):
            s_, lo, hi, _ = evaluate(held, sc, thr)
            rec[f"pss {tag}"] = r(s_["pss"], 2)
            if tag.startswith("model@") and tag != "model@4":
                rec["pss_ci"] = f"[{lo:.2f}, {hi:.2f}]"
                rec["pod"], rec["pofd"] = r(s_["pod"], 2), r(s_["pofd"], 3)
        loso.append(rec)
    loso = pd.DataFrame(loso).sort_values("cold_nights", ascending=False)
    loso.to_csv(out_dir / "india_tier3_loso.csv", index=False)
    print(f"\nLeave-one-station-out (LightGBM core; thr* tuned on pooled val, slightly optimistic):\n"
          f"{loso.to_string(index=False)}", flush=True)

    # ------------------------------------------------------- negative controls
    negs = []
    for name in ("CatBoost", "LightGBM"):
        thr = tuned[name]
        pr = fitted[name].predict(neg[F]) <= thr
        for st, g in neg.assign(alert=pr).groupby("name"):
            negs.append({"model": name, "threshold": thr, "station": st, "rows": len(g),
                         "actual_cold_nights": int(g.label_le4.sum()), "alerts": int(g.alert.sum()),
                         "alert_rate_%": r(100 * g.alert.mean(), 2)})
    negs = pd.DataFrame(negs)
    negs.to_csv(out_dir / "india_tier3_negative_controls.csv", index=False)
    print(f"\nNegative controls (all seasons):\n{negs.to_string(index=False)}", flush=True)

    # ------------------------------------------------------------ tables, logs
    Path(a.tables).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tables).write_text("\n\n".join([
        "## Comparison on the TEST seasons (2018-2024), val-tuned thresholds\n\n"
        + md(comp[comp.rule == "val-tuned"].drop(columns=["rule"])),
        "## Same, nominal thresholds (4 C for regressors, p = 0.5 for the classifier)\n\n"
        + md(comp[comp.rule == "nominal"].drop(columns=["rule", "val_pss_at_thr", "inside_best_CI"])),
        f"## Paired Peirce-skill difference vs strongest baseline ({base_name})\n\n" + md(diffs),
        "## Paired Peirce-skill difference vs CatBoost (the shipped model)\n\n" + md(vs_cb),
        f"## Per-station TEST results (CatBoost, threshold {cb_thr:g})\n\n" + md(per),
        "## Sensor ablation (LightGBM; rows with pressure and wind present; common threshold)\n\n" + md(ab),
        "## Leave-one-station-out (LightGBM, core features)\n\n" + md(loso),
        "## Negative controls\n\n" + md(negs)]) + "\n", encoding="utf-8")

    if a.log:
        for _, rr in comp[comp.rule == "val-tuned"].iterrows():
            log_run("experiments/runs.csv", model=f"{rr['method']} [India core, thr {rr['threshold']}]",
                    dataset=DATASET_TAG, params="rule-derived label Tmin<=4C, 1-day horizon, test 2018-2024",
                    metrics={"precision": rr["precision"], "recall": rr["pod"], "peirce_skill": rr["pss"],
                             "rmse": rr["rmse"], "mae": rr["mae"]})
        print("logged to experiments/runs.csv")

    # ------------------------------------------------------------- final export
    limitations = [
        "Labels are RULE-DERIVED (next-day Tmin <= 4 C), not observed frost or crop damage; skill is skill at the rule's conditions.",
        "Pooled across stations; NOT calibrated or validated for any single site, and not for Warora/Vidarbha (no cold events there).",
        "Station training series contain breaks (e.g. Hisar ~2010); labels are an imperfect reference.",
        "Training RH is derived from GSOD mean temperature and mean dew point; Repo B measures RH directly. "
        "Daily aggregates (tmin, tmax, tmean, mean RH, lags, rolling stats) must be computed exactly as in src/features.py "
        "(build_feature_set + add_daily_range_features).",
        "Decision threshold was tuned on validation seasons using the train-only model; the shipped model is refit on all seasons "
        "and has no held-out evaluation of its own.",
        "Alert threshold (predicted next-day Tmin) sits well above the 4 C label because regression pulls cold-tail predictions "
        "upward; at this operating point roughly 3 of 4 alerts are false alarms (precision ~0.24).",
        "No station-identity features are used, by design (the v1 Europe model failed this).",
    ]
    if a.export:
        cb_test = comp[(comp.method == "CatBoost") & (comp.rule == "val-tuned")].iloc[0]
        final = catboost_model().fit(m[F], m["target_tmin_c"])
        paths = export_catboost_model(
            final, feature_columns=list(F), output_dir="models", version="india_v1",
            classification_threshold=float(cb_thr), label_variant="label_le4",
            training_description=(
                f"India Tier 3: pooled {m.STATION.nunique()} GSOD stations (Indo-Gangetic plains + central), Nov-Mar "
                f"1990-2024, core features (temperature + humidity only), next-day Tmin regression, CatBoost default "
                f"parameters, refit on ALL seasons ({len(m):,} station-days). Reported metrics are from the train-only "
                "model on the 2018-2024 test seasons at the validation-tuned threshold."),
            eval_metrics={"precision": cb_test["precision"], "recall_pod": cb_test["pod"], "pofd": cb_test["pofd"],
                          "peirce_skill": cb_test["pss"], "pss_ci_lo": cb_test["pss_lo"], "pss_ci_hi": cb_test["pss_hi"],
                          "mae": cb_test["mae"], "rmse": cb_test["rmse"]},
            extra_known_limitations=limitations)
        lr_row = comp[(comp.method == "Logistic regression (classifier)") & (comp.rule == "val-tuned")].iloc[0]
        lr_final = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
        lr_final.fit(m[F], m["label_le4"])
        lr_path = export_logreg_json(
            lr_final, list(F), "models/india_v1_logreg.json", tuned["Logistic regression (classifier)"],
            {"version": "india_v1_logreg", "label_variant_targeted": "label_le4",
             "predicts": "probability that next-day Tmin <= 4 C (class-weighted classifier)",
             "training_description": "Same data and refit policy as india_v1 (CatBoost); statistically tied with it on the test seasons "
                                     "(see reports/india_tier3_results.md). Needs only numpy to score.",
             "eval_metrics_test_2018_2024": {"pod": lr_row["pod"], "pofd": lr_row["pofd"], "peirce_skill": lr_row["pss"],
                                              "pss_ci_lo": lr_row["pss_lo"], "pss_ci_hi": lr_row["pss_hi"], "precision": lr_row["precision"]},
             "known_limitations": limitations})
        paths["logreg_path"] = lr_path
        print("exported:", paths)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
