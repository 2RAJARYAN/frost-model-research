"""
india_eval.py — evaluation tools for the Tier 3 (India) experiments.

  * POD / POFD / Peirce skill score from counts (Peirce = POD - POFD).
  * Block bootstrap over (station, season) blocks. Nights within one winter at one
    station are strongly correlated, so whole winters are resampled, not days.
    Models whose Peirce skill differs by less than the interval are NOT ranked.
  * Decision-threshold selection on a validation set (never on test).
  * Climatology + persisted anomaly baseline: predict tomorrow's Tmin as
    climatology(tomorrow) + (today's Tmin - climatology(today)), with climatology
    estimated from TRAINING seasons only, per station and day of the Nov-Mar season.
"""
import numpy as np
import pandas as pd

SEASON_DAYS = 151


# ---------------------------------------------------------------- skill scores
def confusion_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    return (int((y_true & y_pred).sum()), int((~y_true & y_pred).sum()),
            int((y_true & ~y_pred).sum()), int((~y_true & ~y_pred).sum()))   # tp, fp, fn, tn


def skill_from_counts(tp, fp, fn, tn):
    pos, neg = tp + fn, fp + tn
    pod = tp / pos if pos else np.nan
    pofd = fp / neg if neg else np.nan
    return {"pod": pod, "pofd": pofd, "pss": pod - pofd,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "events": pos, "alerts": tp + fp}


def _pss_vec(c):
    """Peirce skill for an array of counts shaped (..., 4) = tp, fp, fn, tn."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return c[..., 0] / (c[..., 0] + c[..., 2]) - c[..., 1] / (c[..., 1] + c[..., 3])


# ------------------------------------------------------------ block bootstrap
def block_counts(blocks, y_true, y_pred):
    """Per-block (tp, fp, fn, tn) as an array (n_blocks, 4)."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    df = pd.DataFrame({"b": np.asarray(blocks), "tp": y_true & y_pred, "fp": ~y_true & y_pred,
                       "fn": y_true & ~y_pred, "tn": ~y_true & ~y_pred})
    return df.groupby("b", sort=True)[["tp", "fp", "fn", "tn"]].sum().to_numpy()


def bootstrap_ci(bc, B=1000, seed=0, level=95):
    """CI of the Peirce skill by resampling blocks with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bc), size=(B, len(bc)))
    draws = _pss_vec(bc[idx].sum(axis=1))
    lo, hi = np.nanpercentile(draws, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def bootstrap_diff_ci(bc_a, bc_b, B=1000, seed=0, level=95):
    """CI of PSS(a) - PSS(b) on the SAME blocks (paired resampling)."""
    assert bc_a.shape == bc_b.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bc_a), size=(B, len(bc_a)))
    d = _pss_vec(bc_a[idx].sum(axis=1)) - _pss_vec(bc_b[idx].sum(axis=1))
    lo, hi = np.nanpercentile(d, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def bootstrap_mean_diff_ci(err_a, err_b, blocks, B=1000, seed=0, level=95):
    """CI of mean(|err_a|) - mean(|err_b|), resampling whole blocks (paired)."""
    df = pd.DataFrame({"b": np.asarray(blocks), "a": np.abs(err_a), "c": np.abs(err_b)})
    g = df.groupby("b").agg(sa=("a", "sum"), sc=("c", "sum"), n=("a", "size")).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    s = g[idx].sum(axis=1)
    d = s[:, 0] / s[:, 2] - s[:, 1] / s[:, 2]
    lo, hi = np.percentile(d, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


# ------------------------------------------------------- threshold selection
def choose_threshold(y_true, score, candidates, event_if_below=True):
    """Threshold that maximises Peirce skill. Use VALIDATION data only.
    event_if_below=True: predicted Tmin <= t is an alert. False: probability >= t is an alert."""
    best, best_t = -np.inf, candidates[0]
    for t in candidates:
        pred = (np.asarray(score) <= t) if event_if_below else (np.asarray(score) >= t)
        s = skill_from_counts(*confusion_counts(y_true, pred))["pss"]
        if not np.isnan(s) and s > best:
            best, best_t = s, t
    return best_t, best


# ----------------------------------------------- climatology + persisted anomaly
def day_of_season(date, season):
    start = pd.to_datetime(pd.Series(np.asarray(season)).astype(int).astype(str) + "-11-01")
    return (pd.to_datetime(pd.Series(np.asarray(date))) - start).dt.days.to_numpy()


def climatology_table(train, smooth=15):
    """Station x day-of-season mean of next-day Tmin (explicit days), from TRAIN seasons only."""
    d = train.assign(dos=day_of_season(train["target_date"], train["season"]))
    t = d.groupby(["STATION", "dos"])["target_tmin_c"].mean().unstack("dos")
    t = t.reindex(columns=range(SEASON_DAYS))
    t = t.T.rolling(smooth, center=True, min_periods=max(3, smooth // 3)).mean().T
    return t.ffill(axis=1).bfill(axis=1)


def climatology_persistence_predict(clim, df):
    pos = clim.index.get_indexer(df["STATION"])
    if (pos < 0).any():
        raise ValueError("stations without training climatology: "
                         f"{sorted(set(df['STATION'][pos < 0]))}")
    arr = clim.to_numpy()
    dos = day_of_season(df["DATE"], df["season"])
    c_today = arr[pos, np.clip(dos, 0, SEASON_DAYS - 1)]
    c_tomorrow = arr[pos, np.clip(dos + 1, 0, SEASON_DAYS - 1)]
    return c_tomorrow + (df["tmin_c"].to_numpy() - c_today)
