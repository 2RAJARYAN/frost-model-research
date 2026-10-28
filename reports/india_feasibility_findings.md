# India Tier-3 Label Feasibility — Findings

Source data: `data/raw/india/feasibility_summary.csv` and
`data/raw/india/tmin_<site>_<source>.csv`. Script:
`scripts/india_label_feasibility.py` (tested in
`scripts/test_india_label_feasibility.py`). Benchmark paper: Attri et al.
(2020, *MAUSAM* 71(1), 95-102) — IMD authors, Indo-Gangetic Plains frost
study, 9 IMD stations, 1980-2015, frost day defined as screen Tmin ≤ 4°C.

## Headline result

**Vidarbha (Warora, Nagpur) cannot support a frost-type label at any
defensible threshold.** Warora: 0 days below 5°C in 44 winters, in both
NASA POWER and ERA5. Nagpur: at most 3 days below 5°C in any source,
spread across only 2 winters. There is nothing to train or evaluate a
cold-injury model on at this location.

**The Indo-Gangetic belt (Punjab/Haryana/Rajasthan) does have usable
event counts.** Every site tested passes a working feasibility rule
(~10+ winters with at least 3 positive days) at a 4°C threshold in at
least one data source.

### Counts at Tmin ≤ 4°C (Nov-Mar seasons, 44 winters; ISD has 42-43)

`pct` = share of Nov-Mar days (not of the full year). `W≥3` = winters
with ≥3 positive days.

| Site | NASA POWER (days / pct / W≥3) | ERA5 (days / pct / W≥3) | ISD station (days / pct / W≥3) |
|---|---|---|---|
| Warora | 0 / 0% / 0 | 0 / 0% / 0 | no ID |
| Nagpur | 1 / 0.02% / 0 | 0 / 0% / 0 | 2 / 0.03% / 0 |
| Delhi | 237 / 3.56% / 29 | 44 / 0.66% / 6 | 24 / 0.38% / 2 |
| Jaipur | 109 / 1.64% / 17 | 68 / 1.02% / 8 | no ID |
| Hisar | 219 / 3.29% / 28 | 130 / 1.95% / 20 | no ID |
| Karnal | 268 / 4.03% / 32 | 79 / 1.19% / 14 | no ID |
| Patiala | 210 / 3.16% / 30 | 158 / 2.37% / 25 | no ID |
| Amritsar | 292 / 4.39% / 35 | 405 / 6.09% / 37 | station ID was wrong, see below |
| **Abohar** | **257 / 3.86% / 29** | **146 / 2.19% / 21** | no ID |

*(These figures predate the `<=` threshold fix below — the script used
`<` when they were generated, so true counts are slightly higher than
shown here. Re-run `scripts/india_label_feasibility.py` to regenerate
with the fix applied; see "What still needs re-running" below.)*

## Site selection (for Tier 3 training/evaluation)

**Primary: Abohar** (Kinnow mandarin citrus belt). **Pooled stations:**
Amritsar, Patiala, Hisar — following Omazić et al.'s own approach of
pooling many stations rather than training on one. Karnal and Delhi
optional additions. Warora/Nagpur retained only as negative-control
sites if useful for illustrating the "no signal here" finding, not as
training data.

**Caveat, not yet verified:** Abohar being the right Kinnow reference
site is from general background knowledge, not a checked source — confirm
with ICAR-CCRI, Punjab Agricultural University (PAU), or the literature
before the paper states it as fact.

## Threshold and framing

**Primary label: Tmin ≤ 4°C**, with 3°C and 5°C as sensitivity runs (not
dropped — reported alongside, same spirit as the broad/strict pair from
the European work). Support for 4°C:
- IMD's own plains Cold Wave criterion (actual Tmin ≤ 4°C)
- Attri et al.'s frost-day definition (screen Tmin ≤ 4°C, based on grass
  minimum reaching 0°C at a screen Tmin around 4.1°C, citing Sentelhas et
  al. 1994 — a Brazilian study used here as a proxy; treat as unverified
  until checked directly)

**The crop-specific (Kinnow) injury threshold is still unsourced.** 4°C
is a general cold-wave/frost convention, not a citrus physiology number.
This needs ICAR-CCRI, PAU, or Kinnow-specific literature before being
presented as validated for the crop itself, not just for general cold.

## What the data show about the sources themselves

1. **NASA POWER roughly tracks Attri et al.'s IMD-station figures** at
   4 of 6 comparable sites (mean days below 4°C/season — Patiala 4.8 vs
   5; Karnal 6.1 vs 5; Hisar 5.0 vs 6; Jaipur 2.5 vs 3), but overshoots at
   Delhi (5.4 vs 2). Periods differ (1981-2025 here vs. 1980-2015 in the
   paper) and the original run used `<` vs. the paper's `<=` — both
   narrow but don't close the Delhi gap.
2. **ERA5 runs low almost everywhere** versus both NASA POWER and Attri
   et al. (e.g. Delhi 1.0 vs 2 days/season) — consistent with reanalysis
   products smoothing out cold extremes.
3. **Label rate is strongly source-dependent.** Delhi at 4°C: 0.38%
   (ISD) → 0.66% (ERA5) → 3.56% (NASA POWER) — a ~9x spread on the same
   site, same threshold. The paper must name which source the label uses
   and discuss this spread explicitly.
4. **ISD undercounts cold nights** (synoptic observation times miss the
   true daily minimum) — Delhi ISD gives ~0.6 days/season vs. Attri's ~2
   (true IMD minimum thermometer). ISD is a weaker label source than
   either gridded product here.
5. **Attri et al.'s own Amritsar figure (42 frost days/season) does not
   reproduce** in either gridded product (~7-9 days/season here). Likely
   a siting or data artifact in the original paper — don't treat Amritsar
   as "the typical Indian case" on the strength of that one number.

## Bugs found and fixed in `scripts/india_label_feasibility.py`

1. **Amritsar's ISD station ID was wrong.** The script used
   `420270-99999` (WMO 42027). That returned data with a mean Tmin of
   8.9°C and a minimum of **-22.6°C** — far too cold for Punjab plains,
   more consistent with a hill-station or far-northern climate. **Fixed**
   to `420710-99999` (WMO **42071**, confirmed via the Rajasansi Wikipedia
   article's citation of Amritsar's WMO climate-normals file,
   `WMO-Normals/RA-II/IN/42071.TXT`). **Not yet re-fetched and verified**
   — this sandbox can't reach NOAA directly; re-run the script in an
   internet-connected environment and sanity-check the result before
   trusting it (expect a Tmin distribution broadly similar to Patiala/
   Hisar, not the implausible one above).
2. **Delhi's ISD ID, independently verified correct.** `421820-99999`
   (WMO 42182) is confirmed as New Delhi/Safdarjung (28.58°N, 77.20°E,
   ICAO VIDD) via two independent sources (meteostat.net, tutiempo.net).
   No fix needed — this one was already right.
3. **Threshold comparison used `<` instead of `<=`.** Both IMD's Cold
   Wave criterion and Attri et al.'s frost-day definition are inclusive
   (`<=`). The strict `<` silently excluded a day at exactly the
   threshold temperature from that threshold's count — e.g. a 4.0°C day
   wasn't counted in the 4°C bucket. **Fixed.** A regression test
   (`test_threshold_is_inclusive_not_strict`) now checks this directly so
   it can't silently regress.
4. **No tests existed at all** (the file was also confusingly named
   `test.py`, which risked pytest trying to collect it as a test module
   rather than running it as a script). Renamed to
   `india_label_feasibility.py`; 5 logic-only unit tests added in
   `scripts/test_india_label_feasibility.py`, including the specific
   known-answer case requested (4 days at 1°C in one winter → 4 positive
   days, 1 winter) and a boundary test for the `<=` fix. All run without
   network access.

## What still needs re-running (requires internet access this sandbox doesn't have)

`scripts/india_label_feasibility.py` fetches live from NASA POWER,
Open-Meteo (ERA5), and NOAA ISD-Lite — none of which this development
sandbox can reach. The code fixes above are applied and unit-tested, but
**not yet re-validated against live data.** Next concrete step: run
```
python scripts/india_label_feasibility.py amritsar
```
in an environment with internet access, confirm Amritsar's corrected ISD
data now looks like Punjab's actual climate (not the -22.6°C outlier),
and regenerate `feasibility_summary.csv` with the `<=` fix applied
(expect all counts to tick up slightly from the table above).

## Still open (not resolved by this analysis)

- Abohar as the Kinnow reference site — unverified, needs CCRI/PAU/guide
  confirmation.
- Kinnow flowering/fruit-set timing relative to Dec-Feb cold — unknown;
  determines whether winter cold even threatens the crop's vulnerable
  stage.
- The crop-specific injury threshold itself — not yet sourced from CCRI/
  PAU/Kinnow literature.
- How the paper frames Warora given this finding (three options are live:
  keep Warora as the Repo B deployment target with Punjab/Haryana as a
  methodology-transfer evaluation region; switch the whole project's
  region; or keep Vidarbha under IMD's departure-based cold-wave
  definition instead of a frost/citrus-injury framing). This doesn't
  block the technical pipeline work below — all three options train on
  the same Punjab/Haryana data — but does need a decision before the
  paper's framing is finalized.

## Next steps for the repo (requires the re-run above first)

1. Extend the data pull to the full feature set: Tmax, Tmean, RH, dew
   point, wind speed (NASA POWER parameters likely `T2M_MAX, T2M, RH2M,
   T2MDEW, WS2M` — verify exact names; Open-Meteo/ERA5 daily variable
   availability for humidity/dew point needs checking, may require
   hourly aggregation).
2. Rebuild labels and features using the existing `src/label_rules.py` /
   `src/features.py` contract — decide whether the broad/strict
   (Td < 0°C) pair still makes sense at a 4°C base threshold, and whether
   to restrict to Nov-Mar rows or use the full year (state the chosen
   denominator explicitly, since prevalence differs ~2.4x between them).
3. **Split by winter season, not by random day** — cold days within one
   winter are correlated. For pooled multi-station training, consider
   leave-one-station-out as an additional evaluation view.
4. Re-run the full model comparison (persistence, physical/threshold,
   logistic regression, RF, XGBoost, LightGBM, CatBoost) at the new,
   much rarer prevalence (~1-6% of winter days vs. Basel's ~29.6%) — this
   is the direct test of whether the classifier-vs-threshold divergence
   found on European data reverses back toward Omazić et al.'s original
   pattern at a more realistic imbalance.
5. Re-check the wind feature-importance finding on Indian data — the
   European "nice-to-have, not must-have" verdict was already flagged as
   provisional pending exactly this.
6. Compare NASA POWER vs. ERA5 as the label source on the same site and
   report how model results shift — given the ~9x prevalence spread
   found above, this could matter more than which model architecture is
   used.
