"""
correlation_engine.py

Backend module for robust time-lagged correlation discovery between
Variable A (cause) and Variable B (effect).

Core function: find_correlations(...) returns:
 - results_df: DataFrame with one row per lag and multiple stats
 - significant_df: subset that meets the conservative criteria
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import math
import random

# ---------------------
# Utilities
# ---------------------

def ensure_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in dataframe")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    df = df.set_index(time_col)
    return df

def hours_to_timedelta(hours: float) -> pd.Timedelta:
    return pd.Timedelta(hours=hours)

# ---------------------
# Matching logic
# ---------------------

def build_pairs_by_lag(
    A: Union[pd.Series, pd.DataFrame],
    B: Union[pd.Series, pd.DataFrame],
    A_time_col: Optional[str] = None,
    B_time_col: Optional[str] = None,
    A_value_col: Optional[str] = None,
    B_value_col: Optional[str] = None,
    lags_hours: Optional[List[float]] = None,
    match_window_hours: float = 4.0,
    agg_B_fn: str = "mean",
    require_nonnull_A: bool = True,
    require_nonnull_B: bool = True,
) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Match A and B observations for each lag and return paired arrays."""

    def to_series(x, time_col, value_col):
        if isinstance(x, pd.Series):
            s = x.copy()
            if not isinstance(s.index, pd.DatetimeIndex):
                raise ValueError("If A or B are Series they must have DatetimeIndex")
            return s.sort_index()
        elif isinstance(x, pd.DataFrame):
            if time_col is None or value_col is None:
                raise ValueError("Must provide time_col and value_col for DataFrame inputs")
            df = x.copy()
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col)
            s = pd.Series(df[value_col].values, index=df[time_col])
            return s
        else:
            raise ValueError("A and B must be pandas Series or DataFrames")

    series_A = to_series(A, A_time_col, A_value_col)
    series_B = to_series(B, B_time_col, B_value_col)

    if lags_hours is None:
        lags_hours = [0.0]

    half_window = hours_to_timedelta(match_window_hours / 2.0)
    results = {}

    for lag in lags_hours:
        lag_delta = hours_to_timedelta(lag)
        A_vals, B_vals, A_times, B_match_times = [], [], [], []

        for tA, valA in series_A.items():
            if require_nonnull_A and pd.isna(valA):
                continue
            start = tA + lag_delta - half_window
            end = tA + lag_delta + half_window
            b_window = series_B[start:end]
            if require_nonnull_B:
                b_window = b_window[~b_window.isna()]
            if b_window.empty:
                continue
            if agg_B_fn == "mean":
                b_agg = b_window.mean()
            elif agg_B_fn == "median":
                b_agg = b_window.median()
            elif agg_B_fn == "sum":
                b_agg = b_window.sum()
            elif agg_B_fn == "max":
                b_agg = b_window.max()
            elif agg_B_fn == "min":
                b_agg = b_window.min()
            else:
                b_agg = b_window.mean()

            A_vals.append(valA)
            B_vals.append(b_agg)
            A_times.append(tA)
            B_match_times.append(b_window.index[0] if len(b_window.index) == 1 else b_window.index.mean())

        if len(A_vals) < 3:
            results[lag] = (np.array([]), np.array([]), np.array([]), np.array([]))
        else:
            results[lag] = (np.array(A_vals), np.array(B_vals), np.array(A_times), np.array(B_match_times))

    return results

# ---------------------
# Statistical tests
# ---------------------

def compute_pearson_spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, Union[float, None]]:
    if len(x) < 3:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    if np.allclose(np.nanstd(x), 0) or np.allclose(np.nanstd(y), 0):
        return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    try:
        pr, pp = pearsonr(x, y)
    except Exception:
        pr, pp = np.nan, np.nan
    try:
        sr, sp = spearmanr(x, y)
    except Exception:
        sr, sp = np.nan, np.nan
    return {"pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp}

def permutation_test_correlation(x: np.ndarray, y: np.ndarray, statistic_fn=lambda a,b: pearsonr(a,b)[0], n_permutations: int = 2000, random_state: Optional[int] = None) -> Dict[str, Any]:
    rng = np.random.RandomState(random_state)
    obs_stat = np.nan
    try:
        obs_stat = statistic_fn(x, y)
    except Exception:
        pass
    null_stats = []
    for _ in range(n_permutations):
        perm = rng.permutation(x)
        try:
            s = statistic_fn(perm, y)
        except Exception:
            s = np.nan
        null_stats.append(s)
    null_stats = np.array(null_stats)
    p_val = np.nan if np.isnan(obs_stat) else (np.sum(np.abs(null_stats) >= abs(obs_stat)) + 1) / (n_permutations + 1)
    return {"obs_stat": obs_stat, "p_value": p_val, "null_stats": null_stats}

def bootstrap_ci_for_r(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 2000, random_state: Optional[int] = None, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.RandomState(random_state)
    n = len(x)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        xs, ys = x[idx], y[idx]
        try:
            r = pearsonr(xs, ys)[0]
        except Exception:
            r = np.nan
        boots.append(r)
    boots = np.array(boots)
    lower = np.nanpercentile(boots, 100 * (alpha / 2.0))
    upper = np.nanpercentile(boots, 100 * (1 - alpha / 2.0))
    return float(lower), float(upper)

# ---------------------
# Multiple testing correction
# ---------------------

def apply_fdr_correction(pvals: List[float], alpha: float = 0.05, method: str = "fdr_bh") -> Tuple[np.ndarray, np.ndarray]:
    pvals = np.array(pvals, dtype=float)
    mask_nan = np.isnan(pvals)
    pvals_for_test = pvals.copy()
    pvals_for_test[mask_nan] = 1.0
    rej, pvals_corrected, _, _ = multipletests(pvals_for_test, alpha=alpha, method=method)
    rej[mask_nan] = False
    return rej, pvals_corrected

# ---------------------
# Main pipeline
# ---------------------

def find_correlations(
    A: Union[pd.Series, pd.DataFrame],
    B: Union[pd.Series, pd.DataFrame],
    A_time_col: Optional[str] = None,
    B_time_col: Optional[str] = None,
    A_value_col: Optional[str] = None,
    B_value_col: Optional[str] = None,
    lags_hours: Optional[List[float]] = None,
    match_window_hours: float = 4.0,
    agg_B_fn: str = "mean",
    min_pairs: int = 10,
    permutation_n: int = 2000,
    bootstrap_n: int = 2000,
    effect_size_thresh: float = 0.25,
    alpha: float = 0.05,
    require_agreement: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if lags_hours is None:
        lags_hours = list(range(0,9)) + [12, 24, 48, 72, 168]

    pairs_dict = build_pairs_by_lag(
        A, B, A_time_col=A_time_col, B_time_col=B_time_col, A_value_col=A_value_col, B_value_col=B_value_col,
        lags_hours=lags_hours, match_window_hours=match_window_hours, agg_B_fn=agg_B_fn
    )

    rows, pvals_for_fdr = [], []
    for lag in lags_hours:
        A_vals, B_vals, _, _ = pairs_dict.get(lag, (np.array([]),)*4)
        n_pairs = len(A_vals)
        row = {"lag_hours": float(lag), "n_pairs": n_pairs, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan, "perm_p": np.nan, "pearson_boot_ci_low": np.nan, "pearson_boot_ci_high": np.nan}
        if n_pairs >= max(3, min_pairs):
            stats = compute_pearson_spearman(A_vals, B_vals)
            row.update(stats)
            perm = permutation_test_correlation(A_vals, B_vals, statistic_fn=lambda a,b: pearsonr(a,b)[0], n_permutations=permutation_n, random_state=random_state)
            row["perm_p"] = perm["p_value"]
            try:
                low, high = bootstrap_ci_for_r(A_vals, B_vals, n_bootstrap=bootstrap_n, random_state=random_state, alpha=alpha)
            except Exception:
                low, high = np.nan, np.nan
            row["pearson_boot_ci_low"], row["pearson_boot_ci_high"] = low, high
            pvals_for_fdr.append(row["pearson_p"])
        else:
            pvals_for_fdr.append(np.nan)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    rej, pvals_corrected = apply_fdr_correction(results_df["pearson_p"].tolist(), alpha=alpha)
    results_df["pearson_p_fdr"], results_df["pearson_reject_fdr"] = pvals_corrected, rej

    sig_mask = []
    for _, r in results_df.iterrows():
        ok = False
        if r["n_pairs"] >= min_pairs and not math.isnan(r["pearson_r"]):
            if abs(r["pearson_r"]) >= effect_size_thresh and not math.isnan(r["pearson_p_fdr"]) and r["pearson_p_fdr"] < alpha:
                if not math.isnan(r["perm_p"]) and r["perm_p"] < alpha:
                    low, high = r["pearson_boot_ci_low"], r["pearson_boot_ci_high"]
                    if not math.isnan(low) and not math.isnan(high) and (low > 0 or high < 0):
                        if require_agreement:
                            if not math.isnan(r["spearman_r"]) and np.sign(r["spearman_r"]) == np.sign(r["pearson_r"]):
                                ok = True
                        else:
                            ok = True
        sig_mask.append(ok)

    results_df["significant"] = sig_mask
    significant_df = results_df[results_df["significant"]].copy().reset_index(drop=True)
    return results_df, significant_df
