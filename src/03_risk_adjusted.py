"""
03_risk_adjusted.py — Risk-adjusted return analysis.

Reproduces Table 4 (Sharpe Ratio) and Figure 3 (Max Drawdown) of the Myth Buster article.

Risk-free rate assumption: 3.5% (long-run mean of 10-year US Treasury 1995-2024,
FRED series DGS10).

Usage:
    python src/03_risk_adjusted.py

Output:
    - Console: Sharpe ratios and max drawdowns
    - results/table4_sharpe.csv
    - results/table5_drawdown.csv
"""

import os
import numpy as np
import pandas as pd
from utils import (
    load_assets, annual_returns, max_drawdown, candidate_years, control_years,
    ZHOU_WINDOWS, RESULTS_DIR, ensure_results_dir
)


# Risk-free rate: long-run mean of 10-year US Treasury yield (FRED DGS10)
RISK_FREE_RATE = 3.5  # percent


def sharpe_ratio(returns, rf=RISK_FREE_RATE):
    """Compute the annualized Sharpe ratio from a series of annual returns."""
    if len(returns) < 2:
        return np.nan
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r == 0:
        return np.nan
    return (mean_r - rf) / std_r


def main():
    ensure_results_dir()
    assets = load_assets()

    # === Table 4: Sharpe ratio ===
    print("=" * 78)
    print(f"Sharpe Ratio over 10-year holding periods (risk-free rate = {RISK_FREE_RATE}%)")
    print("=" * 78)

    sharpe_rows = []
    for asset_name, prices in assets.items():
        sharpes_for_baseline = []
        # 1999, 2008 windows (10y horizon complete)
        for y in [1999, 2008]:
            rets = annual_returns(prices, y, 10)
            if len(rets) >= 5:
                s = sharpe_ratio(rets)
                m = np.mean(rets)
                v = np.std(rets, ddof=1)
                print(f"{asset_name:<18s} {y}  mean={m:+6.2f}%  vol={v:6.2f}%  Sharpe={s:+.3f}")
                sharpe_rows.append({
                    'asset': asset_name, 'window': y,
                    'mean_return_pct': m, 'volatility_pct': v, 'sharpe': s,
                })

        # Baseline mean
        for y in control_years(ZHOU_WINDOWS, 1995, 2014):
            rets = annual_returns(prices, y, 10)
            if len(rets) >= 5:
                s = sharpe_ratio(rets)
                if not np.isnan(s):
                    sharpes_for_baseline.append(s)
        if sharpes_for_baseline:
            mean_sharpe = np.mean(sharpes_for_baseline)
            print(f"{asset_name:<18s} BASELINE                          Sharpe={mean_sharpe:+.3f}")
            sharpe_rows.append({
                'asset': asset_name, 'window': 'BASELINE_MEAN',
                'mean_return_pct': np.nan, 'volatility_pct': np.nan, 'sharpe': mean_sharpe,
            })
        print()

    df_sharpe = pd.DataFrame(sharpe_rows)
    out = os.path.join(RESULTS_DIR, 'table4_sharpe.csv')
    df_sharpe.to_csv(out, index=False, float_format='%.4f')
    print(f"Saved: {out}\n")

    # === Table 5 / Figure 3: Maximum Drawdown ===
    print("=" * 78)
    print("Maximum Drawdown over 10-year holding period")
    print("=" * 78)
    print("\nClaim under test: Kondratieff windows are 'low-point entries'.")
    print("If true, post-window max drawdown should be SMALLER than baseline.\n")

    dd_rows = []
    for asset_name, prices in assets.items():
        baseline_dds = []
        for y in ZHOU_WINDOWS:
            dd = max_drawdown(prices, y, 10)
            if not np.isnan(dd):
                print(f"{asset_name:<18s} {y}  max_dd = {dd:+6.1f}%")
                dd_rows.append({
                    'asset': asset_name, 'window': y, 'max_drawdown_pct': dd,
                })
        for y in control_years(ZHOU_WINDOWS, 1995, 2014):
            dd = max_drawdown(prices, y, 10)
            if not np.isnan(dd):
                baseline_dds.append(dd)
        if baseline_dds:
            mean_dd = np.mean(baseline_dds)
            print(f"{asset_name:<18s} BASELINE  max_dd = {mean_dd:+6.1f}%")
            dd_rows.append({
                'asset': asset_name, 'window': 'BASELINE_MEAN', 'max_drawdown_pct': mean_dd,
            })
        print()

    df_dd = pd.DataFrame(dd_rows)
    out = os.path.join(RESULTS_DIR, 'table5_drawdown.csv')
    df_dd.to_csv(out, index=False, float_format='%.4f')
    print(f"Saved: {out}")

    # Count windows where drawdown is worse than baseline
    print("\n" + "=" * 78)
    print("Verdict: are windows actually 'low-point entries'?")
    print("=" * 78)
    n_worse = 0
    n_total = 0
    for asset_name in assets:
        sub = df_dd[df_dd['asset'] == asset_name]
        baseline = sub[sub['window'] == 'BASELINE_MEAN']['max_drawdown_pct'].values
        if len(baseline) == 0:
            continue
        baseline_val = baseline[0]
        for window_year in ZHOU_WINDOWS:
            window_row = sub[sub['window'] == window_year]
            if window_row.empty:
                continue
            window_val = window_row['max_drawdown_pct'].values[0]
            n_total += 1
            if window_val < baseline_val:  # more negative = worse drawdown
                n_worse += 1
    print(f"\n{n_worse} of {n_total} windows show GREATER drawdown than baseline.")
    print("This is the OPPOSITE of what 'low-point entry' would predict.")


if __name__ == '__main__':
    main()
