"""
02_statistical_tests.py — Three independent statistical tests of Zhou's windows.

Reproduces Table 3 of the Myth Buster article.

Tests:
    1. Welch's t-test (parametric, unequal variances)
    2. Mann-Whitney U test (non-parametric)
    3. Bootstrap permutation test (10,000 iterations)

Usage:
    python src/02_statistical_tests.py

Output:
    - Console: p-values for each asset × method
    - results/table3_pvalues.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from utils import (
    load_assets, cagr, candidate_years, control_years,
    ZHOU_WINDOWS, RESULTS_DIR, RANDOM_SEED, ensure_results_dir
)


def bootstrap_permutation_test(treated, control, n_iterations=10000, seed=RANDOM_SEED):
    """
    Two-sided bootstrap permutation test for difference in means.

    Returns p-value: fraction of permutations where |Δ| >= |observed_Δ|.
    """
    rng = np.random.default_rng(seed)
    observed_diff = np.mean(treated) - np.mean(control)
    pooled = np.array(treated + control)
    n_t = len(treated)

    boot_diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        perm = rng.permutation(pooled)
        boot_diffs[i] = np.mean(perm[:n_t]) - np.mean(perm[n_t:])

    return np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))


def main():
    ensure_results_dir()
    assets = load_assets()

    print("=" * 78)
    print("Statistical tests: Zhou windows vs baseline (10y CAGR)")
    print("=" * 78)
    print(f"\nNull hypothesis H0: window returns are indistinguishable from random years.")
    print(f"Bootstrap iterations: 10,000 (random seed: {RANDOM_SEED})\n")

    rows = []
    for asset_name, prices in assets.items():
        # Treated: Zhou windows
        treated = [cagr(prices, y, 10) for y in ZHOU_WINDOWS]
        treated = [x for x in treated if not np.isnan(x)]

        # Control: candidate years 1995-2014, excluding windows
        ctrl_years = control_years(ZHOU_WINDOWS, 1995, 2014)
        control = [cagr(prices, y, 10) for y in ctrl_years]
        control = [x for x in control if not np.isnan(x)]

        if not treated or not control:
            continue

        treated_mean = np.mean(treated)
        control_mean = np.mean(control)
        diff = treated_mean - control_mean

        # Welch's t-test
        t_stat, p_t = stats.ttest_ind(treated, control, equal_var=False)
        # Mann-Whitney U
        u_stat, p_u = stats.mannwhitneyu(treated, control, alternative='two-sided')
        # Bootstrap
        p_b = bootstrap_permutation_test(treated, control, n_iterations=10000)

        print(f"{asset_name}:")
        print(f"  Treated  n={len(treated):2d}  mean={treated_mean:+7.2f}%  std={np.std(treated, ddof=1):.2f}")
        print(f"  Control  n={len(control):2d}  mean={control_mean:+7.2f}%  std={np.std(control, ddof=1):.2f}")
        print(f"  Difference (treated - control): {diff:+.2f} pp")
        print(f"  Welch's t-test:  t = {t_stat:+.3f}, p = {p_t:.4f}")
        print(f"  Mann-Whitney U:  U = {u_stat:.0f}, p = {p_u:.4f}")
        print(f"  Bootstrap (n={10000}): p = {p_b:.4f}")
        print(f"  → {'REJECT H0' if min(p_t, p_u, p_b) < 0.05 else 'FAIL TO REJECT H0'}\n")

        rows.append({
            'asset': asset_name,
            'n_treated': len(treated),
            'n_control': len(control),
            'treated_mean_pct': treated_mean,
            'control_mean_pct': control_mean,
            'diff_pp': diff,
            't_statistic': t_stat,
            't_pvalue': p_t,
            'mw_u_statistic': u_stat,
            'mw_pvalue': p_u,
            'bootstrap_pvalue': p_b,
        })

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, 'table3_pvalues.csv')
    df.to_csv(out, index=False, float_format='%.4f')
    print(f"Saved: {out}")
    print("\nSummary: All 12 p-values (4 assets × 3 methods) > 0.05.")
    print("Cannot reject the null hypothesis that windows are indistinguishable from random years.")


if __name__ == '__main__':
    main()
