"""
01_event_study.py — Event Study: Zhou Jintao's "wealth windows" vs all candidate years.

Reproduces Table 1 and Figure 1 of the Myth Buster article.

Usage:
    python src/01_event_study.py

Output:
    - Console: ranking and CAGR comparison
    - results/table1_cagr.csv: machine-readable CAGR table
"""

import os
import pandas as pd
import numpy as np
from utils import (
    load_assets, cagr, candidate_years, ZHOU_WINDOWS,
    RESULTS_DIR, ensure_results_dir
)


def main():
    ensure_results_dir()
    assets = load_assets()

    # === Table 1: 10-year CAGR for Zhou windows + baseline + actual best ===
    print("=" * 78)
    print("Table 1: 10-year CAGR by start year and asset class")
    print("=" * 78)

    rows = []
    for asset_name, prices in assets.items():
        # Zhou windows
        window_cagrs = {y: cagr(prices, y, 10) for y in ZHOU_WINDOWS}

        # Baseline distribution: all candidate years 1995-2014
        baseline = []
        for y in candidate_years(1995, 2014):
            r = cagr(prices, y, 10)
            if not np.isnan(r):
                baseline.append((y, r))

        # Find best year and worst year in the candidate set
        baseline_sorted = sorted(baseline, key=lambda x: -x[1])
        best_year, best_cagr = baseline_sorted[0]
        worst_year, worst_cagr = baseline_sorted[-1]
        baseline_returns = [r for _, r in baseline]

        # Rankings of Zhou windows in the full set
        all_pairs = sorted(baseline, key=lambda x: -x[1])
        rank_map = {y: i + 1 for i, (y, _) in enumerate(all_pairs)}

        for year in ZHOU_WINDOWS:
            rank = rank_map.get(year, None)
            rows.append({
                'asset': asset_name,
                'start_year': year,
                'is_window': True,
                'cagr_10y_pct': window_cagrs[year],
                'rank': rank,
                'rank_total': len(baseline) if year in rank_map else None,
            })

        rows.append({
            'asset': asset_name,
            'start_year': 'BASELINE_MEAN',
            'is_window': False,
            'cagr_10y_pct': np.mean(baseline_returns),
            'rank': None,
            'rank_total': len(baseline_returns),
        })
        rows.append({
            'asset': asset_name,
            'start_year': 'BASELINE_MEDIAN',
            'is_window': False,
            'cagr_10y_pct': np.median(baseline_returns),
            'rank': None,
            'rank_total': len(baseline_returns),
        })
        rows.append({
            'asset': asset_name,
            'start_year': f'BEST_{best_year}',
            'is_window': False,
            'cagr_10y_pct': best_cagr,
            'rank': 1,
            'rank_total': len(baseline_returns),
        })
        rows.append({
            'asset': asset_name,
            'start_year': f'WORST_{worst_year}',
            'is_window': False,
            'cagr_10y_pct': worst_cagr,
            'rank': len(baseline_returns),
            'rank_total': len(baseline_returns),
        })

        # Print human-readable summary
        print(f"\n{asset_name}:")
        for y in ZHOU_WINDOWS:
            r = window_cagrs[y]
            rank_str = f"rank {rank_map.get(y, 'NA')}/{len(baseline)}" if y in rank_map else "horizon incomplete"
            print(f"  {y} (Zhou window): {r:+7.2f}%  [{rank_str}]")
        print(f"  Baseline mean : {np.mean(baseline_returns):+7.2f}%")
        print(f"  Baseline median: {np.median(baseline_returns):+7.2f}%")
        print(f"  Actual best   : {best_year} ({best_cagr:+.2f}%)")
        print(f"  Actual worst  : {worst_year} ({worst_cagr:+.2f}%)")

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'table1_cagr.csv')
    df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nSaved: {out_path}")

    # === 2019 5-year horizon test (Section 5 of the article) ===
    print("\n" + "=" * 78)
    print("Test: 2019 5-year CAGR rank (Section 5 of the article)")
    print("=" * 78)

    rows_5y = []
    for asset_name, prices in assets.items():
        pairs = []
        for y in range(2010, 2021):
            r = cagr(prices, y, 5)
            if not np.isnan(r):
                pairs.append((y, r))
        if not pairs:
            continue
        pairs_sorted = sorted(pairs, key=lambda x: -x[1])
        rank_map = {y: i + 1 for i, (y, _) in enumerate(pairs_sorted)}
        rank_2019 = rank_map.get(2019, None)
        cagr_2019 = next((r for y, r in pairs if y == 2019), None)
        best_y, best_r = pairs_sorted[0]
        print(f"{asset_name}: 2019 CAGR {cagr_2019:+.2f}%, rank {rank_2019}/{len(pairs)}; best year {best_y} ({best_r:+.2f}%)")
        rows_5y.append({
            'asset': asset_name,
            'cagr_2019_5y': cagr_2019,
            'rank_2019': rank_2019,
            'n_candidates': len(pairs),
            'best_year': best_y,
            'best_cagr_5y': best_r,
        })

    df_5y = pd.DataFrame(rows_5y)
    out_path_5y = os.path.join(RESULTS_DIR, 'table2_5y.csv')
    df_5y.to_csv(out_path_5y, index=False, float_format='%.4f')
    print(f"\nSaved: {out_path_5y}")


if __name__ == '__main__':
    main()
