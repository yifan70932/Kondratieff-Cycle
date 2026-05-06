"""
utils.py — Shared utilities for the Kondratieff cycle analysis.

Author: Yifan
License: MIT
"""

import os
import pandas as pd
import numpy as np


# Project-wide constants
ZHOU_WINDOWS = [1999, 2008, 2019]  # The three "wealth windows" claimed by Zhou Jintao
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# Random seed for reproducibility — used in all bootstrap/permutation procedures
RANDOM_SEED = 42


def load_assets():
    """Load all four asset price series as a dict of pandas Series indexed by year."""
    sse = pd.read_csv(os.path.join(DATA_DIR, 'sse_annual.csv')).set_index('year')['close']
    sse.name = 'A_shares'

    sp500 = pd.read_csv(os.path.join(DATA_DIR, 'sp500_annual.csv')).set_index('year')['close']
    sp500.name = 'SP500'

    bj = pd.read_csv(os.path.join(DATA_DIR, 'beijing_housing.csv'))
    bj = bj.set_index('year')['price_yuan_per_sqm']
    bj.name = 'Beijing_Housing'

    gold = pd.read_csv(os.path.join(DATA_DIR, 'gold_lbma.csv')).set_index('year')['price_usd_per_oz']
    gold.name = 'Gold'

    return {
        'A_shares': sse,
        'SP500': sp500,
        'Beijing_Housing': bj,
        'Gold': gold,
    }


def load_gdp_growth():
    """Load the US GDP growth rate series 1870-2024."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'us_gdp_growth.csv'))
    return df.set_index('year')['gdp_growth_pct']


def cagr(prices, start_year, n=10):
    """
    Compute the compound annual growth rate (CAGR) over an n-year holding period.

    Parameters
    ----------
    prices : pd.Series
        Asset price series indexed by year.
    start_year : int
        Year of purchase (start of holding period).
    n : int, default 10
        Number of years to hold.

    Returns
    -------
    float
        CAGR in percent. NaN if start or end year missing.
    """
    if start_year not in prices.index or (start_year + n) not in prices.index:
        return np.nan
    p0, p1 = prices.loc[start_year], prices.loc[start_year + n]
    if p0 <= 0:
        return np.nan
    return (np.power(p1 / p0, 1 / n) - 1) * 100


def annual_returns(prices, start_year, n=10):
    """
    Get year-on-year returns over an n-year holding period.

    Returns numpy array of percentage returns.
    """
    rets = []
    for i in range(n):
        y0 = start_year + i
        y1 = start_year + i + 1
        if y0 in prices.index and y1 in prices.index:
            rets.append((prices.loc[y1] / prices.loc[y0] - 1) * 100)
    return np.array(rets)


def max_drawdown(prices, start_year, n=10):
    """
    Compute the maximum drawdown over an n-year holding period.

    Returns the most negative peak-to-trough decline as a percentage (negative number).
    """
    series = []
    for y in range(start_year, start_year + n + 1):
        if y in prices.index:
            series.append(prices.loc[y])
    if len(series) < 2:
        return np.nan
    series = np.array(series)
    cummax = np.maximum.accumulate(series)
    dd = (series - cummax) / cummax
    return dd.min() * 100


def candidate_years(start=1995, end=2014):
    """All candidate buy years in the analysis window (default 1995-2014, inclusive)."""
    return list(range(start, end + 1))


def control_years(treated, start=1995, end=2014):
    """All non-treated candidate years (used as control group)."""
    return [y for y in candidate_years(start, end) if y not in treated]


def ensure_results_dir():
    """Create the results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


if __name__ == '__main__':
    # Self-test: load all data and print summary
    assets = load_assets()
    gdp = load_gdp_growth()
    print(f"Loaded {len(assets)} asset series:")
    for name, series in assets.items():
        print(f"  {name}: {series.index.min()}-{series.index.max()} ({len(series)} obs)")
    print(f"\nGDP growth: {gdp.index.min()}-{gdp.index.max()} ({len(gdp)} obs)")
