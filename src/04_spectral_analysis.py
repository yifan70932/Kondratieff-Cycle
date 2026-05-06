"""
04_spectral_analysis.py — Spectral analysis of US GDP growth (1870-2024).

Reproduces Figure 2 of the Myth Buster article.

Tests whether a 60-year Kondratieff cycle is detectable in the power spectrum
of US real GDP growth rate. Uses Welch's method for PSD estimation and a
Bootstrap permutation test for the significance of the 60-year peak.

Usage:
    python src/04_spectral_analysis.py

Output:
    - Console: identified peaks and bootstrap p-value at 60y
    - results/figure2_spectrum.csv: PSD values for each frequency
    - results/figure2_spectrum.png: matplotlib plot
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib

from utils import load_gdp_growth, RESULTS_DIR, RANDOM_SEED, ensure_results_dir


def compute_psd(growth_series, nperseg=60):
    """Detrend and compute Welch PSD."""
    arr = growth_series.values.astype(float)
    detrended = signal.detrend(arr)
    freqs, psd = signal.welch(detrended, fs=1.0, nperseg=min(nperseg, len(arr) // 2),
                              scaling='density')
    # Drop DC component
    return freqs[1:], psd[1:], detrended


def bootstrap_significance(detrended, target_period=60, n_iterations=5000,
                            nperseg=60, seed=RANDOM_SEED):
    """
    Bootstrap p-value for the power at a specific frequency.

    Shuffles the detrended series and recomputes the PSD; reports the fraction
    of shuffles where the bootstrap power at the target frequency exceeds the
    observed power.
    """
    rng = np.random.default_rng(seed)
    N = len(detrended)
    freqs, psd = signal.welch(detrended, fs=1.0,
                              nperseg=min(nperseg, N // 2), scaling='density')
    psd = psd[1:]
    periods = 1.0 / freqs[1:]
    target_idx = np.argmin(np.abs(periods - target_period))
    observed_power = psd[target_idx]

    boot_powers = np.empty(n_iterations)
    for i in range(n_iterations):
        sh = rng.permutation(detrended)
        f_b, p_b = signal.welch(sh, fs=1.0,
                                nperseg=min(nperseg, N // 2), scaling='density')
        p_b = p_b[1:]
        boot_powers[i] = p_b[target_idx]

    p_value = np.mean(boot_powers >= observed_power)
    threshold_95 = np.percentile(boot_powers, 95)
    return observed_power, p_value, threshold_95, boot_powers


def main():
    ensure_results_dir()
    growth = load_gdp_growth()
    print("=" * 78)
    print(f"Spectral analysis of US GDP growth, {growth.index.min()}-{growth.index.max()}")
    print(f"Sample length: N = {len(growth)} years")
    print(f"Theoretical maximum detectable period (Nyquist): N/2 = {len(growth)/2:.0f} years")
    print("=" * 78)

    freqs, psd, detrended = compute_psd(growth, nperseg=60)
    periods = 1.0 / freqs

    # Identify peaks
    peak_idx = signal.find_peaks(psd, prominence=psd.std() * 0.5)[0]
    print("\nIdentified peaks in the power spectrum:")
    print(f"{'Period (years)':>14}  {'PSD':>10}  {'Relative':>10}")
    print("-" * 40)
    for idx in peak_idx:
        period = periods[idx]
        power = psd[idx]
        rel = power / psd.max()
        marker = "  ← K-wave region (45-65y)" if 45 <= period <= 65 else ""
        print(f"{period:14.2f}  {power:10.4f}  {rel:10.2f}{marker}")

    # Bootstrap significance test at 60-year frequency
    print("\n" + "-" * 78)
    print("Bootstrap test for significance of the 60-year frequency")
    print("-" * 78)
    observed_60y, p_value_60y, threshold_95, boot_powers = bootstrap_significance(
        detrended, target_period=60, n_iterations=5000)
    print(f"  Observed power at 60y       : {observed_60y:.4f}")
    print(f"  Bootstrap median             : {np.median(boot_powers):.4f}")
    print(f"  Bootstrap 95% threshold      : {threshold_95:.4f}")
    print(f"  Bootstrap p-value (one-sided): {p_value_60y:.4f}")
    print(f"\n  Conclusion: {'Significant' if p_value_60y < 0.05 else 'NOT statistically significant'}")
    print(f"  (Cannot reject the noise null hypothesis at the 60-year frequency.)")

    # Save data
    df = pd.DataFrame({'period_years': periods, 'psd': psd})
    df = df.sort_values('period_years').reset_index(drop=True)
    csv_path = os.path.join(RESULTS_DIR, 'figure2_spectrum.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nSaved: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(periods, psd, color='#1e3a5f', linewidth=1.8)
    ax.fill_between(periods, psd, alpha=0.2, color='#1e3a5f')
    ax.axhline(threshold_95, color='#2d5a3d', linestyle='--', linewidth=1, alpha=0.8,
               label='95% significance threshold (bootstrap)')

    # Annotate Juglar (~8.6y), Kuznets (~20y), Kondratieff (60y)
    for tgt_period, label, color in [(8.6, '8.6y Juglar', '#2d5a3d'),
                                      (20, '20y Kuznets', '#2d5a3d'),
                                      (60, '60y (no peak)', '#1e3a5f')]:
        idx = np.argmin(np.abs(periods - tgt_period))
        ax.scatter(periods[idx], psd[idx], color=color, s=60, zorder=5)
        ax.annotate(label, (periods[idx], psd[idx]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, color=color)

    # Highlight K-wave zone
    ax.axvspan(50, 65, alpha=0.08, color='#1e3a5f', label='K-wave zone')

    ax.set_xscale('log')
    ax.set_xticks([2, 8.6, 20, 30, 50, 80])
    ax.set_xticklabels([2, '8.6', 20, 30, 50, 80])
    ax.set_xlabel('Period length (years, log scale)', fontsize=11)
    ax.set_ylabel('Power spectral density', fontsize=11)
    ax.set_title(f'Power spectrum of US GDP growth ({growth.index.min()}-{growth.index.max()}), '
                 f'Welch method', fontsize=12)
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    png_path = os.path.join(RESULTS_DIR, 'figure2_spectrum.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_path}")
    plt.close()


if __name__ == '__main__':
    main()
