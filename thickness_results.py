# Open Use License
#
# You may use, copy, modify, and redistribute this file for any purpose,
# provided that redistributions include this original source code (including this notice).
# This software is provided "as is" without warranty of any kind.

"""
PVDF thickness comparison plotter with propagated uncertainties.

Using the February 27–28, 2024 calibration campaigns (30 µm, 8 µm, 3 µm films),
the script re-creates the classic Q / m^1.3 versus v^3 plots while incorporating
the measurement systematics captured in Oscilloscope Palantir.  For each HDF5
dataset it:

  • loads the `fits` table, extracts the channel-4 `low_pass_max` amplitudes
    and channel-2 QD3Fit outputs, and filters events below 1.1 km/s or 60 mV,
  • converts the measured voltage to charge using the thickness-dependent
    Cremat transfer function (Q = slope · V + intercept),
  • propagates a 2% charge uncertainty, a fixed ±10 m/s velocity uncertainty,
    and the oscilloscope voltage error (30 µm: 5 mV, 8 µm: 10 mV, 3 µm: 20 mV)
    to obtain per-shot σ(Q), σ(m), σ(v³), and σ(Q / m^1.3),
  • performs an outlier-resistant regression of the stopping-law form
    Q = C · m^1.3 · v^3 for each film thickness and reports the fitted C value,
  • renders the log–log scatter of v^3 vs Q / m^1.3 with dataset-specific color
    coding, error bars (without caps), data-driven axis limits, and Simpson
    theoretical overlays, and
  • runs a shared-exponent nonlinear least-squares fit that solves for a single
    pair (α, β) across all thicknesses while allowing each dataset an
    independent normalization C, plotting the transformed v^β vs Q / m^α view
    with matching error bars and hoverable event IDs.

In short, the script converts raw Palantir fits into uncertainty-aware
calibration figures for the thickness study and reports both per-film and
global stopping-law constants.

Shared power-law fit details: each event contributes `log Q`, `log m`, and
`log v`.  For any trial exponents α and β, the per-dataset normalization is
`log C_i = mean(log Q - α log m - β log v)` across events in dataset i.  The
global residual vector (`log Q - log C_i - α log m - β log v`) feeds both a
bounded nonlinear least-squares solver that finds the optimal α/β and a dense
χ² grid that is plotted so you can see how the solution compares to the legacy
(1.3, 3.0) exponents.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import optimize

# -------------------------------------------------------------
# Physical constants / helper functions
# -------------------------------------------------------------

E_CHARGE = 1.602176634e-19  # Coulombs per electron
E_PER_COULOMB = 1.0 / E_CHARGE

# Simpson/Tuzzolino analytical forms (charges returned in electrons)
def pvdf_penetration(mass_g: np.ndarray, vel_kms: np.ndarray) -> np.ndarray:
    """Penetration calibration (Simpson 1985 / Tuzzolino 2001)."""
    return 1.4e13 * np.power(mass_g, 0.7) * np.power(vel_kms, 1.3)


def pvdf_stopping(mass_g: np.ndarray, vel_kms: np.ndarray) -> np.ndarray:
    return 3.8e17 * np.power(mass_g, 1.3) * np.power(vel_kms, 3.0)


def pvdf_stopping_8(mass_g: np.ndarray, vel_kms: np.ndarray) -> np.ndarray:
    return 3.6e18 * np.power(mass_g, 1.3) * np.power(vel_kms, 3.0)


def pvdf_stopping_3(mass_g: np.ndarray, vel_kms: np.ndarray) -> np.ndarray:
    return 6.0e18 * np.power(mass_g, 1.3) * np.power(vel_kms, 3.0)


# Cremat CSA transfer functions (Q [C] = slope * V + intercept)
CSA_CAL: Dict[str, Tuple[float, float]] = {
    "30um": (-8.29338526e-13, -1.24407122e-14),
    "8um": (1.02344806e-12, 5.45524781e-15),
    "3um": (-8.23012166e-13, 2.30821114e-16),
}

# HDF5 sources produced by Oscilloscope Palantir
HDF_ROOT = Path("../hdf5_data")
DATASETS = [
    {
        "label": "30 µm PVDF (Cremat CSA)",
        "key": "30um",
        "path": HDF_ROOT / "2024_02_27_run0_30um_cremat_csa.h5",
        "voltage_unc": 0.005,
    },
    {
        "label": "8 µm PVDF (Cremat CSA)",
        "key": "8um",
        "path": HDF_ROOT / "2024_02_27_run1_8um_cremat_csa.h5",
        "voltage_unc": 0.010,
    },
    {
        "label": "3 µm PVDF (Cremat CSA)",
        "key": "3um",
        "path": HDF_ROOT / "2024_02_28_run1_3um_cremat_csa.h5",
        "voltage_unc": 0.020,
    },
]

COLOR_PALETTE: Dict[str, Dict[str, str]] = {
    "30um": {
        "scatter": "#2e7d32",  # deep green
        "fit": "#43a047",      # mid green
        "theory": "#81c784",   # light green
    },
    "8um": {
        "scatter": "#1565c0",  # deep blue
        "fit": "#1e88e5",      # mid blue
        "theory": "#64b5f6",   # light blue
    },
    "3um": {
        "scatter": "#b71c1c",  # deep red
        "fit": "#d32f2f",      # mid red
        "theory": "#ef5350",   # light red
    },
}

PLOT_ORDER = {
    "30um": 0,
    "3um": 1,
    "8um": 2,  # plot last so 8 µm points sit on top of the denser 3 µm cloud
}


def pad_bounds(lo: float, hi: float, frac: float = 0.01) -> Tuple[float, float]:
    """Return positive bounds expanded multiplicatively for log-scale plotting."""
    lo = float(lo)
    hi = float(hi)

    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("Bounds must be finite numbers.")

    lo = max(lo, 1e-12)
    hi = max(hi, lo * (1.0 + 1e-6))

    if hi <= lo:
        hi = lo * (1.0 + 1e-3)

    lo_padded = max(lo * (1.0 - frac), 1e-12)
    hi_padded = hi * (1.0 + frac)

    if hi_padded <= lo_padded:
        hi_padded = lo_padded * (1.0 + 1e-3)

    return lo_padded, hi_padded


def load_dataset(meta: Dict[str, str]) -> pd.DataFrame:
    """Merge low_pass_max (channel 4) with QD3Fit (channel 2) for one dataset."""
    h5_path = meta["path"]
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")

    df = pd.read_hdf(h5_path, key="fits")

    # Pull channel‑4 low_pass_max results and rename columns
    lp = (
        df[(df["fit_type"] == "low_pass_max") & (df["channel"] == 4)]
        .loc[:, ["event", "value"]]
        .copy()
    )
    lp.rename(columns={"value": "amplitude_v"}, inplace=True)

    # Channel‑2 QD3 fits supply mass (kg) & velocity (m/s)
    qd = (
        df[df["fit_type"].str.startswith("QD3Fit", na=False) & (df["channel"] == 2)]
        .loc[:, ["event", "mass", "v"]]
        .copy()
    )

    merged = pd.merge(qd, lp, on="event", how="inner")
    merged.dropna(subset=["mass", "v", "amplitude_v"], inplace=True)

    # Convert units
    merged["mass_g"] = merged["mass"] * 1e3  # kg -> g
    merged["vel_kms"] = merged["v"] / 1e3    # m/s -> km/s

    # Apply calibration curve and propagate measurement uncertainties
    voltage_unc = float(meta.get("voltage_unc", 0.005))
    sigma_voltage = np.full(len(merged), voltage_unc, dtype=float)
    sigma_vel_kms = np.full(len(merged), 0.010, dtype=float)  # 10 m/s -> 0.01 km/s

    amplitudes = merged["amplitude_v"].to_numpy(dtype=float)
    slope, intercept = CSA_CAL[meta["key"]]
    charge_c = slope * amplitudes + intercept
    merged["charge_c"] = charge_c
    merged["charge_e"] = charge_c * E_PER_COULOMB

    sigma_charge_voltage = np.abs(slope) * sigma_voltage
    sigma_charge_c = np.sqrt(sigma_charge_voltage**2 + (0.02 * charge_c) ** 2)

    vel_kms_arr = merged["vel_kms"].to_numpy(dtype=float)
    mass_g_arr = merged["mass_g"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_charge = np.divide(
            sigma_charge_c,
            charge_c,
            out=np.zeros_like(sigma_charge_c),
            where=charge_c != 0.0,
        )
        rel_vel = np.divide(
            sigma_vel_kms,
            vel_kms_arr,
            out=np.zeros_like(sigma_vel_kms),
            where=vel_kms_arr != 0.0,
        )

    sigma_mass_g = mass_g_arr * np.sqrt(rel_charge**2 + rel_vel**2)

    merged["sigma_voltage"] = sigma_voltage
    merged["sigma_charge_c"] = sigma_charge_c
    merged["sigma_vel_kms"] = sigma_vel_kms
    merged["sigma_mass_g"] = sigma_mass_g

    # Derived axes used in legacy plots
    merged["vel_pow3"] = np.power(merged["vel_kms"], 3.0)
    merged["mass_pow13"] = np.power(merged["mass_g"], 1.3)
    merged["charge_over_mass"] = merged["charge_c"] / merged["mass_pow13"]

    # Drop any non-finite values that would break plotting/log-scaling
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.dropna(subset=["vel_pow3", "charge_over_mass"], inplace=True)

    # Apply requested quality cuts:
    #  • velocity must be >= 1.1 km/s
    #  • low_pass_max amplitude must be >= 0.060 V
    merged = merged[
        (merged["vel_kms"] >= 1.1) &
        (np.abs(merged["amplitude_v"]) >= 0.060)
    ].reset_index(drop=True)

    charge_arr = merged["charge_c"].to_numpy(dtype=float)
    sigma_charge_arr = merged["sigma_charge_c"].to_numpy(dtype=float)
    mass_arr = merged["mass_g"].to_numpy(dtype=float)
    sigma_mass_arr = merged["sigma_mass_g"].to_numpy(dtype=float)
    vel_arr = merged["vel_kms"].to_numpy(dtype=float)
    sigma_vel_arr = merged["sigma_vel_kms"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_charge_post = np.divide(
            sigma_charge_arr,
            charge_arr,
            out=np.zeros_like(sigma_charge_arr),
            where=charge_arr != 0.0,
        )
        rel_mass_post = np.divide(
            sigma_mass_arr,
            mass_arr,
            out=np.zeros_like(sigma_mass_arr),
            where=mass_arr != 0.0,
        )

    sigma_vel_pow3 = np.abs(3.0 * np.power(vel_arr, 2.0) * sigma_vel_arr)
    alpha_fixed = 1.3
    sigma_charge_over_mass = merged["charge_over_mass"].to_numpy(dtype=float) * np.sqrt(
        rel_charge_post**2 + (alpha_fixed * rel_mass_post) ** 2
    )

    merged["sigma_vel_pow3"] = sigma_vel_pow3
    merged["sigma_charge_over_mass"] = sigma_charge_over_mass

    # Attach metadata for diagnostics
    merged["dataset"] = meta["label"]
    merged["calibration"] = meta["key"]

    return merged


def print_dataset_summary(df: pd.DataFrame, label: str) -> None:
    """Emit quick stats for sanity checking."""
    q_min, q_max = df["charge_c"].min(), df["charge_c"].max()
    m_min, m_max = df["mass_g"].min(), df["mass_g"].max()
    v_min, v_max = df["vel_kms"].min(), df["vel_kms"].max()
    print(f"{label:>28s} | N={len(df):4d} | "
          f"charge: [{q_min:.2e}, {q_max:.2e}] C | "
          f"mass: [{m_min:.2e}, {m_max:.2e}] g | "
          f"vel: [{v_min:.2f}, {v_max:.2f}] km/s")


def theoretical_curves(min_mass_g: float, vel_kms: np.ndarray):
    """Return list of theoretical charge/mass curves with palette keys."""
    return [
        ("Simpson 28 µm", "30um", pvdf_stopping(min_mass_g, vel_kms)),
        ("Simpson 8 µm", "8um", pvdf_stopping_8(min_mass_g, vel_kms)),
        ("Simpson 3 µm", "3um", pvdf_stopping_3(min_mass_g, vel_kms)),
    ]


def fit_with_outlier_rejection(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Regress y = C * x through the origin, dropping large residual outliers.

    Uses a ratio-based MAD cut (on y / x) as the first guard, then reinforces
    with residual and relative-error screens so leverage-heavy points cannot
    dominate the fit. Returns the fitted coefficient and a boolean mask for
    inliers.
    """
    if x.size == 0 or y.size == 0:
        return float("nan"), np.zeros_like(x, dtype=bool)

    denom = np.dot(x, x)
    if denom <= 0:
        return float("nan"), np.zeros_like(x, dtype=bool)

    ratio = np.divide(y, x, out=np.zeros_like(y), where=np.abs(x) > 1e-30)
    finite_ratio = np.isfinite(ratio)
    ratio_mask = np.ones_like(x, dtype=bool)
    if np.any(finite_ratio):
        ratio_valid = ratio[finite_ratio]
        ratio_median = np.median(ratio_valid)
        ratio_mad = np.median(np.abs(ratio_valid - ratio_median))
        if ratio_mad > 0:
            ratio_mask = np.abs(ratio - ratio_median) <= 3.5 * ratio_mad
        else:
            tolerance = max(0.1 * abs(ratio_median), 1e-3)
            ratio_mask = np.abs(ratio - ratio_median) <= tolerance

    required = max(3, int(0.5 * len(x)))
    if int(np.count_nonzero(ratio_mask)) < required:
        ratio_mask = np.ones_like(x, dtype=bool)

    x_base = x[ratio_mask]
    y_base = y[ratio_mask]
    denom_base = np.dot(x_base, x_base)
    if denom_base <= 0:
        return float("nan"), np.zeros_like(x, dtype=bool)

    c_initial = float(np.dot(x_base, y_base) / denom_base)
    residuals = y - c_initial * x
    mask = ratio_mask.copy()

    residuals_base = residuals[mask]
    if residuals_base.size:
        median_res = np.median(residuals_base)
        centered = residuals - median_res
        mad = np.median(np.abs(residuals_base - median_res))
        if mad > 0:
            mask &= np.abs(centered) <= 3.5 * mad

    # Supplemental guard: suppress points whose residuals are large relative
    # to the predicted charge so high-leverage shots do not bias the fit.
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = c_initial * x
        rel_error = np.abs(residuals) / np.maximum(np.abs(predicted), 1e-30)

    finite_rel = np.isfinite(rel_error[mask])
    if np.any(finite_rel):
        rel_subset = rel_error[mask][finite_rel]
        rel_median = np.median(rel_subset)
        rel_mad = np.median(np.abs(rel_subset - rel_median))
        if rel_mad > 0:
            rel_threshold = rel_median + 3.0 * rel_mad
            mask &= rel_error <= rel_threshold
        else:
            mask &= rel_error <= (rel_median + 0.5)

    inlier_count = int(np.count_nonzero(mask))
    if inlier_count < required:
        mask = ratio_mask
        inlier_count = int(np.count_nonzero(mask))

    if inlier_count < required:
        mask = np.ones_like(x, dtype=bool)
        inlier_count = len(x)

    x_refined = x[mask]
    y_refined = y[mask]
    denom_refined = np.dot(x_refined, x_refined)
    if denom_refined <= 0:
        return c_initial, mask

    c_refined = float(np.dot(x_refined, y_refined) / denom_refined)
    return c_refined, mask


def fit_global_powerlaw(records: List[Dict]) -> Tuple[float, float, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Fit a shared power law Q = C_i * m^alpha * v^beta across datasets.

    Returns best-fit alpha, beta (shared) and a mapping of dataset key -> C_i,
    with alpha constrained to [0.5, 1.5] and beta constrained to [1.0, 4.0].
    """
    if not records:
        raise ValueError("No records available for global power-law fit.")

    unique_keys: List[str] = []
    key_to_index: Dict[str, int] = {}
    dataset_data: Dict[str, Dict[str, np.ndarray]] = {}
    log_mass_list: List[np.ndarray] = []
    log_vel_list: List[np.ndarray] = []
    log_charge_list: List[np.ndarray] = []
    data_index_list: List[np.ndarray] = []

    for rec in records:
        meta = rec["meta"]
        key = meta["key"]
        df = rec["df"]
        mask = rec.get("mask")
        if isinstance(mask, np.ndarray) and mask.size == len(df):
            df_fit = df.loc[mask]
        else:
            df_fit = df

        df_fit = df_fit[
            (df_fit["charge_c"] > 0) &
            (df_fit["mass_g"] > 0) &
            (df_fit["vel_kms"] > 0)
        ].copy()

        if df_fit.empty:
            continue

        if key not in key_to_index:
            key_to_index[key] = len(unique_keys)
            unique_keys.append(key)

        idx = key_to_index[key]
        mass_vals = df_fit["mass_g"].to_numpy(dtype=float)
        vel_vals = df_fit["vel_kms"].to_numpy(dtype=float)
        charge_vals = df_fit["charge_c"].to_numpy(dtype=float)

        dataset_data[key] = {
            "mass": mass_vals,
            "vel": vel_vals,
            "charge": charge_vals,
        }

        log_mass = np.log(mass_vals)
        log_vel = np.log(vel_vals)
        log_charge = np.log(charge_vals)

        log_mass_list.append(log_mass)
        log_vel_list.append(log_vel)
        log_charge_list.append(log_charge)
        data_index_list.append(np.full(log_mass.shape, idx, dtype=int))

    if not unique_keys:
        raise ValueError("No valid data points for global power-law fit.")

    log_mass_vec = np.concatenate(log_mass_list)
    log_vel_vec = np.concatenate(log_vel_list)
    log_charge_vec = np.concatenate(log_charge_list)
    data_idx_vec = np.concatenate(data_index_list)

    alpha_init = float(np.clip(1.3, 0.5, 1.5))
    beta_init = float(np.clip(3.0, 1.0, 4.0))
    log_c_init: List[float] = []

    for key in unique_keys:
        data = dataset_data[key]
        log_c = np.mean(
            np.log(data["charge"]) -
            alpha_init * np.log(data["mass"]) -
            beta_init * np.log(data["vel"])
        )
        log_c_init.append(float(log_c))

    initial_params = np.array([alpha_init, beta_init, *log_c_init], dtype=float)
    lower_bounds = np.array([0.5, 1.0, *(-np.inf * np.ones(len(unique_keys)))])
    upper_bounds = np.array([1.5, 4.0, *(np.inf * np.ones(len(unique_keys)))])

    def residuals(params: np.ndarray) -> np.ndarray:
        alpha = params[0]
        beta = params[1]
        log_c = params[2:]
        prediction = log_c[data_idx_vec] + alpha * log_mass_vec + beta * log_vel_vec
        return log_charge_vec - prediction

    result = optimize.least_squares(
        residuals,
        initial_params,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=5000,
    )

    if not result.success:
        raise RuntimeError(
            f"Global power-law optimization failed: {result.message}"
        )

    alpha_opt = float(result.x[0])
    beta_opt = float(result.x[1])
    log_c_opt = result.x[2:]

    c_map = {
        key: float(np.exp(log_c_opt[key_to_index[key]]))
        for key in unique_keys
    }

    grid_data = {
        "log_mass_vec": log_mass_vec,
        "log_vel_vec": log_vel_vec,
        "log_charge_vec": log_charge_vec,
        "data_idx_vec": data_idx_vec,
        "unique_keys": unique_keys,
        "key_to_index": key_to_index,
    }

    return alpha_opt, beta_opt, c_map, grid_data


def plot_chi2_map(grid_data: Dict[str, np.ndarray], alpha_best: float, beta_best: float) -> None:
    """Render χ² landscape over α/β bounds."""
    log_mass_vec = grid_data["log_mass_vec"]
    log_vel_vec = grid_data["log_vel_vec"]
    log_charge_vec = grid_data["log_charge_vec"]
    data_idx_vec = grid_data["data_idx_vec"]
    unique_keys = grid_data["unique_keys"]

    alpha_bounds = (0.5, 1.5)
    beta_bounds = (1.0, 4.0)
    alpha_grid = np.linspace(alpha_bounds[0], alpha_bounds[1], 150)
    beta_grid = np.linspace(beta_bounds[0], beta_bounds[1], 150)
    chi2_vals = np.empty((beta_grid.size, alpha_grid.size))

    idx_masks = [data_idx_vec == i for i in range(len(unique_keys))]

    for ib, beta in enumerate(beta_grid):
        beta_term = beta * log_vel_vec
        for ia, alpha in enumerate(alpha_grid):
            offset = log_charge_vec - alpha * log_mass_vec - beta_term
            log_c = np.zeros(len(unique_keys))
            for i, mask in enumerate(idx_masks):
                if np.any(mask):
                    log_c[i] = float(np.mean(offset[mask]))
            residuals = offset - log_c[data_idx_vec]
            chi2_vals[ib, ia] = float(np.sum(residuals**2))

    chi2_vals = np.maximum(chi2_vals, 1e-12)
    chi2_min = float(np.min(chi2_vals))
    chi2_max = float(np.max(chi2_vals))
    has_variation = chi2_max > chi2_min
    if has_variation:
        chi2_norm = (chi2_vals - chi2_min) / (chi2_max - chi2_min)
    else:
        chi2_norm = np.zeros_like(chi2_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    hm = ax.pcolormesh(alpha_grid, beta_grid, chi2_norm, shading="auto", cmap="viridis", norm=norm)
    cs = None
    if has_variation:
        cs_levels = np.linspace(0.0, 1.0, 12) ** 2
        cs = ax.contour(alpha_grid, beta_grid, chi2_norm, levels=cs_levels, colors="white", linewidths=0.8)
        ax.clabel(cs, inline=True, fmt="%.2f", fontsize=8)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"$\chi^2$ landscape for shared power-law fit")
    cbar = fig.colorbar(hm, ax=ax)
    cbar.set_ticks([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.00", "0.10", "0.25", "0.50", "0.75", "1.00"])
    cbar.set_label(r"Normalized $\chi^2$")

    ax.scatter([alpha_best], [beta_best], marker="x", s=120, color="white", linewidths=2, label="Best fit")
    ax.scatter([1.3], [3.0], marker="x", s=120, color="black", linewidths=2, label=r"Legacy $(\alpha=1.3,\ \beta=3.0)$")
    ax.set_xlim(alpha_bounds)
    ax.set_ylim(beta_bounds)
    ax.legend(loc="upper right", frameon=True, facecolor='none')


def plot_global_powerlaw(records: List[Dict], alpha: float, beta: float, c_map: Dict[str, float]) -> None:
    """Create a plot in the transformed domain using shared alpha/beta."""
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlabel(rf"Velocity$^{{{beta:.2f}}}$ [km/s]$^{{{beta:.2f}}}$")
    ax.set_ylabel(rf"Charge / Mass$^{{{alpha:.2f}}}$ [C / g$^{{{alpha:.2f}}}$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    scatter_records = []
    for rec in records:
        df = rec["df"]
        if df.empty:
            continue

        meta = rec["meta"]
        palette = COLOR_PALETTE[meta["key"]]

        vel_kms = df["vel_kms"].to_numpy(dtype=float)
        sigma_vel = df["sigma_vel_kms"].to_numpy(dtype=float)
        mass_g = df["mass_g"].to_numpy(dtype=float)
        sigma_mass = df["sigma_mass_g"].to_numpy(dtype=float)
        charge_c = df["charge_c"].to_numpy(dtype=float)
        sigma_charge = df["sigma_charge_c"].to_numpy(dtype=float)
        events_all = df["event"].to_numpy()

        vel_pow_raw = np.power(vel_kms, beta)
        mass_pow = np.power(mass_g, alpha)
        charge_norm_raw = charge_c / mass_pow

        valid = (
            (vel_pow_raw > 0)
            & np.isfinite(vel_pow_raw)
            & (charge_norm_raw > 0)
            & np.isfinite(charge_norm_raw)
            & np.isfinite(vel_kms)
            & np.isfinite(mass_g)
            & np.isfinite(charge_c)
        )
        valid &= np.isfinite(sigma_vel) & np.isfinite(sigma_mass) & np.isfinite(sigma_charge)

        if not np.any(valid):
            continue

        vel_pow = vel_pow_raw[valid]
        charge_norm = charge_norm_raw[valid]
        events = events_all[valid]

        vel_kms_valid = vel_kms[valid]
        sigma_vel_valid = sigma_vel[valid]
        mass_g_valid = mass_g[valid]
        sigma_mass_valid = sigma_mass[valid]
        charge_c_valid = charge_c[valid]
        sigma_charge_valid = sigma_charge[valid]

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_charge = np.divide(
                sigma_charge_valid,
                charge_c_valid,
                out=np.zeros_like(sigma_charge_valid),
                where=charge_c_valid != 0.0,
            )
            rel_mass = np.divide(
                sigma_mass_valid,
                mass_g_valid,
                out=np.zeros_like(sigma_mass_valid),
                where=mass_g_valid != 0.0,
            )

        sigma_vel_pow = np.abs(beta * np.power(vel_kms_valid, beta - 1.0) * sigma_vel_valid)
        sigma_charge_norm = np.abs(charge_norm) * np.sqrt(rel_charge**2 + (alpha * rel_mass) ** 2)

        scatter = ax.scatter(
            vel_pow,
            charge_norm,
            label=meta["label"],
            s=30,
            alpha=0.9,
            edgecolor="none",
            color=palette["scatter"],
        )

        ax.errorbar(
            vel_pow,
            charge_norm,
            xerr=sigma_vel_pow,
            yerr=sigma_charge_norm,
            fmt="none",
            ecolor=palette["scatter"],
            elinewidth=0.8,
            alpha=0.5,
        )

        scatter_records.append({
            "collection": scatter,
            "events": events,
            "key": meta["key"],
        })

        all_x.append(vel_pow)
        all_y.append(charge_norm)

    if not all_x:
        ax.set_title("Global Power Law Fit – no valid data")
        fig.tight_layout()
        return

    all_x_vals = np.concatenate(all_x)
    all_y_vals = np.concatenate(all_y)

    x_min = float(all_x_vals.min())
    x_max = float(all_x_vals.max())
    y_min = float(all_y_vals.min())
    y_max = float(all_y_vals.max())

    if x_min <= 0 or y_min <= 0:
        raise ValueError("Encountered non-positive values in transformed data.")

    axis_x_min, axis_x_max = pad_bounds(x_min, x_max)
    axis_y_min, axis_y_max = pad_bounds(y_min, y_max)

    ax.set_xlim(axis_x_min, axis_x_max)
    ax.set_ylim(axis_y_min, axis_y_max)

    x_fit = np.geomspace(axis_x_min, axis_x_max, 200)

    for rec in records:
        meta = rec["meta"]
        key = meta["key"]
        palette = COLOR_PALETTE[key]
        c_val = c_map.get(key)
        if c_val is None or c_val <= 0:
            continue

        y_fit = c_val * x_fit
        ax.plot(
            x_fit,
            y_fit,
            color=palette["fit"],
            linestyle="-.",
            linewidth=1.2,
            label=f"{meta['label']} fit (C={c_val:.2e})",
        )

    ax.set_title(
        f"Shared Power Law Fit – α={alpha:.3f}, β={beta:.3f}"
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor='none')
    fig.tight_layout()

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="gray"),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )
    annot.set_visible(False)

    offsets_cache = {info["collection"]: info["collection"].get_offsets() for info in scatter_records}

    def update_annotation(scatter_info, index: int) -> None:
        offsets = offsets_cache[scatter_info["collection"]]
        xy = offsets[index]
        annot.xy = xy
        event_num = scatter_info["events"][index]
        annot.set_text(f"Event {event_num}")
        palette_key = scatter_info["key"]
        annot.get_bbox_patch().set_facecolor(COLOR_PALETTE[palette_key]["theory"])
        annot.get_bbox_patch().set_edgecolor(COLOR_PALETTE[palette_key]["fit"])

    def on_motion(event) -> None:
        visible = annot.get_visible()
        if event.inaxes == ax:
            for scatter_info in scatter_records:
                contains, indices = scatter_info["collection"].contains(event)
                if contains and indices["ind"]:
                    update_annotation(scatter_info, indices["ind"][0])
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if visible:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    ax.set_xlim(axis_x_min, axis_x_max)
    ax.set_ylim(axis_y_min, axis_y_max)
    fig.tight_layout()


def main():
    plt.rcParams.update({
        "axes.labelsize": 18,
        "axes.titlesize": 22,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    })

    records = []
    for meta in DATASETS:
        df = load_dataset(meta)
        print_dataset_summary(df, meta["label"])
        if df.empty:
            continue

        # Fit Q = C * m^alpha * v^beta with alpha=1.3, beta=3.0 by regressing
        # Y = Q / m^alpha against X = v^beta through the origin.
        x = df["vel_pow3"].to_numpy()
        y = df["charge_over_mass"].to_numpy()
        c_fit, inliers = fit_with_outlier_rejection(x, y)
        if not math.isnan(c_fit):
            used = int(np.count_nonzero(inliers))
            total = len(df)
            if used != total:
                print(f"    Using {used}/{total} points after outlier rejection.")
                excluded_events = df.loc[~inliers, "event"].to_list()
                if excluded_events:
                    print(f"    Excluded events: {excluded_events}")
            print(f"    Fitted C (C·g^-1.3·(km/s)^-3) = {c_fit:.3e}")

        records.append({
            "meta": meta,
            "df": df,
            "fit": c_fit,
            "mask": inliers,
        })

    if not records:
        raise RuntimeError("No calibration data was loaded; check HDF paths.")

    records.sort(key=lambda rec: PLOT_ORDER.get(rec["meta"]["key"], 99))

    alpha_global, beta_global, c_map, grid_data = fit_global_powerlaw(records)
    print("\nGlobal power-law fit with shared α, β:")
    print(f"    α = {alpha_global:.4f}")
    print(f"    β = {beta_global:.4f}")
    for rec in records:
        key = rec["meta"]["key"]
        label = rec["meta"]["label"]
        c_val = c_map.get(key)
        if c_val is not None:
            print(f"    {label}: C = {c_val:.3e}")
        else:
            print(f"    {label}: insufficient positive data for C fit")

    # Plot setup
    plt.style.use("classic")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlabel(r"Velocity$^{3.0}$ [km/s]$^{3}$")
    ax.set_ylabel(r"Charge / Mass$^{1.3}$ [C / g$^{1.3}$]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    # Determine min mass for theoretical curves (use lightest grain across datasets)
    min_mass_g = min(rec["df"]["mass_g"].min() for rec in records)
    vel_grid = np.linspace(1.0, 20.0, 400)  # km/s
    vel_grid_pow3 = np.power(vel_grid, 3.0)

    for name, palette_key, q_electrons in theoretical_curves(min_mass_g, vel_grid):
        q_coul = q_electrons * E_CHARGE
        y_vals = q_coul / np.power(min_mass_g, 1.3)
        ax.plot(
            vel_grid_pow3,
            y_vals,
            label=name,
            linestyle="--",
            linewidth=1.5,
            color=COLOR_PALETTE[palette_key]["theory"],
        )

    vel_arrays: List[np.ndarray] = []
    charge_arrays: List[np.ndarray] = []
    scatter_records = []
    fit_records: List[Tuple[str, str, float, np.ndarray]] = []

    for rec in records:
        df = rec["df"]
        meta = rec["meta"]
        palette = COLOR_PALETTE[meta["key"]]
        scatter = ax.scatter(
            df["vel_pow3"],
            df["charge_over_mass"],
            label=meta["label"],
            s=30,
            alpha=0.9,
            edgecolor="none",
            color=palette["scatter"],
        )

        scatter_records.append({
            "collection": scatter,
            "events": df["event"].to_numpy(),
            "key": meta["key"],
        })

        vel_arrays.append(df["vel_pow3"].to_numpy(dtype=float))
        charge_arrays.append(df["charge_over_mass"].to_numpy(dtype=float))

        ax.errorbar(
            df["vel_pow3"],
            df["charge_over_mass"],
            xerr=df["sigma_vel_pow3"],
            yerr=df["sigma_charge_over_mass"],
            fmt="none",
            ecolor=palette["scatter"],
            elinewidth=0.8,
            alpha=0.5,
        )

        c_fit = rec["fit"]
        if not math.isnan(c_fit):
            fit_mask = rec.get("mask")
            if isinstance(fit_mask, np.ndarray) and fit_mask.size == len(df) and np.any(fit_mask):
                df_fit_domain = df.loc[fit_mask]
            else:
                df_fit_domain = df

            if not df_fit_domain.empty:
                fit_records.append((meta["label"], meta["key"], c_fit, df_fit_domain["vel_pow3"].to_numpy(dtype=float)))

    if vel_arrays and charge_arrays:
        x_raw_min = min(arr.min() for arr in vel_arrays)
        x_raw_max = max(arr.max() for arr in vel_arrays)
        y_raw_min = min(arr.min() for arr in charge_arrays)
        y_raw_max = max(arr.max() for arr in charge_arrays)
        axis_x_min, axis_x_max = pad_bounds(x_raw_min, x_raw_max)
        axis_y_min, axis_y_max = pad_bounds(y_raw_min, y_raw_max)
    else:
        axis_x_min, axis_x_max = 1e-1, 1e4
        axis_y_min, axis_y_max = 1e-2, 1e5

    for label, key, c_fit, _domain in fit_records:
        palette = COLOR_PALETTE[key]
        x_fit = np.geomspace(axis_x_min, axis_x_max, 200)
        y_fit = c_fit * x_fit
        ax.plot(
            x_fit,
            y_fit,
            color=palette["fit"],
            linestyle="-.",
            linewidth=1.2,
            label=f"{label} fit (C={c_fit:.2e})",
        )

    ax.set_xlim(axis_x_min, axis_x_max)
    ax.set_ylim(axis_y_min, axis_y_max)

    ax.set_title("PVDF Stopping Charge Yield – February 2024 calibrations")
    ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor='none')
    fig.tight_layout()

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="gray"),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )
    annot.set_visible(False)

    offsets_cache = {info["collection"]: info["collection"].get_offsets() for info in scatter_records}

    def update_annotation(scatter_info, index: int) -> None:
        offsets = offsets_cache[scatter_info["collection"]]
        xy = offsets[index]
        annot.xy = xy
        event_num = scatter_info["events"][index]
        annot.set_text(f"Event {event_num}")
        palette_key = scatter_info["key"]
        annot.get_bbox_patch().set_facecolor(COLOR_PALETTE[palette_key]["theory"])
        annot.get_bbox_patch().set_edgecolor(COLOR_PALETTE[palette_key]["fit"])

    def on_motion(event) -> None:
        visible = annot.get_visible()
        if event.inaxes == ax:
            for scatter_info in scatter_records:
                contains, indices = scatter_info["collection"].contains(event)
                if contains and indices["ind"]:
                    update_annotation(scatter_info, indices["ind"][0])
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if visible:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    plot_global_powerlaw(records, alpha_global, beta_global, c_map)
    plot_chi2_map(grid_data, alpha_global, beta_global)
    plt.show()


if __name__ == "__main__":
    main()
