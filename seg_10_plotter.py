# SPDX-License-Identifier: MIT

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fit_data import best_fit_rows
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_ELEMENTARY_CHARGE = 1.602176634e-19  # Coulombs per electron
_E_PER_C = 1.0 / _ELEMENTARY_CHARGE   # electrons per Coulomb
_CSA_LINEAR_CAL_30UM = (-8.29338526e-13, -1.24407122e-14)  # Q [C] = m*V + b


def _dataset_leaf_name(value) -> str:
    if value is None:
        return ""
    s = str(value).strip().rstrip("/\\")
    leaf = os.path.basename(s)
    return leaf or s


def _read_one_h5(path: str) -> pd.DataFrame:
    try:
        df = pd.read_hdf(path, key="fits")
        df["source_file"] = path
        return df
    except Exception:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
            if not keys:
                raise RuntimeError(f"No datasets in HDF5: {path}")
            df = store[keys[0]]
        df["source_file"] = path
        return df


def _add_qd_cal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "mass" not in df.columns or "v" not in df.columns:
        return df
    m_kg = pd.to_numeric(df["mass"], errors="coerce")
    v_ms = pd.to_numeric(df["v"], errors="coerce")
    m_g = m_kg * 1e3
    v_kms = v_ms / 1e3
    #cal = np.where(
        #np.isfinite(m_g) & np.isfinite(v_kms),
        #np.power(m_g, 1.3) * np.power(v_kms, 3.0),
        #np.nan,
    #)
    cal = np.where(
        np.isfinite(m_g) & np.isfinite(v_kms),
        np.power(m_g,0.28) * np.power(v_kms, 1.1),
        np.nan,
    )
    out = df.copy()
    out["cal"] = cal
    return out


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot QD3 cal vs PVDF q (Cremat CSA 30um) for seg10 data.")
    parser.add_argument(
        "--h5",
        default="../hdf5_data/2026_01_09_LMM_28um_sensor_run10_seg10.h5",
        help="Path to the HDF5 fits file.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output image path (e.g. seg10_cal_vs_pvdf_q.png).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not os.path.exists(args.h5):
        raise FileNotFoundError(f"Missing HDF5 file: {args.h5}")

    df = _read_one_h5(args.h5)
    for col in ("dataset_id", "event", "channel", "fit_type"):
        if col not in df.columns:
            df[col] = np.nan

    qd = df[(df["fit_type"] == "QD3Fit") & (df["channel"] == 2)].copy()
    qdm = df[df["fit_type"] == "QDMFit"].copy()
    low = df[df["fit_type"] == "low_pass_max"].copy()

    qd["event"] = qd["event"].astype(str).str.strip()
    qdm["event"] = qdm["event"].astype(str).str.strip()
    low["event"] = low["event"].astype(str).str.strip()

    qd = best_fit_rows(qd, ["dataset_id", "event"])
    qd = _add_qd_cal(qd)
    if "cal" not in qd.columns:
        qd["cal"] = np.nan
    if "q" not in qd.columns:
        qd["q"] = np.nan
    qd["q"] = pd.to_numeric(qd["q"], errors="coerce")

    qdm = best_fit_rows(qdm, ["dataset_id", "event"])
    qdm = _add_qd_cal(qdm)
    if "cal" not in qdm.columns:
        qdm["cal"] = np.nan

    low["value"] = pd.to_numeric(low["value"], errors="coerce").abs()
    noise_half_pp = 0.02  # 40 mV peak-to-peak -> +/-20 mV per measurement
    low["snr"] = low["value"] / noise_half_pp
    low = low[low["snr"] >= 2.0].copy()
    slope, intercept = _CSA_LINEAR_CAL_30UM
    slope_pos = -slope
    intercept_pos = -intercept
    low["pvdf_q"] = (slope_pos * low["value"].values + intercept_pos) * _E_PER_C
    pvdf_q_err = slope_pos * noise_half_pp * _E_PER_C

    merged = pd.merge(
        low[["dataset_id", "event", "pvdf_q"]],
        qd[["dataset_id", "event", "cal", "q", "mass", "v"]].rename(
            columns={"cal": "cal_qd", "q": "q_qd", "mass": "mass_qd", "v": "v_qd"}
        ),
        on=["dataset_id", "event"],
        how="left",
    )
    merged = pd.merge(
        merged,
        qdm[["dataset_id", "event", "cal", "mass", "v"]].rename(
            columns={"cal": "cal_qdm", "mass": "mass_qdm", "v": "v_qdm"}
        ),
        on=["dataset_id", "event"],
        how="left",
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    use_qd = merged["cal_qd"].notna() & merged["q_qd"].notna() & (merged["q_qd"] >= -2.5)
    merged["cal"] = np.where(use_qd, merged["cal_qd"], merged["cal_qdm"])
    merged["mass"] = np.where(use_qd, merged["mass_qd"], merged["mass_qdm"])
    merged["v"] = np.where(use_qd, merged["v_qd"], merged["v_qdm"])
    merged = merged.dropna(subset=["cal", "pvdf_q"])
    if merged.empty:
        raise RuntimeError("No overlapping QD3Fit/low_pass_max events found after filtering.")

    merged_plot = merged
    datasets = sorted(merged_plot["dataset_id"].dropna().unique().tolist())
    n_colors = max(10, len(datasets))
    cmap = plt.get_cmap("tab20" if n_colors > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for i, ds in enumerate(datasets):
        sub = merged_plot[merged_plot["dataset_id"] == ds]
        if sub.empty:
            continue
        x = sub["cal"].to_numpy(dtype=float)
        y = sub["pvdf_q"].to_numpy(dtype=float)
        color = cmap(i % cmap.N)
        label = _dataset_leaf_name(ds)

        n_points = len(x)
        if n_points >= 2:
            denom = np.dot(x, x)
            slope = np.dot(x, y) / denom if denom != 0 else np.nan
            xfit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            yfit = slope * xfit
            ax.plot(xfit, yfit, color=color, linewidth=1.5, alpha=0.85)
            label = f"{label} (m={slope:.3g}, n={n_points})"
        else:
            label = f"{label} (m=N/A, n={n_points})"

        yerr = np.full_like(y, pvdf_q_err, dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=4.5,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.8,
            label=label,
        )

    #ax.set_xlabel("cal (g^1.0 * (km/s)^2)")
    ax.set_xlabel("cal (g^0.28 * (km/s)^1.1)")
    ax.set_ylabel("pvdf_q (electrons)")
    ax.set_title("QD3 cal vs PVDF q (Cremat CSA 30um) by dataset")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=8, loc="best", frameon=True)

    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=200)

    run_targets = {
        "2026_01_09_LMM_28um_sensor_run1",
        "2026_01_09_LMM_28um_sensor_run10",
    }
    run_mask = merged["dataset_id"].map(_dataset_leaf_name).isin(run_targets)
    surf = merged[run_mask].dropna(subset=["mass", "v", "pvdf_q"])
    surf = surf[(surf["mass"] > 0) & (surf["v"] > 0) & (surf["pvdf_q"] > 0)]
    if not surf.empty:
        fig2 = plt.figure(figsize=(8.5, 6.5))
        ax2 = fig2.add_subplot(111, projection="3d")
        mass_log = np.log10(surf["mass"].to_numpy(float))
        v_log = np.log10(surf["v"].to_numpy(float))
        pvdf_q_log = np.log10(surf["pvdf_q"].to_numpy(float))
        A = np.column_stack([np.ones_like(mass_log), mass_log, v_log])
        coeffs, _, _, _ = np.linalg.lstsq(A, pvdf_q_log, rcond=None)
        logk, alpha, beta = coeffs
        k = 10 ** logk
        ax2.scatter(
            mass_log,
            v_log,
            pvdf_q_log,
            s=22,
            alpha=0.8,
        )
        grid_x = np.linspace(mass_log.min(), mass_log.max(), 30)
        grid_y = np.linspace(v_log.min(), v_log.max(), 30)
        gx, gy = np.meshgrid(grid_x, grid_y)
        gz = logk + alpha * gx + beta * gy
        ax2.plot_surface(gx, gy, gz, color="gray", alpha=0.25, linewidth=0)
        ax2.set_xlabel("log10(mass (kg))")
        ax2.set_ylabel("log10(v (m/s))")
        ax2.set_zlabel("log10(pvdf_q (electrons))")
        ax2.set_title(
            "Log-Log-Log Mass vs Velocity vs PVDF q (run1 + run10)\n"
            f"pvdf_q = {k:.3g} * m^{alpha:.3g} * v^{beta:.3g}"
        )
        fig2.tight_layout()
        if args.save:
            root, ext = os.path.splitext(args.save)
            fig2.savefig(f"{root}_mass_v_pvdf_q{ext or '.png'}", dpi=200)

    rate_h5 = "../hdf5_data/2026_01_12_seg5-10_rate_study.h5"
    csv_runs = [
        (
            "Run0 (seg10)",
            "../data/2026_01_12_run0_metadata.csv",
            "2026_01_12_run0_seg10",
        ),
        (
            "Run1 (seg5)",
            "../data/2026_01_12_run1_metadata.csv",
            "2026_01_12_run1_seg5",
        ),
    ]
    if os.path.exists(rate_h5) and all(os.path.exists(p) for _, p, _ in csv_runs):
        fits_rate = _read_one_h5(rate_h5)
        fits_rate["dataset_leaf"] = fits_rate["dataset_id"].map(_dataset_leaf_name)
        qd_rate = fits_rate[(fits_rate["fit_type"] == "QD3Fit") & (fits_rate["channel"] == 2)].copy()
        qd_rate["mass"] = pd.to_numeric(qd_rate["mass"], errors="coerce")
        qd_rate["v"] = pd.to_numeric(qd_rate["v"], errors="coerce")

        fig3, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=False, sharey=False)
        for ax, (title, csv_path, ds_leaf) in zip(axes, csv_runs):
            csv_df = pd.read_csv(csv_path)
            csv_mass = pd.to_numeric(csv_df.get("Mass (kg)"), errors="coerce")
            csv_vel = pd.to_numeric(csv_df.get("Velocity (m/s)"), errors="coerce")
            csv_mask = (csv_mass > 0) & (csv_vel > 0)
            ax.scatter(
                csv_mass[csv_mask],
                csv_vel[csv_mask],
                s=14,
                color="black",
                alpha=0.55,
                label="CSV",
            )

            run_qd = qd_rate[qd_rate["dataset_leaf"] == ds_leaf]
            run_qd = run_qd[(run_qd["mass"] > 0) & (run_qd["v"] > 0)]
            ax.scatter(
                run_qd["mass"].to_numpy(float),
                run_qd["v"].to_numpy(float),
                s=18,
                color="red",
                alpha=0.8,
                label="H5 QD3Fit",
            )
            ax.set_title(title)
            ax.set_xlabel("mass (kg)")
            ax.set_ylabel("velocity (m/s)")
            ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
            ax.legend(fontsize=8, loc="best", frameon=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
        fig3.tight_layout()
        if args.save:
            root, ext = os.path.splitext(args.save)
            fig3.savefig(f"{root}_run01_mass_v_compare{ext or '.png'}", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
