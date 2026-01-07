# SPDX-License-Identifier: MIT

import sys, os, re, traceback, inspect
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Optional calibration helpers — load if present; otherwise features that rely on them are disabled.
try:
    from fitparams import CSA_pulse  # provided by your project
except Exception:
    CSA_pulse = None

# --- Try to load PVDF q(v) calibration CSVs if available ---
_q_of_v_5 = None
_q_of_v_9 = None
_q_of_v_15 = None
_ELEMENTARY_CHARGE = 1.602176634e-19  # Coulombs per electron
_E_PER_C = 1.0 / _ELEMENTARY_CHARGE   # electrons per Coulomb
# Default fraction of the Gaussian peak (above baseline) used to define the integration window.
_GAUSS_INT_FRAC = 0.10
_CSA_LINEAR_CAL = {
    "Cremat CSA 3um": (-8.23012166e-13, 2.30821114e-16),   # Q [C] = m*V + b
    "Cremat CSA 8um": (1.02344806e-12, 5.45524781e-15),
    "Cremat CSA 30um": (-8.29338526e-13, -1.24407122e-14),
}
try:
    from scipy.interpolate import interp1d
    if os.path.exists('q2v_5nf.csv'):
        df_5 = pd.read_csv('q2v_5nf.csv')
        # q values were applied across 1 segment of a 10 segment detector; divide by 10 for CSA charge.
        df_5['q'] = df_5['q'] / 10.0
        _q_of_v_5 = interp1d(df_5['v'].values, df_5['q'].values, kind='linear', fill_value='extrapolate')
    if os.path.exists('q2v_9-5nf.csv'):
        df_9 = pd.read_csv('q2v_9-5nf.csv')
        # q values were applied across 1 segment of a 10 segment detector; divide by 10 for CSA charge.
        df_9['q'] = df_9['q'] / 10.0
        _q_of_v_9 = interp1d(df_9['v'].values, df_9['q'].values, kind='linear', fill_value='extrapolate')
    if os.path.exists('q2v_15nf.csv'):
        df_15 = pd.read_csv('q2v_15nf.csv')
        # q values were applied across 1 segment of a 10 segment detector; divide by 10 for CSA charge.
        df_15['q'] = df_15['q'] / 10.0
        _q_of_v_15 = interp1d(df_15['v'].values, df_15['q'].values, kind='linear', fill_value='extrapolate')
except Exception:
    pass


def _sanitize(name: str) -> str:
    """Make a safe, compact suffix from a fit_type like 'CSA_pulse' -> 'csa_pulse'."""
    return re.sub(r'[^0-9a-zA-Z]+', '_', str(name)).strip('_').lower()


def _dataset_leaf_name(value) -> str:
    """Return the last path component of a dataset identifier for concise labeling."""
    if value is None:
        return ''
    s = str(value).strip()
    if not s:
        return s
    s = s.rstrip('/\\')
    leaf = os.path.basename(s)
    return leaf or s


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _erf(z):
    """Vectorized error function compatible with/without numpy.special.erf at top-level."""
    try:
        return np.erf(z)  # available in many NumPy builds
    except AttributeError:
        from math import erf as _merf
        return np.vectorize(_merf)(z)


def skew_gaussian_value(x, A, xi, omega, alpha, C):
    """Evaluate the provided skew-Gaussian form at x (vectorized)."""
    t = (x - xi) / omega
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * t**2)
    Phi = 0.5 * (1.0 + _erf(alpha * t / np.sqrt(2.0)))
    return A * 2.0 / omega * phi * Phi + C


# --- Helpers for skew_gaussian peak value / mode ---
def _phi(z):
    """Standard normal PDF, vectorized."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * np.asarray(z) ** 2)


def _Phi(z):
    """Standard normal CDF, vectorized."""
    return 0.5 * (1.0 + _erf(np.asarray(z) / np.sqrt(2.0)))


def skew_gaussian_mode_x(xi: float, omega: float, alpha: float) -> float:
    """Approximate the mode (x at peak) of the Azzalini skew-normal (location xi, scale omega, shape alpha).
    Uses a dense grid search around xi ± 8*omega. Robust and dependency-free.
    """
    if not (np.isfinite(xi) and np.isfinite(omega) and np.isfinite(alpha)):
        return np.nan
    if omega <= 0:
        return np.nan
    # Evaluate the core shape term (without A, C, 2/omega scaling) on t-grid
    tgrid = np.linspace(-8.0, 8.0, 4001)
    vals = _phi(tgrid) * _Phi(alpha * tgrid)
    imax = int(np.nanargmax(vals))
    t_pk = tgrid[imax]
    return float(xi + omega * t_pk)


def skew_gaussian_peak_value(A: float, xi: float, omega: float, alpha: float, C: float) -> Tuple[float, float]:
    """Return (peak_value, x_at_peak) for the skew_gaussian curve.
    peak_value is the peak ABOVE the baseline C, i.e., (peak - C) = A*(2/omega)*phi(t)*Phi(alpha*t)
    at t corresponding to the mode. This matches the requested definition as a difference from C.
    """
    xpk = skew_gaussian_mode_x(xi, omega, alpha)
    if not np.isfinite(xpk):
        return np.nan, np.nan
    t = (xpk - xi) / omega
    val = A * (2.0 / omega) * _phi(t) * _Phi(alpha * t)
    try:
        return float(val), float(xpk)
    except Exception:
        return np.nan, np.nan


def gaussian_integral_within_frac(A: float, mu: float, sigma: float, C: float = 0.0, frac: float = _GAUSS_INT_FRAC) -> float:
    """Analytic area of the fitted Gaussian above the baseline C between the ± points where (y-C) = frac*(peak-C).

    Peak above baseline is A at x = μ; the ± bounds are μ ± σ*sqrt(-2 ln(frac)).
    The baseline-removed integral on that window is A * σ * sqrt(2π) * erf(w/(σ√2))
    with w = σ*sqrt(-2 ln(frac)). Sign follows the fit amplitude so inverted peaks
    stay negative. Note: integral value is translation-invariant, so μ only sets
    the window location, not the magnitude.
    """
    try:
        if not all(np.isfinite(v) for v in (A, sigma)):
            return np.nan
        if sigma <= 0 or not (0 < frac < 1):
            return np.nan
        tmax = np.sqrt(-2.0 * np.log(frac))  # dimensionless half-width factor
        area = A * sigma * np.sqrt(2.0 * np.pi) * _erf(tmax / np.sqrt(2.0))
        return float(area)
    except Exception:
        return np.nan


class FitResultsVisualizer(QMainWindow):
    """
    Visualizer for HDF5 produced by oscope_palantir.py's Export Fits.

    NEW FLOW (per your request):
    • On load, build one DataFrame for EACH unique (dataset_id, fit_type, channel).
    • Each frame contains all events for that trio (deduped to 1 row per event).
    • UI lets you choose an X-frame and a Y-frame, then a parameter from each.
    • We plot only events present in BOTH frames (inner join on event).
    • If duplicates exist per event in a frame, we pick the row with MOST non-NaNs
      (simple & reliable). You can later swap to a "closest t0" rule if desired.
    """

    # Emits (dataset_id, event) when user clicks a point (if selection enabled)
    eventPicked = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fit Results Visualizer (by group: dataset/fit/channel)")
        self.setGeometry(80, 80, 1300, 820)

        self.debug = True  # toggle verbose logging

        # Raw table & per-group frames
        self.raw = pd.DataFrame()   # long HDF table
        self.groups: Dict[str, pd.DataFrame] = {}  # label -> dataframe (event-deduped)
        self.group_meta: Dict[str, Tuple[str, str, int]] = {}  # label -> (dataset_id, fit_type, channel)
        self.group_params: Dict[str, List[str]] = {}  # label -> numeric parameter list

        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Controls row 1: loading & status ---
        row1 = QHBoxLayout(); layout.addLayout(row1)
        btn_load = QPushButton("Load HDF5 Files")
        btn_load.clicked.connect(self.load_files)
        row1.addWidget(btn_load)

        btn_diag = QPushButton("Diagnostics…")
        btn_diag.setToolTip("Show summary + previews of selected frames")
        btn_diag.clicked.connect(self.show_diagnostics)
        row1.addWidget(btn_diag)

        self.lbl_status = QLabel("No files loaded.")
        self.lbl_status.setFixedHeight(20)
        row1.addWidget(self.lbl_status)
        row1.addStretch()

        # --- Controls row 2: dataset filter + PVDF option (kept) ---
        row2 = QHBoxLayout(); layout.addLayout(row2)
        row2.addWidget(QLabel("Dataset filter:"))
        self.ds_combo = QComboBox(); self.ds_combo.currentIndexChanged.connect(self.on_dataset_changed)
        row2.addWidget(self.ds_combo)

        row2.addWidget(QLabel("PVDF q(v) mode:"))
        pvdf_modes = [
            'Off',
            '5nF segmented',
            '9.5nF monolithic',
            '15nF (q2v_15nf.csv)',
        ] + list(_CSA_LINEAR_CAL.keys())
        self.pvdf_combo = QComboBox(); self.pvdf_combo.addItems(pvdf_modes)
        self.pvdf_combo.currentIndexChanged.connect(self.rebuild_groups)  # recalculates derived cols per group
        row2.addWidget(self.pvdf_combo)
        row2.addStretch()

        # --- Controls row 3: X-frame / Y-frame selection + parameter selectors ---
        row3 = QHBoxLayout(); layout.addLayout(row3)

        row3.addWidget(QLabel("X frame:"))
        self.gx_combo = QComboBox(); self.gx_combo.currentIndexChanged.connect(self.on_group_changed)
        row3.addWidget(self.gx_combo)
        row3.addWidget(QLabel("X param:"))
        self.px_combo = QComboBox(); self.px_combo.currentIndexChanged.connect(self.update_plot)
        row3.addWidget(self.px_combo)

        row3.addSpacing(20)

        row3.addWidget(QLabel("Y frame:"))
        self.gy_combo = QComboBox(); self.gy_combo.currentIndexChanged.connect(self.on_group_changed)
        row3.addWidget(self.gy_combo)
        row3.addWidget(QLabel("Y param:"))
        self.py_combo = QComboBox(); self.py_combo.currentIndexChanged.connect(self.update_plot)
        row3.addWidget(self.py_combo)

        row3.addStretch()

        # --- Controls row 3b: overlay dataset (optional) ---
        row3b = QHBoxLayout(); layout.addLayout(row3b)
        self.chk_overlay = QCheckBox("Overlay second dataset")
        self.chk_overlay.stateChanged.connect(self.on_overlay_toggle)
        row3b.addWidget(self.chk_overlay)

        row3b.addSpacing(10)

        row3b.addWidget(QLabel("X frame:"))
        self.gx2_combo = QComboBox(); self.gx2_combo.currentIndexChanged.connect(self.on_overlay_group_changed)
        row3b.addWidget(self.gx2_combo)
        row3b.addWidget(QLabel("X param:"))
        self.px2_combo = QComboBox(); self.px2_combo.currentIndexChanged.connect(self.update_plot)
        row3b.addWidget(self.px2_combo)

        row3b.addSpacing(20)

        row3b.addWidget(QLabel("Y frame:"))
        self.gy2_combo = QComboBox(); self.gy2_combo.currentIndexChanged.connect(self.on_overlay_group_changed)
        row3b.addWidget(self.gy2_combo)
        row3b.addWidget(QLabel("Y param:"))
        self.py2_combo = QComboBox(); self.py2_combo.currentIndexChanged.connect(self.update_plot)
        row3b.addWidget(self.py2_combo)

        row3b.addStretch()

        # --- Controls row 4: scales, fit, export ---
        row4 = QHBoxLayout(); layout.addLayout(row4)
        self.chk_logx = QCheckBox("log X"); self.chk_logx.stateChanged.connect(self.update_plot); row4.addWidget(self.chk_logx)
        self.chk_logy = QCheckBox("log Y"); self.chk_logy.stateChanged.connect(self.update_plot); row4.addWidget(self.chk_logy)
        self.chk_dropna = QCheckBox("drop NaN"); self.chk_dropna.setChecked(False); self.chk_dropna.stateChanged.connect(self.update_plot); row4.addWidget(self.chk_dropna)

        btn_linfit = QPushButton("Linear Fit (through origin)")
        btn_linfit.clicked.connect(self.run_linear_fit)
        row4.addWidget(btn_linfit)

        # NEW: unconstrained linear fit (m, b)
        btn_linfit_free = QPushButton("Linear Fit (m, b)")
        btn_linfit_free.clicked.connect(self.run_linear_fit_free)
        row4.addWidget(btn_linfit_free)

        btn_save = QPushButton("Export merged CSV")
        btn_save.clicked.connect(self.save_csv)
        row4.addWidget(btn_save)
        # Selection/removal controls
        self.chk_select = QCheckBox("Select points")
        row4.addWidget(self.chk_select)
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setToolTip("Remove selected points from current plot and fits")
        self.btn_remove.clicked.connect(self.remove_selected_points)
        row4.addWidget(self.btn_remove)
        self.btn_clear_sel = QPushButton("Clear Selection")
        self.btn_clear_sel.clicked.connect(self.clear_selection)
        row4.addWidget(self.btn_clear_sel)
        self.btn_reset_rm = QPushButton("Reset Removals")
        self.btn_reset_rm.setToolTip("Restore all removed points")
        self.btn_reset_rm.clicked.connect(self.reset_removals)
        row4.addWidget(self.btn_reset_rm)
        row4.addStretch()

        # --- Plot area ---
        self.figure = Figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # cache last plotted merged df for linear fit & export
        self.last_plot_df: Optional[pd.DataFrame] = None
        self.last_overlay_df: Optional[pd.DataFrame] = None
        # interactive selection/removal state
        self._merged_base: Optional[pd.DataFrame] = None
        self._display_to_base = None
        self._current_display_X = None
        self._current_display_Y = None
        self.selected_indices: set = set()  # base indices
        self.removed_indices: set = set()   # base indices
        self._plot_sig: Optional[Tuple[str,str,str,str,str]] = None
        self._scatter = None
        self._overlay_scatter = None
        self._sel_scatter = None
        self._pick_cid = None

        # Overlay controls start disabled until requested
        self._set_overlay_controls_enabled(False)

        if self.debug:
            print("[INIT] Visualizer ready. CSA_pulse present:", CSA_pulse is not None)

    # ---------------- helpers -----------------
    def dprint(self, *a):
        if self.debug:
            try: print(*a)
            except Exception: pass

    # ---------------- I/O -----------------
    def _read_one_h5(self, path: str) -> pd.DataFrame:
        self.dprint(f"[_read_one_h5] Reading {path}")
        try:
            df = pd.read_hdf(path, 'fits')
            self.dprint(f"[_read_one_h5] Loaded 'fits' shape={df.shape}; cols={list(df.columns)[:12]}…")
            df['source_file'] = path
            return df
        except Exception as e1:
            self.dprint(f"[_read_one_h5] 'fits' not found or failed: {e1}")
            with pd.HDFStore(path, mode='r') as store:
                keys = store.keys()
                if not keys:
                    raise RuntimeError("No datasets in HDF5")
                df = store[keys[0]]
            self.dprint(f"[_read_one_h5] Loaded first key {keys[0]} shape={df.shape}")
            df['source_file'] = path
            return df

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select HDF5 Fit Files", "", "HDF5 Files (*.h5 *.hdf *.hdf5)")
        if not files:
            self.dprint("[load_files] cancelled")
            return
        self.dprint(f"[load_files] Selected {len(files)} files")
        dfs, errs = [], []
        for p in files:
            try: dfs.append(self._read_one_h5(p))
            except Exception as e: errs.append(f"{p}: {e}")
        if not dfs:
            msg = "No valid HDF5 tables loaded." + "".join(errs)
            self.dprint("[load_files] ERROR:", msg)
            QMessageBox.warning(self, "No Data", msg)
            return
        raw = pd.concat(dfs, ignore_index=True)
        self.dprint(f"[load_files] raw shape={raw.shape}")
        # Normalize core columns if missing
        for col in ['dataset_id', 'event', 'channel', 'fit_type']:
            if col not in raw.columns: raw[col] = np.nan
        # Types + normalize event spacing
        try: raw['event'] = raw['event'].astype(str).str.strip()
        except Exception as e: self.dprint("[load_files] event astype/strip failed:", e)
        try: raw['channel'] = pd.to_numeric(raw['channel'], errors='coerce').astype('Int64')
        except Exception as e: self.dprint("[load_files] channel coerce failed:", e)
        try: raw['fit_type'] = raw['fit_type'].astype(str)
        except Exception as e: self.dprint("[load_files] fit_type astype failed:", e)

        self.raw = raw
        self.rebuild_groups()

    # ---------------- Grouping / Derived -----------------
    def _pick_best(self, g: pd.DataFrame) -> pd.Series:
        """Row with MOST non-NaNs; ties -> last (stable)."""
        scores = g.notna().sum(axis=1)
        mx = scores.max()
        return g.loc[scores.idxmax()] if (scores==mx).sum()==1 else g[scores==mx].iloc[-1]

    def _coerce_numeric_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        for c in df.columns:
            if c in ('dataset_id','fit_type','channel','event'): continue
            if df[c].dtype == 'object':
                def _flatten(x):
                    if isinstance(x, (list, tuple, np.ndarray)): return x[0] if len(x)==1 else np.nan
                    return x
                df[c] = df[c].map(_flatten)
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def _maybe_add_csa_derivatives(self, df: pd.DataFrame, fit_suffix: str) -> pd.DataFrame:
        if CSA_pulse is None: return df
        req = [f't0', f'C0', f'C1', f'C2', f'T0', f'T1', f'T2']
        if not all(r in df.columns for r in req):
            return df
        try:
            t0, T1, T2 = df['t0'], df['T1'], df['T2']
            peak_t = t0 + T1 * np.log(T2 / T1 + 1.0)
            df['CSA_peak_time'] = peak_t
            sig = inspect.signature(CSA_pulse)
            nparams = len(sig.parameters)
            c0_vals = df['C0'].values
            c1_vals = df['C1'].values
            baseline = c0_vals - c1_vals
            if 'C' in df.columns:
                c_vals = df['C'].values
                baseline = baseline + c_vals
            else:
                c_vals = np.zeros_like(baseline, dtype=float)

            if nparams >= 9:
                peak_vals = CSA_pulse(
                    peak_t.values,
                    t0.values,
                    c0_vals,
                    c1_vals,
                    df['C2'].values,
                    df['T0'].values,
                    df['T1'].values,
                    df['T2'].values,
                    c_vals,
                )
            else:
                peak_vals = CSA_pulse(
                    peak_t.values,
                    t0.values,
                    c0_vals,
                    c1_vals,
                    df['C2'].values,
                    df['T0'].values,
                    df['T1'].values,
                    df['T2'].values,
                )
            df['CSA_peak'] = peak_vals - baseline  # report peak above asymptotic baseline (C0 - C1 [+ C])
        except Exception as e:
            self.dprint(f"[_maybe_add_csa_derivatives] skip due to: {e}")
        return df

    def _maybe_add_skew_peak(self, df: pd.DataFrame) -> pd.DataFrame:
        """For skew_gaussian fits, add:
        - SG_peak: numeric grid max of (skew_gaussian - C) [legacy]
        - peak_value: value above baseline C at mode x (peak - C), per request
        """
        req = ['A', 'xi', 'omega', 'alpha', 'C']
        if not all(r in df.columns for r in req):
            return df
        # Use a fixed grid around xi ± 8*omega to find the peak numerically.
        tgrid = np.linspace(-8.0, 8.0, 2001)

        def _row_peak(row) -> float:
            try:
                A, xi, omega, alpha, C = (row[k] for k in req)
                if not (np.isfinite(A) and np.isfinite(xi) and np.isfinite(omega) and np.isfinite(alpha) and np.isfinite(C)):
                    return np.nan
                if omega <= 0:
                    return np.nan
                x = xi + omega * tgrid
                y = skew_gaussian_value(x, A, xi, omega, alpha, C)
                return float(np.nanmax(y - C))
            except Exception:
                return np.nan

        def _row_peak_value(row) -> float:
            try:
                A, xi, omega, alpha, C = (row[k] for k in req)
                pv, xpk = skew_gaussian_peak_value(A, xi, omega, alpha, C)
                return pv
            except Exception:
                return np.nan

        try:
            df['SG_peak'] = df.apply(_row_peak, axis=1)
            df['peak_value'] = df.apply(_row_peak_value, axis=1)
        except Exception as e:
            self.dprint(f"[_maybe_add_skew_peak] failed: {e}")
        return df

    def _maybe_add_gaussian_integral(self, df: pd.DataFrame, frac: float = _GAUSS_INT_FRAC) -> pd.DataFrame:
        """For gaussian fits, add gauss_int_<pct>: analytic area above baseline within a fixed window.

        Window is μ ± σ * sqrt(-2 ln(frac)), i.e., where the Gaussian remains at
        least `frac` of its peak above baseline (default 10%). The integral is
        signed, preserving negative peaks.
        """
        req = ['A', 'sigma']  # μ and C are optional for the analytic integral
        if df.empty or not all(r in df.columns for r in req):
            return df
        col_name = f"gauss_int_{int(frac*100)}pct"
        try:
            df[col_name] = df.apply(
                lambda row: gaussian_integral_within_frac(
                    row['A'],
                    row['mu'] if 'mu' in df.columns else np.nan,
                    row['sigma'],
                    row['C'] if 'C' in df.columns else 0.0,
                    frac
                ),
                axis=1
            )
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        except Exception as e:
            self.dprint(f"[_maybe_add_gaussian_integral] failed: {e}")
        return df

    def _apply_pvdf_q(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply PVDF q(v) calibration to an available peak/amplitude column.

        Priority for amplitude source:
          1) 'CSA_peak' (from CSA_pulse)
          2) 'peak_value' (skew_gaussian peak above baseline at mode)
          3) 'SG_peak' (legacy skew_gaussian peak above baseline by grid)
          4) 'value' (used by low_pass_max export)

        Writes:
          • 'pvdf_q'  -> calibrated charge in electrons
          • 'pvdf_q_c' -> same charge in Coulombs

        Supports CSV-derived calibrations (5nF segmented, 9.5nF monolithic, 15nF custom)
        and
        linear Cremat CSA transfer functions (3um / 8um / 30um).
        """
        mode = self.pvdf_combo.currentText() if self.pvdf_combo is not None else 'Off'
        if mode == 'Off' or df.empty:
            return df
        # Choose which column to convert
        amp_col = None
        for c in ('CSA_peak', 'peak_value', 'SG_peak', 'value'):
            if c in df.columns:
                amp_col = c
                break
        if amp_col is None:
            return df
        try:
            vals = pd.to_numeric(df[amp_col], errors='coerce').astype(float).values
        except Exception:
            return df

        pvdf_q_e = None
        pvdf_q_c = None
        try:
            if mode.startswith('5nF') and _q_of_v_5 is not None:
                pvdf_q_e = _q_of_v_5(vals)
                pvdf_q_c = pvdf_q_e / _E_PER_C
            elif mode.startswith('9.5nF') and _q_of_v_9 is not None:
                pvdf_q_e = _q_of_v_9(vals)
                pvdf_q_c = pvdf_q_e / _E_PER_C
            elif mode.startswith('15nF') and _q_of_v_15 is not None:
                pvdf_q_e = _q_of_v_15(vals)
                pvdf_q_c = pvdf_q_e / _E_PER_C
            elif mode in _CSA_LINEAR_CAL:
                slope, intercept = _CSA_LINEAR_CAL[mode]
                pvdf_q_c = slope * vals + intercept
                pvdf_q_e = pvdf_q_c * _E_PER_C
        except Exception as e:
            self.dprint("[_apply_pvdf_q]", e)
            pvdf_q_e = None
            pvdf_q_c = None

        if pvdf_q_e is not None:
            df['pvdf_q'] = pvdf_q_e
        if pvdf_q_c is not None:
            df['pvdf_q_c'] = pvdf_q_c

        return df

    def _maybe_add_qd_cal(self, df: pd.DataFrame) -> pd.DataFrame:
        """For QD3Fit/QDMFit groups, add cal = mass_g^1.3 * v_km_s^3 if columns present."""
        if df.empty: return df
        if 'mass' not in df.columns or 'v' not in df.columns:
            return df
        try:
            m_kg = pd.to_numeric(df['mass'], errors='coerce')
            v_ms = pd.to_numeric(df['v'], errors='coerce')
            m_g = m_kg * 1e3
            v_kms = v_ms / 1e3
            cal = np.where(
                np.isfinite(m_g) & np.isfinite(v_kms),
                np.power(m_g, 1.3) * np.power(v_kms, 3.0),
                np.nan,
            )
            df['cal'] = cal
        except Exception as e:
            self.dprint(f"[_maybe_add_qd_cal] failed: {e}")
        return df

    def rebuild_groups(self):
        """Build per-(dataset_id, fit_type, channel) DataFrames, 1 row per event."""
        self.groups.clear(); self.group_meta.clear(); self.group_params.clear()
        if self.raw.empty:
            self._refresh_combos()
            self.update_plot(); self._update_status()
            self.dprint("[rebuild_groups] raw empty; nothing to build")
            return
        # First, dedupe per (dataset_id, event, channel, fit_type)
        self.dprint("[rebuild_groups] raw shape:", self.raw.shape)
        dedup = (self.raw
                 .groupby(['dataset_id','event','channel','fit_type'], dropna=False, as_index=False)
                 .apply(self._pick_best)
                 .reset_index(drop=True))
        self.dprint("[rebuild_groups] after per-event dedupe:", dedup.shape)
        # Then split into groups by (dataset_id, fit_type, channel)
        for (ds, ft, ch), g in dedup.groupby(['dataset_id','fit_type','channel'], dropna=False):
            g2 = g.copy()
            g2['event'] = g2['event'].astype(str).str.strip()
            g2 = self._coerce_numeric_all(g2)
            keep = [c for c in g2.columns if c not in ['source_file']]
            g2 = g2[keep]
            if isinstance(ft, str) and _sanitize(ft) in ('csa','csa_pulse'):
                g2 = self._maybe_add_csa_derivatives(g2, _sanitize(ft))
            if isinstance(ft, str) and _sanitize(ft) in ('skew_gaussian',):
                g2 = self._maybe_add_skew_peak(g2)
            if isinstance(ft, str) and ('gaussian' in _sanitize(ft)):
                g2 = self._maybe_add_gaussian_integral(g2)
            if isinstance(ft, str) and _sanitize(ft) in ('qd3fit','qdmfit'):
                g2 = self._maybe_add_qd_cal(g2)
            # Apply PVDF q(v) conversion to any available peak/amp field
            g2 = self._apply_pvdf_q(g2)
            # ---- Drop columns that are entirely NaN (except key columns) ----
            key_cols = {'dataset_id','fit_type','channel','event'}
            mask_keep = g2.columns.to_series().isin(key_cols) | g2.notna().any()
            dropped = g2.columns[~mask_keep].tolist()
            if dropped:
                self.dprint(f"[group] dropped {len(dropped)} all-NaN cols for ds={ds} | {ft} | ch={ch}: {dropped[:10]}{'…' if len(dropped)>10 else ''}")
            g2 = g2.loc[:, mask_keep]
            # parameters available to choose from (numeric, excluding keys)
            params = [c for c in g2.columns if c not in key_cols and _is_numeric(g2[c])]
            label = f"ds={ds} | {ft} | ch={ch}"
            self.groups[label] = g2
            self.group_meta[label] = (str(ds), str(ft), int(ch) if pd.notna(ch) else -1)
            self.group_params[label] = sorted(params)
            self.dprint(f"[group] built {label}: rows={len(g2)}, params={len(params)}")
        self._refresh_combos()
        self.update_plot()
        self._update_status()

    # ---------------- UI refresh -----------------
    def _refresh_combos(self):
        # Dataset filter options from RAW
        ds_vals = sorted(list({str(x) for x in self.raw['dataset_id'].dropna().unique()})) if 'dataset_id' in self.raw else []
        cur_ds = self.ds_combo.currentText() if self.ds_combo.count() else 'All'
        self.ds_combo.blockSignals(True)
        self.ds_combo.clear(); self.ds_combo.addItem('All'); [self.ds_combo.addItem(d) for d in ds_vals]
        idx = self.ds_combo.findText(cur_ds); self.ds_combo.setCurrentIndex(idx if idx>=0 else 0)
        self.ds_combo.blockSignals(False)

        # Filter groups by dataset (if any)
        ds_filter = self.ds_combo.currentText() if self.ds_combo.count() else 'All'
        labels = [lbl for lbl,(d,_,_) in self.group_meta.items() if ds_filter=='All' or str(d)==ds_filter]
        labels = sorted(labels)
        cur_gx = self.gx_combo.currentText() if self.gx_combo.count() else ''
        cur_gy = self.gy_combo.currentText() if self.gy_combo.count() else ''
        cur_gx2 = self.gx2_combo.currentText() if self.gx2_combo.count() else ''
        cur_gy2 = self.gy2_combo.currentText() if self.gy2_combo.count() else ''
        self.gx_combo.blockSignals(True); self.gy_combo.blockSignals(True)
        self.gx_combo.clear(); [self.gx_combo.addItem(l) for l in labels]
        self.gy_combo.clear(); [self.gy_combo.addItem(l) for l in labels]
        if cur_gx:
            j = self.gx_combo.findText(cur_gx);  self.gx_combo.setCurrentIndex(j if j>=0 else 0)
        if cur_gy:
            j = self.gy_combo.findText(cur_gy);  self.gy_combo.setCurrentIndex(j if j>=0 else (1 if self.gy_combo.count()>1 else 0))
        self.gx_combo.blockSignals(False); self.gy_combo.blockSignals(False)

        self.gx2_combo.blockSignals(True); self.gy2_combo.blockSignals(True)
        self.gx2_combo.clear(); [self.gx2_combo.addItem(l) for l in labels]
        self.gy2_combo.clear(); [self.gy2_combo.addItem(l) for l in labels]
        if cur_gx2:
            j = self.gx2_combo.findText(cur_gx2); self.gx2_combo.setCurrentIndex(j if j>=0 else 0)
        if cur_gy2:
            j = self.gy2_combo.findText(cur_gy2); self.gy2_combo.setCurrentIndex(j if j>=0 else (1 if self.gy2_combo.count()>1 else 0))
        self.gx2_combo.blockSignals(False); self.gy2_combo.blockSignals(False)

        # Populate params for current selections
        self._refresh_param_combos()
        self._refresh_overlay_param_combos()

    def _refresh_param_combos(self):
        gx_lbl = self.gx_combo.currentText()
        gy_lbl = self.gy_combo.currentText()
        px_cur = self.px_combo.currentText() if self.px_combo.count() else ''
        py_cur = self.py_combo.currentText() if self.py_combo.count() else ''
        self.px_combo.blockSignals(True); self.py_combo.blockSignals(True)
        self.px_combo.clear(); self.py_combo.clear()
        if gx_lbl in self.group_params:
            for p in self.group_params[gx_lbl]: self.px_combo.addItem(p)
        if gy_lbl in self.group_params:
            for p in self.group_params[gy_lbl]: self.py_combo.addItem(p)
        if px_cur:
            j = self.px_combo.findText(px_cur); self.px_combo.setCurrentIndex(j if j>=0 else 0)
        if py_cur:
            j = self.py_combo.findText(py_cur); self.py_combo.setCurrentIndex(j if j>=0 else (1 if self.py_combo.count()>1 else 0))
        self.px_combo.blockSignals(False); self.py_combo.blockSignals(False)

    def _refresh_overlay_param_combos(self):
        if not getattr(self, 'chk_overlay', None):
            return
        if not self.chk_overlay.isChecked():
            self.px2_combo.clear(); self.py2_combo.clear()
            return
        gx_lbl = self.gx2_combo.currentText()
        gy_lbl = self.gy2_combo.currentText()
        px_cur = self.px2_combo.currentText() if self.px2_combo.count() else ''
        py_cur = self.py2_combo.currentText() if self.py2_combo.count() else ''
        self.px2_combo.blockSignals(True); self.py2_combo.blockSignals(True)
        self.px2_combo.clear(); self.py2_combo.clear()
        if gx_lbl in self.group_params:
            for p in self.group_params[gx_lbl]:
                self.px2_combo.addItem(p)
        if gy_lbl in self.group_params:
            for p in self.group_params[gy_lbl]:
                self.py2_combo.addItem(p)
        if px_cur:
            j = self.px2_combo.findText(px_cur); self.px2_combo.setCurrentIndex(j if j>=0 else 0)
        if py_cur:
            j = self.py2_combo.findText(py_cur); self.py2_combo.setCurrentIndex(j if j>=0 else (1 if self.py2_combo.count()>1 else 0))
        self.px2_combo.blockSignals(False); self.py2_combo.blockSignals(False)

    def on_dataset_changed(self):
        self._refresh_combos()
        self.update_plot()

    def on_group_changed(self):
        self._refresh_param_combos()
        self.update_plot()

    def on_overlay_toggle(self, _state=None):
        enabled = bool(self.chk_overlay.isChecked())
        self._set_overlay_controls_enabled(enabled)
        self._refresh_overlay_param_combos()
        self.update_plot()

    def on_overlay_group_changed(self):
        self._refresh_overlay_param_combos()
        if self.chk_overlay.isChecked():
            self.update_plot()

    def _set_overlay_controls_enabled(self, enabled: bool):
        for widget in (self.gx2_combo, self.px2_combo, self.gy2_combo, self.py2_combo):
            widget.setEnabled(enabled)

    # ---------------- Plotting -----------------
    def _update_status(self, extra: str = ""):
        raw_n = len(self.raw)
        groups_n = len(self.groups)
        msg = f"raw rows: {raw_n} | groups: {groups_n}"
        if extra:
            msg += f" | {extra}"
        self.lbl_status.setText(msg)
        self.dprint("[status]", msg)

    def _axes_group_label(self, label: str) -> str:
        """Shorten dataset path portions for cleaner axis annotations."""
        meta = self.group_meta.get(label)
        if not meta:
            return label
        ds, ft, ch = meta
        ds_leaf = _dataset_leaf_name(ds)
        ft_str = str(ft) if ft is not None else ''
        ch_str = str(ch) if ch is not None else ''
        return f"ds={ds_leaf} | {ft_str} | ch={ch_str}"

    def _param_label(self, param: str) -> str:
        if param == 'cal':
            return "cal (g^1.3 * (km/s)^3)"
        return param

    def _param_tooltip(self, param: str) -> str:
        if param == 'cal':
            return "cal = m^1.3 * v^3 using mass in g and velocity in km/s"
        return ""

    def _calc_limits(self, df: pd.DataFrame, x: str, y: str) -> Tuple[Tuple[float,float], Tuple[float,float]]:
        xmin, xmax = np.nanmin(df[x].values), np.nanmax(df[x].values)
        ymin, ymax = np.nanmin(df[y].values), np.nanmax(df[y].values)
        self.dprint(f"[limits] x:[{xmin}, {xmax}] y:[{ymin}, {ymax}]")
        def margin(lo, hi):
            if not np.isfinite(lo) or not np.isfinite(hi): return (lo, hi)
            if lo == hi:
                span = 1.0 if lo == 0 else abs(lo) * 0.1
                return (lo-span, hi+span)
            span = hi-lo
            return (lo-0.05*span, hi+0.05*span)
        return margin(xmin,xmax), margin(ymin,ymax)

    def _prepare_xy_dataframe(self, gx_lbl: str, px: str, gy_lbl: str, py: str,
                              dropna: bool = False, context: str = "") -> pd.DataFrame:
        """Return merged dataframe with columns X,Y,event,dataset_id for the given group selections."""
        prefix = f"{context.strip()} " if context else ""
        if not gx_lbl or gx_lbl not in self.groups:
            raise ValueError(f"{prefix}X frame unavailable")
        if not gy_lbl or gy_lbl not in self.groups:
            raise ValueError(f"{prefix}Y frame unavailable")
        if not px:
            raise ValueError(f"{prefix}select an X parameter")
        if not py:
            raise ValueError(f"{prefix}select a Y parameter")

        GX = self.groups[gx_lbl]; GY = self.groups[gy_lbl]
        if px not in GX.columns:
            raise ValueError(f"{prefix}parameter '{px}' missing in X frame")
        if py not in GY.columns:
            raise ValueError(f"{prefix}parameter '{py}' missing in Y frame")

        ds_x = None
        try:
            ds_x = self.group_meta.get(gx_lbl, (None, None, None))[0]
        except Exception:
            ds_x = None

        xdf = GX[['event', px]].copy()
        xdf['event'] = xdf['event'].astype(str).str.strip()
        xdf[px] = pd.to_numeric(xdf[px], errors='coerce')
        if 'dataset_id' in GX.columns:
            xdf['dataset_id'] = GX['dataset_id'].astype(str)
        else:
            xdf['dataset_id'] = str(ds_x) if ds_x is not None else ''

        ydf = GY[['event', py]].copy()
        ydf['event'] = ydf['event'].astype(str).str.strip()
        ydf[py] = pd.to_numeric(ydf[py], errors='coerce')

        ctx_tag = context or 'primary'
        ev_x = set(xdf['event'].dropna().astype(str))
        ev_y = set(ydf['event'].dropna().astype(str))
        inter = ev_x & ev_y
        only_x = sorted(list(ev_x - ev_y))[:10]
        only_y = sorted(list(ev_y - ev_x))[:10]
        self.dprint(f"[events:{ctx_tag}] |X|={len(ev_x)} |Y|={len(ev_y)} |∩|={len(inter)}; sample only-in-X: {only_x}; only-in-Y: {only_y}")

        merged = pd.merge(xdf, ydf, on='event', how='inner')
        colx, coly = px, py
        if px == py:
            cx, cy = f"{px}_x", f"{py}_y"
            if cx in merged.columns and cy in merged.columns:
                colx, coly = cx, cy
        mask = np.isfinite(merged[colx].values) & np.isfinite(merged[coly].values)
        merged = merged.loc[mask]
        if dropna:
            merged = merged.dropna()
        merged = merged.rename(columns={colx: 'X', coly: 'Y'})
        keep_cols = ['X', 'Y', 'event', 'dataset_id']
        for col in keep_cols:
            if col not in merged.columns:
                merged[col] = np.nan
        merged = merged[keep_cols].reset_index(drop=True)
        self.dprint(f"[prepare_xy] {context or 'primary'} => rows={len(merged)} from {gx_lbl}->{px} & {gy_lbl}->{py}")
        return merged

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('white')
        self.last_plot_df = None
        self.last_overlay_df = None
        self._scatter = None
        self._overlay_scatter = None
        self._sel_scatter = None
        self._current_display_X = None
        self._current_display_Y = None
        self._display_to_base = None

        gx_lbl = self.gx_combo.currentText(); gy_lbl = self.gy_combo.currentText()
        px = self.px_combo.currentText(); py = self.py_combo.currentText()
        opx = ''
        opy = ''
        if not gx_lbl or not gy_lbl or gx_lbl not in self.groups or gy_lbl not in self.groups:
            ax.set_title('Load files and select X/Y frames')
            self.canvas.draw(); self._update_status("waiting for selection")
            self.dprint("[update_plot] missing group selection")
            return
        if not px or not py:
            ax.set_title('Select X and Y parameters')
            self.canvas.draw(); self._update_status("waiting for params")
            self.dprint("[update_plot] missing parameter selection")
            return

        drop_na = self.chk_dropna.isChecked()
        try:
            merged = self._prepare_xy_dataframe(gx_lbl, px, gy_lbl, py, dropna=drop_na, context="primary")
        except ValueError as exc:
            msg = str(exc)
            ax.set_title(msg)
            self.canvas.draw(); self._update_status(msg)
            self.dprint("[update_plot] primary prep failed:", msg)
            return

        # establish plot signature; clear selection/removals if context changed (primary only)
        sig = (self.ds_combo.currentText() if self.ds_combo.count() else 'All', gx_lbl, px, gy_lbl, py)
        if self._plot_sig != sig:
            self.selected_indices.clear()
            self.removed_indices.clear()
            self._plot_sig = sig

        if merged.empty:
            ax.set_title('No overlapping events between X and Y frames')
            self.canvas.draw(); self._update_status("0 overlap")
            self.dprint("[update_plot] no overlap")
            return

        base_full = merged[['X','Y','event','dataset_id']].copy()
        try:
            # Base df and apply removals using base indices
            base = base_full.reset_index(drop=True)
            self._merged_base = base
            if self.removed_indices:
                mask = ~base.index.to_series().isin(self.removed_indices)
                disp = base[mask].reset_index(drop=True)
                self._display_to_base = base.index[mask.values].to_numpy()
            else:
                disp = base.copy()
                self._display_to_base = base.index.to_numpy()

            self._current_display_X = disp['X'].values
            self._current_display_Y = disp['Y'].values
        except Exception as e:
            self.dprint("[update_plot] Exception during plotting:", e)
            traceback.print_exc()
            ax.set_title(f"Plot error: {e}")
            self.canvas.draw()
            return

        overlay_disp = None
        overlay_label = None
        overlay_err = None
        if self.chk_overlay.isChecked():
            ogx_lbl = self.gx2_combo.currentText()
            ogy_lbl = self.gy2_combo.currentText()
            opx = self.px2_combo.currentText()
            opy = self.py2_combo.currentText()
            try:
                overlay_full = self._prepare_xy_dataframe(ogx_lbl, opx, ogy_lbl, opy, dropna=drop_na, context="overlay")
            except ValueError as exc:
                overlay_full = None
                overlay_err = str(exc)
                self.dprint("[overlay] skipped:", overlay_err)
            if overlay_full is not None:
                if overlay_full.empty:
                    overlay_err = "overlay has no overlapping events"
                else:
                    overlay_disp = overlay_full[['X','Y']].reset_index(drop=True)
                    self.last_overlay_df = overlay_disp[['X','Y']].copy()
                    overlay_label = f"Overlay ({len(overlay_disp)} pts) {self._axes_group_label(ogx_lbl)} vs {self._axes_group_label(ogy_lbl)}"
        else:
            self.last_overlay_df = None

        limit_source = base_full[['X','Y']]
        if overlay_disp is not None and not overlay_disp.empty:
            limit_source = pd.concat([limit_source, overlay_disp[['X','Y']]], ignore_index=True)

        (xmin,xmax), (ymin,ymax) = self._calc_limits(limit_source, 'X','Y')
        try:
            if self.chk_logx.isChecked():
                if np.any(limit_source['X'] <= 0): self.dprint('[update_plot] WARNING: non-positive X with log scale')
                ax.set_xscale('log')
            if self.chk_logy.isChecked():
                if np.any(limit_source['Y'] <= 0): self.dprint('[update_plot] WARNING: non-positive Y with log scale')
                ax.set_yscale('log')
        except Exception as e:
            self.dprint('[update_plot] scale set failed:', e)

        primary_label = None
        if overlay_disp is not None and not overlay_disp.empty:
            primary_label = f"Primary ({len(disp)} pts)"
        self._scatter = ax.scatter(self._current_display_X, self._current_display_Y,
                                   s=30, alpha=1.0, marker='o', edgecolors='none', zorder=3,
                                   picker=True, label=primary_label)

        if overlay_disp is not None and not overlay_disp.empty:
            self._overlay_scatter = ax.scatter(
                overlay_disp['X'].values, overlay_disp['Y'].values,
                s=36, marker='s', facecolors='none', edgecolors='tab:orange',
                linewidths=1.0, alpha=0.85, zorder=4, label=overlay_label)
        else:
            self._overlay_scatter = None

        # selection overlay
        self._update_selection_overlay(ax)

        if np.isfinite(xmin) and np.isfinite(xmax): ax.set_xlim(xmin, xmax)
        if np.isfinite(ymin) and np.isfinite(ymax): ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
        gx_axis_lbl = self._axes_group_label(gx_lbl)
        gy_axis_lbl = self._axes_group_label(gy_lbl)
        ax.set_xlabel(f"{self._param_label(px)}  [{gx_axis_lbl}]")
        ax.set_ylabel(f"{self._param_label(py)}  [{gy_axis_lbl}]")
        self.px_combo.setToolTip(self._param_tooltip(px))
        self.py_combo.setToolTip(self._param_tooltip(py))
        if self.chk_overlay.isChecked():
            self.px2_combo.setToolTip(self._param_tooltip(opx))
            self.py2_combo.setToolTip(self._param_tooltip(opy))
        else:
            self.px2_combo.setToolTip("")
            self.py2_combo.setToolTip("")
        title = f"{py} vs {px}  (n={len(disp)})"
        if overlay_disp is not None and not overlay_disp.empty:
            title += f"  | overlay n={len(overlay_disp)}"
        elif overlay_err:
            title += f"  | overlay: {overlay_err}"
        ax.set_title(title)
        if (overlay_disp is not None and not overlay_disp.empty) or (primary_label is not None):
            ax.legend()
        self.last_plot_df = disp[['X','Y']].copy()
        if self._pick_cid is None:
            self._pick_cid = self.canvas.mpl_connect('pick_event', self.on_pick)
        status_msg = f"plotted {len(disp)} pts"
        if overlay_disp is not None and not overlay_disp.empty:
            status_msg += f" | overlay {len(overlay_disp)} pts"
        elif overlay_err:
            status_msg += f" | overlay: {overlay_err}"
        self.canvas.draw(); self._update_status(status_msg)
        self.dprint("[update_plot] plotted", len(disp), "points; xlim=", ax.get_xlim(), "ylim=", ax.get_ylim())

    # ---------------- Point selection/removal -----------------
    def _update_selection_overlay(self, ax):
        # Remove old selection overlay
        try:
            if self._sel_scatter is not None and self._sel_scatter in ax.collections:
                ax.collections.remove(self._sel_scatter)
        except Exception:
            pass
        self._sel_scatter = None
        if self._display_to_base is None or self._current_display_X is None:
            return
        if not self.selected_indices:
            self.canvas.draw(); return
        # Map base-selected indices to displayed indices
        sel_mask = [i for i, b in enumerate(self._display_to_base) if b in self.selected_indices]
        if not sel_mask:
            self.canvas.draw(); return
        xs = np.asarray(self._current_display_X)[sel_mask]
        ys = np.asarray(self._current_display_Y)[sel_mask]
        ax = self.figure.axes[0]
        self._sel_scatter = ax.scatter(xs, ys, s=90, facecolors='none', edgecolors='red', linewidths=1.5, zorder=4, label='selected')
        self.canvas.draw()

    def on_pick(self, event):
        if not getattr(self, 'chk_select', None) or not self.chk_select.isChecked():
            return
        if event.artist is not self._scatter:
            return
        if self._current_display_X is None or self._display_to_base is None:
            return
        inds = getattr(event, 'ind', None)
        if inds is None or len(inds) == 0:
            return
        # choose nearest to click among candidates
        x0 = event.mouseevent.xdata; y0 = event.mouseevent.ydata
        try:
            cand = np.array(inds, dtype=int)
            dx = self._current_display_X[cand] - x0
            dy = self._current_display_Y[cand] - y0
            i_disp = int(cand[np.argmin(dx*dx + dy*dy)])
        except Exception:
            i_disp = int(inds[0])
        base_idx = int(self._display_to_base[i_disp])
        if base_idx in self.selected_indices:
            self.selected_indices.remove(base_idx)
        else:
            self.selected_indices.add(base_idx)
        ax = self.figure.axes[0]
        self._update_selection_overlay(ax)
        # Emit dataset_id + event for external listeners (e.g., oscope_palantir)
        try:
            row = self._merged_base.iloc[base_idx]
            ds = str(row.get('dataset_id', '')).strip()
            ev = str(row.get('event', '')).strip()
            if ev:
                self.eventPicked.emit(ds, ev)
        except Exception:
            pass

    def remove_selected_points(self):
        if not self.selected_indices:
            QMessageBox.information(self, "Remove Selected", "No points selected.")
            return
        self.removed_indices.update(self.selected_indices)
        self.selected_indices.clear()
        self.update_plot()

    def clear_selection(self):
        if self.selected_indices:
            self.selected_indices.clear()
            # Refresh overlay without changing removals
            self.update_plot()

    def reset_removals(self):
        if not self.removed_indices:
            return
        self.removed_indices.clear()
        self.selected_indices.clear()
        self.update_plot()

    # ---------------- Fit / Export / Diag -----------------
    def run_linear_fit(self):
        if self.last_plot_df is None or self.last_plot_df.empty:
            QMessageBox.information(self, "Linear Fit", "Nothing plotted yet.")
            return
        X = self.last_plot_df['X'].values; Y = self.last_plot_df['Y'].values
        denom = float(np.sum(X*X))
        m = float(np.sum(X*Y) / denom) if denom != 0 else np.nan
        self.dprint(f"[run_linear_fit] m={m}, denom={denom}, n={len(X)}")
        ax = self.figure.axes[0]
        try:
            xmin, xmax = ax.get_xlim(); xline = np.linspace(xmin, xmax, 100)
            # Remove prior origin-fit line(s) before plotting a new one
            for ln in list(ax.lines):
                lab = ln.get_label()
                if isinstance(lab, str) and ('Linear Fit (through origin' in lab or 'Linear Fit (origin' in lab):
                    try: ax.lines.remove(ln)
                    except Exception: pass
            label = f"Linear Fit (through origin) m={m:.2e}"
            ax.plot(xline, m*xline, label=label)
            ax.legend(); self.canvas.draw()
        except Exception as e:
            self.dprint("[run_linear_fit] plot line failed:", e)
        QMessageBox.information(self, "Linear Fit", f"Slope m = {m:.6g} (Intercept forced to 0)")

    def run_linear_fit_free(self):
        """Ordinary least squares with free intercept: Y = m*X + b."""
        if self.last_plot_df is None or self.last_plot_df.empty:
            QMessageBox.information(self, "Linear Fit (m, b)", "Nothing plotted yet.")
            return
        X = self.last_plot_df['X'].values.astype(float)
        Y = self.last_plot_df['Y'].values.astype(float)
        finite = np.isfinite(X) & np.isfinite(Y)
        X = X[finite]; Y = Y[finite]
        if X.size < 2:
            QMessageBox.information(self, "Linear Fit (m, b)", "Not enough points to fit.")
            return
        # Solve [X, 1] * [m, b] = Y
        A = np.vstack([X, np.ones_like(X)]).T
        try:
            m, b = np.linalg.lstsq(A, Y, rcond=None)[0]
        except Exception as e:
            self.dprint("[run_linear_fit_free] lstsq failed:", e)
            QMessageBox.critical(self, "Linear Fit (m, b)", f"Least squares failed: {e}")
            return
        # R^2
        yhat = m*X + b
        ss_res = float(np.sum((Y - yhat)**2))
        ss_tot = float(np.sum((Y - np.mean(Y))**2))
        r2 = (1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan
        self.dprint(f"[run_linear_fit_free] m={m}, b={b}, r2={r2}, n={X.size}")
        # Plot line over current axes
        ax = self.figure.axes[0]
        try:
            xmin, xmax = ax.get_xlim(); xline = np.linspace(xmin, xmax, 200)
            # remove prior free-fit line if present
            for ln in list(ax.lines):
                lab = ln.get_label()
                if isinstance(lab, str) and lab.startswith('Linear Fit (m, b'):
                    try:
                        ax.lines.remove(ln)
                    except Exception:
                        pass
            label = f"Linear Fit (m, b) m={m:.2e}, b={b:.2e}"
            ax.plot(xline, m*xline + b, label=label)
            ax.legend(); self.canvas.draw()
        except Exception as e:
            self.dprint("[run_linear_fit_free] plot line failed:", e)
        QMessageBox.information(self, "Linear Fit (m, b)", f"Slope m = {m:.6g} \n Intercept b = {b:.6g} \n R² = {r2:.6g}")

    def save_csv(self):
        if self.last_plot_df is None or self.last_plot_df.empty:
            QMessageBox.information(self, "Export", "No merged X/Y data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save merged XY CSV", "", "CSV Files (*.csv)")
        if not path: return
        try:
            self.last_plot_df.to_csv(path, index=False)
            QMessageBox.information(self, "Export", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _dup_stats(self, ds: str, ft: str, ch: int) -> Tuple[int, List[Tuple[str,int]]]:
        """Return (#duplicate events, sample list of (event, count)) for the given group from RAW."""
        try:
            sub = self.raw[(self.raw['dataset_id'].astype(str)==str(ds)) &
                           (self.raw['fit_type'].astype(str)==str(ft)) &
                           (pd.to_numeric(self.raw['channel'], errors='coerce')==int(ch))]
            ev = sub['event'].astype(str).str.strip()
            vc = ev.value_counts()
            dups = vc[vc>1]
            sample = list(dups.head(10).items())
            return int(dups.sum()), sample
        except Exception:
            return 0, []

    def show_diagnostics(self):
        if self.raw.empty and not self.groups:
            QMessageBox.information(self, "Diagnostics", "No data loaded yet.")
            return
        parts = []
        parts.append(f"RAW SHAPE: {self.raw.shape}")
        if not self.raw.empty:
            cols = list(self.raw.columns)
            parts.append("RAW COLUMNS (first 20): " + ", ".join(cols[:20]) + ("…" if len(cols)>20 else ""))
        parts.append(f"GROUPS: {len(self.groups)}")
        for lbl in sorted(self.groups.keys())[:20]:
            g = self.groups[lbl]
            params = self.group_params.get(lbl, [])
            parts.append(f"  - {lbl}: rows={len(g)} | params={len(params)} -> {', '.join(params[:10]) + ('…' if len(params)>10 else '')}")
        # --- Previews for the currently selected X/Y frames ---
        gx_lbl = self.gx_combo.currentText(); gy_lbl = self.gy_combo.currentText()
        if gx_lbl in self.groups:
            G = self.groups[gx_lbl]
            parts.append("X FRAME PREVIEW: " + gx_lbl)
            parts.append(G.head(10).to_string(max_cols=20))
            ds, ft, ch = self.group_meta.get(gx_lbl, ("?","?",-1))
            dup_total, dup_sample = self._dup_stats(ds, ft, ch)
            if dup_total:
                parts.append(f"X dup events in RAW: total rows over unique = {dup_total}; sample: {dup_sample}")
        if gy_lbl in self.groups:
            G = self.groups[gy_lbl]
            parts.append("Y FRAME PREVIEW: " + gy_lbl)
            parts.append(G.head(10).to_string(max_cols=20))
            ds, ft, ch = self.group_meta.get(gy_lbl, ("?","?",-1))
            dup_total, dup_sample = self._dup_stats(ds, ft, ch)
            if dup_total:
                parts.append(f"Y dup events in RAW: total rows over unique = {dup_total}; sample: {dup_sample}")
        # --- Event overlap report for current param choices ---
        px = self.px_combo.currentText(); py = self.py_combo.currentText()
        if gx_lbl in self.groups and gy_lbl in self.groups and px and py:
            GX = self.groups[gx_lbl][['event', px]].copy(); GX['event'] = GX['event'].astype(str).str.strip(); GX[px] = pd.to_numeric(GX[px], errors='coerce')
            GY = self.groups[gy_lbl][['event', py]].copy(); GY['event'] = GY['event'].astype(str).str.strip(); GY[py] = pd.to_numeric(GY[py], errors='coerce')
            ev_x = set(GX['event'].dropna().astype(str)); ev_y = set(GY['event'].dropna().astype(str))
            inter = ev_x & ev_y
            only_x = sorted(list(ev_x - ev_y))[:20]
            only_y = sorted(list(ev_y - ev_x))[:20]
            parts.append(f"EVENT OVERLAP for selected params: |X|={len(ev_x)} |Y|={len(ev_y)} |∩|={len(inter)}")
            if only_x:
                parts.append("  sample only-in-X: " + ", ".join(map(str, only_x)))
            if only_y:
                parts.append("  sample only-in-Y: " + ", ".join(map(str, only_y)))
        if self.chk_overlay.isChecked():
            ogx_lbl = self.gx2_combo.currentText(); ogy_lbl = self.gy2_combo.currentText()
            opx = self.px2_combo.currentText(); opy = self.py2_combo.currentText()
            parts.append(f"OVERLAY selection -> X:{ogx_lbl} [{opx}] | Y:{ogy_lbl} [{opy}]")
            if ogx_lbl in self.groups and ogy_lbl in self.groups and opx and opy:
                try:
                    GX = self.groups[ogx_lbl][['event', opx]].copy(); GX['event'] = GX['event'].astype(str).str.strip(); GX[opx] = pd.to_numeric(GX[opx], errors='coerce')
                    GY = self.groups[ogy_lbl][['event', opy]].copy(); GY['event'] = GY['event'].astype(str).str.strip(); GY[opy] = pd.to_numeric(GY[opy], errors='coerce')
                    ev_x = set(GX['event'].dropna().astype(str)); ev_y = set(GY['event'].dropna().astype(str))
                    inter = ev_x & ev_y
                    parts.append(f"  overlay event overlap: |X|={len(ev_x)} |Y|={len(ev_y)} |∩|={len(inter)}")
                except Exception as e:
                    parts.append(f"  overlay preview failed: {e}")
        dlg_txt = "".join(parts)
        self.dprint("[diagnostics]" + dlg_txt)
        QMessageBox.information(self, "Diagnostics", dlg_txt)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FitResultsVisualizer()
    win.show()
    sys.exit(app.exec_())
