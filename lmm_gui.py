import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd

"""
LMM CSA/Scaler has a calibration of ~0.155 mV/pC (Rough linear fit of scaler data)

Cremat Shaper uses a calibration of 4 V/pC
"""


DEFAULT_RHO_DUST    = 2.7 #Al    #7.5 Fe  # g/cc
DEFAULT_CAPACITANCE = 6e-7  # F
R_SMALL_CM          = 1e-8 * 1e2
R_BIG_CM            = 1e-5 * 1e2 #10 um in cm
N_POINTS            = 1000
DEFAULT_META_PATH = "./2024_09_25_metadata.csv"
DUST_TYPE_DENSITIES = {
    "aluminum": 2.7,
    "iron": 5.7,
}

# Calibration constants for q = K * m^1.3 * v^3.0 (mass in g, v in km/s)
CALIBRATION_K = {
    "3 um" : 6.0e18,
    "8 um" : 3.6e18,
    "30 um": 3.8e17,
}
CREMAT_V_PER_PC = 4.0

# Scaler output table from "Raw Output" sheet in
# "LMM Charge vs Output vs Detector Capacitance.xlsx" (charge in e-, C in F, V out).
SCALER_CHARGES = [
    4e+06, 5.04e+06, 6.34e+06, 7.98e+06, 1e+07, 1.26e+07, 1.59e+07, 2e+07, 2.52e+07, 3.18e+07,
    4e+07, 5.04e+07, 6.34e+07, 7.98e+07, 1e+08, 1.26e+08, 1.59e+08, 2e+08, 2.52e+08, 3.18e+08,
    4e+08, 5.04e+08, 6.34e+08, 7.98e+08, 1e+09, 1.26e+09, 1.59e+09, 2e+09, 2.52e+09, 3.18e+09,
    4e+09, 5.04e+09, 6.34e+09, 7.98e+09, 1e+10, 1.26e+10, 1.59e+10, 2e+10, 2.52e+10, 3.18e+10,
    4e+10, 5.04e+10, 6.34e+10, 7.98e+10, 1e+11,
]
SCALER_CAPACITANCES = [
    3e-07, 4e-07, 5e-07, 6e-07, 7e-07, 8e-07, 9e-07, 1e-06, 1.1e-06, 1.2e-06,
    1.3e-06, 1.4e-06,
]
SCALER_OUTPUTS = [
    [
        0.00569, 0.00716, 0.00901, 0.0113, 0.0143, 0.018, 0.0226, 0.0284, 0.0357, 0.0447,
        0.056, 0.0698, 0.0864, 0.106, 0.126, 0.145, 0.163, 0.178, 0.191, 0.203,
        0.214, 0.225, 0.236, 0.248, 0.261, 0.274, 0.289, 0.307, 0.326, 0.35,
        0.377, 0.41, 0.45, 0.498, 0.558, 0.631, 0.721, 0.833, 0.971, 1.14,
        1.36, 1.63, 1.96, 2.36, 2.84,
    ],
    [
        0.00553, 0.00696, 0.00877, 0.011, 0.0139, 0.0175, 0.022, 0.0276, 0.0347, 0.0435,
        0.0545, 0.068, 0.0843, 0.103, 0.123, 0.143, 0.161, 0.176, 0.189, 0.201,
        0.213, 0.224, 0.235, 0.246, 0.259, 0.272, 0.287, 0.304, 0.324, 0.346,
        0.373, 0.405, 0.444, 0.492, 0.549, 0.62, 0.708, 0.817, 0.952, 1.12,
        1.33, 1.59, 1.91, 2.3, 2.76,
    ],
    [
        0.00538, 0.00678, 0.00853, 0.0107, 0.0135, 0.017, 0.0214, 0.0269, 0.0338, 0.0424,
        0.0531, 0.0663, 0.0822, 0.101, 0.121, 0.141, 0.159, 0.174, 0.188, 0.2,
        0.211, 0.222, 0.233, 0.245, 0.257, 0.27, 0.285, 0.302, 0.321, 0.343,
        0.37, 0.401, 0.439, 0.485, 0.542, 0.611, 0.697, 0.802, 0.934, 1.1,
        1.3, 1.55, 1.86, 2.24, 2.68,
    ],
    [
        0.00524, 0.0066, 0.00831, 0.0105, 0.0132, 0.0165, 0.0208, 0.0262, 0.0329, 0.0413,
        0.0517, 0.0646, 0.0802, 0.0985, 0.119, 0.139, 0.157, 0.172, 0.186, 0.198,
        0.21, 0.221, 0.232, 0.243, 0.255, 0.269, 0.283, 0.3, 0.318, 0.34,
        0.366, 0.397, 0.434, 0.479, 0.534, 0.602, 0.685, 0.789, 0.917, 1.08,
        1.27, 1.52, 1.82, 2.18, 2.61,
    ],
    [
        0.00511, 0.00643, 0.00809, 0.0102, 0.0128, 0.0161, 0.0203, 0.0255, 0.032, 0.0402,
        0.0504, 0.063, 0.0783, 0.0963, 0.116, 0.136, 0.155, 0.171, 0.184, 0.197,
        0.208, 0.219, 0.23, 0.242, 0.254, 0.267, 0.281, 0.297, 0.316, 0.337,
        0.363, 0.393, 0.429, 0.473, 0.527, 0.593, 0.675, 0.776, 0.9, 1.06,
        1.25, 1.48, 1.78, 2.13, 2.55,
    ],
    [
        0.00498, 0.00626, 0.00789, 0.00993, 0.0125, 0.0157, 0.0198, 0.0249, 0.0312, 0.0392,
        0.0492, 0.0615, 0.0765, 0.0942, 0.114, 0.134, 0.153, 0.169, 0.183, 0.195,
        0.207, 0.218, 0.229, 0.24, 0.252, 0.265, 0.279, 0.295, 0.314, 0.335,
        0.36, 0.389, 0.425, 0.468, 0.521, 0.585, 0.665, 0.763, 0.885, 1.04,
        1.22, 1.45, 1.74, 2.08, 2.49,
    ],
    [
        0.00485, 0.00611, 0.00769, 0.00968, 0.0122, 0.0153, 0.0193, 0.0242, 0.0305, 0.0383,
        0.048, 0.06, 0.0747, 0.0921, 0.112, 0.132, 0.151, 0.167, 0.181, 0.194,
        0.205, 0.216, 0.227, 0.239, 0.251, 0.264, 0.278, 0.293, 0.311, 0.332,
        0.357, 0.386, 0.421, 0.463, 0.514, 0.577, 0.655, 0.751, 0.87, 1.02,
        1.2, 1.42, 1.7, 2.04, 2.43,
    ],
    [
        0.00474, 0.00596, 0.0075, 0.00944, 0.0119, 0.015, 0.0188, 0.0237, 0.0297, 0.0373,
        0.0468, 0.0586, 0.073, 0.0901, 0.11, 0.13, 0.149, 0.165, 0.18, 0.192,
        0.204, 0.215, 0.226, 0.237, 0.249, 0.262, 0.276, 0.291, 0.309, 0.33,
        0.354, 0.382, 0.416, 0.458, 0.508, 0.57, 0.646, 0.74, 0.856, 1,
        1.18, 1.4, 1.67, 1.99, 2.38,
    ],
    [
        0.00462, 0.00582, 0.00732, 0.00922, 0.0116, 0.0146, 0.0184, 0.0231, 0.029, 0.0365,
        0.0457, 0.0572, 0.0713, 0.0882, 0.107, 0.128, 0.147, 0.164, 0.178, 0.191,
        0.203, 0.214, 0.225, 0.236, 0.248, 0.26, 0.274, 0.29, 0.307, 0.327,
        0.351, 0.379, 0.412, 0.453, 0.502, 0.563, 0.637, 0.729, 0.843, 0.983,
        1.16, 1.37, 1.63, 1.95, 2.33,
    ],
    [
        0.00451, 0.00568, 0.00715, 0.009, 0.0113, 0.0143, 0.0179, 0.0226, 0.0284, 0.0356,
        0.0447, 0.0559, 0.0697, 0.0863, 0.105, 0.126, 0.145, 0.162, 0.177, 0.189,
        0.201, 0.212, 0.224, 0.235, 0.247, 0.259, 0.273, 0.288, 0.305, 0.325,
        0.348, 0.376, 0.409, 0.448, 0.497, 0.556, 0.629, 0.719, 0.83, 0.967,
        1.14, 1.35, 1.6, 1.92, 2.28,
    ],
    [
        0.00441, 0.00555, 0.00699, 0.0088, 0.0111, 0.0139, 0.0175, 0.022, 0.0277, 0.0348,
        0.0437, 0.0547, 0.0682, 0.0845, 0.103, 0.124, 0.143, 0.16, 0.175, 0.188,
        0.2, 0.211, 0.222, 0.234, 0.245, 0.258, 0.271, 0.286, 0.303, 0.323,
        0.346, 0.373, 0.405, 0.444, 0.492, 0.55, 0.621, 0.709, 0.818, 0.952,
        1.12, 1.32, 1.57, 1.88, 2.24,
    ],
    [
        0.00431, 0.00543, 0.00683, 0.0086, 0.0108, 0.0136, 0.0171, 0.0215, 0.0271, 0.034,
        0.0427, 0.0535, 0.0667, 0.0828, 0.101, 0.122, 0.141, 0.159, 0.174, 0.187,
        0.199, 0.21, 0.221, 0.232, 0.244, 0.256, 0.27, 0.285, 0.301, 0.321,
        0.343, 0.37, 0.402, 0.44, 0.487, 0.543, 0.613, 0.7, 0.806, 0.938,
        1.1, 1.3, 1.55, 1.84, 2.19,
    ],
]


def mass(radius_cm, density_gcc):
    """Convert radius and density into particle mass."""
    return 4.0 / 3.0 * np.pi * density_gcc * radius_cm**3


class LmmGui(tk.Tk):
    def __init__(self):
        """Initialize the GUI state and render the first plot."""
        super().__init__()
        self.title("LMM Calibration Plot")
        self.geometry("1120x840")

        # Bind control values to GUI inputs.
        self.thickness_var = tk.StringVar(value="8 um")
        self.charge_min_var = tk.StringVar(value="1e5")
        self.charge_max_var = tk.StringVar(value="1e8")
        self.charge_steps_var = tk.StringVar(value="4")
        self.segments_var = tk.StringVar(value="10")
        self.density_var = tk.StringVar(value=str(DEFAULT_RHO_DUST))
        self.dust_type_var = tk.StringVar(value="")
        self.capacitance_var = tk.StringVar(value=str(DEFAULT_CAPACITANCE))
        self.isoline_var = tk.StringVar(value="charge")
        self.show_scatter_var = tk.BooleanVar(value=True)
        self.meta_path_var = tk.StringVar(value=DEFAULT_META_PATH)
        self.hdf5_path_var = tk.StringVar(value="")
        self.hdf5_param_var = tk.StringVar(value="")
        self.show_hdf5_var = tk.BooleanVar(value=False)
        # Cache metadata and scaler data to avoid repeated parsing.
        self._meta_cache_path = None
        self._meta_cache_df = None
        self._scaler_data = None
        self._hdf5_cache_path = None
        self._hdf5_cache_df = None

        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        """Build the control panel and plotting canvas."""
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Layout frames for controls and plot.
        controls = ttk.Frame(self, padding=10)
        controls.grid(row=0, column=0, sticky="ns")

        plot_frame = ttk.Frame(self, padding=10)
        plot_frame.grid(row=0, column=1, sticky="nsew")

        # Calibration selection controls.
        ttk.Label(controls, text="Calibration").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            controls, text="3 um", variable=self.thickness_var, value="3 um"
        ).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(
            controls, text="8 um", variable=self.thickness_var, value="8 um"
        ).grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(
            controls, text="30 um", variable=self.thickness_var, value="30 um"
        ).grid(row=3, column=0, sticky="w")

        ttk.Separator(controls, orient="horizontal").grid(
            row=4, column=0, pady=8, sticky="ew"
        )

        # Charge range controls.
        ttk.Label(controls, text="Charge Out of Det (e-)").grid(
            row=5, column=0, sticky="w"
        )
        ttk.Label(controls, text="Min").grid(row=6, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.charge_min_var, width=12).grid(
            row=7, column=0, sticky="w")

        ttk.Label(controls, text="Max").grid(row=8, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.charge_max_var, width=12).grid(
            row=9, column=0, sticky="w")

        ttk.Label(controls, text="Steps").grid(row=10, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.charge_steps_var, width=12).grid(
            row=11, column=0, sticky="w")

        ttk.Label(controls, text="(log spaced)").grid(row=12, column=0, sticky="w")

        ttk.Separator(controls, orient="horizontal").grid(
            row=13, column=0, pady=8, sticky="ew")

        # Segmentation controls.
        ttk.Label(controls, text="Segments (1-10)").grid(
            row=14, column=0, sticky="w")

        ttk.Entry(controls, textvariable=self.segments_var, width=12).grid(
            row=15, column=0, sticky="w")

        ttk.Label(controls, text="Total charge = charge * segments").grid(
            row=16, column=0, sticky="w")

        ttk.Separator(controls, orient="horizontal").grid(
            row=17, column=0, pady=8, sticky="ew")

        # Isoline labeling and capacitance controls.
        ttk.Label(controls, text="Isoline labels").grid(
            row=18, column=0, sticky="w")
        ttk.Radiobutton(
            controls, text="Charge (e-)", variable=self.isoline_var, value="charge"
        ).grid(row=19, column=0, sticky="w")

        ttk.Radiobutton(
            controls, text="Scaler out (V)", variable=self.isoline_var, value="scaler"
        ).grid(row=20, column=0, sticky="w")

        ttk.Radiobutton(
            controls, text="Cremat Shaper", variable=self.isoline_var, value="cremat"
        ).grid(row=21, column=0, sticky="w")

        ttk.Label(controls, text="Detector capacitance (F)").grid(
            row=22, column=0, sticky="w")

        ttk.Entry(controls, textvariable=self.capacitance_var, width=12).grid(
            row=23, column=0, sticky="w")

        ttk.Label(controls, text="Scaler/Cremat uses total charge").grid(
            row=24, column=0, sticky="w")

        ttk.Separator(controls, orient="horizontal").grid(
            row=25, column=0, pady=8, sticky="ew")

        # Dust density controls.
        ttk.Label(controls, text="Dust density (g/cc)").grid(
            row=26, column=0, sticky="w")

        ttk.Entry(controls, textvariable=self.density_var, width=12).grid(
            row=27, column=0, sticky="w")

        ttk.Label(controls, text="Dust type").grid(
            row=28, column=0, sticky="w")

        ttk.Entry(
            controls,
            textvariable=self.dust_type_var,
            width=12,
            state="readonly",
        ).grid(row=29, column=0, sticky="w")

        ttk.Separator(controls, orient="horizontal").grid(
            row=30, column=0, pady=8, sticky="ew")

        # Metadata controls and scatter toggle.
        ttk.Label(controls, text="Metadata CSV").grid(row=31, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.meta_path_var, width=28).grid(
            row=32, column=0, sticky="w")

        ttk.Button(controls, text="Browse", command=self._browse_meta).grid(
            row=33, column=0, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Show accelerator scatter",
            variable=self.show_scatter_var,
        ).grid(row=34, column=0, sticky="w")

        ttk.Separator(controls, orient="horizontal").grid(
            row=35, column=0, pady=8, sticky="ew")

        # HDF5 fit file controls.
        ttk.Label(controls, text="HDF5 fit file").grid(row=36, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.hdf5_path_var, width=28).grid(
            row=37, column=0, sticky="w")

        ttk.Button(controls, text="Browse", command=self._browse_hdf5).grid(
            row=38, column=0, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Show HDF5 scatter",
            variable=self.show_hdf5_var,
        ).grid(row=39, column=0, sticky="w")

        ttk.Label(controls, text="Color by fit param").grid(
            row=40, column=0, sticky="w")
        self.hdf5_param_combo = ttk.Combobox(
            controls,
            textvariable=self.hdf5_param_var,
            values=[],
            state="readonly",
            width=26,
        )
        self.hdf5_param_combo.grid(row=41, column=0, sticky="w")

        ttk.Separator(controls, orient="horizontal").grid(
            row=42, column=0, pady=8, sticky="ew")

        # Plot action.
        ttk.Button(controls, text="Plot", command=self._update_plot).grid(
            row=43, column=0, sticky="ew")

        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

    def _parse_inputs(self):
        """Validate and parse user input into numeric values."""
        try:
            charge_min = float(self.charge_min_var.get())
            charge_max = float(self.charge_max_var.get())
            charge_steps = int(self.charge_steps_var.get())
            segments = int(self.segments_var.get())
            density = float(self.density_var.get())
            capacitance = float(self.capacitance_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Enter numeric values for all inputs.")
            return None

        if charge_min <= 0 or charge_max <= 0:
            messagebox.showerror("Input error", "Charge values must be positive.")
            return None
        if charge_steps <= 0:
            messagebox.showerror("Input error", "Steps must be a positive integer.")
            return None
        if segments < 1 or segments > 10:
            messagebox.showerror("Input error", "Segments must be between 1 and 10.")
            return None
        if density <= 0:
            messagebox.showerror("Input error", "Dust density must be positive.")
            return None
        if capacitance <= 0:
            messagebox.showerror(
                "Input error", "Detector capacitance must be positive."
            )
            return None
        if charge_min > charge_max:
            messagebox.showerror("Input error", "Charge min must be <= charge max.")
            return None

        return charge_min, charge_max, charge_steps, segments, density, capacitance

    def _charge_values(self, charge_min, charge_max, steps):
        """Generate log-spaced charge values for iso-lines."""
        if steps == 1 or charge_min == charge_max:
            return np.array([charge_min])
        return np.logspace(np.log10(charge_min), np.log10(charge_max), steps)

    def _browse_meta(self):
        """Open a file picker for the metadata CSV."""
        path = filedialog.askopenfilename(
            title="Select metadata CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.meta_path_var.set(path)
            self._meta_cache_path = None
            self._meta_cache_df = None

    def _browse_hdf5(self):
        """Open a file picker for the HDF5 fit file."""
        path = filedialog.askopenfilename(
            title="Select HDF5 fit file",
            filetypes=[
                ("HDF5 files", "*.h5 *.hdf *.hdf5"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.hdf5_path_var.set(path)
            self._hdf5_cache_path = None
            self._hdf5_cache_df = None
            self._load_hdf5_data()

    def _load_hdf5_data(self):
        """Load and cache HDF5 fit data for scatter overlay."""
        path = self.hdf5_path_var.get().strip()
        if not path:
            return None
        if path == self._hdf5_cache_path and self._hdf5_cache_df is not None:
            return self._hdf5_cache_df
        if not os.path.isfile(path):
            messagebox.showerror("HDF5 error", f"HDF5 file not found:\n{path}")
            return None
        try:
            df = pd.read_hdf(path, key="fits")
        except Exception:
            try:
                with pd.HDFStore(path, mode="r") as store:
                    keys = store.keys()
                    if not keys:
                        raise RuntimeError("HDF5 file contains no datasets.")
                    df = store[keys[0]]
            except Exception as exc:
                messagebox.showerror("HDF5 error", f"Failed to read HDF5:\n{exc}")
                return None
        if df is None or df.empty:
            messagebox.showerror("HDF5 error", "No rows found in the HDF5 file.")
            return None
        self._hdf5_cache_path = path
        self._hdf5_cache_df = df
        self._refresh_hdf5_param_combo(df)
        return df

    def _refresh_hdf5_param_combo(self, df):
        """Populate the fit-parameter dropdown from numeric HDF5 columns."""
        key_cols = {"dataset_id", "event", "channel", "fit_type"}
        params = []

        def _flatten_value(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                return val[0] if len(val) == 1 else np.nan
            return val

        for col in df.columns:
            if col in key_cols:
                continue
            series = df[col]
            if series.dtype == "object":
                series = series.map(_flatten_value)
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                params.append(col)

        params = sorted(params)
        self.hdf5_param_combo["values"] = params
        if not params:
            self.hdf5_param_var.set("")
        elif self.hdf5_param_var.get() not in params:
            self.hdf5_param_var.set(params[0])

    def _find_first_column(self, df, candidates):
        """Return the first candidate column found in the dataframe."""
        for name in candidates:
            if name in df.columns:
                return name
        return None

    def _coerce_numeric(self, series):
        """Coerce a Series to numeric values, flattening 1-item containers."""
        if series.dtype == "object":
            def _flatten_value(val):
                if isinstance(val, (list, tuple, np.ndarray)):
                    return val[0] if len(val) == 1 else np.nan
                return val
            series = series.map(_flatten_value)
        return pd.to_numeric(series, errors="coerce")

    def _velocity_to_kms(self, series):
        """Convert velocity to km/s using a simple magnitude heuristic."""
        vel = self._coerce_numeric(series)
        finite = vel[np.isfinite(vel)]
        if finite.empty:
            return vel
        median = float(np.nanmedian(finite.to_numpy()))
        if median > 200.0:
            return vel / 1000.0
        return vel

    def _load_meta_data(self):
        """Load and cache accelerator metadata for scatter overlay."""
        path = self.meta_path_var.get().strip()
        if not path:
            return None
        # Reuse cached metadata if the file path has not changed.
        if path == self._meta_cache_path and self._meta_cache_df is not None:
            self._apply_dust_type_density(
                self._extract_meta_dust_type(self._meta_cache_df)
            )
            return self._meta_cache_df
        # Validate the CSV path before loading.
        if not os.path.isfile(path):
            messagebox.showerror("Metadata error", f"CSV not found:\n{path}")
            return None
        # Read the metadata CSV into a dataframe.
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Metadata error", f"Failed to read CSV:\n{exc}")
            return None
        # Ensure required columns are present.
        required = {"Velocity (m/s)", "Mass (kg)"}
        if not required.issubset(df.columns):
            messagebox.showerror(
                "Metadata error",
                "CSV must include columns: 'Velocity (m/s)', 'Mass (kg)'",
            )
            return None
        self._apply_dust_type_density(self._extract_meta_dust_type(df))
        # Cache the dataframe for future calls.
        self._meta_cache_path = path
        self._meta_cache_df = df
        return df

    def _extract_meta_dust_type(self, df):
        """Return the second Dust Type entry from the metadata dataframe."""
        if "Dust Type" not in df.columns or len(df.index) < 2:
            return ""
        raw_value = df["Dust Type"].iloc[1]
        if pd.isna(raw_value):
            return ""
        return str(raw_value).strip()

    def _apply_dust_type_density(self, dust_type):
        """Update the dust type display and density input based on metadata."""
        self.dust_type_var.set(dust_type)
        if not dust_type:
            return
        density = DUST_TYPE_DENSITIES.get(dust_type.lower())
        if density is not None:
            self.density_var.set(str(density))

    def _load_scaler_table(self):
        """Load the hard-coded scaler lookup table into numpy arrays."""
        if self._scaler_data is not None:
            return self._scaler_data
        # Convert hard-coded lists into arrays and validate shape.
        charges = np.array(SCALER_CHARGES, dtype=float)
        capacitances = np.array(SCALER_CAPACITANCES, dtype=float)
        outputs = np.array(SCALER_OUTPUTS, dtype=float)
        if outputs.shape != (len(capacitances), len(charges)):
            messagebox.showerror("Scaler table error", "Scaler table shape mismatch.")
            return None
        self._scaler_data = (charges, capacitances, outputs)
        return self._scaler_data

    def _interp_extrap_1d(self, x, xp, fp):
        """Interpolate with linear extrapolation at the ends."""
        xp = np.asarray(xp, dtype=float)
        fp = np.asarray(fp, dtype=float)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.interp(x_arr, xp, fp)
        if xp.size < 2:
            return y[0] if np.isscalar(x) else y

        left = x_arr < xp[0]
        right = x_arr > xp[-1]
        if np.any(left):
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            y[left] = fp[0] + slope * (x_arr[left] - xp[0])
        if np.any(right):
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            y[right] = fp[-1] + slope * (x_arr[right] - xp[-1])
        return y[0] if np.isscalar(x) else y

    def _charge_to_scaler(self, charge_values, capacitance):
        """Convert total charge to scaler output via the lookup table."""
        scaler_data = self._load_scaler_table()
        if scaler_data is None:
            return None
        charges, capacitances, outputs = scaler_data

        # Interpolate voltage across charge at each capacitance row.
        charge_array = np.atleast_1d(np.asarray(charge_values, dtype=float))
        by_cap = np.vstack(
            [
                self._interp_extrap_1d(charge_array, charges, row)
                for row in outputs
            ]
        )
        # Interpolate across capacitance for each charge point.
        scaler_values = np.array(
            [
                self._interp_extrap_1d(capacitance, capacitances, by_cap[:, idx])
                for idx in range(by_cap.shape[1])
            ]
        )
        return scaler_values[0] if np.isscalar(charge_values) else scaler_values

    def _charge_to_cremat(self, charge_values):
        """Convert total charge to Cremat output using a fixed V/pC slope."""
        charge_array = np.atleast_1d(np.asarray(charge_values, dtype=float))
        q_pc = charge_array * 1.602e-7
        output = CREMAT_V_PER_PC * q_pc
        return output[0] if np.isscalar(charge_values) else output

    def _update_plot(self):
        """Recompute iso-lines and redraw the plot."""
        meta_df = None
        meta_path = self.meta_path_var.get().strip()
        if meta_path:
            if self.show_scatter_var.get() or os.path.isfile(meta_path):
                meta_df = self._load_meta_data()

        parsed = self._parse_inputs()
        if not parsed:
            return

        # Build inputs and mass range for the calibration curves.
        charge_min, charge_max, charge_steps, segments, density, capacitance = parsed
        charges = self._charge_values(charge_min, charge_max, charge_steps)

        r_range = np.linspace(R_SMALL_CM, R_BIG_CM, N_POINTS)
        m_range = mass(r_range, density)  # grams
        m_kg = m_range / 1000.0

        # Compute iso-lines in charge or scaler units.
        k = CALIBRATION_K[self.thickness_var.get()]
        total_charges = charges * segments
        isoline_mode = self.isoline_var.get()
        scaler_outputs = None
        if isoline_mode == "scaler":
            scaler_outputs = self._charge_to_scaler(total_charges, capacitance)
            if scaler_outputs is None:
                return
        elif isoline_mode == "cremat":
            scaler_outputs = self._charge_to_cremat(total_charges)

        # Draw iso-lines for each charge (or scaler output) value.
        self.ax.clear()
        cmap = plt.cm.viridis
        norm = None
        colors = None
        if isoline_mode in ("scaler", "cremat") and scaler_outputs is not None:
            vmin = np.nanmin(scaler_outputs)
            vmax = np.nanmax(scaler_outputs)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
                colors = [cmap(norm(val)) for val in scaler_outputs]
        if colors is None:
            colors = cmap(np.linspace(0.1, 0.9, len(total_charges)))
        for idx, (total_charge, charge) in enumerate(zip(total_charges, charges)):
            vel = np.power(total_charge / (k * np.power(m_range, 1.3)), (1 / 3))
            color = colors[idx]
            if isoline_mode in ("scaler", "cremat"):
                label = f"{scaler_outputs[idx]:.3g} V"
            else:
                label = f"{charge:.2e} e-"
            self.ax.plot(vel, m_kg, color=color, label=label)

        overlay_mass_min = None
        # Overlay accelerator data if enabled.
        if self.show_scatter_var.get():
            if meta_df is not None:
                self.ax.plot(
                    meta_df["Velocity (m/s)"] / 1000.0,
                    meta_df["Mass (kg)"],
                    ".",
                    color="black",
                    alpha=0.5,
                    label="Accelerator",
                )
                masses = meta_df["Mass (kg)"]
                masses = masses[masses > 0]
                if not masses.empty:
                    overlay_mass_min = float(masses.min())

        if self.show_hdf5_var.get():
            hdf5_df = self._load_hdf5_data()
            if hdf5_df is not None:
                mass_col = self._find_first_column(
                    hdf5_df,
                    ["mass", "Mass (kg)", "mass_kg", "m_kg"],
                )
                vel_col = self._find_first_column(
                    hdf5_df,
                    ["v", "velocity", "Velocity (m/s)", "vel", "v_ms", "v_km_s"],
                )
                if not mass_col or not vel_col:
                    messagebox.showerror(
                        "HDF5 error",
                        "HDF5 file must include mass and velocity columns (e.g., 'mass' and 'v').",
                    )
                else:
                    mass_vals = self._coerce_numeric(hdf5_df[mass_col])
                    vel_vals = self._velocity_to_kms(hdf5_df[vel_col])
                    mask = (
                        np.isfinite(mass_vals)
                        & np.isfinite(vel_vals)
                        & (mass_vals > 0)
                        & (vel_vals > 0)
                    )
                    param_col = self.hdf5_param_var.get().strip()
                    color_vals = None
                    if param_col:
                        if param_col in hdf5_df.columns:
                            color_vals = self._coerce_numeric(hdf5_df[param_col])
                            mask &= np.isfinite(color_vals)
                        else:
                            param_col = ""
                    if np.any(mask):
                        if color_vals is not None:
                            if norm is not None:
                                point_colors = cmap(norm(color_vals[mask]))
                            else:
                                cmin = np.nanmin(color_vals[mask])
                                cmax = np.nanmax(color_vals[mask])
                                if np.isfinite(cmin) and np.isfinite(cmax) and cmin != cmax:
                                    point_norm = plt.Normalize(
                                        vmin=cmin,
                                        vmax=cmax,
                                        clip=True,
                                    )
                                    point_colors = cmap(point_norm(color_vals[mask]))
                                else:
                                    point_colors = "tab:orange"
                        else:
                            point_colors = "tab:orange"
                        self.ax.scatter(
                            vel_vals[mask],
                            mass_vals[mask],
                            s=36,
                            c=point_colors,
                            alpha=0.85,
                            edgecolors="none",
                            label="HDF5",
                            zorder=4,
                        )
                        hdf5_mass_min = float(np.nanmin(mass_vals[mask]))
                        if overlay_mass_min is None:
                            overlay_mass_min = hdf5_mass_min
                        else:
                            overlay_mass_min = min(overlay_mass_min, hdf5_mass_min)

        # Apply chart styling and bounds.
        self.ax.set_xlabel("Velocity [km/s]")
        self.ax.set_ylabel("Mass [kg]")
        self.ax.set_yscale("log")
        self.ax.grid(True)
        title = (
            f"LMM Calibration, {self.thickness_var.get()} film, "
            f"{segments} segments, {density:g} g/cc"
        )
        if isoline_mode == "scaler":
            title = f"{title}, {capacitance:g} F"
        elif isoline_mode == "cremat":
            title = f"{title}, Cremat Shaper"
        self.ax.set_title(title)
        x_max = max(10.0, float(np.max(vel)) * 1.05)
        self.ax.set_xlim(0, min(20.0, x_max))
        y_min = float(np.min(m_kg))
        y_max = float(np.max(m_kg))
        y_lower = y_min / 2.0
        if overlay_mass_min is not None:
            y_lower = overlay_mass_min
        self.ax.set_ylim(y_lower, y_max * 2.0)
        self.ax.legend()

        # Refresh the canvas.
        self.canvas.draw_idle()


if __name__ == "__main__":
    # Launch the GUI application.
    app = LmmGui()
    app.mainloop()
