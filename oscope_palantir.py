"""
Oscilloscope Palantir
----------------------

Purpose:
    To provide a GUI interface to analyze ``.trc`` oscilloscope waveforms. The
    tool loads waveform events, allows interactive selection and fitting of
    pulses (QD3, QDM, CSA, etc.), applies Savitzky–Golay filtering, and exports
    figures or fit parameters for further study.
"""

import sys, os, re, csv, lmfit, matplotlib
import numpy as np, pandas as pd, scipy.signal as sig
from   scipy.signal import fftconvolve
from   scipy.optimize import curve_fit

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRegExp
from PyQt5.QtGui import QRegExpValidator, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QLabel, QCheckBox, QDialog, QInputDialog,
    QProgressDialog, QSizePolicy, QTableWidget, QTableWidgetItem, QLineEdit, QDialogButtonBox)

import qtawesome as qta

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

# My files with supporting functions
from readTrcDoner import Trc
from fitparams    import (
    FitParamsDialog,
    getMetaData,
    metaMatch,
    CSA_pulse,
    QD3Fit,
    QDMFit,
    QD3Cal,
    QDMCal,
    skew_gaussian,
    gaussian
)
from results_plotter import FitResultsVisualizer
import traceback

# Mapping of available fit functions for the dynamic row
FIT_LIB = {
    "QD3Fit": (QD3Fit, ["t0", "q", "v", "C"]),
    "QDMFit": (QDMFit, ["t0", "q", "v", "C"]),
    "CSA_pulse": (CSA_pulse, ["t0", "C0", "C1", "C2", "T0", "T1", "T2", "C"]),
    "skew_gaussian": (skew_gaussian, ["A", "xi", "omega", "alpha", "C"]),
    "gaussian": (gaussian, ["A", "mu", "sigma", "C"]),
    # Analysis-only utility: compute and plot FFT of selection in a popup
    "FFT": (None, []),
    # Special measurement: Savitzky–Golay low-pass then take max (invert -> min)
    "low_pass_max": (None, ["width"])  # width = SG window samples (odd)
}

# Set plot defaults
matplotlib.rcParams['axes.labelsize']  = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['axes.grid']       = True

class FitInfoWindow(QWidget):
    """Window showing fit results."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Fit Information")
        self.resize(800, 400)
        layout = QVBoxLayout(self)

        # Text area summarizing fits for the currently displayed event
        self.summary_box = QtWidgets.QTextEdit()
        self.summary_box.setReadOnly(True)
        layout.addWidget(self.summary_box)

        
        # --- Filter toggle row for dataset scoping ---
        toggle_row = QHBoxLayout()
        self.filter_checkbox = QCheckBox("Only current dataset")
        self.filter_checkbox.setToolTip("When checked, show only fits from the selected folder")
        self.filter_checkbox.setChecked(True)  # default ON to prevent cross-dataset bleed
        self.filter_checkbox.toggled.connect(lambda _: self._refresh())
        toggle_row.addWidget(self.filter_checkbox)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        # cache holders used by _refresh()
        self._last_results = {}
        self._last_event = None
# Table containing all fit records
        self.table = QTableWidget()
        layout.addWidget(self.table)

        
    def update_info(self, results, current_event=None):
        """Populate the table and summary text with fit information.
        - Table: shows all results (optionally filtered to current dataset via checkbox)
        - Summary: shows only rows for current_event in the current dataset
        """
        # cache for refresh
        self._last_results = results
        self._last_event = current_event
    
        
        filter_to_current = bool(getattr(self, "filter_checkbox", None) and self.filter_checkbox.isChecked())
# Optional filtering for table
        if hasattr(self, 'filter_checkbox') and self.filter_checkbox.isChecked():
            ds = None
            try:
                ds = self.parent().dataset_id
            except Exception:
                ds = None
            if ds is not None:
                results = {k: v for k, v in results.items() if isinstance(v, dict) and v.get('dataset_id') == ds}
    
        pattern = re.compile(r'^evt_(\d+)_(.+)_ch(\d+)(?:_\d+)?$')
        records = []
        summary_lines = []
        current_ds = None
        try:
            current_ds = self.parent().dataset_id
        except Exception:
            current_ds = None
    
        for key, rec in results.items():
            m = pattern.match(key)
            if not m:
                continue
            event, fit_type, ch_str = m.groups()
            ch = int(ch_str)
    
            row = {"event": event, "channel": ch, "fit_type": fit_type, "dataset_id": rec.get("dataset_id")}
            params = rec.get("params", ())
            ft = fit_type.lower()
    
            if ft.startswith("qd3"):
                # t0, q, v, C (+ derived fields if present)
                row.update({
                    "t0": params[0] if len(params) > 0 else None,
                    "q": params[1] if len(params) > 1 else None,
                    "v": params[2] if len(params) > 2 else None,
                    "C": params[3] if len(params) > 3 else None,
                })
                if "charge" in rec: row["charge"] = rec["charge"]
                if "mass" in rec: row["mass"] = rec["mass"]
                if "radius" in rec: row["radius"] = rec["radius"]
                if "impact_time" in rec: row["impact_time"] = rec["impact_time"]
            elif ft.startswith("qdm"):
                row.update({
                    "t0": params[0] if len(params) > 0 else None,
                    "q": params[1] if len(params) > 1 else None,
                    "v": params[2] if len(params) > 2 else None,
                    "C": params[3] if len(params) > 3 else None,
                })
                if "charge" in rec: row["charge"] = rec["charge"]
                if "mass" in rec: row["mass"] = rec["mass"]
                if "radius" in rec: row["radius"] = rec["radius"]
            elif ft.startswith("csa"):
                row.update({
                    "t0": params[0] if len(params) > 0 else None,
                    "C0": params[1] if len(params) > 1 else None,
                    "C1": params[2] if len(params) > 2 else None,
                    "C2": params[3] if len(params) > 3 else None,
                    "T0": params[4] if len(params) > 4 else None,
                    "T1": params[5] if len(params) > 5 else None,
                    "T2": params[6] if len(params) > 6 else None,
                    "C": params[7] if len(params) > 7 else None,
                })
            else:
                # Generic: infer names from record
                names = rec.get("param_names", [f"p{i}" for i in range(len(params))])
                for n, v in zip(names, params):
                    row[n] = v
                if "impact_time" in rec: row["impact_time"] = rec["impact_time"]
                if "radius" in rec: row["radius"] = rec["radius"]
                if "value" in rec: row["value"] = rec["value"]
                if "t_at" in rec: row["t_at"] = rec["t_at"]
    
            records.append(row)
    
            # Summary for current event - only from current dataset
            if current_event is not None and str(event) == str(current_event) and ((current_ds is None and row.get("dataset_id") in (None, current_ds)) or (row.get("dataset_id") == current_ds)):
                # Make a compact line excluding index columns
                exclude = {"event", "channel", "fit_type", "dataset_id"}
                param_items = [f"{k}={self._fmt(v)}" for k, v in row.items() if k not in exclude]
                summary_lines.append(f"{fit_type} ch{ch}: " + ", ".join(param_items))
    
        # Populate table
        if not records:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
        else:
            columns = sorted({col for r in records for col in r.keys()})
            self.table.setColumnCount(len(columns))
            self.table.setHorizontalHeaderLabels(columns)
            self.table.setRowCount(len(records))
    
            for r, rec in enumerate(records):
                for c, col in enumerate(columns):
                    item = QTableWidgetItem(self._fmt(rec.get(col, "")))
                    self.table.setItem(r, c, item)
    
        text = "\n".join(summary_lines) if summary_lines else "No fits for this event."
        self.summary_box.setPlainText(text)
    def _refresh(self):
        """Re-apply the current filter to the table and summary."""
        self.update_info(getattr(self, "_last_results", {}), getattr(self, "_last_event", None))

    
    def _fmt(self, val):
        import numbers, math
        if isinstance(val, numbers.Real) and not isinstance(val, bool):
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return ""
            return f"{val:.3e}"
        return str(val)



class SciFitParamsDialog(QDialog):
    """
    Parameter editor that displays and edits values in scientific notation.
    Emits params_changed(dict) live while editing.
    """
    params_changed = pyqtSignal(dict)

    def __init__(self, params_dict, func, t_sel, y_sel, names, bounds=None, parent=None, precision=8):
        super().__init__(parent)
        self.setWindowTitle("Adjust Parameters (Scientific Notation)")
        self._names = list(names)
        self._precision = int(precision)
        self._func = func
        self._bounds = bounds
        self._editors = {}
        self._last_valid = dict(params_dict)

        main = QVBoxLayout(self)
        form = QGridLayout()
        main.addLayout(form)

        # Scientific number regex: optional sign, digits (with optional decimal), optional exponent
        sci_rx = QRegExp(r'^[\+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][\+\-]?\d+)?$')
        validator = QRegExpValidator(sci_rx, self)

        # Build a row per parameter
        for row, name in enumerate(self._names):
            val = float(params_dict.get(name, 0.0))
            lbl = QLabel(name)
            le = QLineEdit(f"{val:.6e}") #% self._precision)
            le.setObjectName(name)
            le.setAlignment(Qt.AlignRight)
            le.setValidator(validator)
            le.setToolTip("Enter a number (scientific notation OK), e.g., 1.23e-6")

            # Bounds hint
            hint = ""
            if bounds and len(bounds) == 2:
                lo, hi = bounds
                try:
                    lo_v = lo[self._names.index(name)]
                    hi_v = hi[self._names.index(name)]
                    lo_t = "-∞" if not (isinstance(lo_v, (int,float)) and lo_v == lo_v) else f"{lo_v:.%de}" % self._precision
                    hi_t = "∞"  if not (isinstance(hi_v, (int,float)) and hi_v == hi_v) else f"{hi_v:.%de}" % self._precision
                    hint = f"[{lo_t}, {hi_t}]"
                except Exception:
                    hint = ""

            hint_lbl = QLabel(hint)
            hint_lbl.setAlignment(Qt.AlignLeft)

            form.addWidget(lbl, row, 0)
            form.addWidget(le,  row, 1)
            form.addWidget(hint_lbl, row, 2)
            self._editors[name] = le

            # Live update and consistent formatting on finish
            le.textChanged.connect(self._maybe_emit)
            le.editingFinished.connect(lambda n=name: self._reformat(n))

        # Buttons 
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)
        # --- Extra buttons (match v2 behavior) ---
        self.btn_plot  = QPushButton("Plot")
        self.btn_refit = QPushButton("Refit")
        #main.addWidget(self.btn_plot)
        main.addWidget(self.btn_refit)

        # Clicking Plot just emits current params (useful if validation passed)
        self.btn_plot.clicked.connect(self._maybe_emit)

        # Refit: run a quick curve_fit from current values and update fields
        def _do_refit():
            import numpy as _np
            from scipy.optimize import curve_fit as _curve_fit
            parsed = self._parse_all()
            if parsed is None:
                QMessageBox.warning(self, "Invalid input", "Please correct parameter values before refitting.")
                return
            p0 = [parsed[n] for n in self._names]
            try:
                if self._bounds is not None:
                    popt, _ = _curve_fit(self._func, _np.array(t_sel), _np.array(y_sel), p0=p0, bounds=self._bounds, maxfev=20000)
                else:
                    popt, _ = _curve_fit(self._func, _np.array(t_sel), _np.array(y_sel), p0=p0, maxfev=20000)
            except Exception as e:
                QMessageBox.warning(self, "Refit failed", str(e))
                return
            # Update editors and emit
            for n, v in zip(self._names, popt):
                le = self._editors.get(n)
                if le is not None:
                    le.blockSignals(True)
                    try:
                        le.setText(f"{float(v):.{self._precision}e}")
                    finally:
                        le.blockSignals(False)
            self._maybe_emit()
        self.btn_refit.clicked.connect(_do_refit)
    
        # Initial emit
        self._maybe_emit()

    def _parse_all(self):
        out = {}
        for name, le in self._editors.items():
            txt = le.text().strip()
            try:
                out[name] = float(txt)
            except Exception:
                return None
        return out

    def _reformat(self, name):
        le = self._editors.get(name)
        if le is None:
            return
        try:
            v = float(le.text().strip())
            le.blockSignals(True)
            le.setText(f"{v:.%de}" % self._precision)
            le.blockSignals(False)
        except Exception:
            pass

    def _maybe_emit(self):
        parsed = self._parse_all()
        if parsed is None:
            return  # do not emit until valid
        self._last_valid = parsed
        self.params_changed.emit(parsed)

    def _on_accept(self):
        parsed = self._parse_all()
        if parsed is None:
            QMessageBox.warning(self, "Invalid input", "Please correct parameter values.")
            return
        self._last_valid = parsed
        self.accept()

    def getParams(self):
        return dict(self._last_valid)


class BatchFitDialog(QDialog):
    """Collects event range and time window for batch fitting."""
    def __init__(self, parent, evt_min, evt_max, tmin=None, tmax=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Fit Configuration")
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        layout.addLayout(grid)

        # Event range inputs
        grid.addWidget(QLabel("Start Event"), 0, 0)
        self.start_evt = QLineEdit(str(evt_min))
        self.start_evt.setToolTip("First event number to include")
        grid.addWidget(self.start_evt, 0, 1)

        grid.addWidget(QLabel("End Event"), 1, 0)
        self.end_evt = QLineEdit(str(evt_max))
        self.end_evt.setToolTip("Last event number to include")
        grid.addWidget(self.end_evt, 1, 1)

        # Time window inputs (seconds)
        grid.addWidget(QLabel("Time Start [s]"), 2, 0)
        self.t_start = QLineEdit("" if tmin is None else f"{float(tmin):.9f}")
        self.t_start.setToolTip("Lower bound of fit window in seconds")
        grid.addWidget(self.t_start, 2, 1)

        grid.addWidget(QLabel("Time End [s]"), 3, 0)
        self.t_end = QLineEdit("" if tmax is None else f"{float(tmax):.9f}")
        self.t_end.setToolTip("Upper bound of fit window in seconds")
        grid.addWidget(self.t_end, 3, 1)

        note = QLabel("Uses the currently selected Fit + Channel + Invert.")
        layout.addWidget(note)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self):
        try:
            ev0 = int(self.start_evt.text().strip())
            ev1 = int(self.end_evt.text().strip())
        except Exception:
            raise ValueError("Invalid event range")
        try:
            t0 = float(self.t_start.text().strip())
            t1 = float(self.t_end.text().strip())
        except Exception:
            raise ValueError("Invalid time window")
        if t1 <= t0:
            raise ValueError("Time End must be > Time Start")
        if ev1 < ev0:
            raise ValueError("End Event must be >= Start Event")
        return ev0, ev1, t0, t1


class FeatureScanDialog(QDialog):
    """Collects event range, channel, and threshold for feature scanning."""
    def __init__(self, parent, evt_min, evt_max, ch_default=1, thr_default=5.0):
        super().__init__(parent)
        self.setWindowTitle("Feature Scan Configuration")
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        layout.addLayout(grid)

        grid.addWidget(QLabel("Start Event"), 0, 0)
        self.start_evt = QLineEdit(str(evt_min))
        grid.addWidget(self.start_evt, 0, 1)

        grid.addWidget(QLabel("End Event"), 1, 0)
        self.end_evt = QLineEdit(str(evt_max))
        grid.addWidget(self.end_evt, 1, 1)

        grid.addWidget(QLabel("Channel"), 2, 0)
        self.ch_combo = QComboBox()
        self.ch_combo.addItems([str(i) for i in range(1, 5)])
        self.ch_combo.setCurrentText(str(ch_default))
        grid.addWidget(self.ch_combo, 2, 1)

        grid.addWidget(QLabel("Threshold × std"), 3, 0)
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.1, 1e3)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setSingleStep(0.5)
        self.thr_spin.setValue(float(thr_default))
        grid.addWidget(self.thr_spin, 3, 1)

        self.abs_chk = QCheckBox("Use absolute value")
        self.abs_chk.setChecked(True)
        layout.addWidget(self.abs_chk)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self):
        try:
            ev0 = int(self.start_evt.text().strip())
            ev1 = int(self.end_evt.text().strip())
        except Exception:
            raise ValueError("Invalid event range")
        if ev1 < ev0:
            raise ValueError("End Event must be >= Start Event")
        ch = int(self.ch_combo.currentText())
        thr = float(self.thr_spin.value())
        use_abs = bool(self.abs_chk.isChecked())
        return ev0, ev1, ch, thr, use_abs


class FeatureResultsDialog(QDialog):
    """Shows a list of matched events; allows jumping to selection."""
    def __init__(self, parent, matches):
        super().__init__(parent)
        self.setWindowTitle("Feature Scan Results")
        self._parent = parent
        self._matches = list(matches)
        self._action = None  # 'apply', 'rescan', or None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Matches: {len(matches)}"))

        self.list_widget = QtWidgets.QListWidget()
        for ev in matches:
            self.list_widget.addItem(ev)
        layout.addWidget(self.list_widget)

        btns = QDialogButtonBox(parent=self)
        self.btn_apply = QPushButton("Apply Filter")
        self.btn_rescan = QPushButton("Rescan…")
        self.btn_goto = QPushButton("Go to Selected")
        self.btn_close = QPushButton("Close")
        btns.addButton(self.btn_apply, QDialogButtonBox.AcceptRole)
        btns.addButton(self.btn_rescan, QDialogButtonBox.ActionRole)
        btns.addButton(self.btn_goto, QDialogButtonBox.ActionRole)
        btns.addButton(self.btn_close, QDialogButtonBox.RejectRole)
        layout.addWidget(btns)

        self.btn_goto.clicked.connect(self._goto)
        self.list_widget.itemDoubleClicked.connect(lambda _: self._goto())
        self.btn_apply.clicked.connect(self._apply)
        self.btn_rescan.clicked.connect(self._rescan)
        self.btn_close.clicked.connect(self.reject)

    def _goto(self):
        it = self.list_widget.currentItem()
        if not it:
            return
        key = it.text()
        # Use parent's switching method
        try:
            self._parent._switch_to_event(key)
        except Exception:
            pass
        self.accept()

    def _apply(self):
        self._action = 'apply'
        self.accept()

    def _rescan(self):
        self._action = 'rescan'
        self.accept()

    def action(self):
        return self._action

    def matches(self):
        return list(self._matches)

class FitFilterDialog(QDialog):
    """Dialog to configure filtering events based on stored fit parameters."""
    def __init__(self, parent, fit_types, ch_default=1):
        super().__init__(parent)
        self.setWindowTitle("Fit Parameter Filter")
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        layout.addLayout(grid)

        # Fit type
        grid.addWidget(QLabel("Fit Type"), 0, 0)
        self.fit_combo = QComboBox()
        self.fit_combo.addItems(list(fit_types))
        grid.addWidget(self.fit_combo, 0, 1)

        # Channel
        grid.addWidget(QLabel("Channel"), 1, 0)
        self.ch_combo = QComboBox()
        self.ch_combo.addItems([str(i) for i in range(1, 5)])
        self.ch_combo.setCurrentText(str(ch_default))
        grid.addWidget(self.ch_combo, 1, 1)

        # Parameter
        grid.addWidget(QLabel("Parameter"), 2, 0)
        self.param_combo = QComboBox()
        grid.addWidget(self.param_combo, 2, 1)

        # Operator
        grid.addWidget(QLabel("Operator"), 3, 0)
        self.op_combo = QComboBox()
        self.op_combo.addItems([">", ">=", "<", "<=", "==", "between"])
        grid.addWidget(self.op_combo, 3, 1)

        # Threshold(s)
        sci_rx = QRegExp(r'^[\+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][\+\-]?\d+)?$')
        self._validator = QRegExpValidator(sci_rx, self)

        grid.addWidget(QLabel("Value"), 4, 0)
        self.thr1_edit = QLineEdit("1.0e3")
        self.thr1_edit.setValidator(self._validator)
        grid.addWidget(self.thr1_edit, 4, 1)

        self.thr2_row_lbl = QLabel("and")
        self.thr2_edit = QLineEdit("1.0e6")
        self.thr2_edit.setValidator(self._validator)
        grid.addWidget(self.thr2_row_lbl, 5, 0)
        grid.addWidget(self.thr2_edit, 5, 1)

        # Update parameter list when fit type changes, and show/hide second threshold
        def refresh_params():
            fit_name = self.fit_combo.currentText()
            # Start with known params from FIT_LIB
            base = []
            if fit_name in FIT_LIB:
                base = list(FIT_LIB[fit_name][1])
            # Add known derived fields that may be present in records
            derived = ["charge", "mass", "radius", "impact_time"]
            self.param_combo.clear()
            self.param_combo.addItems(base + derived)
        def refresh_op():
            is_between = (self.op_combo.currentText() == "between")
            self.thr2_row_lbl.setVisible(is_between)
            self.thr2_edit.setVisible(is_between)
        self.fit_combo.currentTextChanged.connect(refresh_params)
        self.op_combo.currentTextChanged.connect(refresh_op)
        refresh_params()
        refresh_op()

        note = QLabel("Filter to events where the chosen fit on the selected channel has a parameter matching the criterion.")
        layout.addWidget(note)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self):
        fit_name = self.fit_combo.currentText().strip()
        ch = int(self.ch_combo.currentText())
        param = self.param_combo.currentText().strip()
        op = self.op_combo.currentText()
        try:
            v1 = float(self.thr1_edit.text().strip())
        except Exception:
            raise ValueError("Invalid numeric value")
        v2 = None
        if op == "between":
            try:
                v2 = float(self.thr2_edit.text().strip())
            except Exception:
                raise ValueError("Invalid upper bound value")
            if v2 < v1:
                raise ValueError("Upper bound must be >= lower bound")
        return fit_name, ch, param, op, v1, v2

class FitWorker(QThread):
    """Background thread running curve_fit with cancellation support."""

    result = pyqtSignal(object, object)

    def __init__(self, func, t_sel, y_sel, p0, bounds=None, parent=None):
        super().__init__(parent)
        self.func = func
        self.t_sel = t_sel
        self.y_sel = y_sel
        self.p0 = p0
        self.bounds = bounds
        self.cancelled = False

    def run(self):
        def wrapped(tt, *pp):
            if self.cancelled:
                raise RuntimeError("Fit cancelled")
            return self.func(tt, *pp)

        try:
            if self.bounds is not None:
                popt, _ = curve_fit(wrapped, self.t_sel, self.y_sel, p0=self.p0,
                                    bounds=self.bounds, maxfev=20000)
            else:
                popt, _ = curve_fit(wrapped, self.t_sel, self.y_sel, p0=self.p0,
                                    maxfev=20000)
            self.result.emit(popt, None)
        except Exception as e:
            self.result.emit(None, e)


class OscilloscopeAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset_id = None  # current source folder identifier
        self._all_event_keys = []
        self.feature_filter_active = False
        self.feature_filter_keys = []

        self.setWindowTitle("Oscilloscope Waveform Analyzer")
        self.setGeometry(100, 100, 1500, 900)

        self.accent = "#00aaff"
        self.dark_mode = True
        self.unit_width = 130

        self.trc             = Trc()
        self.event_files     = {}
        self.current_data    = {}
        self.results         = {}
        self.fit_regions     = {}
        self.span_selectors  = {}
        self.sg_filtered     = {}
        self.metaArr         = None
        self._child_windows  = []  # keep references to child windows (e.g., Results Plotter)

        def wrap_control(label_text, control):
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(QLabel(label_text))
            lay.addWidget(control)
            w.setFixedWidth(self.unit_width)
            # Keep compact row height for wrapped controls
            w.setMaximumHeight(24)
            return w

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(4)
        # Secondary window for displaying fit results
        self.fit_info_window = FitInfoWindow(self)

        ####################
        # Initialize Controls
        # This creates a row (horizontal box) to put buttons and such into
        ####################
        ctrl_layout = QHBoxLayout()
        # This adds that box to the overall layout, can have multiple such boxes...
        layout.addLayout(ctrl_layout)
        layout.setAlignment(ctrl_layout, Qt.AlignLeft)

        # Folder selection for trc folder
        btn_folder = QPushButton("Select Folder (F)")
        btn_folder.setIcon(qta.icon('fa5s.folder-open', color=self.accent))
        btn_folder.setToolTip("Choose folder containing TRC files (F)")
        btn_folder.clicked.connect(self.select_folder)
        btn_folder.setFixedWidth(self.unit_width)
        ctrl_layout.addWidget(btn_folder)

        self.event_combo = QComboBox()
        self.event_combo.currentIndexChanged.connect(self.load_event)
        self.event_combo.setFixedWidth(self.unit_width)
        ctrl_layout.addWidget(self.event_combo)

        # Label to display the currently selected source folder
        self.folder_label = QLabel("Select Folder To Continue")
        self.folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.folder_label.setToolTip("")
        self.folder_label.setMaximumHeight(20)
        ctrl_layout.addWidget(self.folder_label)

        # Push following buttons to the far right
        ctrl_layout.addStretch()

        # Help Button
        btn_help = QPushButton("(H)elp")
        btn_help.setIcon(qta.icon('fa5s.question-circle', color=self.accent))
        btn_help.setToolTip("Open help dialog describing UI and functionality (H)")
        btn_help.clicked.connect(self.show_help)
        btn_help.setFixedWidth(self.unit_width)
        ctrl_layout.addWidget(btn_help)

        # Theme toggle
        self.btn_theme = QPushButton("Light Mode")
        self.btn_theme.setIcon(qta.icon('fa5s.sun', color=self.accent))
        self.btn_theme.setToolTip("Toggle light/dark theme (Ctrl+T)")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_theme.setFixedWidth(self.unit_width)
        ctrl_layout.addWidget(self.btn_theme)

        # Row: Navigation / Plotting (right-aligned session controls)
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)
        layout.addLayout(nav_layout)
        layout.setAlignment(nav_layout, Qt.AlignLeft)

        # Button to iterate to prev trc files.
        self.btn_prev = QPushButton("Prev (<)")
        self.btn_prev.setIcon(qta.icon('fa5s.arrow-left', color=self.accent))
        self.btn_prev.setToolTip("Load previous event (<)")
        self.btn_prev.clicked.connect(self.prev_event)
        self.btn_prev.setFixedWidth(self.unit_width)
        self.btn_prev.setMaximumHeight(24)
        nav_layout.addWidget(self.btn_prev)

        # Button to iterate to next trc files.
        self.btn_next = QPushButton("Next (>)")
        self.btn_next.setIcon(qta.icon('fa5s.arrow-right', color=self.accent))
        self.btn_next.setToolTip("Load next event (>)")
        self.btn_next.clicked.connect(self.next_event)
        self.btn_next.setFixedWidth(self.unit_width)
        self.btn_next.setMaximumHeight(24)
        nav_layout.addWidget(self.btn_next)

        # Decimation factor for plotting
        self.decim_spin = QSpinBox()
        self.decim_spin.setRange(1, 100)
        self.decim_spin.setValue(2)
        self.decim_spin.setToolTip("Plot every Nth sample; 1=all data, 100=1% of points")
        self.decim_spin.valueChanged.connect(self._on_decim_change)
        self.decim_spin.setFixedWidth(60)
        self.decim_spin.setMaximumHeight(24)
        nav_layout.addWidget(wrap_control("Decim:", self.decim_spin))
        nav_layout.addStretch()

        # Session I/O controls (right side of the same row)
        self.btn_clear = QPushButton("Clear Event Fits")
        self.btn_clear.setIcon(qta.icon('fa5s.eraser', color=self.accent))
        self.btn_clear.setToolTip("Remove all fits on this event (E)")
        self.btn_clear.clicked.connect(self.clear_fits)
        self.btn_clear.setFixedWidth(self.unit_width)
        self.btn_clear.setMaximumHeight(24)
        nav_layout.addWidget(self.btn_clear)

        self.btn_save_fig = QPushButton("Save Figure")
        self.btn_save_fig.setIcon(qta.icon('fa5s.save', color=self.accent))
        self.btn_save_fig.setToolTip("Save current plot as an image (Ctrl+S)")
        self.btn_save_fig.clicked.connect(self.save_figure_transparent)
        self.btn_save_fig.setFixedWidth(self.unit_width)
        self.btn_save_fig.setMaximumHeight(24)
        nav_layout.addWidget(self.btn_save_fig)

        self.btn_export = QPushButton("Export Fits")
        self.btn_export.setIcon(qta.icon('fa5s.file-export', color=self.accent))
        self.btn_export.setToolTip("Export all fits to an HDF5 file (Ctrl+E)")
        self.btn_export.clicked.connect(self.export_fits)
        self.btn_export.setFixedWidth(self.unit_width)
        self.btn_export.setMaximumHeight(24)
        nav_layout.addWidget(self.btn_export)

        self.btn_import = QPushButton("Load Fits")
        self.btn_import.setIcon(qta.icon('fa5s.file-import', color=self.accent))
        self.btn_import.setToolTip("Load fits from an HDF5 file and merge into session (Ctrl+I)")
        self.btn_import.clicked.connect(self.import_fits)
        self.btn_import.setFixedWidth(self.unit_width)
        self.btn_import.setMaximumHeight(24)
        nav_layout.addWidget(self.btn_import)

        self.btn_clear_data = QPushButton("Clear Data")
        self.btn_clear_data.setIcon(qta.icon('fa5s.trash', color=self.accent))
        self.btn_clear_data.setToolTip("Remove loaded data and ALL fits (Shift+D)")
        self.btn_clear_data.clicked.connect(self.clear_data)
        self.btn_clear_data.setFixedWidth(self.unit_width)
        self.btn_clear_data.setMaximumHeight(24)
        nav_layout.addWidget(self.btn_clear_data)

        self.btn_show_info = QPushButton("Fit Info (U)")
        self.btn_show_info.setIcon(qta.icon('fa5s.info-circle', color=self.accent))
        self.btn_show_info.setToolTip("Show table of all fit results (U)")
        self.btn_show_info.clicked.connect(self.show_fit_info)
        self.btn_show_info.setFixedWidth(self.unit_width)
        self.btn_show_info.setMaximumHeight(24)
        nav_layout.addWidget(self.btn_show_info)

        # Row: Metadata / Impact
        meta_layout = QHBoxLayout()
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(6)
        layout.addLayout(meta_layout)
        layout.setAlignment(meta_layout, Qt.AlignLeft)

        # Button to find and load metadata file
        self.btn_load_meta = QPushButton("(L)oad Metadata")
        self.btn_load_meta.setIcon(qta.icon('fa5s.file-import', color=self.accent))
        self.btn_load_meta.setToolTip("Load metadata file (L)")
        self.btn_load_meta.clicked.connect(self.load_metadata)
        self.btn_load_meta.setFixedWidth(self.unit_width)
        meta_layout.addWidget(self.btn_load_meta)

        # Run the metamatch routine....
        self.btn_meta = QPushButton("(M)etaMatch")
        self.btn_meta.setIcon(qta.icon('fa5s.search', color=self.accent))
        self.btn_meta.setToolTip("Match event to metadata (M). Requires QD3 fit.")
        self.btn_meta.clicked.connect(self.run_meta_match)
        self.btn_meta.setFixedWidth(self.unit_width)
        meta_layout.addWidget(self.btn_meta)

        # Distance spin box for impact_time
        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0.0, 10.0)
        self.dist_spin.setDecimals(3)
        self.dist_spin.setSingleStep(0.001)
        self.dist_spin.setValue(1.51)                # Det3 to Center LEIL m (Defaults ~ 1.82)
        self.dist_spin.setFixedWidth(60)
        dist_widget = wrap_control("D3Dist:", self.dist_spin)
        dist_widget.findChild(QLabel).setToolTip("Det3 to Target distance (m)")
        meta_layout.addWidget(dist_widget)

        # Button to mark impact time using current QD3/QDM fit and distance
        self.btn_mark_impact = QPushButton("Mark Impact (T)")
        self.btn_mark_impact.setIcon(qta.icon('fa5s.location-arrow', color=self.accent))
        self.btn_mark_impact.setToolTip("Plot vertical line at t0 + distance/velocity on all channels (T)")
        self.btn_mark_impact.clicked.connect(self.mark_impact_line)
        self.btn_mark_impact.setFixedWidth(self.unit_width)
        meta_layout.addWidget(self.btn_mark_impact)

        # Display metadata match output
        self.meta_label = QLabel("MetaMatch: ")
        self.meta_label.setWordWrap(True)
        self.meta_label.setFixedHeight(20)
        self.meta_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        meta_layout.addWidget(self.meta_label)

        # Row: Session I/O moved to nav row (right-aligned)


        ####################
        # Button row for SG filtering
        ####################
        sg_layout = QHBoxLayout()
        sg_layout.setContentsMargins(0, 0, 0, 0)
        sg_layout.setSpacing(6)
        layout.addLayout(sg_layout)
        layout.setAlignment(sg_layout, Qt.AlignLeft)

        # Savitzky-Golay
        self.btn_sg = QPushButton("(S)G Filter")
        self.btn_sg.setIcon(qta.icon('fa5s.filter', color=self.accent))
        self.btn_sg.setToolTip("Apply Savitzky-Golay filter (S)")
        self.btn_sg.clicked.connect(self.run_sg_filter)
        self.btn_sg.setFixedWidth(self.unit_width)
        self.btn_sg.setMaximumHeight(24)
        sg_layout.addWidget(self.btn_sg)

        #Channel selector label
        #channel selector
        self.sg_ch = QComboBox()
        self.sg_ch.addItems([str(i) for i in range(1,5)])
        self.sg_ch.setCurrentText("1")
        self.sg_ch.setMaximumHeight(24)
        self.sg_ch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sg_layout.addWidget(wrap_control("Chan:", self.sg_ch))

        # Sample window label
        self.sg_win = QSpinBox()
        self.sg_win.setRange(10,9999)
        self.sg_win.setSingleStep(2)
        self.sg_win.setValue(200)
        self.sg_win.setFixedWidth(60)
        self.sg_win.setMaximumHeight(24)
        sg_layout.addWidget(wrap_control("Width:", self.sg_win))

        # Button for batch csa
        self.btn_batch_sg = QPushButton("Batch Run")
        self.btn_batch_sg.setIcon(qta.icon('fa5s.play', color=self.accent))
        self.btn_batch_sg.setToolTip("Run SG filter on all events (Ctrl+B)")
        self.btn_batch_sg.setFixedWidth(self.unit_width)
        self.btn_batch_sg.setMaximumHeight(24)
        sg_layout.addWidget(self.btn_batch_sg)

        # Button to clear
        self.btn_clear_sg = QPushButton("Clear Fit")
        self.btn_clear_sg.setIcon(qta.icon('fa5s.times', color=self.accent))
        self.btn_clear_sg.setToolTip("Remove SG filter overlay (Shift+S)")
        self.btn_clear_sg.clicked.connect(self.clear_sg_filter)
        self.btn_clear_sg.setFixedWidth(self.unit_width)
        self.btn_clear_sg.setMaximumHeight(24)
        sg_layout.addWidget(self.btn_clear_sg)

        # Right-aligned scanning/filtering controls on same row
        sg_layout.addStretch()

        self.btn_results_plotter = QPushButton("Results Plotter")
        self.btn_results_plotter.setIcon(qta.icon('fa5s.chart-line', color=self.accent))
        self.btn_results_plotter.setToolTip("Open Results Plotter using current in-memory fits (P)")
        self.btn_results_plotter.clicked.connect(self.open_results_plotter)
        self.btn_results_plotter.setFixedWidth(self.unit_width)
        self.btn_results_plotter.setMaximumHeight(24)
        sg_layout.addWidget(self.btn_results_plotter)

        self.btn_feature_scan = QPushButton("Feature Scan")
        self.btn_feature_scan.setIcon(qta.icon('fa5s.search-plus', color=self.accent))
        self.btn_feature_scan.setToolTip("Scan events for large excursions vs. stddev (Ctrl+F)")
        self.btn_feature_scan.setFixedWidth(self.unit_width)
        self.btn_feature_scan.setMaximumHeight(24)
        self.btn_feature_scan.clicked.connect(self.run_feature_scan)
        sg_layout.addWidget(self.btn_feature_scan)

        self.btn_fit_filter = QPushButton("Fit Filter")
        self.btn_fit_filter.setIcon(qta.icon('fa5s.filter', color=self.accent))
        self.btn_fit_filter.setToolTip("Filter events where a fit parameter meets a criterion (Ctrl+Shift+F)")
        self.btn_fit_filter.setFixedWidth(self.unit_width)
        self.btn_fit_filter.setMaximumHeight(24)
        self.btn_fit_filter.clicked.connect(self.run_fit_filter)
        sg_layout.addWidget(self.btn_fit_filter)

        self.btn_clear_filter = QPushButton("Clear Filter")
        self.btn_clear_filter.setIcon(qta.icon('fa5s.undo', color=self.accent))
        self.btn_clear_filter.setToolTip("Restore the full event list (Ctrl+K)")
        self.btn_clear_filter.setFixedWidth(self.unit_width)
        self.btn_clear_filter.setMaximumHeight(24)
        self.btn_clear_filter.clicked.connect(self.clear_event_filter)
        sg_layout.addWidget(self.btn_clear_filter)

        ####################
        # Dynamic fit row
        ####################
        ctrl_layout5 = QHBoxLayout()
        layout.addLayout(ctrl_layout5)
        layout.setAlignment(ctrl_layout5, Qt.AlignLeft)

        self.dyn_func_combo = QComboBox()
        self.dyn_func_combo.addItems(sorted(FIT_LIB.keys()))
        self.dyn_func_combo.setToolTip("Q=QD3Fit, D=QDMFit, C=CSA_pulse, W=skew_gaussian, G=gaussian, Shift+F=FFT, X=low_pass_max")
        self.dyn_func_combo.setFixedWidth(self.unit_width+2)
        self.dyn_func_combo.setMaximumHeight(24)
        ctrl_layout5.addWidget(self.dyn_func_combo)

        # Channel selection then optional invert
        self.dyn_ch_combo = QComboBox()
        self.dyn_ch_combo.addItems([str(i) for i in range(1,5)])
        self.dyn_ch_combo.setCurrentText("1")
        self.dyn_ch_combo.setMaximumHeight(24)
        self.dyn_ch_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ctrl_layout5.addWidget(wrap_control("Chan:", self.dyn_ch_combo))
        # Small gap between channel and invert
        ctrl_layout5.addSpacing(2)

        self.dyn_invert = QCheckBox(":(I)nvert")
        self.dyn_invert.setFixedWidth(self.unit_width+2)
        self.dyn_invert.setMaximumHeight(24)
        ctrl_layout5.addWidget(self.dyn_invert)

        # Primary fit actions
        self.btn_dyn = QPushButton("Run Fit")
        self.btn_dyn.setIcon(qta.icon('fa5s.level-down-alt', color=self.accent))
        self.btn_dyn.setToolTip("Execute selected fit (Return)")
        self.btn_dyn.clicked.connect(self.run_dynamic_fit)
        self.btn_dyn.setFixedWidth(self.unit_width)
        self.btn_dyn.setMaximumHeight(24)
        ctrl_layout5.addWidget(self.btn_dyn)

        self.btn_adjust_dyn = QPushButton("(A)djust Fit")
        self.btn_adjust_dyn.setIcon(qta.icon('fa5s.sliders-h', color=self.accent))
        self.btn_adjust_dyn.setToolTip("Edit initial parameters (A)")
        self.btn_adjust_dyn.clicked.connect(self.adjust_dynamic_params)
        self.btn_adjust_dyn.setFixedWidth(self.unit_width)
        self.btn_adjust_dyn.setMaximumHeight(24)
        ctrl_layout5.addWidget(self.btn_adjust_dyn)

        self.btn_clear_dyn = QPushButton("Clea(r) Fit")
        self.btn_clear_dyn.setIcon(qta.icon('fa5s.times', color=self.accent))
        self.btn_clear_dyn.setToolTip("Remove dynamic fit (R)")
        self.btn_clear_dyn.clicked.connect(self.clear_dynamic_fit)
        self.btn_clear_dyn.setFixedWidth(self.unit_width)
        self.btn_clear_dyn.setMaximumHeight(24)
        ctrl_layout5.addWidget(self.btn_clear_dyn)

        # Clear all fits for selected channel on current event
        self.btn_clear_chan = QPushButton("Clear Chan Fits")
        self.btn_clear_chan.setIcon(qta.icon('fa5s.eraser', color=self.accent))
        self.btn_clear_chan.setToolTip("Remove all fits for selected channel on this event (Ctrl+R)")
        self.btn_clear_chan.clicked.connect(self.clear_channel_fits)
        self.btn_clear_chan.setFixedWidth(self.unit_width)
        self.btn_clear_chan.setMaximumHeight(24)
        ctrl_layout5.addWidget(self.btn_clear_chan)

        # Batch run across multiple events for the selected fit/channel
        self.btn_batch_dyn = QPushButton("Batch Run")
        self.btn_batch_dyn.setIcon(qta.icon('fa5s.play', color=self.accent))
        self.btn_batch_dyn.setToolTip("Run selected fit over a time window across a range of events (B)")
        self.btn_batch_dyn.clicked.connect(self.run_batch_fit)
        self.btn_batch_dyn.setFixedWidth(self.unit_width)
        self.btn_batch_dyn.setMaximumHeight(24)
        ctrl_layout5.addWidget(self.btn_batch_dyn)

        # Optional: use SG-filtered data if present when running fits/FFT (moved to end)
        self.use_sg_toggle = QCheckBox("Use SG")
        self.use_sg_toggle.setToolTip("Use SG-filtered data if available for fits/FFT (Ctrl+Shift+S)")
        self.use_sg_toggle.setFixedWidth(self.unit_width+2)
        self.use_sg_toggle.setMaximumHeight(24)
        ctrl_layout5.addWidget(self.use_sg_toggle)

        ctrl_layout5.addStretch()

        ####################
        # Finalize init.
        ####################
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Disable controls until a valid folder is selected
        self.set_controls_enabled(False)

        # Apply initial theme
        self.apply_theme()

        # Global shortcuts so hotkeys work even when dropdowns have focus
        self._install_global_shortcuts()
        self._shortcuts_installed = True

    def _switch_to_event(self, key):
        """Programmatically select an event by its key string, if present."""
        if not key:
            return
        idx = self.event_combo.findText(key)
        if idx != -1:
            self.event_combo.setCurrentIndex(idx)

    def set_controls_enabled(self, enabled):
        widgets = [
            self.event_combo, self.btn_prev, self.btn_next, self.decim_spin,
            self.btn_load_meta, self.btn_meta, self.dist_spin, self.btn_mark_impact,
            self.btn_clear, self.btn_save_fig, self.btn_export, self.btn_import,
            self.btn_clear_data, self.btn_show_info,
            self.btn_sg, self.sg_ch, self.sg_win,
            self.btn_batch_sg, self.btn_clear_sg, self.btn_results_plotter, self.btn_feature_scan, self.btn_fit_filter, self.btn_clear_filter,
            self.dyn_func_combo, self.btn_dyn, self.btn_batch_dyn, self.dyn_ch_combo,
            self.dyn_invert, self.use_sg_toggle, self.btn_adjust_dyn, self.btn_clear_dyn, self.btn_clear_chan
        ]
        for w in widgets:
            w.setEnabled(enabled)

    def toggle_theme(self):
        """Switch between dark and light themes."""
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        if self.dark_mode:
            self.btn_theme.setText("Light Mode")
            self.btn_theme.setIcon(qta.icon('fa5s.sun', color=self.accent))
        else:
            self.btn_theme.setText("Dark Mode")
            self.btn_theme.setIcon(qta.icon('fa5s.moon', color=self.accent))

    def apply_theme(self):
        """Apply the current theme to widgets and plots."""
        if self.dark_mode:
            bg = "#1e1e1e"
            fg = "#f0f0f0"
            widget_bg = "#2b2b2b"
            border = "#555555"
            button_bg = "#333333"
            button_hover = "#444444"
        else:
            bg = "#f0f0f0"
            fg = "#1e1e1e"
            widget_bg = "#ffffff"
            border = "#999999"
            button_bg = "#e0e0e0"
            button_hover = "#d0d0d0"

        self.plot_bg_color = bg
        self.fg_color = fg
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {bg};
                color: {fg};
            }}
            QPushButton {{
                background-color: {button_bg};
                border: 1px solid {self.accent};
                padding: 4px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {button_hover};
            }}
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QTableWidget {{
                background-color: {widget_bg};
                color: {fg};
                border: 1px solid {self.accent};
            }}
            QComboBox QAbstractItemView {{
                background-color: {widget_bg};
                color: {fg};
                border: 1px solid {self.accent};
                selection-background-color: {self.accent};
                selection-color: {fg};
            }}
            QTableWidget::item:selected {{
                background-color: {self.accent};
                color: {fg};
            }}
            QCheckBox::indicator {{
                border: 1px solid {self.accent};
                width: 13px;
                height: 13px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.accent};
            }}
            QLabel {{ color: {fg}; }}
            """
        )

        self.figure.set_facecolor(self.plot_bg_color)
        for ax in self.figure.axes:
            ax.set_facecolor(self.plot_bg_color)
            ax.tick_params(colors=fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_color(fg)
            leg = ax.get_legend()
            if leg:
                for text in leg.get_texts():
                    text.set_color(fg)
        self.canvas.draw()

    def select_folder(self):
        default_dir = '/media/ext_drive/PVDF_Data'
        #default_dir = '../dust_testing/2025_03_11_run0/'
        #default_dir = './2025_06_09_run1/'
        folder = QFileDialog.getExistingDirectory(
            self, "Select TRC Folder", default_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.ReadOnly
        )
        if not folder:
            return
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Selected path is not a directory")
            self.set_controls_enabled(False)
            return
        self.selected_folder = folder
        #from pathlib import Path
        #self.dataset_id = str(Path(folder).resolve())
        self.dataset_id = os.path.abspath(folder)
        self.folder_label.setText(f"Folder: {folder}")
        self.folder_label.setToolTip(folder)

        try:
            files = [os.path.join(folder, f)
                     for f in os.listdir(folder)
                     if f.lower().endswith('.trc')]
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not read folder:\n{e}")
            self.event_files.clear()
            self.event_combo.clear()
            self.set_controls_enabled(False)
            return
        if not files:
            QMessageBox.warning(self, "No Files", "Selected folder contains no .trc files")
            self.event_files.clear()
            self.event_combo.clear()
            self.set_controls_enabled(False)
            return

        pat = re.compile(r"-(\d+)\.trc$")
        events = {}
        for f in files:
            m = pat.search(f)
            if m:
                events.setdefault(m.group(1), []).append(f)
        if not events:
            QMessageBox.warning(self, "No Events", "No valid event files were found in the folder")
            self.event_files.clear()
            self.event_combo.clear()
            self.set_controls_enabled(False)
            return

        self.event_files = events
        # Preserve full list of events and clear any active filter
        self._all_event_keys = sorted(events.keys(), key=lambda x: int(re.search(r"(\d+)$", x).group(1)) if re.search(r"(\d+)$", x) else x)
        self.feature_filter_active = False
        self.feature_filter_keys = []
        self.event_combo.clear()
        self.event_combo.addItems(self._all_event_keys)
        self.set_controls_enabled(True)

    def load_metadata(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Metadata CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            self.metaArr = getMetaData(path)
            self.meta_label.setText(f"Loaded {len(self.metaArr)} metadata rows.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load metadata: {e}")

    def load_event(self, idx):
        evt = self.event_combo.currentText()
        
        # Clear current display data but preserve fit results
        self.current_data.clear()
        self.fit_regions.clear() 
        self.span_selectors.clear()
        self.sg_filtered.clear()
        self.meta_label.setText("MetaMatch result will appear here.")
        
        if not evt:
            return

        # Load waveform data
        for path in self.event_files.get(evt, []):
            m = re.match(r"C([1-4])", os.path.basename(path))
            if not m:
                continue
            ch = int(m.group(1))
            try:
                t, y, meta = self.trc.open(path)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load {os.path.basename(path)}: {e}")
                continue
            self.current_data[ch] = (t, y - np.mean(y), meta)

        if not self.current_data:
            QMessageBox.warning(self, "No Data", f"No waveforms were loaded for event {evt}")
            return

        # Plot waveforms first
        self.plot_waveforms()
        
        # Now recall and replot any existing fits for this event
        self.recall_fits_for_event(evt)
        self.fit_info_window.update_info(self.results, evt)

    def recall_fits_for_event(self, evt):
        """Recall and replot all fits for the given event"""
        
        # Find all fit results for this event
        evt_keys = [k for k in self.results.keys() if k.startswith(f"evt_{evt}_")]
        
        if not evt_keys:
            return  # No fits to recall
        
        pattern = re.compile(r'^evt_\d+_(.+)_ch(\d+)(?:_\d+)?$')

        # Process each fit type
        for key in evt_keys:
            rec = self.results.get(key, {})
            if rec.get('dataset_id') is not None and rec.get('dataset_id') != self.dataset_id:
                continue
            m = pattern.match(key)
            if not m:
                continue
            fit_type, ch_str = m.groups()
            ch = int(ch_str)

            self.recall_dynamic_fit(evt, ch, key, fit_type)

            # If this record stored an impact_time, re-draw the impact line across all axes
            it = rec.get('impact_time')
            if it is not None:
                for ax in self.figure.axes:
                    # Remove any existing impact lines (dash-dot) before re-adding
                    for ln in list(ax.lines):
                        if ln.get_linestyle() == '-.':
                            ax.lines.remove(ln)
                    ax.axvline(it, color='tab:purple', linestyle='-.', linewidth=1.2, label='Impact')
                    ax.legend()

        self.canvas.draw()
        # Also update the clear_fits method to be more specific


    def clear_fits(self):
        """Clear fits for current event only"""
        evt = self.event_combo.currentText()
        if not evt:
            return
            
        # Remove all stored fit results for this event
        keys_to_remove = [k for k in list(self.results.keys()) if k.startswith(f"evt_{evt}_") and (self.results[k].get('dataset_id') is None or self.results[k].get('dataset_id') == self.dataset_id)]
        for k in keys_to_remove:
            del self.results[k]

        # Clear plotted overlays for this event (by gid or legacy label rules)
        evt_prefix = f"evt_{evt}_"
        for ax in self.figure.axes:
            lines_to_remove = []
            for line in list(ax.lines):
                gid = line.get_gid()
                if isinstance(gid, str) and gid.startswith(evt_prefix):
                    lines_to_remove.append(line)
                elif line.get_linestyle() == '-.' or str(line.get_label()).endswith('Fit'):
                    lines_to_remove.append(line)
            for line in lines_to_remove:
                try:
                    line.remove()
                except Exception:
                    pass
            ax.legend()
        self.canvas.draw()
        self.fit_info_window.update_info(self.results, self.event_combo.currentText())

    def prev_event(self):
            idx = self.event_combo.currentIndex()
            if idx > 0:
                    self.event_combo.setCurrentIndex(idx - 1)

    def next_event(self):
            idx = self.event_combo.currentIndex()
            if idx < self.event_combo.count() - 1:
                    self.event_combo.setCurrentIndex(idx + 1)
            else:
                    QMessageBox.information(self, "No Next Event", "Already at the last event.")

    def keyPressEvent(self, event):  
        if getattr(self, '_shortcuts_installed', False):
            return super().keyPressEvent(event)
        key = event.key()
        # Navigation
        if key == Qt.Key_Period:
            if hasattr(self, 'btn_next') and self.btn_next.isEnabled():
                self.next_event()
        elif key == Qt.Key_Comma:
            if hasattr(self, 'btn_prev') and self.btn_prev.isEnabled():
                self.prev_event()
        # Clear dynamic fit
        elif key == Qt.Key_R:
            if hasattr(self, 'btn_clear_dyn') and self.btn_clear_dyn.isEnabled():
                self.clear_dynamic_fit()
        # SG filter
        elif key == Qt.Key_S:
            if hasattr(self, 'btn_sg') and self.btn_sg.isEnabled():
                self.run_sg_filter()
        # MetaMatch
        elif key == Qt.Key_M:
            if hasattr(self, 'btn_meta') and self.btn_meta.isEnabled():
                self.run_meta_match()
        # Folder select (always allowed)
        elif key == Qt.Key_F:
            self.select_folder()
        # Load metadata
        elif key == Qt.Key_L:
            if hasattr(self, 'btn_load_meta') and self.btn_load_meta.isEnabled():
                self.load_metadata()
        # Help (always allowed)
        elif key == Qt.Key_H:
            self.show_help()
        # Fit info
        elif key == Qt.Key_U:
            if hasattr(self, 'btn_show_info') and self.btn_show_info.isEnabled():
                self.show_fit_info()
        # Mark impact
        elif key == Qt.Key_T:
            if hasattr(self, 'btn_mark_impact') and self.btn_mark_impact.isEnabled():
                self.mark_impact_line()

        # Dynamic fit row hotkeys (guarded by combo enabled state)
        elif key == Qt.Key_Q:
            if hasattr(self, 'dyn_func_combo') and self.dyn_func_combo.isEnabled():
                self.dyn_func_combo.setCurrentText("QD3Fit")
        elif key == Qt.Key_D:
            if hasattr(self, 'dyn_func_combo') and self.dyn_func_combo.isEnabled():
                self.dyn_func_combo.setCurrentText("QDMFit")
        elif key == Qt.Key_C:
            if hasattr(self, 'dyn_func_combo') and self.dyn_func_combo.isEnabled():
                self.dyn_func_combo.setCurrentText("CSA_pulse")
        elif key == Qt.Key_W:
            if hasattr(self, 'dyn_func_combo') and self.dyn_func_combo.isEnabled():
                self.dyn_func_combo.setCurrentText("skew_gaussian")
        elif key == Qt.Key_G:
            if hasattr(self, 'dyn_func_combo') and self.dyn_func_combo.isEnabled():
                self.dyn_func_combo.setCurrentText("gaussian")
        elif key == Qt.Key_X:
            if hasattr(self, 'dyn_func_combo') and self.dyn_func_combo.isEnabled():
                self.dyn_func_combo.setCurrentText("low_pass_max")
        # Channel numeric keys
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8):
            if hasattr(self, 'dyn_ch_combo') and self.dyn_ch_combo.isEnabled() and self.dyn_ch_combo.count() > 0:
                desired = str(key - Qt.Key_0)
                # Only set if that channel exists in the combo
                idx = self.dyn_ch_combo.findText(desired)
                if idx != -1:
                    self.dyn_ch_combo.setCurrentText(desired)
        # Invert toggle
        elif key == Qt.Key_I:
            if hasattr(self, 'dyn_invert') and self.dyn_invert.isEnabled():
                self.dyn_invert.setChecked(not self.dyn_invert.isChecked())
        # Run / adjust
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if hasattr(self, 'btn_dyn') and self.btn_dyn.isEnabled():
                self.run_dynamic_fit()
        elif key == Qt.Key_A:
            if hasattr(self, 'btn_adjust_dyn') and self.btn_adjust_dyn.isEnabled():
                self.adjust_dynamic_params()
        else:
            super().keyPressEvent(event)

    def _install_global_shortcuts(self):
        # Hold references to prevent GC
        self._shortcuts = []

        def add_shortcut(key, handler):
            sc = QtWidgets.QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

        # Navigation
        add_shortcut(Qt.Key_Period, lambda: (self.btn_next.isEnabled() and self.next_event()))
        add_shortcut(Qt.Key_Comma,  lambda: (self.btn_prev.isEnabled() and self.prev_event()))

        # Common actions
        add_shortcut(Qt.Key_F, self.select_folder)
        add_shortcut(Qt.Key_L, lambda: (self.btn_load_meta.isEnabled() and self.load_metadata()))
        add_shortcut(Qt.Key_H, self.show_help)
        add_shortcut(Qt.Key_U, lambda: (self.btn_show_info.isEnabled() and self.show_fit_info()))
        add_shortcut(Qt.Key_S, lambda: (self.btn_sg.isEnabled() and self.run_sg_filter()))
        add_shortcut(Qt.Key_M, lambda: (self.btn_meta.isEnabled() and self.run_meta_match()))
        add_shortcut(Qt.Key_T, lambda: (self.btn_mark_impact.isEnabled() and self.mark_impact_line()))
        add_shortcut(Qt.Key_R, lambda: (self.btn_clear_dyn.isEnabled() and self.clear_dynamic_fit()))
        # Additional accelerators
        add_shortcut(Qt.CTRL + Qt.Key_S, lambda: (self.btn_save_fig.isEnabled() and self.save_figure_transparent()))
        add_shortcut(Qt.CTRL + Qt.Key_E, lambda: (self.btn_export.isEnabled() and self.export_fits()))
        add_shortcut(Qt.CTRL + Qt.Key_I, lambda: (self.btn_import.isEnabled() and self.import_fits()))
        add_shortcut(Qt.SHIFT + Qt.Key_D, lambda: (self.btn_clear_data.isEnabled() and self.clear_data()))
        add_shortcut(Qt.Key_E, lambda: (self.btn_clear.isEnabled() and self.clear_fits()))
        add_shortcut(Qt.CTRL + Qt.Key_T, lambda: (self.btn_theme.isEnabled() and self.toggle_theme()))

        # SG row extras
        add_shortcut(Qt.CTRL + Qt.Key_B, lambda: (self.btn_batch_sg.isEnabled() and self.btn_batch_sg.click()))
        add_shortcut(Qt.SHIFT + Qt.Key_S, lambda: (self.btn_clear_sg.isEnabled() and self.clear_sg_filter()))
        add_shortcut(Qt.Key_P, lambda: (self.btn_results_plotter.isEnabled() and self.open_results_plotter()))
        add_shortcut(Qt.CTRL + Qt.Key_F, lambda: (self.btn_feature_scan.isEnabled() and self.run_feature_scan()))
        add_shortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_F, lambda: (self.btn_fit_filter.isEnabled() and self.run_fit_filter()))
        add_shortcut(Qt.CTRL + Qt.Key_K, lambda: (self.btn_clear_filter.isEnabled() and self.clear_event_filter()))

        # Dynamic row extras
        add_shortcut(Qt.Key_B, lambda: (self.btn_batch_dyn.isEnabled() and self.run_batch_fit()))
        add_shortcut(Qt.SHIFT + Qt.Key_R, lambda: (self.btn_clear_chan.isEnabled() and self.clear_channel_fits()))
        add_shortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_S, lambda: (self.use_sg_toggle.isEnabled() and self.use_sg_toggle.setChecked(not self.use_sg_toggle.isChecked())))

        # Dynamic fit selection
        add_shortcut(Qt.Key_Q, lambda: (self.dyn_func_combo.isEnabled() and self.dyn_func_combo.setCurrentText("QD3Fit")))
        add_shortcut(Qt.Key_D, lambda: (self.dyn_func_combo.isEnabled() and self.dyn_func_combo.setCurrentText("QDMFit")))
        add_shortcut(Qt.Key_C, lambda: (self.dyn_func_combo.isEnabled() and self.dyn_func_combo.setCurrentText("CSA_pulse")))
        add_shortcut(Qt.Key_W, lambda: (self.dyn_func_combo.isEnabled() and self.dyn_func_combo.setCurrentText("skew_gaussian")))
        add_shortcut(Qt.Key_G, lambda: (self.dyn_func_combo.isEnabled() and self.dyn_func_combo.setCurrentText("gaussian")))
        add_shortcut(Qt.Key_X, lambda: (self.dyn_func_combo.isEnabled() and self.dyn_func_combo.setCurrentText("low_pass_max")))
        # Shift+F selects FFT without overriding plain F (Select Folder)
        add_shortcut(Qt.SHIFT + Qt.Key_F, lambda: (self.dyn_func_combo.isEnabled() and self.dyn_func_combo.setCurrentText("FFT")))

        # Channel selection 1-4
        for n in (1, 2, 3, 4):
            add_shortcut(Qt.Key_0 + n, lambda n=n: (
                self.dyn_ch_combo.isEnabled() and self.dyn_ch_combo.findText(str(n)) != -1 and self.dyn_ch_combo.setCurrentText(str(n))
            ))

        # Invert toggle
        add_shortcut(Qt.Key_I, lambda: (self.dyn_invert.isEnabled() and self.dyn_invert.setChecked(not self.dyn_invert.isChecked())))

        # Run/Adjust
        add_shortcut(Qt.Key_Return, lambda: (self.btn_dyn.isEnabled() and self.run_dynamic_fit()))
        add_shortcut(Qt.Key_Enter,  lambda: (self.btn_dyn.isEnabled() and self.run_dynamic_fit()))
        add_shortcut(Qt.Key_A,      lambda: (self.btn_adjust_dyn.isEnabled() and self.adjust_dynamic_params()))

    def plot_waveforms(self):
        self.figure.clf()
        decim = self.decim_spin.value()
        sorted_chs = sorted(self.current_data.keys())
        for idx, ch in enumerate(sorted_chs, start=1):
            ax = self.figure.add_subplot(len(sorted_chs), 1, idx)
            ax.set_facecolor(self.plot_bg_color)
            t, y, meta = self.current_data[ch]
            # apply decimation
            t_plot = t[::decim]
            y_plot = y[::decim]
            ax.plot(t_plot, y_plot, color='C0', label=f"Raw CH{ch}")
            ax.set_ylabel(f"CH{ch} [V]", color=self.fg_color)
            if idx == len(sorted_chs):
                ax.set_xlabel("Time [s]", color=self.fg_color)
            ax.tick_params(colors=self.fg_color)
            for spine in ax.spines.values():
                spine.set_color(self.fg_color)
            sel = SpanSelector(
                ax,
                lambda xmin, xmax, ch=ch: self.on_select(ch, xmin, xmax),
                'horizontal', useblit=True,
                props=dict(alpha=0.3, facecolor='orange')
            )
            self.span_selectors[ch] = sel
        self.figure.tight_layout()
        self.canvas.draw()

    def on_select(self, channel, xmin, xmax):
        self.fit_regions[channel] = (xmin, xmax)
        ax = self.figure.axes[sorted(self.current_data.keys()).index(channel)]
        for line in ax.lines:
            if line.get_linestyle() == '--':
                line.remove()
        ax.axvline(xmin, color='red', linestyle='--')
        ax.axvline(xmax, color='red', linestyle='--')
        self.canvas.draw()

    def run_meta_match(self):
        # Collect TIMESTAMP WINDOW FROM O-SCOPE
        timeArr = list()
        j = 0
        _, _, meta = self.current_data[1]
        while j<len(meta['TRIGGER_TIME']):
            timeArr.append(float(meta['TRIGGER_TIME'][j]))
            j += 1
        timeTemp = timeArr[0]*1e10 + timeArr[1]*1e8 + timeArr[2]*1e6 + timeArr[3]*1e4 + timeArr[4]*1e2 + timeArr[5]
        time = timeTemp    #this is in yyyymmddhhmmss.ms format 

        if self.metaArr is None:
            QMessageBox.warning(self, "No Metadata", "No metadata loaded. Use 'Load Metadata'.")
            return
        evt = self.event_combo.currentText()
        qd_keys = [k for k in self.results if k.startswith(f"evt_{evt}_QD3Fit_ch")]
        if not qd_keys:
            QMessageBox.warning(self, "No QD3 Fit", "Perform a QD3 fit first.")
            return
        res = self.results[qd_keys[0]]
        t0, q, v = res['params']
        match = metaMatch(self.metaArr, time, v)
        if not match:
            text = "No matching metadata found."
        else:
            # only display selected fields
            fields = ['UTC Timestamp', 'Velocity (m/s)', 'Charge (C)', 'Radius (m)']
            text = "; ".join(f"{fld}: {match.get(fld,'N/A')}" for fld in fields)

    def mark_impact_line(self):
        """Plot a vertical line at t0 + distance/velocity across all visible channels.
        Uses t0 and v from a present QD3Fit or QDMFit. If both are present, prompt.
        Stores the computed impact time on the chosen fit record.
        """
        evt = self.event_combo.currentText()
        if not evt:
            QMessageBox.warning(self, "No Event", "Select an event first.")
            return

        # Gather candidate fit keys for current event (and current dataset)
        qd_keys = []
        qdm_keys = []
        for k, rec in self.results.items():
            if not k.startswith(f"evt_{evt}_"):
                continue
            if rec.get('dataset_id') is not None and rec.get('dataset_id') != self.dataset_id:
                continue
            if f"evt_{evt}_QD3Fit_" in k or k.startswith(f"evt_{evt}_QD3Fit_ch"):
                qd_keys.append(k)
            elif f"evt_{evt}_QDMFit_" in k or k.startswith(f"evt_{evt}_QDMFit_ch"):
                qdm_keys.append(k)

        if not qd_keys and not qdm_keys:
            QMessageBox.warning(self, "No Fit Found", "Perform a QD3 or QDM fit first.")
            return

        # Choose fit type if both present
        use_type = None
        if qd_keys and qdm_keys:
            items = ["QD3Fit", "QDMFit"]
            sel, ok = QInputDialog.getItem(self, "Choose Fit", "Use which fit for impact time?", items, 0, False)
            if not ok:
                return
            use_type = sel
        elif qd_keys:
            use_type = "QD3Fit"
        else:
            use_type = "QDMFit"

        keys = qd_keys if use_type == "QD3Fit" else qdm_keys

        # If multiple of the chosen type exist, prompt to select one
        choose_key = None
        if len(keys) == 1:
            choose_key = keys[0]
        else:
            # Build human-friendly list: e.g., "ch1 (fit 1)" based on key
            disp = []
            for k in keys:
                m = re.search(r"_ch(\d+)(?:_(\d+))?$", k)
                ch = m.group(1) if m else "?"
                idx = m.group(2) if (m and m.group(2)) else "1"
                disp.append(f"ch{ch} (fit {idx}) :: {k}")
            sel, ok = QInputDialog.getItem(self, "Choose Fit Instance", "Select specific fit:", disp, 0, False)
            if not ok:
                return
            # Extract key back out
            choose_key = sel.split("::", 1)[1].strip()

        rec = self.results.get(choose_key)
        if not rec:
            QMessageBox.warning(self, "Selection Error", "Selected fit record not found.")
            return

        params = rec.get('params', ())
        if len(params) < 3:
            QMessageBox.warning(self, "Bad Fit Params", "Selected fit does not include t0 and v.")
            return

        t0 = float(params[0])
        v  = float(params[2])
        d  = float(self.dist_spin.value())
        if v == 0:
            QMessageBox.warning(self, "Invalid Velocity", "Velocity is zero; cannot compute impact time.")
            return

        # Physically consistent: t_impact = t0 + distance / velocity
        t_impact = t0 + d / v

        # Remove existing impact lines and draw new one on all visible axes
        for ax in self.figure.axes:
            for ln in list(ax.lines):
                if ln.get_linestyle() == '-.':
                    ax.lines.remove(ln)
            label = f"Impact ({use_type.split('Fit')[0]})" if use_type else "Impact"
            ax.axvline(t_impact, color='tab:purple', linestyle='-.', linewidth=1.2, label=label)
            ax.legend()

        # Store impact_time on the chosen record for export/recall
        rec['impact_time'] = t_impact
        self.results[choose_key] = rec

        self.canvas.draw()
        self.fit_info_window.update_info(self.results, self.event_combo.currentText())

    def run_sg_filter(self):
        ch = int(self.sg_ch.currentText()); data = self.current_data.get(ch)
        if data is None:
            QMessageBox.warning(self, "No Data", f"CH{ch} not loaded."); return
        t, y, _ = data; w = self.sg_win.value()
        if w>len(y): QMessageBox.warning(self, "Window Too Large", "SG window > data length."); return
        y_sg = sig.savgol_filter(y, w, 2)
        self.sg_filtered[ch] = (t, y_sg)
        ax = self.figure.axes[sorted(self.current_data).index(ch)]
        for ln in ax.get_lines():
            if ln.get_label()=='SG Filter': ax.lines.remove(ln)
        ax.plot(t, y_sg, color="tab:green", label="SG Filter")
        ax.legend()
        self.canvas.draw()
        # End run_sg_filter

    def _guess_params(self, func_name, names, t_sel, y_sel):
        """Return initial parameter guesses and bounds for curve_fit."""
        guess = {}
        lower = []
        upper = []

        if func_name in ("QD3Fit", "QD3Fit_soft"):
            # heuristic similar to run_qd3_fit, but robust to short windows
            try:
                # choose an odd window <= len(y_sel)-1 and not too small
                desired = 501
                max_allowed = max(5, len(y_sel) - 1)
                w = min(desired, max_allowed)
                if w % 2 == 0:
                    w = max(5, w - 1)
                y_smooth = sig.savgol_filter(y_sel, w, 2)
            except Exception:
                y_smooth = y_sel
            if len(y_smooth) >= 3:
                y_min = float(np.min(y_smooth))
                diff = np.diff(y_smooth)
                if diff.size > 0:
                    rise = int(np.argmax(diff))
                    fall = int(np.argmin(diff))
                    dt = float(t_sel[rise] - t_sel[fall]) if 0 <= rise < len(t_sel) and 0 <= fall < len(t_sel) else 0.0
                    det_vel = 0.19 / dt if dt != 0 else 1.0
                    t0_guess = float(t_sel[fall]) if 0 <= fall < len(t_sel) else float(t_sel[0])
                else:
                    det_vel = 1.0
                    t0_guess = float(t_sel[0])
                guess.update({"t0": t0_guess, "q": y_min, "v": det_vel})
        elif func_name in ("QDMFit", "QDMobileFit"):
            try:
                desired = 201
                max_allowed = max(5, len(y_sel) - 1)
                w = min(desired, max_allowed)
                if w % 2 == 0:
                    w = max(5, w - 1)
                y_smooth = sig.savgol_filter(y_sel, w, 2)
            except Exception:
                y_smooth = y_sel
            if len(y_smooth) >= 3:
                y_min = float(np.min(y_smooth))
                diff = np.diff(y_smooth)
                if diff.size > 0:
                    rise = int(np.argmax(diff))
                    fall = int(np.argmin(diff))
                    dt = float(t_sel[rise] - t_sel[fall]) if 0 <= rise < len(t_sel) and 0 <= fall < len(t_sel) else 0.0
                    det_vel = 0.133 / dt if dt != 0 else 1.0
                    t0_guess = float(t_sel[fall] - 5e-6) if 0 <= fall < len(t_sel) else float(t_sel[0])
                else:
                    det_vel = 1.0
                    t0_guess = float(t_sel[0])
                guess.update({"t0": t0_guess, "q": y_min, "v": det_vel})
        elif func_name == "skew_gaussian":
            area = np.trapz(y_sel, t_sel)
            peak_idx = np.argmax(y_sel)
            guess.update({
                "A": area,
                "xi": t_sel[peak_idx],
                "omega": (t_sel[-1] - t_sel[0]) / 6,
                "alpha": 0.0,
            })
        elif func_name == "gaussian":
            peak_idx = int(np.argmax(y_sel))
            amp = float(y_sel[peak_idx]) - float(np.min(y_sel))
            width = (t_sel[-1] - t_sel[0]) / 6
            guess.update({
                "A": amp,
                "mu": t_sel[peak_idx],
                "sigma": max(width, 1e-9),
            })
        else:
            if "t0" in names:
                guess["t0"] = t_sel[0]

        amp = max(y_sel) - min(y_sel)
        if "q" in names and "q" not in guess:
            guess["q"] = amp
        if "A" in names and "A" not in guess:
            guess["A"] = amp
        if "v" in names and "v" not in guess:
            guess["v"] = 1.0
        if "tau" in names:
            guess["tau"] = (t_sel[-1] - t_sel[0]) / 2
        if "beta" in names:
            guess["beta"] = 1.0
        if "edge_w" in names and "edge_w" not in guess:
            guess["edge_w"] = 1e-6
        if "C0" in names:
            guess["C0"] = y_sel[0]
        if "C1" in names:
            guess["C1"] = 0.0
        if "C2" in names:
            guess["C2"] = max(y_sel)
        if "T0" in names:
            guess["T0"] = (t_sel[-1] - t_sel[0]) / 10
        if "T1" in names:
            guess["T1"] = (t_sel[-1] - t_sel[0]) / 10
        if "T2" in names:
            guess["T2"] = (t_sel[-1] - t_sel[0]) / 5
        if "C" in names:
            guess["C"] = float(np.mean(y_sel))

        for n in names:
            if n == "v":
                lower.append(0)
                upper.append(np.inf)
            elif n == "t0" and func_name in ("QD3Fit", "QD3Fit_soft", "QDMFit", "QDMobileFit"):
                lower.append(t_sel[0])
                upper.append(t_sel[-1])
            elif n == "omega":
                lower.append(0)
                upper.append(np.inf)
            else:
                lower.append(-np.inf)
                upper.append(np.inf)

        return guess, lower, upper

    def run_dynamic_fit(self):
        func_name = self.dyn_func_combo.currentText()
        func, names = FIT_LIB[func_name]
        ch = int(self.dyn_ch_combo.currentText())
        region = self.fit_regions.get(ch)
        # Choose data source: SG if toggle is on and available, else raw
        if hasattr(self, 'use_sg_toggle') and self.use_sg_toggle.isChecked() and (ch in self.sg_filtered):
            t, y = self.sg_filtered[ch]
        else:
            t, y, _ = self.current_data[ch]
        if region is not None:
            mask = (t >= region[0]) & (t <= region[1])
            t_sel, y_sel = t[mask], y[mask]
        else:
            t_sel, y_sel = t, y
        # FFT: analysis-only branch; show popup with spectrum and return
        if func_name == "FFT":
            if t_sel.size < 8:
                QMessageBox.warning(self, "Selection Too Small", "Select a larger time window for FFT.")
                return
            try:
                self._show_fft(t_sel, y_sel, ch, self.event_combo.currentText())
            except Exception as e:
                QMessageBox.warning(self, "FFT Error", str(e))
            return
        # Only pre-invert selection for functional fits; low_pass_max handles polarity internally
        if self.dyn_invert.isChecked() and func_name != "low_pass_max":
            y_sel = -y_sel

        # Special-case measurement that doesn't use curve_fit
        if func_name == "low_pass_max":
            invert_flag = self.dyn_invert.isChecked()
            # Default width from SG control if present, else 201
            w = int(self.sg_win.value()) if hasattr(self, 'sg_win') else 201
            # If an existing result for this region exists, reuse its width (suppress warnings here)
            existing_key = self._select_fit_key(func_name, ch, self.event_combo.currentText(), quiet=True) if ch in self.fit_regions else None
            if existing_key and existing_key in self.results:
                try:
                    w = int(self.results[existing_key].get('params', (w,))[0])
                except Exception:
                    pass
            # Ensure odd and within bounds
            w = max(5, w)
            if w % 2 == 0:
                w += 1
            # Cap to selection length and ensure odd
            w = min(w, max(5, len(y_sel) - 1))
            if w % 2 == 0:
                w -= 1
            w = max(5, w)
            # Prepare data (apply invert for picking minima)
            y_proc = -y_sel if invert_flag else y_sel
            try:
                y_f = sig.savgol_filter(y_proc, w, 2)
            except Exception as e:
                QMessageBox.warning(self, "SG Error", f"Failed to filter: {e}")
                return
            idx = int(np.argmax(y_f)) if y_f.size else 0
            t_peak = float(t_sel[idx]) if len(t_sel) else None
            val_proc = float(y_f[idx]) if y_f.size else None
            if t_peak is None or val_proc is None:
                QMessageBox.warning(self, "No Data", "Selection too small for SG filter.")
                return
            val = -val_proc if invert_flag else val_proc
            # Plot marker on original polarity
            ax = self.figure.axes[sorted(self.current_data.keys()).index(ch)]
            # Remove prior marker if any for this selection
            # Build unique key and index
            evt = self.event_combo.currentText()
            base = f"evt_{evt}_{func_name}_ch{ch}"
            idx_num = 1
            key = base
            while key in self.results:
                idx_num += 1
                key = f"{base}_{idx_num}"
            label = f"{func_name} {idx_num}" if idx_num > 1 else f"{func_name}"
            line, = ax.plot([t_peak], [val], marker='o', color='tab:green', label=label)
            line.set_gid(key)
            if region is not None:
                ax.axvline(region[0], color="red", linestyle="--", alpha=0.7)
                ax.axvline(region[1], color="red", linestyle="--", alpha=0.7)
            ax.legend(); self.canvas.draw()
            # Store record
            self.results[key] = {
                "params": (int(w),),
                "param_names": ["width"],
                "fit_region": region if region is not None else (float(t_sel[0]), float(t_sel[-1])),
                "dataset_id": self.dataset_id,
                "inverted": invert_flag,
                "value": float(val),
                "t_at": float(t_peak),
                "line": line,
            }
            self.fit_info_window.update_info(self.results, self.event_combo.currentText())
            return

        # Standard curve_fit path
        guess, lower, upper = self._guess_params(func_name, names, t_sel, y_sel)
        p0 = [guess.get(n, 0.0) for n in names]
        bounds = None
        if any(np.isfinite(lower)) or any(np.isfinite(upper)):
            bounds = (lower, upper)

        self.fit_progress = QProgressDialog("Fit in progress...", "Cancel Fit", 0, 0, self)
        self.fit_progress.setWindowModality(Qt.ApplicationModal)
        self.fit_progress.setAutoClose(False)
        self.fit_progress.canceled.connect(self._cancel_fit)
        self.fit_progress.show()

        self.fit_worker = FitWorker(func, t_sel, y_sel, p0, bounds)
        current_evt = self.event_combo.currentText()
        self.fit_worker.result.connect(
            lambda popt, err, fn=func_name, nm=names, ch_=ch, reg=region, tt=t_sel, yy=y_sel, ev=current_evt:
                self._finish_dynamic_fit(popt, err, fn, nm, ch_, reg, tt, yy, ev))
        self.fit_worker.start()

    def _show_fft(self, t_sel, y_sel, ch, evt):
        # Compute FFT using rfft; detrend and window
        t_sel = np.asarray(t_sel)
        y_sel = np.asarray(y_sel)
        dt = np.median(np.diff(t_sel))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid sampling interval for FFT.")
        y_dt = y_sel - np.mean(y_sel)
        n = len(y_dt)
        win = np.hanning(n)
        y_win = y_dt * win
        yf = np.fft.rfft(y_win)
        xf = np.fft.rfftfreq(n, dt)
        # Amplitude scaling (two-sided to single-sided)
        amp = (2.0 / n) * np.abs(yf)

        # Build popup window
        dlg = QDialog(self)
        dlg.setWindowTitle(f"FFT ch{ch} • evt {evt}")
        v = QVBoxLayout(dlg)
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        # Add navigation toolbar for zoom/pan/home, same as main view
        toolbar = NavigationToolbar(canvas, dlg)
        v.addWidget(toolbar)
        ax = fig.add_subplot(111)
        # Use log-log scales by default; exclude zero/negative
        mask = (xf > 0) & (amp > 0)
        ax.plot(xf[mask], amp[mask], color='tab:blue')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude [V]')
        ax.grid(True)
        fig.tight_layout()
        v.addWidget(canvas)
        btns = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        v.addWidget(btns)
        self._child_windows.append(dlg)
        dlg.resize(700, 450)
        dlg.show()

    def run_batch_fit(self):
        """Run the selected fit across a user-specified time window and event range."""
        # Preconditions
        if not hasattr(self, 'event_files') or not self.event_files:
            QMessageBox.warning(self, "No Folder", "Select a folder with TRC files first.")
            return
        func_name = self.dyn_func_combo.currentText()
        if func_name not in FIT_LIB:
            QMessageBox.warning(self, "No Fit Selected", "Choose a fit function first.")
            return
        ch = int(self.dyn_ch_combo.currentText())
        invert_flag = self.dyn_invert.isChecked()

        # Derive defaults for dialog from current selection/loaded data
        # Build a list of (int_value, key_string) for events to preserve zero padding
        ev_pairs = []
        for i in range(self.event_combo.count()):
            key = self.event_combo.itemText(i)
            try:
                iv = int(key)
            except Exception:
                m = re.search(r"(\d+)$", key)
                if not m:
                    continue
                iv = int(m.group(1))
            ev_pairs.append((iv, key))
        if not ev_pairs:
            QMessageBox.warning(self, "No Events", "No events found in the current folder.")
            return
        ev_pairs.sort(key=lambda x: x[0])
        evt_min = ev_pairs[0][0]
        evt_max = ev_pairs[-1][0]

        # Time defaults: use current region if present for channel, else current trace bounds
        tmin_def = None
        tmax_def = None
        if ch in self.fit_regions:
            tmin_def, tmax_def = self.fit_regions[ch]
        elif ch in self.current_data:
            tt, _, _ = self.current_data[ch]
            tmin_def, tmax_def = float(tt[0]), float(tt[-1])

        dlg = BatchFitDialog(self, evt_min, evt_max, tmin_def, tmax_def)
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            ev0, ev1, t0_win, t1_win = dlg.values()
        except Exception as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        # Build list of target event KEYS preserving zero-padding
        target_events = [key for (iv, key) in ev_pairs if ev0 <= iv <= ev1]
        if not target_events:
            QMessageBox.information(self, "No Events", "No events in the specified range.")
            return

        func, names = FIT_LIB[func_name]

        # Progress dialog
        prog = QProgressDialog("Batch fitting...", "Cancel", 0, len(target_events), self)
        prog.setWindowModality(Qt.ApplicationModal)
        prog.setAutoClose(True)
        prog.show()

        added = 0
        # Iterate events
        for i, evt in enumerate(target_events, start=1):
            prog.setValue(i - 1)
            prog.setLabelText(f"Fitting {func_name} ch{ch} on event {evt} ({i}/{len(target_events)})")
            QApplication.processEvents()
            if prog.wasCanceled():
                break

            # Load trace for this event and channel
            paths = self.event_files.get(evt, [])
            ch_path = None
            for p in paths:
                base = os.path.basename(p)
                if re.match(fr"C{ch}.*", base):
                    ch_path = p
                    break
            if ch_path is None:
                # No data for this channel in this event
                continue

            try:
                t, y, meta = self.trc.open(ch_path)
                y = y - np.mean(y)
            except Exception:
                continue

            # Select window
            mask = (t >= t0_win) & (t <= t1_win)
            t_sel = t[mask]
            y_sel = y[mask]
            # If window is invalid for this event, skip
            if t_sel.size < (len(names) + 3):
                continue
            if invert_flag:
                y_sel = -y_sel

            if func_name == "low_pass_max":
                # Measurement path: SG filter and take max (or min if inverted)
                w = int(self.sg_win.value()) if hasattr(self, 'sg_win') else 201
                if w % 2 == 0: w += 1
                w = min(w, max(5, len(y_sel) - 1))
                if w % 2 == 0:
                    w -= 1
                y_proc = -y_sel if invert_flag else y_sel
                try:
                    y_f = sig.savgol_filter(y_proc, w, 2)
                    ii = int(np.argmax(y_f)) if y_f.size else 0
                    t_peak = float(t_sel[ii]) if len(t_sel) else None
                    val_proc = float(y_f[ii]) if y_f.size else None
                    if t_peak is None or val_proc is None:
                        continue
                    val = -val_proc if invert_flag else val_proc
                except Exception:
                    continue
                base = f"evt_{evt}_{func_name}_ch{ch}"
                idx = 1
                key = base
                while key in self.results:
                    idx += 1
                    key = f"{base}_{idx}"
                rec = {
                    "params": (int(w),),
                    "param_names": ["width"],
                    "fit_region": (float(t0_win), float(t1_win)),
                    "dataset_id": self.dataset_id,
                    "inverted": invert_flag,
                    "value": float(val),
                    "t_at": float(t_peak),
                }
                if self.event_combo.currentText() == evt and ch in self.current_data:
                    ax = self.figure.axes[sorted(self.current_data.keys()).index(ch)]
                    label = f"{func_name} {idx}" if idx > 1 else f"{func_name}"
                    line, = ax.plot([t_peak], [val], marker='o', color='tab:green', label=label)
                    line.set_gid(key)
                    rec["line"] = line
                    ax.legend()
                self.results[key] = rec
                added += 1
                continue

            # Functional fit path
            try:
                guess, lower, upper = self._guess_params(func_name, names, t_sel, y_sel)
                p0 = [guess.get(n, 0.0) for n in names]
                bounds = (lower, upper) if (any(np.isfinite(lower)) or any(np.isfinite(upper))) else None
                if bounds is not None:
                    popt, _ = curve_fit(func, t_sel, y_sel, p0=p0, bounds=bounds, maxfev=20000)
                else:
                    popt, _ = curve_fit(func, t_sel, y_sel, p0=p0, maxfev=20000)
            except Exception:
                continue
            base = f"evt_{evt}_{func_name}_ch{ch}"
            idx = 1
            key = base
            while key in self.results:
                idx += 1
                key = f"{base}_{idx}"
            rec = {
                "params": tuple(popt),
                "param_names": names,
                "fit_region": (float(t0_win), float(t1_win)),
                "dataset_id": self.dataset_id,
                "inverted": invert_flag,
            }
            if func_name == "QD3Fit":
                try:
                    charge, mass, rad = QD3Cal(popt[1], popt[2])
                    rec.update({"charge": charge, "mass": mass, "radius": rad})
                except Exception:
                    pass
            elif func_name == "QDMFit":
                try:
                    charge, mass, rad = QDMCal(popt[1], popt[2])
                    rec.update({"charge": charge, "mass": mass, "radius": rad})
                except Exception:
                    pass
            if self.event_combo.currentText() == evt and ch in self.current_data:
                ax = self.figure.axes[sorted(self.current_data.keys()).index(ch)]
                y_fit = func(t_sel, *popt)
                if invert_flag:
                    y_fit = -y_fit
                label = f"{func_name} Fit {idx}" if idx > 1 else f"{func_name} Fit"
                line, = ax.plot(t_sel, y_fit, color='tab:green', label=label)
                line.set_gid(key)
                rec["line"] = line
                ax.legend()
            self.results[key] = rec
            added += 1

        prog.setValue(len(target_events))
        self.canvas.draw()
        # Update the info window for the current event
        self.fit_info_window.update_info(self.results, self.event_combo.currentText())
        if added == 0:
            QMessageBox.information(self, "Batch Complete", "No fits were added. Check time window and channel files.")
        else:
            QMessageBox.information(self, "Batch Complete", f"Added {added} fits.")

    def _cancel_fit(self):
        if hasattr(self, "fit_worker") and self.fit_worker is not None:
            self.fit_worker.cancelled = True

    def _finish_dynamic_fit(self, popt, err, func_name, names, ch, region, t_sel, y_sel, evt):
        self.fit_progress.close()
        self.fit_worker = None
        if err is not None:
            if isinstance(err, RuntimeError) and "cancelled" in str(err).lower():
                QMessageBox.information(self, "Fit Cancelled", "Fit was cancelled.")
            else:
                QMessageBox.warning(self, "Fit Failed", str(err))
            return

        base = f"evt_{evt}_{func_name}_ch{ch}"
        idx = 1
        key = base
        while key in self.results:
            idx += 1
            key = f"{base}_{idx}"

        # Ensure a region is stored for later selection
        if region is None:
            region = (t_sel[0], t_sel[-1])

        self.results[key] = {
            "params": tuple(popt),
            "param_names": names,
            "fit_region": region,
            "dataset_id": self.dataset_id,
            "inverted": self.dyn_invert.isChecked(),
        }

        if func_name == "QD3Fit":
            charge, mass, rad = QD3Cal(popt[1], popt[2])
            self.results[key].update({"charge": charge, "mass": mass, "radius": rad})
        elif func_name == "QDMFit":
            charge, mass, rad = QDMCal(popt[1], popt[2])
            self.results[key].update({"charge": charge, "mass": mass, "radius": rad})

        y_fit = func_name and FIT_LIB[func_name][0](t_sel, *popt)
        if self.dyn_invert.isChecked():
            y_fit = -y_fit
        ax = self.figure.axes[sorted(self.current_data.keys()).index(ch)]
        label = f"{func_name} Fit {idx}" if idx > 1 else f"{func_name} Fit"
        line, = ax.plot(t_sel, y_fit, color='tab:green', label=label)
        line.set_gid(key)
        self.results[key]["line"] = line
        ax.legend()
        self.canvas.draw()
        self.fit_info_window.update_info(self.results, evt)

    def _select_fit_key(self, func_name, ch, evt, quiet: bool = False):
        region = self.fit_regions.get(ch)
        if region is None:
            if not quiet:
                QMessageBox.warning(self, "No Region", "Select a region enclosing the fit first.")
            return None
        prefix = f"evt_{evt}_{func_name}_ch{ch}"
        matches = []
        for k, rec in self.results.items():
            if not k.startswith(prefix):
                continue
            if self.results[k].get('dataset_id') is not None and self.results[k].get('dataset_id') != self.dataset_id:
                continue
            fit_reg = rec.get("fit_region")
            if fit_reg and region[0] <= fit_reg[0] and region[1] >= fit_reg[1]:
                matches.append(k)
        if len(matches) != 1:
            if not quiet:
                QMessageBox.warning(self, "Ambiguous Selection", "Highlight exactly one fit region.")
            return None
        return matches[0]

    def adjust_dynamic_params(self):
        func_name = self.dyn_func_combo.currentText()
        ch = int(self.dyn_ch_combo.currentText())
        evt = self.event_combo.currentText()
        key = self._select_fit_key(func_name, ch, evt)
        if key is None:
            return
        rec = self.results[key]
        # Special-case: low_pass_max width adjustment
        if func_name == "low_pass_max":
            try:
                cur_w = int(rec.get("params", (self.sg_win.value() if hasattr(self, 'sg_win') else 201,))[0])
            except Exception:
                cur_w = 201
            # Build simple dialog with QSpinBox for width
            dlg = QDialog(self)
            dlg.setWindowTitle("Low-pass Max Settings")
            lay = QVBoxLayout(dlg)
            row = QHBoxLayout(); lay.addLayout(row)
            row.addWidget(QLabel("SG window (samples, odd):"))
            sp = QSpinBox(); sp.setRange(5, 9999); sp.setSingleStep(2)
            # Ensure odd
            if cur_w % 2 == 0: cur_w += 1
            sp.setValue(cur_w)
            row.addWidget(sp)
            btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
            lay.addWidget(btns)
            btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
            if dlg.exec_() != QDialog.Accepted:
                return
            w = int(sp.value())
            if w % 2 == 0:
                w += 1
            # Recompute marker
            t, y, _ = self.current_data[ch]
            region = self.fit_regions.get(ch) or rec.get("fit_region")
            mask = (t >= region[0]) & (t <= region[1])
            t_sel, y_sel = t[mask], y[mask]
            invert_flag = rec.get("inverted", False)
            y_proc = -y_sel if invert_flag else y_sel
            try:
                y_f = sig.savgol_filter(y_proc, w, 2)
                i = int(np.argmax(y_f)) if y_f.size else 0
                t_peak = float(t_sel[i]) if len(t_sel) else None
                val_proc = float(y_f[i]) if y_f.size else None
                if t_peak is None or val_proc is None:
                    return
                val = -val_proc if invert_flag else val_proc
            except Exception as e:
                QMessageBox.warning(self, "SG Error", str(e)); return
            ax = self.figure.axes[sorted(self.current_data.keys()).index(ch)]
            line = rec.get("line")
            if line in ax.lines:
                line.remove()
            new_line, = ax.plot([t_peak], [val], marker='o', color='tab:green', label=f"low_pass_max")
            new_line.set_gid(key)
            rec.update({
                "params": (int(w),),
                "param_names": ["width"],
                "value": float(val),
                "t_at": float(t_peak),
                "line": new_line,
                "fit_region": region,
            })
            ax.legend(); self.canvas.draw(); self.fit_info_window.update_info(self.results, evt)
            return
        names = rec["param_names"]
        params = rec["params"]
        func, _ = FIT_LIB[func_name]
        region = self.fit_regions.get(ch) or rec.get("fit_region")
        # Choose data source for adjustment preview: SG if toggle is on and available
        if hasattr(self, 'use_sg_toggle') and self.use_sg_toggle.isChecked() and (ch in self.sg_filtered):
            t, y = self.sg_filtered[ch]
        else:
            t, y, _ = self.current_data[ch]
        mask = (t >= region[0]) & (t <= region[1])
        t_sel, y_sel = t[mask], y[mask]
        if rec.get("inverted", False):
            y_sel = -y_sel
        guess, lower, upper = self._guess_params(func_name, names, t_sel, y_sel)
        bounds = (lower, upper) if any(np.isfinite(lower)) or any(np.isfinite(upper)) else None
        dlg = SciFitParamsDialog({n: p for n, p in zip(names, params)}, func, t_sel, y_sel, names, bounds, parent=self)
        ax = self.figure.axes[sorted(self.current_data.keys()).index(ch)]
        line = rec.get("line")
        orig_params = params
        orig_inverted = rec.get("inverted", False)
        invert_flag = self.dyn_invert.isChecked()

        def plot_current(param_dict):
            nonlocal line
            new_params = [param_dict[n] for n in names]
            y_fit = func(t_sel, *new_params)
            if invert_flag:
                y_plot = -y_fit
            else:
                y_plot = y_fit
            if line in ax.lines:
                line.set_data(t_sel, y_plot)
            else:
                idx_match = re.search(r"_(\d+)$", key)
                idx = idx_match.group(1) if idx_match else ""
                label = f"{func_name} Fit {idx}".strip()
                line, = ax.plot(t_sel, y_plot, color='tab:green', label=label)
                line.set_gid(key)
            self.canvas.draw_idle()

        dlg.params_changed.connect(plot_current)

        result = dlg.exec_()
        if result == QDialog.Accepted:
            params_dict = dlg.getParams()
            plot_current(params_dict)
            rec["params"] = tuple(params_dict[n] for n in names)
            rec["inverted"] = invert_flag
            rec["fit_region"] = region
            rec["line"] = line
            if func_name == "QD3Fit":
                charge, mass, rad = QD3Cal(params_dict["q"], params_dict["v"])
                rec.update({"charge": charge, "mass": mass, "radius": rad})
            elif func_name == "QDMFit":
                charge, mass, rad = QDMCal(params_dict["q"], params_dict["v"])
                rec.update({"charge": charge, "mass": mass, "radius": rad})
            ax.legend()
            self.canvas.draw()
            self.fit_info_window.update_info(self.results, evt)
        else:
            invert_flag = orig_inverted
            plot_current({n: v for n, v in zip(names, orig_params)})
            rec["line"] = line
            self.canvas.draw()

    def clear_dynamic_fit(self):
        func_name = self.dyn_func_combo.currentText()
        ch = int(self.dyn_ch_combo.currentText())
        evt = self.event_combo.currentText()
        key = self._select_fit_key(func_name, ch, evt)
        if key is None:
            return
        rec = self.results.pop(key)
        ax = self.figure.axes[sorted(self.current_data.keys()).index(ch)]
        line = rec.get("line")
        if line in ax.lines:
            line.remove()
        ax.legend()
        self.canvas.draw()
        self.fit_info_window.update_info(self.results, self.event_combo.currentText())

    def clear_channel_fits(self):
        """Remove all fits for the selected channel on the current event."""
        evt = self.event_combo.currentText()
        if not evt:
            return
        ch = int(self.dyn_ch_combo.currentText())
        # Build regex for keys matching this event and channel
        pat = re.compile(rf'^evt_{re.escape(str(evt))}_.+_ch{ch}(?:_\d+)?$')
        keys = [k for k in list(self.results.keys()) if pat.match(k)]
        if not keys:
            QMessageBox.information(self, "Clear Channel Fits", f"No fits found for CH{ch} on event {evt}.")
            return
        # Remove overlays by gid
        prefix = f"evt_{evt}_"
        for ax in self.figure.axes:
            for line in list(ax.lines):
                gid = line.get_gid()
                if isinstance(gid, str) and pat.match(gid):
                    try:
                        ax.lines.remove(line)
                    except Exception:
                        pass
            ax.legend()
        # Remove records
        for k in keys:
            self.results.pop(k, None)
        self.canvas.draw()
        self.fit_info_window.update_info(self.results, evt)

    def recall_dynamic_fit(self, evt, ch, key, func_name):
        if ch not in self.current_data:
            return
        rec = self.results[key]
        params = rec.get("params", ())
        names = rec.get("param_names", [])
        t, y, _ = self.current_data[ch]
        region = rec.get("fit_region")
        if region is not None:
            mask = (t >= region[0]) & (t <= region[1])
            t_sel, y_sel = t[mask], y[mask]
            self.fit_regions[ch] = region
        else:
            t_sel, y_sel = t, y
        inverted = rec.get("inverted", False)
        ax = self.figure.axes[sorted(self.current_data.keys()).index(ch)]
        # Special handling for low_pass_max
        if func_name == "low_pass_max":
            w = int(params[0]) if params else 201
            # Ensure odd and valid
            w = max(5, w)
            if w % 2 == 0:
                w += 1
            w = min(w, max(5, len(y_sel) - 1))
            if w % 2 == 0:
                w -= 1
            y_proc = -y_sel if inverted else y_sel
            try:
                y_f = sig.savgol_filter(y_proc, w, 2)
                i = int(np.argmax(y_f)) if y_f.size else 0
                t_peak = float(t_sel[i]) if len(t_sel) else None
                val_proc = float(y_f[i]) if y_f.size else None
                if t_peak is None or val_proc is None:
                    return
                val = -val_proc if inverted else val_proc
            except Exception:
                return
            line = rec.get("line")
            if line in ax.lines:
                line.remove()
            idx_match = re.search(r"_(\d+)$", key)
            idxs = idx_match.group(1) if idx_match else ""
            label = f"{func_name} {idxs}".strip()
            new_line, = ax.plot([t_peak], [val], marker='o', color='tab:green', label=label)
            new_line.set_gid(key)
            rec["line"] = new_line
            if region is not None:
                ax.axvline(region[0], color="red", linestyle="--", alpha=0.7)
                ax.axvline(region[1], color="red", linestyle="--", alpha=0.7)
            ax.legend()
            return

        # Default path for functional fits
        func = FIT_LIB.get(func_name, (None,))[0]
        if func is None:
            return
        y_sel_eff = -y_sel if inverted else y_sel
        y_fit = func(t_sel, *params)
        if inverted:
            y_fit = -y_fit
        line = rec.get("line")
        if line in ax.lines:
            line.remove()
        idx_match = re.search(r"_(\d+)$", key)
        idx = idx_match.group(1) if idx_match else ""
        label = f"{func_name} Fit {idx}".strip()
        new_line, = ax.plot(t_sel, y_fit, color='tab:green', label=label)
        new_line.set_gid(key)
        rec["line"] = new_line
        if region is not None:
            ax.axvline(region[0], color="red", linestyle="--", alpha=0.7)
            ax.axvline(region[1], color="red", linestyle="--", alpha=0.7)
        ax.legend()

    def export_fits(self):
        if not self.results:
            QMessageBox.warning(self, "No Fits", "There are no fit results to export.")
            return
        df = self._build_fits_dataframe()
        if df is None or df.empty:
            QMessageBox.warning(self, "No Fits", "There are no fit results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Fits", "", "HDF5 Files (*.h5)")
        if not path:
            return
        try:
            with pd.HDFStore(path) as store:
                store['fits'] = df
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export fits:\n{e}")
            return
        QMessageBox.information(self, "Exported", f"Exported {len(df)} fit records to {path}")

    def show_fit_info(self):
        """Display the fit information window."""
        self.fit_info_window.update_info(self.results, self.event_combo.currentText())
        self.fit_info_window.show()
        self.fit_info_window.raise_()

    # End export_fits

    def _build_fits_dataframe(self):
        """Build a DataFrame from self.results in the same schema as export."""
        if not self.results:
            return pd.DataFrame()
        records = []
        pattern = re.compile(r'^evt_(\d+)_(.+)_ch(\d+)(?:_\d+)?$')
        for key, rec in self.results.items():
            m = pattern.match(key)
            if not m:
                continue
            event, fit_type, ch_str = m.groups()
            ch = int(ch_str)
            row = {'event': event, 'channel': ch, 'fit_type': fit_type, 'dataset_id': rec.get('dataset_id')}
            params = rec.get('params', ())
            ft = fit_type.lower()
            if ft.startswith('qd3'):
                row.update({
                    't0': params[0] if len(params)>0 else None,
                    'q':  params[1] if len(params)>1 else None,
                    'v':  params[2] if len(params)>2 else None,
                    'C':  params[3] if len(params)>3 else None,
                    'charge': rec.get('charge', None),
                    'mass':   rec.get('mass', None),
                    'radius': rec.get('radius', None),
                })
                row['impact_time'] = rec.get('impact_time', None)
            elif ft.startswith('qdm'):
                row.update({
                    't0': params[0] if len(params)>0 else None,
                    'q':  params[1] if len(params)>1 else None,
                    'v':  params[2] if len(params)>2 else None,
                    'C':  params[3] if len(params)>3 else None,
                    'charge': rec.get('charge', None),
                    'mass':   rec.get('mass', None),
                    'radius': rec.get('radius', None),
                })
                row['impact_time'] = rec.get('impact_time', None)
            elif ft.startswith('csa'):
                row.update({
                    't0': params[0] if len(params)>0 else None,
                    'C0': params[1] if len(params)>1 else None,
                    'C1': params[2] if len(params)>2 else None,
                    'C2': params[3] if len(params)>3 else None,
                    'T0': params[4] if len(params)>4 else None,
                    'T1': params[5] if len(params)>5 else None,
                    'T2': params[6] if len(params)>6 else None,
                    'C':  params[7] if len(params)>7 else None,
                })
            else:
                names = rec.get('param_names', [f'p{i}' for i in range(len(params))])
                for n, v in zip(names, params):
                    row[n] = v
                # Common optional fields for generic/special fits (e.g., low_pass_max)
                if 'impact_time' in rec:
                    row['impact_time'] = rec['impact_time']
                if 'radius' in rec:
                    row['radius'] = rec['radius']
                if 'value' in rec:
                    row['value'] = rec['value']
                if 't_at' in rec:
                    row['t_at'] = rec['t_at']
            # Persist inverted flag for round-trip accuracy
            row['inverted'] = bool(rec.get('inverted', False))
            records.append(row)
        return pd.DataFrame.from_records(records)

    def open_results_plotter(self):
        """Open the Results Plotter window seeded with current in-memory fits."""
        df = self._build_fits_dataframe()
        if df is None or df.empty:
            QMessageBox.information(self, "Results Plotter", "No fits to visualize. Perform or load fits first.")
            return
        # Apply current event list filters (Feature Scan / Fit Filter) to the DataFrame
        try:
            visible_events = [self.event_combo.itemText(i) for i in range(self.event_combo.count())]
            if visible_events:
                mask = df['event'].astype(str).isin(set(visible_events))
                df = df.loc[mask].copy()
        except Exception:
            pass
        if df is None or df.empty:
            QMessageBox.information(self, "Results Plotter", "No fits match the current event filter(s). Clear filters or adjust criteria.")
            return
        try:
            win = FitResultsVisualizer()
            # Seed with in-memory DataFrame and build groups
            win.raw = df.copy()
            win.rebuild_groups()
            # Default dataset filter to current dataset if present
            ds = getattr(self, 'dataset_id', None)
            if ds:
                idx = win.ds_combo.findText(str(ds))
                if idx >= 0:
                    win.ds_combo.setCurrentIndex(idx)
                    win.on_dataset_changed()
            # Connect pick signal to jump to event in this window
            try:
                win.eventPicked.connect(self._on_results_pick)
            except Exception:
                pass
            win.show()
            self._child_windows.append(win)  # keep reference
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Results Plotter", f"Failed to open Results Plotter:\n{e}\n\n{tb}")

    def _on_results_pick(self, ds_id: str, ev: str):
        """Respond to a selection in Results Plotter by switching event here."""
        # If dataset matches current, just switch event if present
        cur_ds = getattr(self, 'dataset_id', None)
        target_ev = str(ev).strip()
        if not target_ev:
            return
        if cur_ds and ds_id and str(cur_ds) != str(ds_id):
            # Different dataset; for now, only switch if the event exists in current list
            pass
        # Try to find the event in the current combo (handles zero padding)
        idx = self.event_combo.findText(target_ev)
        if idx == -1 and hasattr(self, '_all_event_keys'):
            # Try to match by numeric equivalence
            try:
                tev = int(target_ev)
                for i in range(self.event_combo.count()):
                    txt = self.event_combo.itemText(i)
                    try:
                        if int(txt) == tev:
                            idx = i; break
                    except Exception:
                        continue
            except Exception:
                pass
        if idx >= 0:
            self.event_combo.setCurrentIndex(idx)

    def import_fits(self):
        """Load fits from an HDF5 file and merge into the session.
        Expected key: 'fits' with columns including at least
        ['event','channel','fit_type'] and parameter columns.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Load Fits", "", "HDF5 Files (*.h5 *.hdf5)")
        if not path:
            return
        try:
            with pd.HDFStore(path) as store:
                ks = set(store.keys())
                key = '/fits' if '/fits' in ks else ('fits' if 'fits' in ks else None)
                if key is None:
                    QMessageBox.warning(self, "Invalid File", "HDF5 file does not contain a 'fits' dataset.")
                    return
                df = store[key]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load fits:\n{e}")
            return

        if df is None or len(df) == 0:
            QMessageBox.information(self, "No Data", "No fit records found in file.")
            return

        required = {'event', 'channel', 'fit_type'}
        if not required.issubset(set(df.columns)):
            QMessageBox.warning(self, "Invalid Data", f"'fits' dataset missing required columns: {sorted(required)}")
            return

        def _is_num(x):
            try:
                return np.isfinite(float(x))
            except Exception:
                return False
        def _to_bool(x):
            try:
                if isinstance(x, (bool, np.bool_)):
                    return bool(x)
                if x is None:
                    return False
                s = str(x).strip().lower()
                if s in ('1','true','t','yes','y','on'):
                    return True
                if s in ('0','false','f','no','n','off','nan',''):
                    return False
                # Fallback: numeric nonzero -> True
                return float(x) != 0.0
            except Exception:
                return False

        base_keys = set(self.results.keys())
        added = 0
        added_current_ds = 0

        # Helper to normalize event key to match loaded dataset (preserve leading zeros)
        def _normalize_event_key(ev_raw):
            s = str(ev_raw).strip()
            # If numeric, try to match an existing key by integer equality
            try:
                s_int = int(s)
            except Exception:
                return s
            keys = getattr(self, '_all_event_keys', None)
            if not keys:
                keys = list(getattr(self, 'event_files', {}).keys())
            for k in keys or []:
                try:
                    if int(k) == s_int:
                        return k
                except Exception:
                    continue
            return str(s_int)

        # Iterate row-wise and rebuild records
        for _, row in df.iterrows():
            try:
                ev_raw = row['event']
                # Normalize event to match keys in this dataset (retain zero padding)
                ev_str = _normalize_event_key(ev_raw)
                ch = int(row['channel'])
                fit_type = str(row['fit_type'])
            except Exception:
                continue

            # Build parameter names and values
            names = None
            params = []
            ft_lower = fit_type.lower()
            if fit_type in FIT_LIB:
                names = list(FIT_LIB[fit_type][1])
                # Only include names present in the row
                vals = []
                ok = True
                for n in names:
                    if n in row and _is_num(row[n]):
                        vals.append(float(row[n]))
                    else:
                        ok = False
                        break
                if ok:
                    params = vals
                else:
                    names = None  # fall back to generic discovery

            if names is None:
                # Generic: collect numeric columns that are not meta
                meta_cols = {'event','channel','fit_type','dataset_id','impact_time','charge','mass','radius'}
                param_candidates = [c for c in df.columns if c not in meta_cols and _is_num(row.get(c))]
                # If pN style, sort by index; else preserve DataFrame order
                def p_index(c):
                    m = re.match(r'^p(\d+)$', str(c))
                    return int(m.group(1)) if m else None
                p_cols = [(c, p_index(c)) for c in param_candidates]
                if any(idx is not None for _, idx in p_cols):
                    p_cols = sorted([(c, idx) for c, idx in p_cols if idx is not None], key=lambda x: x[1])
                    names = [c for c, _ in p_cols]
                else:
                    names = list(param_candidates)
                params = [float(row[n]) for n in names]

            # Dataset ID handling
            ds_val = None
            if 'dataset_id' in row:
                try:
                    ds_val = row['dataset_id']
                    # Treat NaN as missing
                    if isinstance(ds_val, float) and not np.isfinite(ds_val):
                        ds_val = None
                except Exception:
                    ds_val = None
            # If dataset is missing in file, assume current dataset
            if ds_val is None:
                ds_val = getattr(self, 'dataset_id', None)

            rec = {
                'params': tuple(params),
                'param_names': list(names),
                'dataset_id': ds_val,
                'inverted': _to_bool(row['inverted']) if 'inverted' in row else False,
            }
            # Optional extras
            if 'impact_time' in row and _is_num(row['impact_time']):
                rec['impact_time'] = float(row['impact_time'])
            if 'charge' in row and _is_num(row['charge']):
                rec['charge'] = float(row['charge'])
            if 'mass' in row and _is_num(row['mass']):
                rec['mass'] = float(row['mass'])
            if 'radius' in row and _is_num(row['radius']):
                rec['radius'] = float(row['radius'])
            if 'value' in row and _is_num(row['value']):
                rec['value'] = float(row['value'])
            if 't_at' in row and _is_num(row['t_at']):
                rec['t_at'] = float(row['t_at'])

            base = f"evt_{ev_str}_{fit_type}_ch{ch}"
            key = base
            idx = 1
            while key in self.results:
                idx += 1
                key = f"{base}_{idx}"

            self.results[key] = rec
            added += 1
            try:
                if rec.get('dataset_id') == self.dataset_id:
                    added_current_ds += 1
            except Exception:
                pass

        # If a current event is visible, redraw overlays from imported fits that match dataset
        evt = self.event_combo.currentText()
        if evt:
            self.recall_fits_for_event(evt)
        self.fit_info_window.update_info(self.results, evt)

        QMessageBox.information(
            self,
            "Loaded Fits",
            f"Imported {added} fit records from\n{os.path.basename(path)}\n"
            + (f"({added_current_ds} match current dataset)" if added_current_ds else "")
        )

    def clear_sg_filter(self):
        ch = self.sg_ch.currentText()
        # SG is stored in self.sg_filtered, not self.results
        self.sg_filtered.pop(int(ch), None)
        for ax in self.figure.axes:
            for line in list(ax.lines):
                if line.get_label() == 'SG Filter':
                    ax.lines.remove(line)
            ax.legend()
        self.canvas.draw()

    def save_figure_transparent(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "", "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            # make figure and axes backgrounds transparent
            self.figure.patch.set_facecolor('none')
            for ax in self.figure.axes:
                ax.patch.set_facecolor('none')
            # save with transparent background
            self.figure.savefig(path, transparent=True, dpi=300)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save figure:\n{e}")
            return
        QMessageBox.information(self, "Saved", f"Figure saved to {path}")

    def _on_decim_change(self, val):
        # Replot waveforms and restore any fits
        evt = self.event_combo.currentText()
        if not evt:
            return
        # Redraw base waveforms with new decimation
        self.plot_waveforms()
        # Recall all existing fits for this event
        self.recall_fits_for_event(evt)

    def clear_data(self):
        """Remove all stored fit results and clear plot annotations"""
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            "Are you sure you want to clear all fit data? This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Clear the results dict
        self.results.clear()
        # Remove plotted fit lines and impact lines
        for ax in self.figure.axes:
            lines_to_remove = [ln for ln in ax.lines if ln.get_linestyle() == '-.' or ln.get_label().endswith('Fit')]
            for ln in lines_to_remove:
                ax.lines.remove(ln)
            ax.legend()
        self.canvas.draw()
        self.fit_info_window.update_info(self.results, self.event_combo.currentText())

    def run_feature_scan(self):
        """Scan events for excursions above threshold×std on a selected channel."""
        # Build event (int, key) list preserving zero padding
        ev_pairs = []
        for i in range(self.event_combo.count()):
            key = self.event_combo.itemText(i)
            m = re.search(r"(\d+)$", key)
            if not m:
                continue
            ev_pairs.append((int(m.group(1)), key))
        if not ev_pairs:
            QMessageBox.warning(self, "No Events", "No events found in the current folder.")
            return
        ev_pairs.sort(key=lambda x: x[0])
        evt_min = ev_pairs[0][0]
        evt_max = ev_pairs[-1][0]

        # Default channel from dynamic or SG selection if available
        ch_default = 1
        try:
            ch_default = int(self.dyn_ch_combo.currentText())
        except Exception:
            try:
                ch_default = int(self.sg_ch.currentText())
            except Exception:
                ch_default = 1

        # interactive rescan/review/apply loop
        prev = dict(ev0=evt_min, ev1=evt_max, ch=ch_default, thr=5.0, use_abs=True)
        while True:
            scan_dlg = FeatureScanDialog(self, prev['ev0'], prev['ev1'], ch_default=prev['ch'], thr_default=prev['thr'])
            scan_dlg.abs_chk.setChecked(prev.get('use_abs', True))
            if scan_dlg.exec_() != QDialog.Accepted:
                return  # cancel
            try:
                ev0, ev1, ch, thr_mult, use_abs = scan_dlg.values()
            except Exception as e:
                QMessageBox.warning(self, "Invalid Input", str(e))
                continue
            prev.update(ev0=ev0, ev1=ev1, ch=ch, thr=thr_mult, use_abs=use_abs)

            targets = [key for (iv, key) in ev_pairs if ev0 <= iv <= ev1]
            if not targets:
                QMessageBox.information(self, "No Events", "No events in the specified range.")
                continue

            prog = QProgressDialog("Scanning...", "Cancel", 0, len(targets), self)
            prog.setWindowModality(Qt.ApplicationModal)
            prog.setAutoClose(True)
            prog.show()

            matches = []
            for i, key in enumerate(targets, start=1):
                prog.setValue(i - 1)
                prog.setLabelText(f"Scanning event {key} ({i}/{len(targets)})")
                QApplication.processEvents()
                if prog.wasCanceled():
                    break
                paths = self.event_files.get(key, [])
                ch_path = None
                for p in paths:
                    if re.match(fr"C{ch}.*", os.path.basename(p)):
                        ch_path = p
                        break
                if ch_path is None:
                    continue
                try:
                    _, y, _ = self.trc.open(ch_path)
                except Exception:
                    continue
                y = y - np.mean(y)
                sd = float(np.std(y))
                if sd <= 0:
                    continue
                peak = float(np.max(np.abs(y))) if use_abs else float(np.max(y))
                if peak > thr_mult * sd:
                    matches.append(key)

            prog.setValue(len(targets))

            res_dlg = FeatureResultsDialog(self, matches)
            res_dlg.exec_()
            act = res_dlg.action()
            if act == 'apply':
                # Apply event filter using match list
                self.apply_event_filter(matches)
                return
            elif act == 'rescan':
                # loop to rescan with possibly adjusted defaults
                continue
            else:
                # cancel/close; do nothing
                return

    def apply_event_filter(self, keys):
        """Filter the event list to the provided keys, compounding with any existing filter."""
        if not keys:
            QMessageBox.information(self, "No Matches", "No events to filter to.")
            return
        # If a filter is already active, compound by intersecting with current filtered set
        avail = set(self.feature_filter_keys) if self.feature_filter_active else set(self._all_event_keys)
        filt = [k for k in keys if k in avail]
        if not filt:
            QMessageBox.information(self, "No Matches", "No matching events found in current dataset.")
            return
        self.feature_filter_active = True
        self.feature_filter_keys = list(filt)
        # Preserve current selection if possible
        current = self.event_combo.currentText()
        self.event_combo.blockSignals(True)
        self.event_combo.clear()
        # sort by numeric suffix if possible
        filt_sorted = sorted(filt, key=lambda x: int(re.search(r"(\d+)$", x).group(1)) if re.search(r"(\d+)$", x) else x)
        self.event_combo.addItems(filt_sorted)
        self.event_combo.blockSignals(False)
        # restore selection or select first
        idx = self.event_combo.findText(current)
        if idx == -1 and self.event_combo.count() > 0:
            idx = 0
        if idx != -1:
            self.event_combo.setCurrentIndex(idx)

    def run_fit_filter(self):
        """Filter listed events by fit parameter criteria (e.g., QD3Fit v > 1e3)."""
        # Available fit types are the ones we know how to name
        fit_types = sorted(FIT_LIB.keys())
        ch_default = 1
        try:
            ch_default = int(self.dyn_ch_combo.currentText())
        except Exception:
            pass

        dlg = FitFilterDialog(self, fit_types, ch_default=ch_default)
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            fit_name, ch, param, op, v1, v2 = dlg.values()
        except Exception as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        # Build a map from numeric event id -> actual event key (preserve zero padding)
        ev_map = {}
        for key in self._all_event_keys:
            m = re.search(r"(\d+)$", key)
            if not m:
                continue
            ev_map.setdefault(int(m.group(1)), []).append(key)

        pattern = re.compile(r'^evt_(\d+)_(.+)_ch(\d+)(?:_\d+)?$')
        matched_event_ints = set()
        for key, rec in self.results.items():
            m = pattern.match(key)
            if not m:
                continue
            ev_str, ftype, ch_str = m.groups()
            if ftype != fit_name:
                continue
            if int(ch_str) != int(ch):
                continue
            # Restrict to current dataset if tagged
            if rec.get('dataset_id') is not None and rec.get('dataset_id') != self.dataset_id:
                continue
            # Pull parameter value
            val = None
            if param in rec:
                val = rec.get(param)
            else:
                names = rec.get('param_names', [])
                try:
                    idx = names.index(param)
                    p = rec.get('params', ())
                    if idx < len(p):
                        val = p[idx]
                except Exception:
                    val = None
            # Skip missing values
            try:
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    continue
            except Exception:
                continue

            ok = False
            try:
                if op == ">":
                    ok = (val > v1)
                elif op == ">=":
                    ok = (val >= v1)
                elif op == "<":
                    ok = (val < v1)
                elif op == "<=":
                    ok = (val <= v1)
                elif op == "==":
                    ok = (val == v1)
                elif op == "between":
                    ok = (v1 <= val <= (v2 if v2 is not None else v1))
            except Exception:
                ok = False
            if ok:
                try:
                    matched_event_ints.add(int(ev_str))
                except Exception:
                    pass

        # Translate numeric ids back to actual event keys
        matches = []
        for evn in sorted(matched_event_ints):
            matches.extend(ev_map.get(evn, []))

        if not matches:
            QMessageBox.information(self, "No Matches", "No events met the selected criterion.")
            return

        # Show preview list and allow apply
        res_dlg = FeatureResultsDialog(self, matches)
        res_dlg.exec_()
        if res_dlg.action() == 'apply':
            self.apply_event_filter(matches)
        # else close/rescan handled by user closing dialog

    def clear_event_filter(self):
        """Restore full event list after applying a feature filter."""
        if not self._all_event_keys:
            return
        self.feature_filter_active = False
        self.feature_filter_keys = []
        current = self.event_combo.currentText()
        self.event_combo.blockSignals(True)
        self.event_combo.clear()
        self.event_combo.addItems(self._all_event_keys)
        self.event_combo.blockSignals(False)
        idx = self.event_combo.findText(current)
        if idx == -1 and self.event_combo.count() > 0:
            idx = 0
        if idx != -1:
            self.event_combo.setCurrentIndex(idx)

    #Implement qd3 batch routine with progress dialog and cancellation:
    def show_help(self):
        """
        Display a help dialog summarizing the tool and each button/function.
        """
        help_text = (
            "Oscilloscope Palantir\n\n"
            "Purpose: Interactive analysis of LeCroy .trc waveforms — load events, filter, fit pulses, match metadata, and export results.\n\n"
            "Controls (by row):\n"
            "• Select Folder (F), Event dropdown, Folder path label.\n"
            "• Navigation: Prev/Next (, .), Decim.  Session: Clear Event Fits (E), Save (Ctrl+S), Export (Ctrl+E), Load (Ctrl+I), Clear Data (Shift+D), Fit Info (U).\n"
            "• Metadata/Impact: Load Metadata (L), MetaMatch (M), D3Dist, Mark Impact (T).\n"
            "• SG Row: SG Filter (S), Chan, Width, SG Batch (Ctrl+B), Clear SG (Shift+S) | Results Plotter (P), Feature Scan (Ctrl+F), Fit Filter (Ctrl+Shift+F), Clear Filter (Ctrl+K).\n"
            "• Dynamic Row: Fit Func (QD3Fit/QDMFit/CSA/skew_gaussian/gaussian/FFT/low_pass_max), Chan (1–8), Invert (I), Run (Enter), Adjust (A), Clear (R), Clear Chan Fits (Shift+R), Batch (B), Use SG (Ctrl+Shift+S).\n\n"
            "Notes:\n"
            "• Fits run on raw data by default; QD3/QDM use SG only for initial parameter guessing.\n"
            "• ‘Use SG’ applies SG-filtered data (if present) to Run Fit, Adjust Fit, and FFT.\n"
            "• FFT opens a popup spectrum with pan/zoom toolbar in log–log scale.\n"
            "• low_pass_max performs an SG low‑pass in the selection and reports the extremum.\n\n"
            "Hotkeys:\n"
            "  Navigation:  , / .  (Prev/Next),  T (Mark Impact)\n"
            "  Files:       F (Select Folder),  Ctrl+S (Save),  Ctrl+E (Export),  Ctrl+I (Load),  Shift+D (Clear Data)\n"
            "  Meta:        L (Load Metadata),  M (MetaMatch)\n"
            "  SG/Filters:  S (SG),  Ctrl+B (SG Batch),  Shift+S (Clear SG),  P (Results Plotter),  Ctrl+F (Feature Scan),  Ctrl+Shift+F (Fit Filter),  Ctrl+K (Clear Filter)\n"
            "  Dynamic:     Enter (Run),  A (Adjust),  R (Clear Fit),  Shift+R (Clear Chan Fits),  B (Batch),  Ctrl+Shift+S (Toggle Use SG)\n"
            "               Q/D/C/W/G/X selects QD3/QDM/CSA/skew/gauss/low_pass_max; 1–8 selects channel; I toggles invert.\n"
            "  Info:        H (Help),  U (Fit Info)\n"
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Help")
        dlg.resize(500, 400)
        layout = QVBoxLayout(dlg)
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(help_text)
        layout.addWidget(text)
        btn_close = QPushButton("Close")
        btn_close.setToolTip("Close this window")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)
        dlg.exec_()
# END class OscilloscopeAnalyzer(QMainWindow):

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('fusion')
    win = OscilloscopeAnalyzer()
    win.show()
    sys.exit(app.exec_())

# END
