# Open Use License
#
# You may use, copy, modify, and redistribute this file for any purpose,
# provided that redistributions include this original source code (including this notice).
# This software is provided "as is" without warranty of any kind.

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QMessageBox
from scipy.signal import fftconvolve
from scipy.special import erf
from scipy.optimize import curve_fit
import scipy.signal as sig
import numpy as np
import csv

def QDMCal(q,v, rho=7500):
	c      = 1e-12 / 20 * 0.95 * 2.35 #Cal
	charge = c * q	
	mass   = abs(2 * 2.2e6 * charge / (v**2))
	rad    = np.power((3.0 * mass / (4.0 * np.pi * rho)), 1/3) # m
	return charge, mass, rad

def QD3Cal(q,v, rho=7500):
	c      = 1e-12 / 20 * 0.95  #Cal
	charge = c * q	
	mass   = abs(2 * 2.2e6 * charge / (v**2))
	rad    = np.power(float(3.0 * mass / (4.0 * np.pi * rho)), 1/3) # m
	return charge, mass, rad


# Utility to load metadata CSV into a list of dicts
def getMetaData(metaFile):
    metaArr = []
    with open(metaFile) as csv_file:
        csv_in = csv.DictReader(csv_file)
        for row in csv_in:
            metaArr.append(row)
    return metaArr

# Function to match an event to a metaArr event based on time and velocity criteria
def metaMatch(metaArr, time, velocity):
    matchWindow = 1.0  # seconds

    # 1) Try UTC Timestamp
    close_events = [
        row for row in metaArr
        if abs(float(row.get('UTC Timestamp', 0)) - time) <= matchWindow
    ]

    # 2) If none, fall back to Local Time or Local Timestamp
    if not close_events:
        close_events = [
            row for row in metaArr
            if abs(float(row.get('Local Time', row.get('Local Timestamp', 0))) - time) <= matchWindow
        ]
        if not close_events:
            return False

    # 3) Filter by velocity tolerance
    acceptable = [
        row for row in close_events
        if abs(float(row['Velocity (m/s)']) - velocity) <= 0.08 * velocity
    ]
    if not acceptable:
        return False

    # 4) Pick the best (closest in velocity)
    best_event = min(
        acceptable,
        key=lambda r: abs(float(r['Velocity (m/s)']) - velocity)
    )
    return best_event

# Model functions for fitting
def CSA_pulse(x, t0, C0, C1, C2, T0, T1, T2, C):
    return (
        C0
        - np.heaviside(t0 - x, 1) * C1 * np.exp(-(((x - t0) ** 2.0) / T0**2.0))
        + np.heaviside(x - t0, 0) * (
            C2 * (1.0 - np.exp(-(x - t0) / T1)) * np.exp(-(x - t0) / T2)
            - C1
        )
    ) + C

def QDMFit(Time, t0, q, v, C):
    v = v * 100
    PUT_l, PUT_d = 15.3, 2.0 #14.5, 1.2
    tau1, tau2 = 0.4e-3, 0.4e-3
    dt1 = PUT_d / v; dt2 = (PUT_l - 2 * PUT_d) / v
    signal = np.zeros(len(Time))
    idx1 = np.where((Time > t0) & (Time < t0 + dt1))
    signal[idx1] = q / dt1 * (Time[idx1] - t0)
    idx2 = np.where((Time > t0 + dt1) & (Time < t0 + dt1 + dt2))
    signal[idx2] = q * np.exp(-(Time[idx2] - (t0 + dt1)) / tau1)
    qq = signal[idx2[0][-1]] if len(idx2[0]) > 0 else 0
    idx3 = np.where((Time > t0 + dt1 + dt2) & (Time < t0 + 2 * dt1 + dt2))
    signal[idx3] = qq - q / dt1 * (Time[idx3] - (t0 + dt1 + dt2))
    qq = signal[idx3[0][-1]] if len(idx3[0]) > 0 else qq
    idx4 = np.where(Time > t0 + 2 * dt1 + dt2)
    signal[idx4] = qq * np.exp(-(Time[idx4] - (t0 + 2 * dt1 + dt2)) / tau2)
       # Sav-Gol Method
	#convert time into num samples for 
    dt          = Time[1] - Time[0]
    window = 3e-6 #seconds
    num_samples = int(window / dt)
    if num_samples < 1:
        num_samples = 1
    smoothed    = sig.savgol_filter(signal, num_samples, 1)

    return smoothed + C

def skew_gaussian(x, A, xi, omega, alpha, C):
    """
    Skew-Gaussian (Azzalini) PDF shape for fitting.

    Parameters
    ----------
    x : array_like
        Independent variable.
    A : float
        Amplitude (area under the curve = A).
    xi : float
        Location parameter (mean of the underlying Gaussian).
    omega : float
        Scale parameter (standard deviation of the underlying Gaussian).
    alpha : float
        Shape (skew) parameter.  alpha=0 → ordinary Gaussian.

    Returns
    -------
    y : ndarray
        Skew-Gaussian evaluated at each x.
    """
    t = (x - xi) / omega
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * t**2)
    Phi = 0.5 * (1.0 + erf(alpha * t / np.sqrt(2.0)))
    return A * 2.0 / omega * phi * Phi + C

def QD3Fit(Time, t0, q, v, C):
    # --- parameters ---
    v = v * 100  # convert to appropriate units
    PUT_l, PUT_d = 20.0, 0.90
    tau1, tau2 = 0.13e-3, 0.15e-3 # 0.12, 0.15 default
    dt1 = PUT_d / v
    dt2 = (PUT_l - 2 * PUT_d) / v

    # --- original piecewise signal ---
    signal = np.zeros_like(Time)
    # rising edge
    idx1 = (Time > t0) & (Time < t0 + dt1)
    signal[idx1] = (q / dt1) * (Time[idx1] - t0)
    # plateau decay
    idx2 = (Time >= t0 + dt1) & (Time < t0 + dt1 + dt2)
    signal[idx2] = q * np.exp(-(Time[idx2] - (t0 + dt1)) / tau1)

    # falling edge
    if idx2.any():
        qq = signal[idx2][-1]
    else:
        qq = 0.0
    idx3 = (Time >= t0 + dt1 + dt2) & (Time < t0 + 2*dt1 + dt2)
    signal[idx3] = qq - (q / dt1) * (Time[idx3] - (t0 + dt1 + dt2))
    if idx3.any():
        qq = signal[idx3][-1]
    idx4 = (Time >= t0 + 2*dt1 + dt2)
    signal[idx4] = qq * np.exp(-(Time[idx4] - (t0 + 2*dt1 + dt2)) / tau2)

    # Sav-Gol Method
	#convert time into num samples for 
    dt          = Time[1] - Time[0]
    window = 3e-6 #seconds
    num_samples = int(window / dt)
    if num_samples < 1:
        num_samples = 1
    smoothed    = sig.savgol_filter(signal, num_samples, 1)

    return smoothed + C

class FitParamsDialog(QDialog):
    params_changed = pyqtSignal(dict)

    def __init__(self, params, func=None, t_sel=None, y_sel=None, names=None, bounds=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adjust Fit Parameters")
        layout = QFormLayout(self)
        self.edits = {}
        self.func = func
        self.t_sel = t_sel
        self.y_sel = y_sel
        self.names = names or list(params.keys())
        self.bounds = bounds

        for name, val in params.items():
            spin = QDoubleSpinBox(self)
            spin.setDecimals(6)
            # ensure a non‐zero range even if val==0
            mag = abs(val) if abs(val)>1e-12 else 1.0
            spin.setRange(-10*mag, 10*mag)
            spin.setValue(val)
            layout.addRow(name, spin)
            self.edits[name] = spin

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self.refit_btn = buttons.addButton("Refit", QDialogButtonBox.ActionRole)
        self.plot_btn = buttons.addButton("Plot", QDialogButtonBox.ActionRole)
        self.refit_btn.clicked.connect(self.refit)
        self.plot_btn.clicked.connect(self.emit_params)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def refit(self):
        if self.func is None or self.t_sel is None or self.y_sel is None:
            return
        p0 = [self.edits[n].value() for n in self.names]
        try:
            if self.bounds is not None:
                popt, _ = curve_fit(self.func, self.t_sel, self.y_sel, p0=p0,
                                    bounds=self.bounds, maxfev=20000)
            else:
                popt, _ = curve_fit(self.func, self.t_sel, self.y_sel, p0=p0,
                                    maxfev=20000)
            for n, val in zip(self.names, popt):
                self.edits[n].setValue(val)
            self.emit_params()
        except Exception as e:
            QMessageBox.warning(self, "Refit Failed", str(e))

    def getParams(self):
        return {name: spin.value() for name, spin in self.edits.items()}

    def emit_params(self):
        self.params_changed.emit(self.getParams())



def gaussian(x, A, mu, sigma, C):
    """
    Standard Gaussian peak with constant offset.

    Parameters
    ----------
    x : array_like
        Independent variable (time).
    A : float
        Peak amplitude (height).
    mu : float
        Peak center.
    sigma : float
        Standard deviation (width > 0).
    C : float
        Constant baseline offset.

    Returns
    -------
    y : ndarray
        Gaussian evaluated at each x.
    """
    x = np.asarray(x)
    sigma = np.asarray(sigma)
    # avoid divide-by-zero
    sigma = np.where(sigma == 0, 1e-15, sigma)
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + C
