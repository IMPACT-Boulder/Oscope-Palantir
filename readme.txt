# Oscilloscope Palantir
GUI tool for interactively analyzing LeCroy `.trc` oscilloscope waveform files, fitting pulses, applying filters, matching external metadata, and exporting results.

## Highlights
- Load multi‑channel events directly from `.trc` files and auto‑group files by *event id*.
- Plot raw waveforms with decimation for responsive UI on large traces.
- Drag‑select a time window per channel to define a fit region.
- Run model fits (QD3Fit, QDMFit, CSA_pulse, skew_gaussian, gaussian) on the selected region; adjust parameters live.
- Optional inversion for negative-going pulses.
- Savitzky–Golay (SG) smoothing overlay per channel; option to run fits/FFT on SG-filtered data.
- FFT analysis on the selected region (log–log spectrum in a popup with zoom/pan toolbar).
- Optional metadata matching from a CSV (e.g., velocity/charge/radius fields) once a QD3 fit exists.
- Export all fit results to a single HDF5 file for downstream analysis.
- Save the current figure (transparent background supported).
- Keyboard shortcuts for fast navigation and fitting.

## Repo Structure
- `oscope_palantir.py` — main PyQt5 GUI.
- `fitparams.py`       — model functions (QD3, QDM, CSA, skew Gaussian), calibration helpers, and the parameter-edit dialog.
- `readTrcDoner.py`    — compact LeCroy `.trc` reader returning `(time[s], voltage[V], metadata)`.

> If you keep additional viewers (e.g., a results HDF5 viewer), place them under a `results_plotter/` directory. This README describes the core GUI.

## Environment & Installation
**Recommended:** Python 3.9+ on Windows/macOS/Linux.

Create and activate a virtual environment (example name: `oscope-palantir`):
```bash
python -m venv oscope-palantir
# Linux/macOS
source oscope-palantir/bin/activate
# Windows (PowerShell)
.\oscope-palantir\Scripts\Activate.ps1
```

Install dependencies (edit `requirements.txt` as needed):
```bash
pip install -r requirements.txt
# To update the lockfile from the active env:
pip freeze > requirements.txt
```

Deactivate the venv when finished:
```bash
deactivate
```

Remove the environment (if needed):
```bash
rm -rf oscope-palantir
```

### Core Python Dependencies
`PyQt5`, `qtawesome`, `numpy`, `pandas`, `scipy`, `matplotlib`, `lmfit`

## Running the App
From the project directory (with the venv activated):
```bash
python oscope_palantir.py
```

## File Naming Assumptions
- Channels are inferred from filenames beginning with `C1`, `C2`, `C3`, `C4` (case‑insensitive).
- Files are grouped into *events* by the last `-<digits>.trc` suffix. Example set for event 1042:
  - `C1-1042.trc`, `C2-1042.trc`, `C3-1042.trc`, `C4-1042.trc`

## Typical Workflow
1. **Select Folder** containing `.trc` files. The app discovers all events and populates the event dropdown.
2. **Pick an event**; all available channels for that event are plotted.
3. (Optional) **Set Decimation** to plot every *N*th sample for speed.
4. **Drag‑select** a fit region on a channel (orange overlay). You can select different regions per channel.
5. Choose a **Fit Function** and **Channel**, optionally **Invert**, then press **Run Fit**.
6. Use **Adjust Fit** to tweak parameters and instantly update the overlay.
7. (Optional) **SG Filter**: Choose a channel and window width, then apply/clear the SG overlay. Enable **Use SG** in the dynamic row to run fits/FFT on the filtered data.
8. (Optional) **Load Metadata** CSV and run **MetaMatch** to associate a fitted event with a metadata row (e.g., by time & velocity).
9. **Export Fits** to HDF5 or **Save Figure** for reporting.

## Fit Models (Parameters)
- **QD3Fit(t, t0, q, v, C)** — Piecewise transient with smoothing; parameters include time offset `t0`, scale `q`, velocity proxy `v`, and offset `C`.
- **QDMFit(t, t0, q, v, C)** — Variant with different geometry/time constants and smoothing; same parameter names.
- **CSA_pulse(t, t0, C0, C1, C2, T0, T1, T2, C)** — Charge-sensitive amplifier‑like response with pre/post behavior and baseline.
- **skew_gaussian(x, A, xi, omega, alpha, C)** — Azzalini skew‑Gaussian; `A` (area), `xi` (location), `omega` (scale), `alpha` (skew), `C` (offset).

### Initial Guessing & Bounds
The app computes heuristic initial guesses from the selected region (e.g., rise/fall indices, area under the curve). Bounds ensure physically meaningful values (e.g., `v ≥ 0`).

### Calibration Helpers
For QD3/QDM fits, the app derives approximate **charge**, **mass**, and **radius** from fitted `q` and `v`. Density can be changed in code; default assumes ~7500 kg/m³.

## Metadata Matching
- Load a CSV with columns like `UTC Timestamp`, `Local Time`, `Velocity (m/s)`, `Charge (C)`, `Radius (m)`, etc.
- MetaMatch constructs a timestamp from the `.trc` trigger time and looks for rows within ±1 s, then chooses the closest by velocity (±8% tolerance).

> Tip: Run a QD3 fit first; MetaMatch requires `v` from a QD3 result.

## Exported Data (HDF5)
**Menu:** *Export Fits* → choose a destination HDF5 file. The tool writes a single table `/fits` with one row per fit. Columns include:
- `event`, `channel`, `fit_type`
- QD3/QDM: `t0`, `q`, `v`, `C`, derived `charge`, `mass`, `radius`, optional `impact_time`
- CSA: `t0`, `C0`, `C1`, `C2`, `T0`, `T1`, `T2`, `C`
- Additionally, the stored fit **region** and **inversion flag** are retained internally and reapplied when recalling fits for an event.

> You can open the HDF5 in Python (pandas) to post‑process, filter, or plot fits across a run.

## Keyboard Shortcuts
- Navigation: `,` (Prev), `.` (Next), `T` (Mark Impact)
- Files: `F` (Select Folder), `Ctrl+S` (Save), `Ctrl+E` (Export), `Ctrl+I` (Load), `Shift+D` (Clear Data)
- Meta: `L` (Load Metadata), `M` (MetaMatch)
- SG/Filters: `S` (SG Filter), `Ctrl+B` (SG Batch), `Shift+S` (Clear SG), `P` (Results Plotter), `Ctrl+F` (Feature Scan), `Ctrl+Shift+F` (Fit Filter), `Ctrl+K` (Clear Filter)
- Dynamic fits: `Enter` (Run), `A` (Adjust), `R` (Clear Fit), `Shift+R` (Clear Chan Fits), `B` (Batch), `Ctrl+Shift+S` (Toggle Use SG)
- Fit selection: `Q` (QD3Fit), `D` (QDMFit), `C` (CSA), `W` (skew_gaussian), `G` (gaussian), `X` (low_pass_max)
- Channel select: `1–8`; Invert: `I`
- Info: `H` (Help), `U` (Fit Info)

## Notes & Tips
- **Decimation** only affects plotting; fits always use the full‑resolution samples inside the selected region.
- **Transparent figures:** Use **Save Figure** to export with a transparent background (useful for posters).
- **Multiple fits per event/channel:** The app supports multiple fits; each new run gets a unique suffix and is tracked in the **Fit Info** window.
- **FFT:** Select a region, choose FFT in the dynamic dropdown, and Run. The popup uses log–log scales with a pan/zoom toolbar.
- **Use SG:** When enabled (and an SG overlay exists for the channel), Run Fit / Adjust Fit / FFT operate on the SG‑filtered signal instead of raw.

## Known Limitations / Roadmap
- Fit info display can lag after switching back to waveform.
- Batch SG/fit processing is partially implemented.
- Displayed fit should be selectable per channel when >1 are shown.
- Auto‑region suggestion around rise/fall for QD3/CSA in batch mode.
- Package the app and pin a cross‑platform dependency set.

## Troubleshooting
- **“No Files”**: Verify the folder has `.trc` files with names like `C1-####.trc`.
- **Event dropdown is empty**: Filenames must end with `-<digits>.trc`.
- **MetaMatch requires a QD3 fit**: Run QD3 before MetaMatch to provide a velocity.
- **SG window too large**: Reduce the SG width to less than the number of samples in the trace.

## License / Credits
- `.trc` reading logic adapted from public LeCroy waveform templates (see comments in `readTrcDoner.py`).

- Core contributors: Alex Doner and collaborators.

---
*Questions or ideas? Message Alex at Alex.doner@lasp.colorado.edu.
