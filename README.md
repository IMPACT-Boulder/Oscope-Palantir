# Oscilloscope Palantir

> **Open Use License**  
> You may use, copy, modify, and redistribute this repository for any purpose, provided that redistributions include this original source code (including this notice). This software is provided "as is" without warranty of any kind.

## Overview
Oscilloscope Palantir is a PyQt5 desktop application for reviewing multi-channel LeCroy `.trc` oscilloscope captures, selecting pulse regions, fitting physics-informed models, and exporting the results for downstream PVDF calibration studies. The 2025 builds emphasize fast iteration on large campaigns: every zoom, filter, or fit state is cached per dataset, and auxiliary tools (Results Plotter, feature scans, co-adds, etc.) can be launched without leaving the GUI.

## Key Capabilities (2025 release)
- **Event-aware loader** – Point Palantir at a folder of `.trc` files and it auto-groups channels by event id, tags the dataset path, and preserves zoom/pan ranges on the channels you select as you step through events.
- **Per-channel axis preservation** – Choose which channels should keep their zoom/pan across events; unselected channels auto-scale on each redraw.
- **Dynamic fitting suite** – Drag out a fit window per channel and run QD3Fit, QDMFit, CSA_pulse, skew_gaussian, gaussian, low_pass_max (SG-based extremum) or FFT from the same control row. Fits can be inverted, batch-applied, adjusted live, and tied to SG-filtered data via the “Use SG” toggle.
- **Savitzky–Golay pipeline** – Apply SG overlays per channel, batch them across events (multi-threaded), and transparently cache the filtered traces under `.palantir_sg` so they reload instantly when you revisit an event/width pair.
- **Waveform co-addition** – Combine multiple events on a selected channel (sum or average). The co-add dialog handles interpolation, shows ±1σ envelopes, and can overlay the contributing traces for context.
- **Metadata + impact tools** – Load an external CSV, run MetaMatch once a QD3/QDM fit exists, and drop an impact-time marker (distance / velocity) across all channels; the computed impact time is stored on the fit record.
- **Fit management hub** – View every stored fit in the Fit Info window (with dataset scoping), import/export HDF5 files (including event-quality tags), and launch the Results Plotter to make campaign-level scatter plots that link back to the active GUI via event picking.
- **Event discovery & filtering** – Use Feature Scan (threshold × σ), Fit Filter (param-based queries), Quality Filter (per-event 0–5 ratings), or manual filter application to focus the event list. Filters compound until you hit Clear Filter.
- **Quality & session context** – Tag each event with a quality score, keep running quality metadata per dataset, and document work via the built-in help dialog and keyboard shortcuts.
- **Analysis companions** – Additional scripts (thickness, segmentation, penetration, velocity-ratio) consume the exported HDF5 data for publication-ready plots.

## Repository Map
- `oscope_palantir.py` – Main PyQt5 GUI with fitting, SG filtering, filtering utilities, axis persistence, and export/import logic.
- `fitparams.py` – Model definitions, helper functions, and the FitParamsDialog used for manual parameter tweaks.
- `readTrcDoner.py` – Lightweight `.trc` reader that returns `(time, voltage, metadata)` arrays.
- `results_plotter.py` – Standalone PyQt tool for visualizing batches of fit results (scatter plots, dataset grouping, event pick linking).
- `thickness_results.py` – Recreates PVDF thickness comparison plots with propagated uncertainties.
- `segmented_results.py` – Specialized segmented PVDF calibration plotter for the May 2025 campaigns.
- `penetration_results.py` – Focused analysis for penetration-only datasets exported from Palantir.
- `velocity_ratio_vs_radius.py` – Helper for plotting velocity ratios versus particle radius using Palantir exports.
- `requirements.txt` – Python dependency list (PyQt5, qtawesome, numpy, pandas, scipy, matplotlib, lmfit, etc.).
- `icon.png`, `palantir.desktop` – Desktop integration assets.

## Installation
1. Install Python 3.9+ and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Freeze the environment for reproducibility: `pip freeze > requirements.txt`.

## Running the GUI
```bash
source .venv/bin/activate   # if not already active
python oscope_palantir.py
```
The app uses the Fusion style for consistent rendering across platforms.

## UI Tour & Workflow
1. **Top row** – Select a folder (F), pick an event, assign a quality score (0–5), and check the active folder path. Use **Help** (H) for shortcuts and **Light/Dark Mode** (Ctrl+T) to toggle themes.
2. **Navigation / session row** – Step through events (`,` / `.`), set decimation for plotting, and co-add waveforms. Session tools on the right clear fits (`E`), save figures (Ctrl+S), export/import fits (Ctrl+E / Ctrl+I), clear all data (Shift+D), and open Fit Info (`U`).
3. **Metadata / impact row** – Load a metadata CSV (`L`), run MetaMatch (`M`), set the detector distance, and mark impact time (`T`) across channels.
4. **Savitzky–Golay & discovery row** – Apply SG filters (`S`), choose the channel + window, run SG Batch (Ctrl+B), clear overlays (Shift+S), and use the **Preserve Axes** menu to pin zoom/pan on selected channels. Launch Results Plotter (`P`), Feature Scan (Ctrl+F), Fit Filter (Ctrl+Shift+F), Quality Filter, or Clear Filter (Ctrl+K).
5. **Dynamic fit row** – Choose the model (QD3/QDM/CSA/skew/gauss/low_pass_max/FFT), channel, invert toggle (I), and fit decimation. Run (Enter), adjust (`A`), clear (`R`), clear channel fits (Shift+R), batch process (`B`), or force the fit/FFT to use SG-filtered data (`Ctrl+Shift+S`). FFT results open in a separate window with its own NavigationToolbar.

Typical workflow:
- Select a folder, optionally apply SG filtering, drag-select a region (orange span) per channel, then run fits. Repeat for the channels of interest, review parameters/derived charge + mass in the Fit Info window, mark impacts, and export to HDF5 when satisfied.

## Button Reference (main window)
**Top row**
- `Select Folder (F)` – Choose a folder of `.trc` files and load the event list.
- `(H)elp` – Open the help dialog with shortcuts and feature notes.
- `Light Mode` / `Dark Mode` – Toggle the UI theme.

**Navigation / session row**
- `Prev (<)` – Load the previous event.
- `Next (>)` – Load the next event.
- `Co-Add Waves` – Combine selected events into a summed or averaged waveform.
- `Clear Event Fits` – Remove all fits for the current event.
- `Save Figure` – Save the current plot to an image file (transparent background supported).
- `Export Fits` – Export all fits to an HDF5 file.
- `Load Fits` – Import fits from an HDF5 file and merge into the session.
- `Clear Data` – Clear all loaded data and fits from memory.
- `Fit Info (U)` – Open the fit summary table.

**Metadata / impact row**
- `(L)oad Metadata` – Load a metadata CSV for MetaMatch.
- `(M)etaMatch` – Find the best metadata match using the active QD3/QDM fit.
- `Mark Impact (T)` – Plot a vertical impact marker using distance/velocity.

**SG / discovery row**
- `(S)G Filter` – Run Savitzky–Golay filtering on the selected channel.
- `Batch Run` – Run SG filtering across a range of events.
- `Clear Fit` – Remove the SG overlay for the current event/channel.
- `Preserve Axes` – Select which channels keep their zoom/pan across events.
- `Results Plotter` – Open the dataset-level results visualization window.
- `Feature Scan` – Search events for large excursions vs σ.
- `Fit Filter` – Filter the event list using fit parameter constraints.
- `Quality Filter` – Filter events by quality rating threshold.
- `Clear Filter` – Restore the full event list.

**Dynamic fit row**
- `Run Fit` – Execute the selected fit on the chosen channel/region.
- `(A)djust Fit` – Edit initial parameters before running the fit.
- `Clea(r) Fit` – Remove the active dynamic fit result.
- `Clear Chan Fits` – Remove all fits on the selected channel for this event.
- `Batch Run` – Run the selected fit across multiple events.

**Toggles**
- `:(I)nvert` – Invert polarity for fitting (also flips low_pass_max extremum).
- `Use SG` – Use SG-filtered data for fits/FFT when available.

## Analysis, Filtering, & Persistence
- **Axis preservation** – Every subplot can record its x/y limits after mouse zoom, pan, or scroll. Use the Preserve Axes menu to keep those limits on a channel across events; unselected channels auto-scale on each redraw.
- **SG caching** – Filtered traces are cached per dataset/event/channel/window in `.palantir_sg`. Clearing the SG overlay removes the cache for that entry; deleting the folder resets everything.
- **Feature Scan** – Define an event range, channel, σ-multiplier, and absolute/positive-only mode. The scanner loads each waveform, computes σ, and flags events whose peaks exceed the threshold. Apply the resulting filter list or rescan with new settings.
- **Fit Filter** – Query stored fits (current dataset aware) by model, channel, parameter, and comparison (>, between, etc.). The resulting event list can be inspected before filtering the dropdown.
- **Quality filter** – Rate events with the Quality spin box, then filter to those ≥ a chosen score. Ratings are saved in-memory and exported with your fits.
- **Manual filters** – All filters compound. The Clear Filter command restores the original event order maintained in `_all_event_keys`.
- **Fit Info window** – Tabular summary of every saved fit, including derived metrics (charge, mass, radius, SG metadata). Toggle “Only current dataset” as needed.
- **Results Plotter** – Opens a dedicated PyQt window that builds scatter plots/histograms from the current in-memory fits. Picking an event in the plotter jumps the main GUI to that event (if present).
- **Import/Export** – HDF5 exports include fit parameters, derived values, fit regions, inversion flags, SG context, dataset id, and event quality table. Imports merge with in-memory results so you can pick up where a previous session ended.
- **Impact tracking** – Mark Impact uses distance/velocity to compute a predicted impact time, plots the vertical marker on every channel, and stores the value on the associated fit record for export.

## Dynamic Fit Functions & Initialization Heuristics
Palantir seeds each fit with fast heuristics so the first curve-fit iteration lands close to the pulse. All fits run on the selected time window; the **Invert** toggle flips polarity for fitting (except `low_pass_max`, which uses invert to choose a minimum vs maximum). If **Use SG** is enabled, the SG-filtered trace is used for fitting/initialization.

- **QD3Fit** – Attempts multiple Savitzky–Golay smoothings (window lengths inferred from the time step, targeting ~1 us / 10 us / 20 us) and estimates `t0` from the strongest negative slope, `q` from the smoothed minimum, and `v` from the rise/fall time (`v ≈ 0.19 / Δt`). The best window is chosen by a quick chi^2 check against the model. `C` starts at the mean of the selected window. Model: a piecewise ramp/decay pulse with time scales derived from `v`, smoothed by Savitzky–Golay, plus a constant offset. In code terms: `dt1 = PUT_d / (v*100)`, `dt2 = (PUT_l - 2*PUT_d) / (v*100)` and
  `signal(t) = { q/dt1*(t-t0) for t0<t<t0+dt1; q*exp(-(t-(t0+dt1))/tau1) for t0+dt1<=t<t0+dt1+dt2; qq - q/dt1*(t-(t0+dt1+dt2)) for t0+dt1+dt2<=t<t0+2*dt1+dt2; qq*exp(-(t-(t0+2*dt1+dt2))/tau2) for t>=t0+2*dt1+dt2 }`, then `SG(signal) + C`.
- **QDMFit** – Applies a shorter SG smoothing (~201 samples) and estimates `t0` from the negative-slope index shifted earlier by ~5 us, `q` from the smoothed minimum, and `v ≈ 0.133 / Δt`. Baseline `C` starts at the mean. Model: same piecewise ramp/decay structure as QD3 but with QDM constants (`PUT_l=15.3`, `PUT_d=2.0`, `tau1=tau2=0.4e-3`) and the same Savitzky–Golay smoothing before adding `C`.
- **CSA_pulse** – Uses generic pulse heuristics: `t0` at the window start, `C0` from the first sample, `C2` from the max, `C` from the mean, and characteristic times (`T0/T1/T2`) set to fractions of the window length. Model: `C0 - H(t0-x)*C1*exp(-((x-t0)^2)/T0^2) + H(x-t0)*(C2*(1-exp(-(x-t0)/T1))*exp(-(x-t0)/T2) - C1) + C` where `H` is the Heaviside step.
- **skew_gaussian** – Seeds `A` from the area under the curve, `xi` at the peak location, `omega` to ~1/6 of the window width, and `alpha = 0` (symmetric). Model: `y = A * 2/omega * phi(t) * Phi(alpha*t) + C` with `t=(x-xi)/omega`, `phi(t)=(1/sqrt(2*pi))*exp(-t^2/2)`, `Phi(z)=0.5*(1+erf(z/sqrt(2)))`.
- **gaussian** – Seeds `A` from peak-to-min amplitude, `mu` at the peak, and `sigma` to ~1/6 of the window width. Model: `y = A*exp(-0.5*((x-mu)/sigma)^2) + C`.
- **low_pass_max** – Not a curve fit: removes a linear baseline (fit to the first 20% of the window), applies an SG low-pass (odd window, derived from the SG control or the prior fit), then reports the extremum time/value. `Invert` flips the search from max to min. Procedure: `baseline(t)=m*t+b`, `y_corr=y-baseline`, `y_f=SG(y_corr,w,2)`, pick `max(y_f)` (or `min` when inverted).
- **FFT** – Applies a Hanning window and displays a log–log spectrum in a dedicated viewer (no fit parameters stored). Procedure: `Y = rfft(y * hann)`, `A = (2/N)*|Y|`.

## Secondary Analysis Workflows
After exporting fits you can:
- Run `results_plotter.py` independently to compare datasets or generate publication plots without the GUI.
- Use `thickness_results.py`, `segmented_results.py`, `penetration_results.py`, or `velocity_ratio_vs_radius.py` to recreate the PVDF charge-yield figures used in recent campaigns. Each script expects a Palantir HDF5 export and focuses on a specific derived metric (thickness sweep, segmented witness plates, penetration-only subsets, or velocity ratio vs radius respectively).

## Keyboard Shortcuts (abbreviated)
- **Navigation** – `,` / `.` (prev/next), `T` (Mark Impact)
- **Files & session** – `F` (folder), `Ctrl+S` (save figure), `Ctrl+E` (export), `Ctrl+I` (import), `Shift+D` (clear data)
- **Metadata** – `L` (load CSV), `M` (MetaMatch)
- **SG & filters** – `S` (SG), `Ctrl+B` (batch SG), `Shift+S` (clear SG), `P` (Results Plotter), `Ctrl+F` (Feature Scan), `Ctrl+Shift+F` (Fit Filter), `Ctrl+K` (Clear Filter)
- **Fitting** – `Enter` (run), `A` (adjust), `R` (clear fit), `Shift+R` (clear channel fits), `B` (batch), `Ctrl+Shift+S` (Use SG), `Q/D/C/W/G/X` (model select), number keys (channel), `I` (invert)
- **Info** – `H` (help), `U` (Fit Info)

## Troubleshooting
- **No events listed** – Ensure filenames end with `-<digits>.trc` (e.g., `C3-1042.trc`).
- **No channel detected** – File names must begin with `C1`–`C4`. Mixed casing is tolerated.
- **Feature/fit filters return nothing** – Clear filters (Ctrl+K) to restore the base event list, then re-run with less restrictive criteria.
- **MetaMatch unavailable** – Run a QD3 or QDM fit first; `v` from that fit seeds the metadata lookup.
- **SG cache issues** – Delete the `.palantir_sg` folder inside the dataset directory to force a fresh SG run.


## Author
Alex Doner

## Contact
Questions or ideas? Message Alex at `Alex.doner@lasp.colorado.edu`.
