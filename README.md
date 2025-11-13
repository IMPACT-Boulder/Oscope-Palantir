# Oscilloscope Palantir

> **Open Use License**  
> You may use, copy, modify, and redistribute this repository for any purpose, provided that redistributions include this original source code (including this notice). This software is provided "as is" without warranty of any kind.

## Overview
Oscilloscope Palantir is a PyQt5 desktop application for reviewing multi-channel LeCroy `.trc` oscilloscope captures, selecting pulse regions, fitting physics-informed models, and exporting the results for downstream PVDF calibration studies. The 2025 builds emphasize fast iteration on large campaigns: every zoom, filter, or fit state is cached per dataset, and auxiliary tools (Results Plotter, feature scans, co-adds, etc.) can be launched without leaving the GUI.

## Key Capabilities (2025 release)
- **Event-aware loader** – Point Palantir at a folder of `.trc` files and it auto-groups channels by event id, tags the dataset path, and remembers per-channel zoom/pan ranges as you step through events.
- **Persistent axes + Reset Axes** – Manual zoom/pan actions are captured per channel so that subsequent redraws preserve your view; the Reset Axes control clears those saved domains to return to auto-scaling when needed.
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
1. **Folder / Event row** – Select a folder (F), navigate events (`,` / `.`), adjust decimation, assign per-event quality (0–5), and—if desired—co-add waveforms across events. Zoom or pan any subplot; the view is remembered per channel until you press **Reset Axes**.
2. **Session controls** – Clear fits (`E`), clear all data (`Shift+D`), save figures (Ctrl+S, transparent background supported), export/import fits (Ctrl+E / Ctrl+I), and pop open the Fit Info window (`U`).
3. **Metadata / Impact row** – Load a metadata CSV (`L`), run MetaMatch (`M`), specify detector distance, and drop impact-time markers (`T`) that appear on every channel and persist with the fit record.
4. **Savitzky–Golay & Discovery row** – Apply SG filters (`S`), choose the channel + window, run SG Batch (Ctrl+B), clear overlays (Shift+S), reset axes, and launch:
   - **Results Plotter** (`P`) for dataset-level charts,
   - **Feature Scan** (Ctrl+F) to search large event ranges for peaks,
   - **Fit Filter** (Ctrl+Shift+F) for param-based queries,
   - **Quality Filter** for rating-based filtering,
   - **Clear Filter** (Ctrl+K) to restore the full event list.
5. **Dynamic fit row** – Choose the model (QD3/QDM/CSA/skew/gauss/low_pass_max/FFT), channel, invert toggle, and decimation for fit math. Run (Enter), adjust (`A`), clear (`R`), clear all fits on a channel (Shift+R), batch process (`B`), or force the fit/FFT to use any cached SG trace (`Ctrl+Shift+S`). FFT results open in a separate window with its own NavigationToolbar.

Typical workflow:
- Select a folder, optionally apply SG filtering, drag-select a region (orange span) per channel, then run fits. Repeat for the channels of interest, review parameters/derived charge + mass in the Fit Info window, mark impacts, and export to HDF5 when satisfied.

## Analysis, Filtering, & Persistence
- **Axis persistence** – Every subplot records its x/y limits after mouse zoom, pan, or scroll; those limits are reapplied when the event redraws, enabling apples-to-apples comparisons across events. Reset Axes wipes the cached limits.
- **SG caching** – Filtered traces are cached per dataset/event/channel/window in `.palantir_sg`. Clearing the SG overlay removes the cache for that entry; deleting the folder resets everything.
- **Feature Scan** – Define an event range, channel, σ-multiplier, and absolute/positive-only mode. The scanner loads each waveform, computes σ, and flags events whose peaks exceed the threshold. Apply the resulting filter list or rescan with new settings.
- **Fit Filter** – Query stored fits (current dataset aware) by model, channel, parameter, and comparison (>, between, etc.). The resulting event list can be inspected before filtering the dropdown.
- **Quality filter** – Rate events with the Quality spin box, then filter to those ≥ a chosen score. Ratings are saved in-memory and exported with your fits.
- **Manual filters** – All filters compound. The Clear Filter command restores the original event order maintained in `_all_event_keys`.
- **Fit Info window** – Tabular summary of every saved fit, including derived metrics (charge, mass, radius, SG metadata). Toggle “Only current dataset” as needed.
- **Results Plotter** – Opens a dedicated PyQt window that builds scatter plots/histograms from the current in-memory fits. Picking an event in the plotter jumps the main GUI to that event (if present).
- **Import/Export** – HDF5 exports include fit parameters, derived values, fit regions, inversion flags, SG context, dataset id, and event quality table. Imports merge with in-memory results so you can pick up where a previous session ended.
- **Impact tracking** – Mark Impact uses distance/velocity to compute a predicted impact time, plots the vertical marker on every channel, and stores the value on the associated fit record for export.

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

## Contact
Questions or ideas? Message Alex at `Alex.doner@lasp.colorado.edu`.
