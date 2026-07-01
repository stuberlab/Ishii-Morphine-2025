# FigS9_ppdh_nac (marimo) — notes

Marimo conversion of `Opioid_pPDH_analysis.ipynb` — **Figure S9**: histological analysis of
**MOR, pPDH (PPD), and c-Fos** in the nucleus accumbens (AcbC / AcbSh) after acute morphine,
from QuPath cell detections. Saline vs Morphine (`condition_colors = gray, green`).

## What I changed (conversion + paths + one bug)
- Converted with `marimo convert`.
- Repointed paths to `DATA_ROOT` (env `OPIOID_DATA_ROOT`):
  `metapath` / `analysis_resultpath` = `07_nac_histology_hcr/`, figures -> repo `../figures`.
  Removed `\\10.159.50.7\...\Opioid_pPPD` and the `C:\Users\...\Desktop\QuPATH_for_cFosPPDH`
  paths (3 places).
- **Raw QuPath output path is configurable:** `rawdatapath = env OPIOID_QUPATH_ROOT` (else
  `07_nac_histology_hcr/QuPath_cFosPPDH`). Used for the per-cell detection CSVs
  (`cell_quantification_roundness/`), the aligned cells/polygons, and the spatial templates.
- **Fixed an undefined variable:** `channels` was used but never defined in the original
  (it only worked from leftover kernel state). Added `channels = list(channel_dict.keys())`
  (`['FITC','Cy3','Cy5']`, matching the `[FITC_mean_corr, Cy3_meas, Cy5_mean_corr]` order).
- Analysis otherwise unchanged: cell classification (MOR+/PPD+/c-Fos+ by channel thresholds),
  master_df build, cleanup, per-region/condition counts, t-tests + reports, representative
  images, and the AcbC/AcbSh spatial density analysis.

## Panels / table
- Representative sections + spatial density maps (S9 A–G).
- Per-region/condition counts of MOR+, PPD+, c-Fos+, MOR+/PPD+ in AcbC/AcbSh, Saline vs
  Morphine (t-tests). Writes `*_count_by_region_and_condition_ttest*.csv` + `*_ttest_report*.txt`.

## Inputs (deposit `07_nac_histology_hcr/`) — all PROVIDE
- meta: `analysis_master.xlsx`, `Opioid_pPDH_meta.csv`
- QuPath raw output (large): the `cFOS_PPD_GFP_QPATH` tree with `cell_quantification_roundness/`
  (AcbC/AcbSh per-annotation CSVs), `aligned_cells/`, `aligned_polygons/`, `template/`. Point
  `OPIOID_QUPATH_ROOT` at it, or place it as `07_nac_histology_hcr/QuPath_cFosPPDH`.
- (The notebook can also use its saved `master_df_*.csv` / `section_meta_*.csv` if you prefer
  not to reprocess the raw CSVs — those are written into the group folder.)

## Notes / to confirm on a real run
- The raw master_df build reads per-annotation CSVs from the QuPath folder; if you have the
  saved `master_df_20260513_001335.csv`, you can skip that (drop it in the group folder).
- Cell-positivity thresholds: FITC/Cy5 use `*_thr_pos`, Cy3 uses a mean+SD rule (`Cy3_threshold`).
