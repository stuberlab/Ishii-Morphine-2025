# Fig3_MORcKO (marimo) — notes

Standalone marimo notebook for **Figure 3H–K** (MOR conditional knockout, VTA), built from
`MORcKO_analysis.ipynb` (216 cells). Scoped to the **minimum needed for the figure panels +
table**.

## Panels produced
- **3J** — mean c-Fos+ cell density coronal planes, `VTA-GFP` (eYFP) vs `VTA-Cre`, from the
  c-Fos heatmap zarr.
- **3K** — per-region Student t-test (eYFP vs Cre) on per-region-mean-normalized density,
  restricted to the acute-morphine responsive regions from Figure 3B, BH-corrected;
  injection-site (VTA spill) regions removed from the plot.
- **Table 6 (cKO part)** — `Figure3_MORcKO_normalized_density_ttest_stat_df.csv`.

## Inputs (deposit)
- `03_mor_cko_vta/MORcKO.csv` (meta; uses `Usable` + `Staining_Batch == 1`).
- `03_mor_cko_vta/Ex_639_Ch2_stitched_long_merge_Annotated_counts_clean_with_density.csv`.
- `03_mor_cko_vta/Ex_639_Ch2_stitched_heatmap_array/` (zarr) + `..._fnamelist.npy` — for 3J.
- `01_main_cfos_morphine/Figure3B_effect_size_df.csv` — acute-responsive region list.

## Deliberately excluded (not needed for these panels/tables)
- Raw preprocessing (building the master CSV from `*_cells.npy` / `*_intensities.npy` /
  `*_transformed_to_Atlas.npy`).
- Starter-cell (GFP) normalization: **not used** (confirmed). 3K tests the
  per-region-mean-normalized `normalized_density`, so the GFP channel is not loaded.
- Batch checks, per-region spatial maps (NAc/MD/Ce/IL), cluster analysis, the video, and the
  voxel-wise GLM/t-test sections.

## Cleanup
- Paths → `DATA_ROOT`/Figshare; removed `\\10.158.246.229\...`, `\\10.159.50.7\...`,
  `G:\My Drive\...`. Visualization via `brain_vis`. Guarded the 3J cell on the zarr's presence.

## To confirm on a real run
- `normalized_density` = density / per-region-mean across the cKO cohort (matches the tested
  variable in the source). Verify this is the manuscript's "normalized c-Fos+ cell density".
- 3J z-planes use the whole-brain representative set `[84,104,117,153,186,220]` (cmax=30).
- Expected 3K hits: CM, MD, PC, AcbSh.
