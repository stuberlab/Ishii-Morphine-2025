# Fig4_acute_opioids (marimo) — notes

Marimo conversion of your cleaned `Opioid_Figure_S7_cleaned_final_Table7.ipynb`
(**Figure 4** — acute opioid c-Fos mapping: Saline / Morphine / Fentanyl / Buprenorphine /
Oxycodone; **Table 7**). This is the "acute" dataset (old "Figure 4" in earlier numbering).

## What I changed (only conversion + paths)
- Converted the notebook to marimo with `marimo convert` (preserves your cell logic and the
  UMAP/HDBSCAN params: `n_neighbors=3`, `min_dist=0.05`, `random_state=12`,
  `HDBSCAN(min_cluster_size=2)`).
- Repointed paths to the deposit: `DATA_ROOT` (env `OPIOID_DATA_ROOT`) ->
  `metapath`/`analysis_resultpath`/`FIGS7_STATS_DIR` = `04_acute_opioids_4drugs/`,
  `atlaspath` = `shared/atlas/`, figures -> repo `../figures`. Removed the
  `\\10.159.50.7\...\Acute_drugs_for_revision` and `\\10.158.246.229\...\Fentanyl` paths.
- Removed a leftover Jupyter `display()` and de-duplicated a `Path` import (marimo rules).
- `brain_vis` imports were already in your notebook — kept as-is.
- Panels 4A/B/C/E/F, clustering, stats, Table 7 export are otherwise unchanged.

## Added after conversion (per request)
- **In-notebook GLM regression for 4D — restored from the original `acute_drug_analysis.ipynb`.**
  The cleaned notebook loaded precomputed `{condition}_betas.npy`; the regression is now back:
  voxel-wise OLS from the heatmap zarr, design = 4 drug-condition dummies (Saline = baseline) +
  Sex + BW + Age (standardized) + constant, on the `acronym == 'CH'` per-subject variables.
  Panel 4D uses the resulting `betas_dict` (in memory). **No files are written** by this cell
  (beta / adjusted-heatmap zarr export is left to you).
- **Subject alignment fixed.** The zarr / `fname_list` hold **52** imaged subjects (incl. 6
  Cocaine + 2 excluded); the meta `acute_drug_revision_full_list.csv` lists the **44** usable,
  non-Cocaine subjects. The load cell now `reindex`es to `fname_list` and keeps `Usable == True`
  (drops the 8), so the regression/analysis runs on 44. (The original `.loc[fname_list]` would
  KeyError on the 8 missing.)

## Panels / table
- **4A** whole-brain c-Fos+ counts per drug.
- **4D** GLM beta-coefficient maps — **computed in-notebook** from the zarr (no precomputed betas).
- **4B/E/F** UMAP + HDBSCAN clustering of TreeFDR-rejected regions.
- **4C / 4F** cluster- and region-level statistics.
- **Table 7** single-table export -> `Table 7.xlsx` + `Table7_Figure4_single_table.csv`.

## Inputs (deposit `04_acute_opioids_4drugs/`) — PROVIDE from the lab drive
`\\10.159.50.7\LabCommon\Ken\data\Acute_drugs_for_revision\` (result/ + meta/), raw zarr from
`\\10.158.246.229\DataCommon\SmartSPIM2\Ken\Fentanyl`:
- `acute_drug_revision_full_list.csv` (meta; 44 usable subjects)
- `Ex_639_Ch2_stitched_long_merge_Annotated_counts_clean_with_density.csv`
- `Ex_639_Ch2_stitched_heatmap_array/` (zarr, 52 subjects) + `fname_list.npy` — **used by the
  regression** (betas are no longer needed as inputs).
- TreeFDR/GLM CSVs (loaded via glob with fallback names): `glm_stat_df_post_TreeBH.csv`,
  `TreeFDRS_pvalue_{condition}_glm_without_Cocaine_GLM.csv` (per drug), `glm_stat_df_{condition}_*.csv`.
- (The notebook writes the `{condition}_betas.npy`, the `Figure4_*` stat CSVs + `Table 7.xlsx` here.)

## Requires
`umap-learn`, `hdbscan`, `adjustText`, `statsmodels`, `scikit-learn` (all already in requirements.txt).
