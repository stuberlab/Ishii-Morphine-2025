# Fig6_semiNMF (marimo) — notes

Marimo conversion of the semi-NMF section of `Opioid_Figure 5.ipynb` — **Figure 6**: spatial
semi-NMF factorization of the whole-brain morphine c-Fos dataset. The factorization itself is
**external** (run separately); this notebook **loads** the factor results and does the
statistics, factor maps, and per-region spatial analysis.

## Scope
Kept the semi-NMF section (`# Semi-nmf` onward) plus the data-prep it needs (meta, atlas,
`merge_df`, curated acronyms). **Excluded** the predictive-activity-maps section of the original
notebook — that is already covered by `Fig5_chronic_earlyWD.py` (same source notebook).

## The factorization is a dependency, not run here
The semi-NMF decomposition is produced by the **`fos`** package (modified Linderman-lab
semi-NMF code), run separately on the c-Fos heatmaps. This notebook only reads its outputs:
`params.mat` (→ `factors`, `count_loadings`), `params.pkl`, and `factors.zarr`. They live in
`FACTOR_DIR` (env `OPIOID_FACTOR_RESULTS`, else `GROUP/spatial_clustering_results`).

To regenerate those results: `fos` is installable from GitHub (in `requirements.txt`:
`fos @ git+https://github.com/kenjp1223/fos`), and the cleaned driver that produces them is
`notebooks/fos_counts_clean.ipynb` in that repo (load `OP_cFos_heatmap_array` + fnamelist +
meta → downsample → `fit_poisson_seminmf` with 22 factors, sparsity 1e-2 → write
`params.pkl`/`params.mat`/`npy/factor{k}.npy`/`factors.zarr`).

## What it produces
- **Table:** `factor_anova_stat_df.csv` — per-factor one-way ANOVA on loadings (Saline vs each
  condition) with BH correction + Saline-vs-condition post-hoc t-tests. Written to the group folder.
- Loading strip/point plots per factor (significant-factor panels), the average-loading heatmap,
  and loadings sorted by drug group.
- Factor weight maps rendered on coronal planes from `factors.zarr` (all 22 factors + the
  significant ones), and per-region (NAc/Ce/BLA/La/VeP/VTA/OFC/Insular/PVT/Cl…) spatial factor
  distributions — the Figure 6 main panels plus supplements.
- `spatial_factors_for_subregion(_sum)_{region}.npy` intermediates (written to the group folder).
- The drug-classifier confusion matrix (supplement).

Note: the `factors/factor{0,1,2}.npy` that **Figure 8** and **Figure S16** consume are exports
of these same semi-NMF factors (from `FACTOR_DIR`); provide them under `06_spatial_gene_merfish/factors/`.

## Cleanup (conversion + paths only; analysis unchanged)
- Converted with `marimo convert`; rebuilt the star-import header (`from contour_visualization
  import *` breaks marimo static analysis) into an explicit import cell.
- **brain_vis swap:** all visualization/region helpers now come from `brain_vis`
  (`from brain_vis import overlap_contour, set_transparency, get_subregions`). The two deprecated
  vendored modules are aliased to it — `import brain_vis as cv2` and
  `import brain_vis as create_mask_for_region` — so every `cv2.overlap_contour` /
  `create_mask_for_region.get_subregions` call resolves to `brain_vis` unchanged.
- Added the standard config cell (`DATA_ROOT`/`GROUP` = `01_main_cfos_morphine`/`ATLAS` =
  `shared/atlas`/`FIG_OUT` = `../figures`) with `assert GROUP.exists()`, matching the other
  group-01 notebooks. Repointed every `\\10.159.50.7\...`, `G:\My Drive\...`, and ClearMap atlas
  path: atlas CSV/contour/annotation tif + curated/ancestor acronym pickles ← `shared/atlas`;
  meta + result CSVs + written tables/intermediates ← the group folder; factor results ←
  `FACTOR_DIR`.
- Figure/panel keys renamed `Figure5*` → `Figure6*` so saved panels match the manuscript.
- Disabled the stray `clean_curated_acronyms.csv` export (already in `shared/atlas`).

## Inputs (deposit `01_main_cfos_morphine/`)
- In deposit / group: `OP_meta.csv`, `long_pivoted_heatmap_df_with_normalized_density.csv`,
  `Ex_639_Ch2_stitched_long_merge_Annotated_counts_with_leaf_with_density_with_normalized_density.csv`.
- `shared/atlas/`: atlas info CSV, `Kim_ref_adult_FP-label_v2.9_contour_map.tif`,
  `Kim_ref_adult_FP-label_v4.0.tif`, `curated_acronym.pickle`, `ancestor_curated_acronym.pickle`.
- **PROVIDE (semi-NMF results):** `params.mat`, `params.pkl`, `factors.zarr` in `FACTOR_DIR`.

## Requires
`brain_vis`, `dask`, `zarr`, `scipy`, `statsmodels`, `scikit-learn`, `seaborn`, `adjustText`,
`tifffile` (all in `requirements.txt`).
