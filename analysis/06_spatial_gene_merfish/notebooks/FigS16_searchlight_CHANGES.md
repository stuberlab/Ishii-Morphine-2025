# FigS16_searchlight (marimo) — notes

Marimo conversion of `Example usage of Neurolight_dontshare.ipynb` — **Figure S16**:
searchlight-inspired local (voxel-neighborhood) correlation between opioid-receptor gene
expression and the morphine semi-NMF activity pattern (factor 1). Uses your **`neurolight`**
package (`nl.to_zarr`, `nl.SearchlightCorrelation`, `neurolight.stats.fdr_correction`).

## What I changed (conversion + paths only)
- Converted with `marimo convert`.
- Added a small config cell and repointed every scattered path to `DATA_ROOT`:
  `GROUP = 06_spatial_gene_merfish`, `ATLAS = shared/atlas`. Removed all
  `\\10.159.50.7\...\neurolight\test_result`, `\\...\Opioid_cFos`, and ClearMap atlas paths
  (this was a `_dontshare` example, now sharable).
- `factor{idx}.npy` <- `GROUP/factors/`; `gene_expression.zarr` + `gene_df.csv` <- `GROUP`;
  searchlight intermediates (`neuro.zarr`, `genes.zarr`, `results.zarr`) -> `OUT_DIR`
  (`OPIOID_SEARCHLIGHT_OUT`, else `GROUP/searchlight_output`); figures -> `../figures/Figure_S16`.
- Analysis unchanged: build neuro/gene searchlight zarrs (RADIUS=5, sphere kernel),
  `SearchlightCorrelation.run` (r + p maps), FDR correction, spatial correlation-map figures
  with Oprm1/Oprd1/Oprk1.

## Dependencies
- **`neurolight`** — installed from GitHub (in `requirements.txt`):
  `neurolight @ git+https://github.com/kenjp1223/neurolight`.
- `brain_vis` (already a dependency), `dask`, `zarr`, `tifffile`.

## Inputs (deposit `06_spatial_gene_merfish/`)
- `gene_expression.zarr`, `gene_df.csv` (from the Fig 8 preprocessing).
- `factors/factor1.npy` (morphine semi-NMF factor; from Figure 6).
- atlas from `shared/atlas`.
