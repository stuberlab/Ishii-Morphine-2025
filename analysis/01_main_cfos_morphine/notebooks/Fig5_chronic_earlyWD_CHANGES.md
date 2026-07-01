# Fig5_chronic_earlyWD (marimo) — notes

Standalone marimo notebook for **Figure 5** (chronic morphine vs. early withdrawal), built
from `Opioid_Figure 4.ipynb` (old filename numbering). Same main c-Fos dataset.

## Panels
- **5B** difference map (Chronic − Early-WD β), from the Figure-1 `{condition}_betas.npy`.
- **5C** UMAP of BRANCH-rejected regions colored by HDBSCAN cluster.
- **5D** clustered scaled-density heatmap (+ HDBSCAN single-linkage tree inset).
- **5E** per-cluster scaled density (Chronic vs Early-WD, t-test + BH).
- **5F** per-region post-hoc in selected regions (t-test + BH).
- **5G–J** predictive-activity maps in PFC (IL), central amygdala (Ce), accumbens (Acb), VTA.

## Inputs (from the deposit)
- `OP_cFos_full_result.csv`, `OP_meta.csv`, atlas/ontology.
- Figure-1 regression β maps `{condition}_betas.npy` (Chronic/Withdrawal + per-region).
- Figure-2 BRANCH output `TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv` (defines rejected regions).
- Writes the UMAP embedding `Figure5E_umap_embedding.csv` (used later by the TRAP figure).
- **No zarr needed** for these panels.

## Cleanup vs. original
- Paths → `DATA_ROOT`/Figshare; removed all `\\10.159.50.7\...` paths and the local
  `create_mask_for_region` / repeated `set_transparency` defs (now `brain_vis`).
- The 4 near-identical per-region map blocks (PFC/Acb/Ce/VTA) are factored into one
  `plot_region_maps(target, hemi, pannel_key)` called in a loop.
- **Dropped the heavy per-voxel `spatial_cell_count` computation** (reshaped the full
  444 MB heatmap zarr 4×): it is not used by these panels — it produced
  `spatial_factors_for_subregion_*.npy` consumed by the semi-NMF / supplement work.
  Re-add there if needed.
- UMAP computed once (seed 42) and kept in memory instead of save/reload roundtrips.
- Resolved marimo single-definition by renaming the reused `data` variable into
  `umap_input` → `data_full`; scipy/sklearn/umap/hdbscan/adjustText/statannotations imported
  as cell-local `_aliases`. Dropped exploratory cells (draw_umap, raw heatmap previews, the
  effect-size-overlay supplement, empty cells).
- Added `umap-learn`, `hdbscan`, `adjustText`, `statannotations` to `requirements.txt`.

## Not in this notebook
- The **semi-NMF** analysis (next figure) and the two Figure-4 supplements remain in the
  original notebook — to be done separately.
- Panel letters G–J follow the manuscript (G PFC, H Ce, I Acb, J VTA); confirm against the
  current figure. UMAP/HDBSCAN are stochastic — seeded, but verify the cluster labels/order
  on a real run.
