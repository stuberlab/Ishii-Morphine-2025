# FigS4_sex_analysis (marimo) — notes

Standalone marimo notebook for **Figure S4** (sex differences), built from
`Opioid_Figure sex analysis.ipynb`. Same main c-Fos dataset as Figs 1/2.

## Structure
- **S4A — sex beta map:** loads `Sex_d_betas.npy` (produced by the Figure 1 regression,
  `Fig1_S2_whole_brain_cFos`, and stored in the Figshare deposit), renders with
  `brain_vis.overlap_contour(..., overlap_black=True)` → `FigureS4A_Sex_d_betacoef.*`.
- **Region GLM:** per-region negative-binomial GLM + chi-square LRT (full vs. sex-removed),
  writes `FigureS4_Sex_glm_stat_df.csv` for the R TreeBH (BRANCH) step.
- **BRANCH read (guarded):** reads `TreeFDRS_pvalue_Figure1_Sex_glm.csv` if present;
  otherwise prints a notice and S4C/S4D skip.
- **S4D swarmplot** and **S4C sunburst** (`brain_vis.sunburst_app.run_app`, imported lazily
  so the dash dependency doesn't block the rest). Launch the sunburst app with
  `sex_sunburst_app.run()`.

## Cleanup vs. original
- Paths → `DATA_ROOT`/Figshare layout; removed all `\\10.159.50.7\...` and `G:\My Drive\...`.
- Visualization via `brain_vis` (was `contour_visualization` / `active_sunburst`); sunburst
  `cmin/cmax` → `data_cmin/data_cmax` to match the `brain_vis.sunburst_app` API.
- Reuses the existing `OP_cFos_full_result.csv` (already has `density` + `normalized_density`);
  no separate `..._normalized_density.csv` needed.
- Dropped cruft: the mislabeled Chronic-vs-WD cell that saved `Table Sex.xlsx`, duplicate
  "rejected acronyms" cells, the unused `Acute_Morphine_betas` load, and empty cells.

## Dependencies / to confirm
- `Sex_d_betas.npy` must exist in the deposit (run `Fig1_S2_whole_brain_cFos` first).
- The BRANCH parts require running the TreeBH R script to produce
  `TreeFDRS_pvalue_Figure1_Sex_glm.csv` (the Figure 2 cleanup will set up that R pipeline).
- `get_subregions(atlasmeta, 8, ...)` keeps the original root id (8 = grey) — verify if your
  ontology differs.
