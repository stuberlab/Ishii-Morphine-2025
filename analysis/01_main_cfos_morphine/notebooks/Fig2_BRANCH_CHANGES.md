# Fig2_BRANCH (marimo) — notes

Standalone marimo notebook for **Figure 2** (BRANCH hierarchical testing), built from
`Opioid_Figure 2.ipynb`. Same main c-Fos dataset.

## Pipeline
1. **Region GLM + LRT (2B):** per-region negative-binomial GLM (drug conditions + sex + BW +
   age + constant) vs. a reduced model with the drug-condition terms removed → chi-square
   p-value. Writes `Figure2_C_glm_stat_df.csv`.
2. **BRANCH (R):** `Fig2_TreeBH.R` reads that CSV and writes
   `TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv` (Simes = BRANCH) +
   `TreeFDRF_...` (Fisher, for comparison). TreeFDRS is already in the deposit.
3. **Read back + visualize:** rejected regions, post-hoc **Table 3** (Chronic vs Early-WD on
   rejected regions, FDR), sunburst (**2C–G**, interactive Dash via `brain_vis.sunburst_app`),
   discovery + per-condition density heatmap, and the **2H** condition correlation matrix.

## Cleanup vs. original
- Paths → `DATA_ROOT`/Figshare; removed `rootpath/data` setup.
- Visualization via `brain_vis` (was `active_sunburst` + `contour_visualization`); sunburst
  `cmin/cmax` → `data_cmin/data_cmax` per the `brain_vis.sunburst_app` API.
- **Standardized on the Simes / `TreeFDRS_*` output (= BRANCH)** everywhere. The original
  mixed in the Fisher `TreeFDRF_*` variant (which is not in the repo) for the sunburst and
  discovery heatmap.
- BRANCH read and all downstream cells are **guarded**: they run when the TreeBH output (and,
  for the sunburst, `Acute_Morphine_betas.npy` from Fig 1) are present, else print a notice.
- Reused `OP_cFos_full_result.csv` / `OP_cFos_heatmap.csv`; dropped the in-cell GLM diagnostic
  `sns.heatmap`, the unused logistic-regression classification block, and `importlib.reload`.

## Not in this notebook / to confirm
- **2I–J scatter plots** (Acute vs Chronic, Acute vs Re-exposure) are not here — they live in
  the clustering notebook (Figs 5/6 pass).
- Panels **2A/2B** are schematics (no code). **2D–G** (hypothalamus/striatal subtree details +
  LZ/MBO, STRv/sAMY maps) are produced by interacting with the sunburst app and exporting.
- `Acute_Morphine_betas.npy` must exist in the deposit (run `Fig1_S2_whole_brain_cFos` first).
- `get_subregions(atlasmeta, 8, ...)` keeps the original root id (8 = grey).
