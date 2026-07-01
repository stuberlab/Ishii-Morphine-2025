# Fig7_TRAP2 (marimo) — notes

Standalone marimo notebook for **Figure 7** (TRAP2 acute-morphine ensemble), built from
`Opioid_Figure 6.ipynb` (internally used `figure_key='Figure6'`; this is manuscript Figure 7).
Scoped to the minimum for the panels + table.

## Panels / table
- **7D** whole-brain double+ counts per condition (scatter). (The region "discovery + density"
  heatmap variant is omitted here — it needs the extra `overlap_density_pivoted_..._without_zscore.csv`.)
- **7E-G** accuracy / efficiency / fidelity per subject over the responsive regions; t-test
  Acute-TRAP vs each condition, BH. Factored into `metric_plot(variable, ylabel, key)`.
- **7I** fidelity by Figure-5 cluster (from `Figure5E_umap_embedding.csv`).
- **7J** fidelity in the selected regions.
- **Table 8** — per-region fidelity (F1) t-test, Acute-TRAP vs each condition, BH per comparison
  (`Table8_TRAP_fidelity.xlsx`), matching the uploaded Table 8 layout.
- **7H** representative double+ beta maps (guarded on `overlap_{condition}_betas.npy`).

## Conditions
`AcM-AcM` = Acute-TRAP (reference), `Sal-AcM` = Saline-TRAP, `AcM-ChM` = Chronic-TRAP,
`AcM-WDM` = EarlyWD-TRAP.

## Inputs (deposit `05_trap2_ensemble/`)
- `OPTRAP_meta.csv`, `total_long_merge_Annotated_counts_with_leaf_with_density.csv`.
- Optional: `TreeFDRS_pvalue_Figure7_D_glm_stat_df_no_batch.csv` (BRANCH region set; if absent,
  the notebook falls back to the Figure-5 clustering regions), `overlap_{condition}_betas.npy` (7H).
- Cross-ref: `01_main_cfos_morphine/Figure5E_umap_embedding.csv`.

## Cleanup + fixes
- Paths -> `DATA_ROOT`/Figshare; `contour_visualization`/`create_mask_for_region` -> `brain_vis`.
- **Accuracy/efficiency labels:** implemented per the manuscript definitions
  (accuracy = double+/tdTomato+ = `overlap_over_Ex_561`; efficiency = double+/c-Fos+ =
  `overlap_over_Ex_639`). The **original notebook's E/F axis labels appear swapped** relative to
  these — please confirm which is intended.
- Metric panels factored into one function; the 2-condition "dash" variants (E'/F'/G') dropped.
- The GLM (region NB GLM + LRT dropping conditions) that feeds the BRANCH region set was left in
  the original; here the region set is read from TreeBH output if present, else the Figure-5
  regions. Re-add the GLM + `Opioid_Figure 6_TreeBH.R` if you want to regenerate it.

## Excluded (not needed for these panels/tables)
- **7K-M** accumbens spatial sub-panels (heavy per-voxel subregion computation + zarr).
- The "Figure 6-supplemental figure 1" per-region fidelity supplement.

## To confirm on a real run
- E/F accuracy-vs-efficiency assignment (see above).
- 7I/7J region set = Figure-5 clustering regions; cluster labels from `Figure5E_umap_embedding.csv`.
