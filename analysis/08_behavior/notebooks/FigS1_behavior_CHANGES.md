# FigS1_behavior (marimo) — notes

Standalone marimo notebook for **Figure S1** (behavioral characterization), built from
`OF_analysis.ipynb`. Scoped to the minimum for the panels + table. No brain-imaging deps.

## Panels / table
- **S1C** open-field quantification: per-variable lineplots (individual + mean) with
  paired-t-test-vs-saline significance.
- **S1E** linear-track quantification: saline-normalized heatmaps by variable group
  (Basic / Behavior / Kinematic) with significance overlay.
- **S1F** behavior-based condition-similarity clustermap (Pearson corr of saline-centered,
  z-scored condition means; hierarchical clustering).
- **Table 2** all paired-t-test statistics -> `FigureS1_all_statistics.xlsx`
  (sheet `All_stats_combined`).

## Stats
Paired Student's t-test (each condition vs Saline, paired by ID) + **Bonferroni** correction
across non-saline comparisons within each variable (matches the manuscript). Factored into
`paired_ttest_vs_saline(...)`. (The original's optional pingouin `rm_anova` is not included.)

## Inputs (deposit `08_behavior/`) — all PROVIDE from the lab drive
`\\10.159.50.7\LabCommon\Ken\data\Opioid_revision_behavior\`
- `meta\Opioid_revision_meta.csv`
- `data\Opioid_revision_OF_results.csv`  (open field)
- `data\LT\Opioid_revision_LT_merge_df_260310.csv`  (pre-merged linear track)

## Excluded (not needed for the panels/table)
- The one-time raw linear-track merge (concatenating `all_days_*_260310.xlsx` sheets:
  General / Supervised / Kinematics) — reproduced upstream to make the merged CSV above.
- The occupancy heatmaps (S1B) come from raw tracking, not in this notebook.
- A stray random-noise demo heatmap cell.

## Conditions
Saline, Acute, Chronic, Early W.D. (Withdrawal_Morphine), Re-Exposure, Late W.D.
Day1-6 map to these in the linear-track table.
