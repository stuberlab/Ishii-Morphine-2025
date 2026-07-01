# Data manifest — what's in the Figshare deposit vs. still needed

Status of inputs for each cleaned notebook. "In deposit" = copied. "COPY (zarr)" = large,
copy yourself. "PROVIDE" = not in the repo (lab drive). "REGEN" = appears after running an
earlier cleaned notebook against the deposit.

## shared/atlas  — DONE
Kim v4.0 + v2.9 contour atlases, atlas_info CSV, curated/ancestor acronym pickles,
clean_curated_acronyms.csv.

## 01_main_cfos_morphine
- In deposit: OP_cFos_full_result.csv, OP_cFos_heatmap.csv, OP_cFos_fnamelist.npy,
  OP_cFos_heatmap_array/ (zarr), OP_meta.csv, TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv,
  random_disjoint_nodes_*.npy
- REGEN (written by running the notebooks): `{condition}_betas.npy`, `Sex_d_betas.npy`
  (Fig1 regression); `Figure2_C_glm_stat_df.csv` (Fig2 GLM); `Figure5E_umap_embedding.csv`
  (Fig5 clustering).
- PROVIDE: **Figure3B_effect_size_df.csv** — produced by the Figure-3B selection
  (`Opioid_Figure 3.ipynb`, not yet cleaned). Needed by groups 02 and 03. Bring from the
  lab drive or regenerate.
- Figure 6 (`Fig6_semiNMF.py`, semi-NMF factor analysis):
  - In deposit/group: `long_pivoted_heatmap_df_with_normalized_density.csv`,
    `Ex_639_Ch2_stitched_long_merge_Annotated_counts_with_leaf_with_density_with_normalized_density.csv`.
  - In `shared/atlas/`: `Kim_ref_adult_FP-label_v4.0.tif` (annotation) in addition to the v2.9
    contour + atlas info CSV + curated/ancestor acronym pickles already listed.
  - **PROVIDE (semi-NMF results)** in `FACTOR_DIR` (env `OPIOID_FACTOR_RESULTS`, else
    `GROUP/spatial_clustering_results`): `params.mat`, `params.pkl`, `factors.zarr` — outputs of
    the external `fos` package (modified Linderman-lab semi-NMF), NOT run by this notebook.
  - REGEN: `factor_anova_stat_df.csv` (factor ANOVA table) + `spatial_factors_for_subregion*.npy`.
  - The `factors/factor{0,1,2}.npy` needed by groups 06/searchlight are exports of these factors.

## 02_mor_kor_labeling
- In deposit: MORKOR_Opioid.csv, MORKOR_cFos_full_result.csv, MORKOR_overlap_fnamelist.pickle,
  overlap_over_Ex_561/639_Ch1/2_stitched_heatmap.csv
- COPY (zarr), needed for 3F maps:
  - `data\spatial\MORKOR_overlap_heatmap_array`  (132 MB)  -> `Figshare\02_mor_kor_labeling\MORKOR_overlap_heatmap_array`
  - `data\spatial\MORKOR_TRAP_heatmap_array`      (296 MB)  -> `Figshare\02_mor_kor_labeling\MORKOR_tdTomato_heatmap_array`  (RENAME on copy; this is the tdTomato heatmap, not a TRAP experiment)
- Cross-ref: Figure3B_effect_size_df.csv (see group 01).

## 03_mor_cko_vta   — all PROVIDE (nothing in repo)
- MORcKO.csv (meta)
- Ex_639_Ch2_stitched_long_merge_Annotated_counts_clean_with_density.csv
- COPY (zarr): `Ex_639_Ch2_stitched_heatmap_array` + `Ex_639_Ch2_stitched_fnamelist.npy`
  (lab drive: `\\10.158.246.229\DataCommon\SmartSPIM2\Ken\MORcKO\...`) -> `Figshare\03_mor_cko_vta\`
- Cross-ref: Figure3B_effect_size_df.csv (group 01).

## 05_trap2_ensemble
- In deposit: OPTRAP_meta.csv
- PROVIDE: total_long_merge_Annotated_counts_with_leaf_with_density.csv  (7D-J + Table 8; lab drive OPTRAP\result)
- COPY (zarr) for 7H MEAN double+ maps: `overlap_heatmap_array` + `fnamelist.npy`
  -> `Figshare\05_trap2_ensemble\`. (7H = mean over subjects per condition, NOT a regression.)
- OPTIONAL: TreeFDRS_pvalue_Figure7_D_glm_stat_df_no_batch.csv (R TreeBH region set; else Fig5 regions used).
- Cross-ref: 01_main_cfos_morphine/Figure5E_umap_embedding.csv (clusters; REGEN by Fig5).

### TRAP heatmap zarrs — locating them
- Per-subject heatmap zarrs are `{channel}_heatmap_array` for
  channel in [`overlap`, `Ex_561_Ch1_stitched`, `Ex_639_Ch2_stitched`], ordered by `fnamelist.npy`.
  `overlap` is needed for 7H; the other two only for the (not-built) 7K-M accumbens panels.
- ⚠️ The original notebook loads them from `\\10.159.50.7\LabCommon\Ken\data\`**`OpioidTRAP`**`\`
  (note: "OpioidTRAP", NOT the "OPTRAP" analysis folder). If that folder is missing, check
  `\\10.159.50.7\LabCommon\Ken\data\OPTRAP\` and the raw folder
  `\\10.158.246.229\DataCommon\SmartSPIM2\Ken\OPTRAP\`.

## 08_behavior   — all PROVIDE (lab drive `\\10.159.50.7\...\Opioid_revision_behavior`)
- `Opioid_revision_meta.csv`  (meta\)
- `Opioid_revision_OF_results.csv`  (data\)
- `Opioid_revision_LT_merge_df_260310.csv`  (data\LT\)
- All small CSVs; no zarr. Notebook writes `FigureS1_all_statistics.xlsx` (= Table 2).

## 04_acute_opioids_4drugs   — all PROVIDE (lab drive `\\10.159.50.7\...\Acute_drugs_for_revision`)
- `acute_drug_revision_full_list.csv` (meta\)
- `Ex_639_Ch2_stitched_long_merge_Annotated_counts_clean_with_density.csv` (result\)
- COPY (zarr): `Ex_639_Ch2_stitched_heatmap_array` + `fname_list.npy`  (raw zarr from
  `\\10.158.246.229\...\Ken\Fentanyl`)
- beta maps: `Acute_{Morphine,Fentanyl,Buprenorphine,Oxycodone}_betas.npy`, `constant_betas.npy`
- TreeFDR/GLM CSVs: `glm_stat_df_post_TreeBH.csv`,
  `TreeFDRS_pvalue_{condition}_glm_without_Cocaine_GLM.csv` (×4), `glm_stat_df_{condition}_*.csv`
- Notebook re-writes `FigureS7_*` stat CSVs + `Table 7.xlsx` into this folder.

## 07_nac_histology_hcr   — all PROVIDE (QuPath output + lab drive)
- meta: `analysis_master.xlsx`, `Opioid_pPDH_meta.csv`
- QuPath tree — place its subfolders **directly in `07_nac_histology_hcr/`**:
  `cell_quantification_roundness/` (AcbC/AcbSh per-annotation CSVs), `aligned_cells/`,
  `aligned_polygons/`, `template/`. (Notebook default `OPIOID_QUPATH_ROOT` = the group folder;
  set the env var if you keep the tree elsewhere, e.g. your Desktop.)
- `master_df_20260513_001335.csv` + `section_meta_*.csv` — saved intermediates; let you skip
  reprocessing the raw `cell_quantification_roundness/` CSVs.
### HCR in-situ (Figure 8 M-P) — `Fig8_NAc_ISH.py`
- In deposit: `NAc_ISH_meta_v3.xlsx` (+ `NAc_ISH_subject_metav3.xlsx`) — copied from the repo.
- PROVIDE: `total_merge_cell_df.csv` (collected spot-detection results; the raw FFT/Cellpose/
  big-fish pipeline that generates it is excluded from the notebook).

## 06_spatial_gene_merfish
- GENERATED by `Fig8_allen_preprocessing.py` (or provide precomputed): `gene_expression.zarr`,
  `cluster_expression.zarr`, `gene_df.csv`, `cluster_details.csv` -> written into this group.
- PROVIDE: `factors/factor{0,1,2}.npy` — semi-NMF spatial factors (from Figure 6).
- PROVIDE in `shared/atlas/` (for preprocessing): `template_25_coronal.tif`,
  `Kim_ref_adult_v1_brain.tif` (Allen CCF template + Kim reference).
- Allen ABC-atlas cache auto-downloads to the working dir (`OPIOID_ALLEN_PROJECT`); needs `ants`.
- Searchlight (S16) — `FigS16_searchlight.py`: reads `gene_expression.zarr`, `gene_df.csv`,
  `factors/factor1.npy`; writes searchlight zarrs to `OPIOID_SEARCHLIGHT_OUT` (else
  `GROUP/searchlight_output`). Needs the **`neurolight`** package installed.
