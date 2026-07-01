# Ishii-Morphine-2025 — Analysis Map

This document maps every manuscript figure/table to the notebook that produces it
and the data source it draws from. The repository is being reorganized **by data
source** (not by figure number) because several notebooks span multiple figures and
several figures draw on the same dataset.

> Status: folder skeleton + documentation only. No files have been moved or edited yet.
> Raw imaging data (light-sheet stacks, registered cell maps, atlas volumes, MERFISH
> voxel data) are **not** in this repo — they will be deposited on **Figshare** and
> referenced from each group's README.
>
> **Figshare deposit path:** `C:\Users\stuberadmin\OneDrive - UW\Opioid_Wholebrain_manuscript\Figshare`
> mirrors these same data-source groups (raw data destination).
>
> ⚠️ **The current notebooks are NOT a reliable ground truth.** The mapping below is a
> starting hypothesis. We will verify and rebuild figure-by-figure, **starting from
> Figure 1**, against the manuscript and the actual data — not by trusting existing
> notebook contents.

---

## ⚠️ Important: notebook numbers ≠ manuscript figure numbers

The notebook filenames were created during analysis and are **offset** from the final
manuscript numbering. Mapping below is by content, not by filename. Renaming notebooks
to match manuscript panels is part of the later cleanup pass.

| Notebook (current name) | Produces (manuscript) |
|---|---|
| `Opioid_Figure 1.ipynb` | Figure 1 |
| `Opioid_Figure 1_lateral analysis.ipynb` | Figure S5 |
| `Opioid_Figure 2.ipynb` (+ `Opioid_Figure 2_TreeBH.R`) | Figure 2, S4, S5 |
| `Opioid_Figure 3.ipynb` | Figure 3B (acute-responsive region selection) |
| `Opioid_Figure 3 MOR-KOR.ipynb` | Figure 3C–G, S7, S8 |
| _(MOR cKO notebook — TBD)_ | Figure 3H–K (MORfl/fl::Cre vs ::eYFP) |
| `Opioid_Figure 4.ipynb` | Figure 5 (UMAP/HDBSCAN clustering) |
| `Opioid_Figure 5.ipynb` (+ `Opioid_Figure 6_TreeBH.R`) | Figure 5 (predictive maps G–J) + Figure 6 (semi-NMF) |
| `Opioid_Figure 6.ipynb` | Figure 7 (TRAP2) |
| `Opioid_Figure 7.ipynb` | Figure 8, S14 |
| `Opioid_Figure_MOR_correlation.ipynb` | Figure S15, S16 |
| `Opioid_Figure Acute drug analysis.ipynb` | Figure 4 (4 opioid compounds) |
| `Opioid_Figure Acute drug analysis_backup.ipynb` | **DUPLICATE — delete** |
| `update_labels.ipynb` | shared preprocessing (atlas label transfer) |

---

## Path & data convention

- **All data (raw + processed) lives in the Figshare deposit**, organized in these
  same data-source groups. The repo is **code-only**.
- Each notebook defines one configurable root near the top:
  `DATA_ROOT = Path(...)  # = the Figshare deposit`.
- Every data load is a **relative subpath that mirrors the Figshare layout**, e.g.
  `DATA_ROOT/"shared/atlas/Kim_ref_adult_FP-label_v4.0.tif"`,
  `DATA_ROOT/"01_main_cfos_morphine/OP_cFos_full_result.csv"`.
- **No absolute lab paths** (`\\10.159.50.7\...`) anywhere — these are stripped during
  cleanup.
- To reproduce: download the Figshare deposit, set `DATA_ROOT` to it, run the notebook.
- Repo per-group folders hold `notebooks/`, `tables/`, `figures/` (outputs); `data/`
  contains only a pointer note.

Figshare deposit currently populated: `shared/atlas/` (Kim v4.0 + v2.9 contour atlas,
atlas_info CSV, curated/ancestor acronym pickles).

## Data-source groups

### 01 — Main whole-brain c-Fos morphine dataset
Core dataset: n = 43 animals, 5–6 drug conditions (saline, acute, chronic, early-WD,
re-exposure, late-WD). Whole-brain c-Fos+ cell density on the Unified (Kim) atlas.

- **Manuscript figures:** 1, 2, 5, 6, S2, S3, S4, S5, S6, S10, S11, S12, S13
- **Tables:** 3 (BRANCH + sex), 4 (cluster post-hoc), 7 (semi-NMF factor ANOVA)
- **Notebooks:** `Opioid_Figure 1`, `Opioid_Figure 1_lateral analysis`, `Opioid_Figure 2`
  (+ `_TreeBH.R`), `Opioid_Figure 3` (3B), `Opioid_Figure 4`, `Opioid_Figure 5`
  (+ `Opioid_Figure 6_TreeBH.R`)
- **In-repo processed data:** `OP_cFos_full_result.csv`, `OP_cFos_heatmap.csv`,
  `OP_cFos_fnamelist.npy`, `OP_cFos_heatmap_array/` (zarr), `OP_meta.csv`,
  `TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv`, `random_disjoint_nodes_*.npy`
- **Figshare (raw):** registered per-animal cell-count voxel maps (650×450×300),
  light-sheet stacks, `*_betas.npy` regression maps.

> **Figure 3 spans three data sources:** 3B comes from the main c-Fos dataset (Group 01,
> acute-responsive region selection); 3C–G + S8 from MOR/KOR labeling (Group 02);
> 3H–K from the MOR conditional knockout (Group 03). Table 6 covers Groups 02 and 03.

### 02 — MOR/KOR retro-orbital labeling
KOR-Cre / MOR-Cre animals, retro-orbital AAV-PHP.eB H2B-tdTomato, acute morphine.
Double+ (c-Fos+ / tdTomato+) quantification. **Also includes KOR/MOR co-labeling data**
(cells co-expressing both receptors) used for Figure 3.

- **Manuscript figures:** 3C–G, S7, S8
- **Tables:** 6 (shared with Group 03)
- **Notebooks:** `Opioid_Figure 3 MOR-KOR`
- **In-repo processed data:** `MORKOR_cFos_full_result.csv`,
  `MORKOR_overlap_fnamelist.pickle`, `MORKOR_overlap_heatmap_array/` (zarr),
  `overlap_over_Ex_561_Ch1_stitched_heatmap.csv`,
  `overlap_over_Ex_639_Ch2_stitched_heatmap.csv`, `MORKOR_Opioid.csv`
- **Depends on:** Group 01 (`Figure3B_effect_size_df.csv` = acute-responsive regions)
- **Figshare (raw):** registered tdTomato + c-Fos channel maps; KOR/MOR co-label maps.

### 03 — MOR conditional knockout (VTA)
MORfl/fl mice with VTA injection of Cre (`MORfl/fl::Cre`, MOR deletion) vs eYFP control
(`MORfl/fl::eYFP`). Whole-brain c-Fos after acute morphine — tests whether VTA MOR is
causally required for the brain-wide activation pattern. n = 7 (::eYFP), 6 (::Cre).

- **Manuscript figures:** 3H–K
- **Tables:** 6 (shared with Group 02)
- **Notebooks:** TBD — verify during the Figure 3 rebuild (current notebooks are not ground truth).
- **In-repo processed data:** TBD (to be located/confirmed).
- **Figshare (raw):** whole-brain c-Fos maps for ::Cre and ::eYFP animals; VTA injection validation.

### 04 — Acute opioids (4 compounds)
Saline / morphine / fentanyl / buprenorphine / oxycodone, single acute dose.

- **Manuscript figures:** 4
- **Tables:** 5
- **Notebooks:** `Opioid_Figure Acute drug analysis`
- **In-repo processed data:** `acute_drug_revision_full_list.csv`,
  `TreeFDRS_pvalue_Acute_*_glm_without_Cocaine_GLM.csv`, `glm_stat_df_post_TreeBH.csv`
- **Figshare (raw):** per-channel intensity / atlas-transformed `.npy`, fname lists.

### 05 — TRAP2 neural-ensemble overlap
TRAP2 × c-Fos double-labeling; 4 conditions (Acute/Saline/Chronic/EarlyWD-TRAP).

- **Manuscript figures:** 7
- **Tables:** 8
- **Notebooks:** `Opioid_Figure 6`
- **In-repo processed data:** `OPTRAP_meta.csv`,
  `total_long_merge_Annotated_counts_with_leaf_with_density.csv`
  (NOTE: `MORKOR_TRAP_heatmap_array` is NOT TRAP2 data — it's the MOR/KOR tdTomato heatmap,
  belongs to Group 02, and has been renamed `MORKOR_tdTomato_heatmap_array`.)
- **Depends on:** Group 01 cluster assignments (`Figure5E_umap_embedding.csv`)
- **Figshare (raw):** registered double+ cell maps.

### 06 — Spatial gene expression / ABI MERFISH
Voxel-level MERFISH (1122 genes) correlated with semi-NMF spatial factors
(factor 1 "Morphine", factor 2 "Withdrawal").

- **Manuscript figures:** 8, S14, S15, S16
- **Tables:** 9
- **Notebooks:** `Opioid_Figure 7`, `Opioid_Figure_MOR_correlation`
- **In-repo processed data:** `gene_list_with_correlation.csv`,
  `*_factor_to_gene_list.csv`, `*_gene_list.csv`, `cluster_details.csv`
- **Depends on:** Group 01 semi-NMF factor maps.
- **Figshare (raw):** ABI MERFISH voxelized expression volumes (1122 genes),
  Allen→Kim atlas alignment.

### 07 — NAc histology / HCR in situ
20-µm coronal sections, HCR ISH (Kcnip1, Calcr) + Fos in nucleus accumbens; MOR/pPDH/c-Fos IHC.

- **Manuscript figures:** 8H–P, S9
- **Tables:** —
- **Notebooks:** currently inside `Opioid_Figure 7` (to be split out) + section-level analysis
- **In-repo processed data:** `NAc_ISH_meta_v3.xlsx`, `NAc_ISH_subject_metav3.xlsx`
- **Figshare (raw):** confocal section images.

### 08 — Behavioral characterization
Separate cohort (n = 20); open-field + linear-track; locomotion, somatic signs.

- **Manuscript figures:** S1
- **Tables:** 2
- **Notebooks:** none in repo yet — behavior analysis code to be added.
- **Figshare (raw):** tracking data / scored behavior.

### shared — atlas, ontology, helpers
Used by all groups.

- **Atlas/ontology:** `Kim_ref_adult_FP-label_v4.0.tif`,
  `Kim_ref_adult_FP-label_v2.9_contour_map.tif`,
  `atlas_info_KimRef_FPbasedLabel_v4.0_..._cleaned_acronyms.csv`,
  `clean_curated_acronyms.csv`, `ancestor_curated_acronym.pickle`,
  `curated_acronym.pickle`
- **Helper modules:** `active_sunburst.py`, `create_mask_for_region.py`,
  `contour_visualization.py`, `allen_merfish_kim_voxelization.py`, `update_labels.ipynb`
- **Table 1** (region acronym ↔ name lookup) lives here.

---

## Cleanup issues found (for the next pass)

1. **Hardcoded lab paths** — `Opioid_Figure 4/5/6/7`, `Acute drug analysis`, and
   `MOR_correlation` hardcode `\\10.159.50.7\...` UNC paths. Earlier notebooks
   (`Figure 1/1_lateral/2/3/3 MOR-KOR`) already use clean relative paths
   (`rootpath/data`, `rootpath/meta`). Standardize all on relative paths.
2. **Two path conventions** — early notebooks use `data/result`; later ones use
   `result` directly under root. Unify.
3. **Duplicate notebook** — `Opioid_Figure Acute drug analysis_backup.ipynb` is a
   copy of the non-backup version. Delete.
4. **Figure filename config** — notebook `figure_key`/`pannel_key` variables emit
   filenames using the *old* figure numbers. Update so output names match the final
   manuscript panel labels.
5. **`__pycache__/`, `desktop.ini`** — strip from repo; add to `.gitignore`.
6. **README figure list** mentions a `Figure 8` notebook that doesn't exist and an
   outdated description — update after renaming.

## Proposed next-phase file moves
For each group, move its notebook(s) → `notebooks/`, processed CSV/npy/zarr → `data/`,
the matching `Table N.xlsx` → `tables/`, and rendered panels → `figures/`. Atlas,
ontology, and helper `.py` go to `shared/`. (Not executed yet — pending your go-ahead.)
