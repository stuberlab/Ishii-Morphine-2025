# Fig8_NAc_ISH (marimo) — notes

Marimo notebook for the NAc **HCR in-situ** quantification — **Figure 8 M–P**: Kcnip1 / Calcr
(+ Fos, Drd1/Drd2) in nucleus accumbens core (AcbC) / shell (AcbSh), Saline vs
Acute / Chronic / Early-WD morphine. Built from `NAc_ISH_Analysis_ThirdBatch.ipynb`.

## Scope — analysis + panels only (raw imaging excluded)
The original is a full raw-imaging pipeline with a mid-notebook kernel switch. I kept the
**statistical-analysis + plotting** section (from `# Statistical analysis` onward) and
**excluded the raw pipeline** (image rename, FFT filters, Cellpose segmentation, cytosol
dilation, big-fish spot detection) and the raw-image representative-visualization cells — those
need the Apotome raw images + Cellpose/big-fish/Fiji and can't run in one kernel. The kept
section starts from the collected spot-detection table `total_merge_cell_df.csv`.

## What it produces
- Positive-cell calls per gene (`*_regressed_counts > 5`), Core/Shell split, per-section means.
- Proportion of Fos+ / Kcnip1+ / Calcr+ cells by condition in AcbC / AcbSh (stripplot+pointplot,
  t-test + BH via statannotations) — panels O/P.
- Fos+ cell proportion along the A–P axis (Kim_z / AP mm), anterior vs posterior — panels M/N.
- Calcr/Kcnip1 (and Drd1/Drd2) overlap distributions.
- **Table:** `ttest_result_df.csv`; also writes `group_cell_df.csv`.

## Cleanup
- Converted with `marimo convert`; repointed paths to `DATA_ROOT` (group 07 for
  meta/results, `../figures/Figure8_NAc_ISH`). Removed `\\10.159.50.7\...\NAc_ISH` and the
  Apotome raw path.

## Inputs (deposit `07_nac_histology_hcr/`)
- `NAc_ISH_meta_v3.xlsx` (section meta; `Usable`) — copied to the deposit.
- **PROVIDE:** `total_merge_cell_df.csv` — the collected spot-detection results (output of the
  excluded raw pipeline). Drop it in the group folder.

## Requires
`statannotations`, `seaborn`, `pandas`, `scikit-image`, `tifffile` (all in requirements.txt).
