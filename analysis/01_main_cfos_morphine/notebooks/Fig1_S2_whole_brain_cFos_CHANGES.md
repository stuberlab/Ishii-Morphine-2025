# Fig1_S2_whole_brain_cFos — cleanup notes

Cleaned from `notebooks/Opioid_Figure 1.ipynb` + `Opioid_Figure 1_lateral analysis.ipynb`
(originals left untouched). Generates main-text panels **1B, 1C, 1D** and supplements
**S2** (brain-wide density maps) and **S5** (lateralization). Same computation; the changes
are organizational.

**Canonical file: `Fig1_S2_whole_brain_cFos.py`** (marimo). The `.ipynb` is the earlier
export and does NOT include S5 — edit/keep the marimo `.py`.

## S5 lateralization (added)
- Ported from `Opioid_Figure 1_lateral analysis.ipynb`: voxel-wise paired left/right
  t-tests per condition (right hemisphere flipped to align), BH-FDR; saline-vs-condition
  two-sample t-tests as a positive control; t-value maps saved as
  `Lateralization_analysis_tvalmap_<condition>.{png,pdf}`; significant-voxel proportions
  printed.
- Reuses existing reactive cells (`atlas_img`, `heatmap_da`, `metadf_2`, `brain_voxels`,
  `curated_zplanes`) instead of reloading. scipy/statsmodels imported as cell-local
  aliases to respect marimo's single-definition rule. Uses `brain_vis.overlap_contour`.

## Paths
- All inputs now resolve from `DATA_ROOT` (the Figshare deposit), via the Figshare layout:
  `GROUP = DATA_ROOT/"01_main_cfos_morphine"`, `ATLAS = DATA_ROOT/"shared/atlas"`.
- Set with the `OPIOID_DATA_ROOT` env var, or edit the default in the path cell.
- Removed the old `rootpath/data|meta|atlas` setup and the stray hardcoded
  `data/opioid_cfos/result` path used for the beta maps.
- Rendered panels now save to `../figures` (repo `analysis/01_main_cfos_morphine/figures/`)
  instead of `figure/Figure1/`.

## Imports
- All visualization/region helpers now come from the **`brain_vis`** package
  (https://github.com/kenjp1223/brainvis), the canonical replacement for the outdated
  `contour_visualization` / `contour_visualization2` modules:
  `from brain_vis import overlap_contour, set_transparency, get_subregions`.
- Added `brain-vis @ git+https://github.com/kenjp1223/brainvis` to `requirements.txt`
  (installed from GitHub, not vendored into the repo).
- Removed the locally-defined `set_transparency` (now `brain_vis.set_transparency`) and the
  `create_mask_for_region.get_subregions` call (now `brain_vis.get_subregions`).
- The vendored `contour_visualization.py`, `create_mask_for_region.py`, `active_sunburst.py`
  in `analysis/shared/helpers/` are superseded by `brain_vis` and can be deleted
  (see `analysis/shared/helpers/README.md`).
- Dropped unused imports: `read_roi`, `shutil`, `adjustText`, `statannotations`,
  and the unused `flatten_extend` helper.

## Beta maps (intermediate)
- The original had the `np.save` of `{variable}_betas.npy` commented out while later panels
  loaded those files — so it only ran if betas already existed. Now the save is **enabled**
  and writes to `GROUP/`, so the notebook regenerates them end-to-end.

## Filenames
- Panel outputs use `figure_key`+`pannel_key` set correctly per panel
  (`Figure1B`, `Figure1C`, `Figure1D_<condition>_betacoef`, `Figure1D_Saline_mean`).
  In the original, `pannel_key` leaked between cells (e.g. raw-count maps were saved as
  `Figure1B_..._rawcount`).

## Removed cells (not used for the published panels)
- "Unbiased preprocessing" exploratory block (re-fit OLS on covariates only; overwrote
  `models`; outputs never saved