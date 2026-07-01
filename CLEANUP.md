# Repo cleanup — run locally

The cleaned code lives in `analysis/` but is **not yet git-tracked**, while the legacy staging
folders (`data/`, `meta/`, `notebooks/`) and a few one-off scripts still are. Run the following
from the repo root on your machine to finish the cleanup. (Nothing here touches the Figshare
deposit, which is already clean.)

## What's unnecessary and why

| Path | Why remove |
|------|-----------|
| `notebooks/` (root) | Raw source notebooks (`Opioid_Figure *.ipynb`, `update_labels.ipynb`) + vendored helpers (`contour_visualization.py`, `create_mask_for_region.py`, `active_sunburst.py`, `allen_merfish_kim_voxelization.py`) + `__pycache__`. All superseded by the cleaned `analysis/*/notebooks/*.py` (which now use `brain_vis`). ~80 MB. |
| `data/` (root) | Byte-for-byte duplicate of the Figshare deposit (atlas tifs, result CSVs, `random_disjoint_nodes_*.npy`). ~370 MB. |
| `meta/` (root) | Duplicate of deposit meta (`OP_meta.csv`, `MORKOR_Opioid.csv`, `OPTRAP_meta.csv`, `NAc_ISH_*.xlsx`, `clean_curated_acronyms.csv`). |
| `clear_notebook_outputs.py`, `simple_path_update.py`, `simple_string_replace.py`, `update_paths.py` | One-off conversion scripts, no longer needed. |
| `desktop.ini` (×2), `__pycache__/` | OS / Python junk. |

## Commands

```bash
cd Ishii-Morphine-2025

# 1. remove the legacy staging folders + one-off scripts + junk (from disk and git)
git rm -r --quiet data meta notebooks
git rm --quiet clear_notebook_outputs.py simple_path_update.py simple_string_replace.py update_paths.py

# 2. track the new code-only structure + the added .gitignore
#    (.gitignore excludes each group's data/figures/tables folders and the per-folder
#     READMEs, so this only tracks the notebooks, *_CHANGES.md, shared/helpers, and
#     analysis/MAPPING.md + DATA_MANIFEST.md)
git add analysis/ .gitignore README.md CLEANUP.md requirements.txt

# 3. sanity check: working tree should now be analysis/ + top-level files only
git status

# 4. commit
git commit -m "Clean up: code-only repo (analysis/), drop staged data/meta/legacy notebooks"
```

`.gitignore` already excludes `data/`, `meta/`, `notebooks/`, `.claude/`, `__pycache__/`, and
`desktop.ini`, so they won't creep back in.

## Optional

`analysis/shared/helpers/` (`contour_visualization.py`, `create_mask_for_region.py`,
`active_sunburst.py`, `allen_merfish_kim_voxelization.py`) is no longer imported by any cleaned
notebook — everything uses `brain_vis`, and the MERFISH voxelization is inlined in
`Fig8_allen_preprocessing.py`. Keep the folder as reference, or `git rm` it too if you want the
repo strictly to the notebooks.
