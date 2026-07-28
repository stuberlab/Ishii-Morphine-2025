# Fig3_MORKOR (marimo) — notes

Standalone marimo notebook for **Figure 3C–G (+ S8 tdTomato)**, built from
`Opioid_Figure 3 MOR-KOR.ipynb` (97 cells). Scoped to the minimum for the panels + table.

## Panels / tables produced
- **3G** — per-region scatter of double+ / tdTomato+ (`overlap_over_Ex_561_Ch1_stitched`)
  across the acute-morphine responsive regions (Figure 3B), Student t-test (MOR vs KOR) + BH.
  Table: `Table6_MORvsKOR_double.xlsx`.
- **S8 (tdTomato)** — same for raw tdTomato+ density (`Ex_561_Ch1_stitched_density`).
  Table: `Table6_MORvsKOR_tdTomato.xlsx`.
- **3F** — mean double+/tdTomato+ density coronal planes per genotype (needs the overlap +
  TRAP zarrs; the cell is guarded and skips if absent).

The per-region t-test + BH is factored into `genotype_ttest(variable, table_name)` and the
scatter into `genotype_scatter(...)`, each called for double+ and tdTomato.

## Inputs (deposit)
- `02_mor_kor_labeling/MORKOR_Opioid.csv` (meta; `Usable`).
- `02_mor_kor_labeling/MORKOR_cFos_full_result.csv` (region table; has the `overlap_over_*`
  and `*_density` columns).
- For 3F: `MORKOR_overlap_heatmap_array/`, `MORKOR_tdTomato_heatmap_array/` (zarr),
  `MORKOR_overlap_fnamelist.pickle`.
- Cross-ref: `01_main_cfos_morphine/Figure3B_effect_size_df.csv`.

## Cleanup
- Paths → `DATA_ROOT`/Figshare; `contour_visualization`/`create_mask_for_region` → `brain_vis`.
- Resolved the original's variable clobbering (it overwrote the same `MOR-Cre_vs_KOR-Cre_*`
  columns for both double+ and tdTomato and exported `Table 4` / `Table 4_tdTomato`) by
  computing separate stat frames and exporting `Table6_MORvsKOR_double` / `_tdTomato`.
- Table numbering updated 4 → 6 (manuscript current).

## Excluded (not needed for these panels/tables)
- The DR and VTA spatial-distribution S8 sub-panels (heavy per-voxel spatial computation),
  the raw double+ density maps, and the spatial voxel t-test section.

## To confirm on a real run
- KOR-Cre is the negative control; MOR vs KOR direction as in the source.
- 3F normalized map = mean over subjects of (overlap / TRAP) heatmaps, cmax=0.3, planes
  [84,104,117,153,186,220].

## Renaming
- The tdTomato+ heatmap zarr was misnamed `MORKOR_TRAP_heatmap_array` in the original
  (legacy label from the shared overlap pipeline; there is NO TRAP experiment in MOR/KOR).
  Renamed to **`MORKOR_tdTomato_heatmap_array`**. Copy the source zarr to the new name.
