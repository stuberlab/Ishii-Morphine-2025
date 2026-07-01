import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 3H–K — MOR conditional knockout (VTA)

    MORfl/fl mice, VTA injection of Cre (`VTA-Cre`, MOR deletion) vs eYFP control (`VTA-GFP`).
    Whole-brain c-Fos after acute morphine (batch 1 only). Minimum code for the figure panels
    + table:

    - **3J** representative coronal planes of mean c-Fos+ cell density, eYFP vs Cre.
    - **3K** per-region quantification (Student t-test + BH) restricted to the acute-morphine
      responsive regions from Figure 3B; significant decreases (CM, MD, PC, AcbSh) in Cre.
    - **Table 6 (cKO part)** — the per-region t-test results.

    Inputs from the deposit `03_mor_cko_vta/`: `MORcKO.csv`, the c-Fos region table
    `Ex_639_Ch2_stitched_long_merge_Annotated_counts_clean_with_density.csv`, and the c-Fos
    heatmap zarr `Ex_639_Ch2_stitched_heatmap_array/` (+ fnamelist). Cross-reference:
    `01_main_cfos_morphine/Figure3B_effect_size_df.csv` (acute-responsive regions).
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import tifffile
    import pickle
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import dask.array as da

    from brain_vis import overlap_contour, set_transparency, get_subregions
    return (
        Path,
        da,
        get_subregions,
        mpl,
        np,
        os,
        overlap_contour,
        pd,
        pickle,
        plt,
        set_transparency,
        sns,
        tifffile,
    )


@app.cell
def _(Path, os):
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    GROUP = DATA_ROOT / "03_mor_cko_vta"
    GROUP01 = DATA_ROOT / "01_main_cfos_morphine"   # for Figure3B_effect_size_df.csv
    ATLAS = DATA_ROOT / "shared" / "atlas"
    FIG_OUT = Path("../figures")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    figure_key = "Figure3"
    assert GROUP.exists(), f"DATA_ROOT not found: {DATA_ROOT}. Set OPIOID_DATA_ROOT."
    return ATLAS, FIG_OUT, GROUP, GROUP01, figure_key


@app.cell
def _(plt):
    plt.rcParams.update({
        'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.edgecolor': 'black',
        'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
        'legend.facecolor': 'none', 'legend.edgecolor': 'none', 'text.color': 'black',
        'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Config + load""")
    return


@app.cell
def _(ATLAS, GROUP, pd, pickle, tifffile):
    cfos_channel = 'Ex_639_Ch2_stitched'                 # c-Fos channel
    Conditions = ['VTA-GFP', 'VTA-Cre']                  # eYFP control, Cre (MOR deletion)
    condition_colors = ['lime', 'magenta']
    atlas_resolution = (20, 20, 50)
    spill_radius = 300                                   # um around the VTA injection site

    metadf = pd.read_csv(GROUP / "MORcKO_subset.csv", index_col=False)
    metadf = metadf[(metadf.Usable == True) & (metadf.Staining_Batch == 1)]

    atlas_df = pd.read_csv(
        ATLAS / "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv",
        index_col=False)
    contour_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v2.9_contour_map.tif")
    atlas_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v4.0.tif")
    with open(ATLAS / "curated_acronym.pickle", "rb") as _h:
        curated_acronyms = pickle.load(_h)

    print("subjects per condition:")
    print(metadf[metadf.Condition.isin(Conditions)].groupby('Condition').size())
    return (
        Conditions,
        atlas_df,
        atlas_img,
        atlas_resolution,
        cfos_channel,
        condition_colors,
        contour_img,
        curated_acronyms,
        metadf,
        spill_radius,
    )


@app.cell
def _(Conditions, GROUP, metadf, pd):
    # c-Fos region table; per-region-mean normalization (variable tested in 3K)
    cFos_merge_df = pd.read_csv(
        GROUP / "Ex_639_Ch2_stitched_long_merge_Annotated_counts_clean_with_density.csv", index_col=False)
    cFos_merge_df = cFos_merge_df[cFos_merge_df.fname.isin(metadf.fname)]
    cFos_merge_df = cFos_merge_df[cFos_merge_df.Condition.isin(Conditions)]
    _mean = (cFos_merge_df[['acronym', 'density']].groupby('acronym').mean()
             .rename(columns={'density': 'mean_density_by_acronym'}).reset_index())
    cFos_merge_df = cFos_merge_df.merge(_mean, on='acronym', how='left')
    cFos_merge_df['normalized_density'] = cFos_merge_df['density'] / cFos_merge_df['mean_density_by_acronym']
    return (cFos_merge_df,)


@app.cell
def _(GROUP01, atlas_df, np, pd):
    # acute-morphine responsive regions (from Figure 3B) + their subregions
    _eff = pd.read_csv(GROUP01 / "Figure3B_effect_size_df.csv", index_col='acronym')
    original_acute_rejected_acronyms = _eff[_eff.Saline_vs_Acute_Morphine_posthoc_qadj < 0.05].index.values
    acute_rejected_acronyms = original_acute_rejected_acronyms.copy()
    for _acr in original_acute_rejected_acronyms:
        _subs = atlas_df.loc[atlas_df.parent_acronym == _acr, 'acronym'].values
        acute_rejected_acronyms = np.concatenate([acute_rejected_acronyms, _subs])
    print(len(original_acute_rejected_acronyms), "acute-responsive regions")
    return acute_rejected_acronyms, original_acute_rejected_acronyms


@app.cell
def _(atlas_df, atlas_img, atlas_resolution, get_subregions, np, spill_radius):
    # regions near the VTA injection site (to exclude from the 3K plot)
    _vid = atlas_df.loc[atlas_df.acronym == 'VTA', 'id'].values[0]
    _subids = get_subregions(atlas_df, _vid, return_original=True)['id'].values
    _zs = np.concatenate([np.where(atlas_img == ID)[0] for ID in _subids])
    _zc = int(np.mean(_zs))
    _half = spill_radius // atlas_resolution[2] // 2
    _ids = np.unique(atlas_img[max(0, _zc - _half):_zc + _half, :, :])
    _id2acr = dict(zip(atlas_df['id'], atlas_df['acronym']))
    candidate_spill_acronyms = [_id2acr[i] for i in _ids if i in _id2acr]
    return (candidate_spill_acronyms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure 3K — per-region quantification (t-test + BH), Table 6 (cKO)""")
    return


@app.cell
def _(Conditions, GROUP, acute_rejected_acronyms, cFos_merge_df, figure_key, original_acute_rejected_acronyms, pd):
    from scipy.stats import ttest_ind as _ttest_ind
    from statsmodels.stats import multitest as _multitest

    _sub = cFos_merge_df[cFos_merge_df.acronym.isin(acute_rejected_acronyms)]
    _rows = []
    for _acr in _sub.acronym.unique():
        _a = _sub.loc[(_sub.acronym == _acr) & (_sub.Condition == Conditions[0]), 'normalized_density'].values
        _b = _sub.loc[(_sub.acronym == _acr) & (_sub.Condition == Conditions[1]), 'normalized_density'].values
        _s, _p = _ttest_ind(_a, _b)
        _rows.append({'acronym': _acr, 'pvalue': _p, 'statistic': _s})
    ttest_stat_df = pd.DataFrame(_rows)

    # BH correction across the (primary) acute-responsive regions
    _m = ttest_stat_df.acronym.isin(original_acute_rejected_acronyms)
    _rej, _q, _, _ = _multitest.multipletests(ttest_stat_df.loc[_m, 'pvalue'], alpha=0.05, method='fdr_bh')
    ttest_stat_df.loc[_m, 'qvalue'] = _q
    ttest_stat_df.loc[_m, 'rejected'] = _rej
    ttest_stat_df.to_csv(GROUP / f'{figure_key}_MORcKO_normalized_density_ttest_stat_df.csv', index=False)
    print("significant (q<0.05):", ttest_stat_df[ttest_stat_df.qvalue < 0.05].acronym.values)
    return (ttest_stat_df,)


@app.cell
def _(Conditions, FIG_OUT, acute_rejected_acronyms, cFos_merge_df, candidate_spill_acronyms, condition_colors, figure_key, np, plt, sns, ttest_stat_df):
    # 3K stripplot — acute-responsive regions, injection-site regions removed, sorted by q
    _inj = list(np.intersect1d(acute_rejected_acronyms, candidate_spill_acronyms)) + ['VTA']
    _regs = ttest_stat_df[ttest_stat_df.acronym.isin(np.setdiff1d(acute_rejected_acronyms, _inj))].sort_values('qvalue').acronym.values
    _regs = list(_regs) + ['IF', 'VTA']
    _data = cFos_merge_df[cFos_merge_df.acronym.isin(_regs)]
    (_fig, _ax) = plt.subplots(1, 1, figsize=(9, 1.2))
    sns.stripplot(data=_data, hue='Condition', y='normalized_density', x='acronym', dodge=True,
                  order=_regs, hue_order=Conditions, ax=_ax, palette=condition_colors, alpha=0.4, size=3)
    sns.pointplot(data=_data, hue='Condition', y='normalized_density', x='acronym', order=_regs,
                  dodge=0.5 - 0.5 / len(_regs), hue_order=Conditions, ax=_ax, palette=condition_colors,
                  markers="o", markersize=3, linestyle="none", linewidth=0.5)
    sns.despine()
    _ax.set_xticklabels(_regs, rotation=-45, fontsize=9)
    if _ax.get_legend() is not None:
        _ax.get_legend().remove()
    _ax.set_xlabel(''); _ax.set_ylabel('Normalized density', fontsize=11)
    _fig.savefig(FIG_OUT / f'{figure_key}K_MORcKO_quantification.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}K_MORcKO_quantification.pdf', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3J — mean c-Fos density maps (eYFP vs Cre)

    Needs the c-Fos heatmap zarr `Ex_639_Ch2_stitched_heatmap_array/` in the deposit.
    """)
    return


@app.cell
def _(ATLAS, Conditions, FIG_OUT, GROUP, atlas_img, cfos_channel, condition_colors, contour_img, da, figure_key, mpl, np, overlap_contour, pd, plt, set_transparency):
    _zarr = GROUP / f"{cfos_channel}_heatmap_array"
    if _zarr.exists():
        _fnamelist = np.load(GROUP / f"{cfos_channel}_fnamelist.npy", allow_pickle=True)
        _meta = pd.read_csv(GROUP / "MORcKO_subset.csv").set_index('fname').loc[_fnamelist].reset_index()
        _keep = ((_meta.Staining_Batch == 1) & (_meta.Usable == True)).values
        _da = da.from_zarr(str(_zarr))[_keep]
        _meta = _meta[_keep].reset_index(drop=True)

        _zplanes = [84, 104, 117, 153, 186, 220]
        _imy, _imx = slice(25, 425), slice(50, 600)
        _cmap = plt.cm.YlGn
        _vmax = 30
        for _cond in Conditions:
            _theatmap = np.nanmean(_da[(_meta.Condition == _cond).values], axis=0).reshape(atlas_img.shape)
            _theatmap = np.asarray(_theatmap); _theatmap[np.isnan(_theatmap)] = 0
            (_fig, _axs) = plt.subplots(1, len(_zplanes), figsize=(3 * len(_zplanes), 3), sharey=True)
            _fig.subplots_adjust(wspace=0.25, hspace=0.3)
            for _i, _ax in enumerate(_axs):
                (__, _ov) = overlap_contour(_theatmap[_zplanes[_i], _imy, _imx],
                                            contour_img[_zplanes[_i], _imy, _imx],
                                            cmin=0, cmax=_vmax, colormap=_cmap)
                _tr = set_transparency(_ov, (atlas_img == 0)[_zplanes[_i], _imy, _imx])
                _ax.imshow(_tr); _ax.axis('off'); _ax.set_ylabel(_cond, color='black')
            _cax = _fig.add_axes([0.92, 0.15, 0.015, 0.7])
            mpl.colorbar.ColorbarBase(_cax, cmap=_cmap, norm=mpl.colors.Normalize(vmin=0, vmax=_vmax)).set_label('c-Fos+ cells', fontsize=8)
            _fig.savefig(FIG_OUT / f'{figure_key}J_{_cond}_cFos_density.png', bbox_inches='tight', dpi=216)
            _fig.savefig(FIG_OUT / f'{figure_key}J_{_cond}_cFos_density.pdf', bbox_inches='tight', dpi=216)
    else:
        print("c-Fos heatmap zarr not found:", _zarr.name, "-> 3J maps skipped (copy the zarr into the deposit).")
    return


if __name__ == "__main__":
    app.run()
