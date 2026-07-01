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
    # Figure 3C–G (+ S8) — MOR/KOR retro-orbital labeling

    KOR-Cre / MOR-Cre mice, retro-orbital AAV-PHP.eB H2B-tdTomato, acute morphine. KOR acts as
    a negative control. Minimum code for the figure panels + table:

    - **3G** per-region scatter of double+ / tdTomato+ across the acute-morphine responsive
      regions (Figure 3B), Student t-test (MOR vs KOR) + BH -> **Table 6 (double+)**.
    - **S8 (tdTomato)** the same for raw tdTomato+ density -> **Table 6 (tdTomato)**.
    - **3F** representative coronal planes of mean double+ / tdTomato+ density per genotype
      (needs the overlap + TRAP heatmap zarrs; guarded).

    Inputs (deposit `02_mor_kor_labeling/`): `MORKOR_Opioid.csv`,
    `MORKOR_cFos_full_result.csv`, and (for 3F) `MORKOR_overlap_heatmap_array/`,
    `MORKOR_tdTomato_heatmap_array/`, `MORKOR_overlap_fnamelist.pickle`. Cross-ref:
    `01_main_cfos_morphine/Figure3B_effect_size_df.csv`.
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
    GROUP = DATA_ROOT / "02_mor_kor_labeling"
    GROUP01 = DATA_ROOT / "01_main_cfos_morphine"   # Figure3B_effect_size_df.csv
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
def _(ATLAS, GROUP, get_subregions, pd, pickle, tifffile):
    Conditions = ['Acute_Morphine']
    genotypes = ['KOR-Cre', 'MOR-Cre']            # KOR = negative control, MOR = test
    genotype_colors = ['gray', 'orange']

    metadf = pd.read_csv(GROUP / "MORKOR_Opioid.csv", index_col=False)
    metadf = metadf[metadf.Usable]

    atlas_df = pd.read_csv(
        ATLAS / "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv",
        index_col=False)
    contour_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v2.9_contour_map.tif")
    atlas_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v4.0.tif")

    total_df = pd.read_csv(GROUP / "MORKOR_cFos_full_result.csv", index_col=False)
    total_df = total_df[total_df.Condition.isin(Conditions)]
    _rm_ids = atlas_df[(atlas_df.acronym == 'HB') | (atlas_df.acronym == 'CBL')]['id'].values
    _rm = pd.concat([get_subregions(atlas_df, _i, return_original=True) for _i in _rm_ids], axis=0)
    _sub_atlas = atlas_df.set_index(['id']).drop(_rm['id'].values)
    total_df = total_df[total_df.acronym.isin(_sub_atlas.acronym.unique())]

    print(total_df.groupby('Genotype').ID.nunique())
    return atlas_df, atlas_img, contour_img, genotype_colors, genotypes, metadf, total_df


@app.cell
def _(GROUP01, np, pd, total_df):
    # acute-morphine responsive regions (Figure 3B), restricted to regions present here
    effect_size_df = pd.read_csv(GROUP01 / "Figure3B_effect_size_df.csv", index_col='acronym')
    _acute = effect_size_df[effect_size_df.Saline_vs_Acute_Morphine_posthoc_qadj < 0.05].index.values
    acute_rejected_acronyms = np.array([a for a in _acute if a in set(total_df.acronym)])
    print(len(acute_rejected_acronyms), "acute-responsive regions present in the MOR/KOR data")
    return acute_rejected_acronyms, effect_size_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Per-region t-test (MOR vs KOR) helper + Table 6""")
    return


@app.cell
def _(GROUP, acute_rejected_acronyms, effect_size_df, np, pd, total_df):
    from scipy.stats import ttest_ind as _ttest_ind
    from statsmodels.stats import multitest as _multitest

    def genotype_ttest(variable, table_name):
        _rows = []
        for _acr in total_df.acronym.unique():
            _a = total_df[(total_df.Genotype == 'MOR-Cre') & (total_df.acronym == _acr)][variable].values.astype('float')
            _b = total_df[(total_df.Genotype == 'KOR-Cre') & (total_df.acronym == _acr)][variable].values.astype('float')
            _s, _p = _ttest_ind(_a, _b)
            _rows.append({'acronym': _acr, 'pvalue': 1.0 if np.isnan(_p) else _p, 'tvalue': _s,
                          'df': len(_a) + len(_b) - 2, 'delta': np.mean(_a) - np.mean(_b)})
        _sdf = pd.DataFrame(_rows).set_index('acronym')
        _rej, _q, _, _ = _multitest.multipletests(_sdf.loc[acute_rejected_acronyms, 'pvalue'], alpha=0.05, method='fdr_bh')
        _sdf.loc[acute_rejected_acronyms, 'corrected_pvalue'] = _q
        _sdf.loc[acute_rejected_acronyms, 'rejected'] = _rej
        # export merged with the Figure 3B effect sizes
        _out = effect_size_df.join(_sdf.add_prefix('MORvsKOR_'), how='left')
        _out.to_excel(GROUP / table_name)
        return _sdf

    double_stats = genotype_ttest('overlap_over_Ex_561_Ch1_stitched', 'Table6_MORvsKOR_double.xlsx')
    tdt_stats = genotype_ttest('Ex_561_Ch1_stitched_density', 'Table6_MORvsKOR_tdTomato.xlsx')
    print("double+ significant:", double_stats[double_stats.get('corrected_pvalue') < 0.05].index.values)
    return double_stats, tdt_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure 3G — double+ / tdTomato+ scatter; S8 — tdTomato+ scatter""")
    return


@app.cell
def _(FIG_OUT, acute_rejected_acronyms, atlas_df, double_stats, figure_key, genotype_colors, genotypes, np, plt, sns, tdt_stats, total_df):
    def genotype_scatter(variable, stats_df, ylabel, out_key):
        _order = acute_rejected_acronyms[np.argsort(stats_df.loc[acute_rejected_acronyms, 'pvalue'].values)]
        _data = total_df[total_df.acronym.isin(acute_rejected_acronyms)]
        (_fig, _ax) = plt.subplots(1, 1, figsize=(9, 1.2))
        sns.stripplot(data=_data, hue='Genotype', y=variable, x='acronym', dodge=True, order=_order,
                      hue_order=genotypes, ax=_ax, palette=genotype_colors, alpha=0.4, size=3)
        sns.pointplot(data=_data, hue='Genotype', y=variable, x='acronym', order=_order,
                      dodge=0.5 - 0.5 / len(_order), hue_order=genotypes, ax=_ax, palette=genotype_colors,
                      markers="o", markersize=5, linestyle="none", linewidth=0.5)
        sns.despine()
        _ax.set_xticklabels([atlas_df[atlas_df.acronym == f].cleaned_acronym.values[0] for f in _order],
                            fontsize=11, rotation=-45)
        if _ax.get_legend() is not None:
            _ax.get_legend().remove()
        _ax.set_xlabel(''); _ax.set_ylabel(ylabel, fontsize=12)
        # significance stars from the BH-corrected q-values
        for _i, _acr in enumerate(_order):
            _q = stats_df.loc[_acr, 'corrected_pvalue']
            _stars = '***' if _q < 1e-3 else '**' if _q < 1e-2 else '*' if _q < 0.05 else ''
            if _stars:
                _ax.text(_i, _ax.get_ylim()[1], _stars, ha='center', va='bottom', fontsize=8)
        _fig.savefig(FIG_OUT / f'{figure_key}{out_key}.png', bbox_inches='tight', dpi=216)
        _fig.savefig(FIG_OUT / f'{figure_key}{out_key}.pdf', bbox_inches='tight')

    genotype_scatter('overlap_over_Ex_561_Ch1_stitched', double_stats, 'double+/tdTomato+', 'G')
    genotype_scatter('Ex_561_Ch1_stitched_density', tdt_stats, 'tdTomato+ density\n(cells/mm3)', 'S8_tdTomato')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 3F — mean double+/tdTomato+ density maps per genotype

    Normalized double+ / tdTomato+ heatmaps (needs the overlap + TRAP zarrs; guarded).
    """)
    return


@app.cell
def _(ATLAS, FIG_OUT, GROUP, atlas_img, contour_img, figure_key, genotypes, mpl, np, overlap_contour, pd, pickle, plt, set_transparency, da):
    _overlap_zarr = GROUP / "MORKOR_overlap_heatmap_array"
    _tdt_zarr = GROUP / "MORKOR_tdTomato_heatmap_array"
    _fn = GROUP / "MORKOR_overlap_fnamelist.pickle"
    if _overlap_zarr.exists() and _tdt_zarr.exists() and _fn.exists():
        with open(_fn, "rb") as _h:
            _fnamelist = pickle.load(_h)
        _meta = pd.read_csv(GROUP / "MORKOR_Opioid.csv")
        _meta = _meta[_meta.Usable].set_index('fname').loc[_fnamelist].reset_index()
        _ov = da.from_zarr(str(_overlap_zarr))
        _tdt = da.from_zarr(str(_tdt_zarr))
        _norm = _ov / _tdt

        _zplanes = [84, 104, 117, 153, 186, 220]
        _imy, _imx = slice(25, 425), slice(50, 600)
        _vmax = 0.3
        for _g in genotypes:
            _mask = (_meta.Genotype == _g).values
            _hm = np.nanmean(_norm[_mask], axis=0).reshape(atlas_img.shape)
            _hm = np.asarray(_hm); _hm[np.isnan(_hm)] = 0
            (_fig, _axs) = plt.subplots(1, len(_zplanes), figsize=(3 * len(_zplanes), 3), sharey=True)
            _fig.subplots_adjust(wspace=0.25, hspace=0.3)
            for _i, _ax in enumerate(_axs):
                (__, _oimg) = overlap_contour(_hm[_zplanes[_i], _imy, _imx], contour_img[_zplanes[_i], _imy, _imx],
                                              cmin=0, cmax=_vmax, colormap=plt.cm.Reds)
                _tr = set_transparency(_oimg, (atlas_img == 0)[_zplanes[_i], _imy, _imx])
                _ax.imshow(_tr); _ax.axis('off'); _ax.set_ylabel(_g, color='black')
            _cax = _fig.add_axes([0.92, 0.15, 0.015, 0.7])
            mpl.colorbar.ColorbarBase(_cax, cmap=plt.cm.Reds, norm=mpl.colors.Normalize(vmin=0, vmax=_vmax)).set_label('double+/tdTomato+', fontsize=8)
            _fig.savefig(FIG_OUT / f'{figure_key}F_{_g}_double_over_tdTomato.png', bbox_inches='tight', dpi=216)
            _fig.savefig(FIG_OUT / f'{figure_key}F_{_g}_double_over_tdTomato.pdf', bbox_inches='tight', dpi=216)
    else:
        print("overlap/TRAP zarr or fnamelist not found -> 3F maps skipped (copy them into the deposit).")
    return


if __name__ == "__main__":
    app.run()
