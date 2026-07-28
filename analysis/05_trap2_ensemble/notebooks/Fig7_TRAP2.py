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
    # Figure 7 — TRAP2 acute-morphine neural ensemble

    TRAP2 x c-Fos double-labeling. 4 conditions (labeled `Cond1-Cond2`, first = TRAP schedule,
    second = c-Fos schedule): `AcM-AcM` (Acute-TRAP, reference), `Sal-AcM` (Saline-TRAP),
    `AcM-ChM` (Chronic-TRAP), `AcM-WDM` (EarlyWD-TRAP).

    - **7D** whole-brain double+ counts + per-region double+ density heatmap.
    - **7E-G** accuracy (double+/tdTomato+), efficiency (double+/c-Fos+), fidelity (F1) per
      subject, t-tests Acute-TRAP vs each (BH).
    - **7I** fidelity by Figure-5 cluster; **7J** fidelity in selected regions -> **Table 8**.
    - **7H** MEAN double+ density maps per condition (from the overlap heatmap zarr; guarded).

    Inputs (deposit `05_trap2_ensemble/`): `OPTRAP_meta.csv`,
    `total_long_merge_Annotated_counts_with_leaf_with_density.csv`; optional
    `overlap_density_pivoted_heatmap_df_without_zscore.csv` (7D heatmap),
    `TreeFDRS_pvalue_Figure7_D_glm_stat_df_no_batch.csv` (BRANCH region set),
    `overlap_heatmap_array/` (zarr) + `fnamelist.npy` (7H). Cross-ref:
    `01_main_cfos_morphine/Figure5E_umap_embedding.csv` (clusters).
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
    import matplotlib.pyplot as plt
    import seaborn as sns

    from brain_vis import overlap_contour, set_transparency, get_subregions
    return (
        Path,
        get_subregions,
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
    GROUP = DATA_ROOT / "05_trap2_ensemble"
    GROUP01 = DATA_ROOT / "01_main_cfos_morphine"   # Figure5E_umap_embedding.csv
    ATLAS = DATA_ROOT / "shared" / "atlas"
    FIG_OUT = Path("../figures")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    figure_key = "Figure7"
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
    Conditions = ['AcM-AcM', 'Sal-AcM', 'AcM-ChM', 'AcM-WDM']  # reference (Acute-TRAP) first
    Condition_figure_name = [r'$Acute^{TRAP}$', r'$Saline^{TRAP}$', r'$Chronic^{TRAP}$', r'$EarlyWD^{TRAP}$']
    Condition_color = ['lime', 'gray', 'orange', 'cyan']
    REF = 'AcM-AcM'

    metadf = pd.read_csv(GROUP / "OPTRAP_meta.csv", index_col=False)
    metadf = metadf[metadf.Condition.isin(Conditions)]

    atlas_df = pd.read_csv(
        ATLAS / "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv",
        index_col=False)
    contour_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v2.9_contour_map.tif")
    atlas_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v4.0.tif")
    with open(ATLAS / "curated_acronym.pickle", "rb") as _h:
        curated_acronyms = pickle.load(_h)

    total_df = pd.read_csv(GROUP / "total_long_merge_Annotated_counts_with_leaf_with_density.csv", index_col=False)
    total_df = total_df[total_df.Condition.isin(Conditions)]
    _rm_ids = atlas_df[(atlas_df.acronym == 'HB') | (atlas_df.acronym == 'CBL')]['id'].values
    _rm = pd.concat([get_subregions(atlas_df, _i, return_original=True) for _i in _rm_ids], axis=0)
    _sub = atlas_df.set_index(['id']).drop(_rm['id'].values)
    total_df = total_df[total_df.acronym.isin(_sub.acronym.unique())]
    return (
        Condition_color,
        Condition_figure_name,
        Conditions,
        REF,
        atlas_df,
        atlas_img,
        contour_img,
        curated_acronyms,
        metadf,
        total_df,
    )


@app.cell
def _(GROUP, GROUP01, curated_acronyms, pd):
    embedding_df = pd.read_csv(GROUP01 / "Figure5E_umap_embedding.csv", index_col=0)
    _tf = GROUP / "TreeFDRS_pvalue_Figure7_D_glm_stat_df_no_batch.csv"
    if _tf.exists():
        _t = pd.read_csv(_tf, index_col=False)
        rejected_acronyms = _t[(_t.acronym.isin(curated_acronyms)) & (_t.rejected == True)].acronym.values
        print(len(rejected_acronyms), "BRANCH-rejected regions (TreeBH)")
    else:
        rejected_acronyms = embedding_df.index.values
        print("TreeBH output not found -> using Figure-5 clustering regions:", len(rejected_acronyms))
    return embedding_df, rejected_acronyms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure 7D - whole-brain double+ counts""")
    return


@app.cell
def _(Condition_color, Condition_figure_name, Conditions, FIG_OUT, figure_key, plt, sns, total_df):
    _t = (total_df[total_df.parent_acronym == 'grey'][['Condition', 'ID', 'overlap_newcounts']]
          .groupby(['Condition', 'ID']).sum().reset_index())
    _t = _t[_t.overlap_newcounts > 0]
    (_fig, _ax) = plt.subplots(1, 1, figsize=(1, 2.5))
    sns.stripplot(data=_t, order=Conditions, y='Condition', x='overlap_newcounts', ax=_ax, alpha=0.25, size=3, palette=Condition_color)
    sns.pointplot(data=_t, order=Conditions, y='Condition', x='overlap_newcounts', ax=_ax, palette=Condition_color,
                  markers="o", markersize=5, linestyle="none", linewidth=0.5)
    _ax.set_yticks(range(len(Conditions))); _ax.set_yticklabels(Condition_figure_name, rotation=0)
    _ax.set_xlim(0,); _ax.set_ylabel(''); _ax.set_xlabel('Whole brain\ndouble+ cells'); sns.despine()
    _fig.savefig(FIG_OUT / f'{figure_key}D_total_cells.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}D_total_cells.pdf', bbox_inches='tight', dpi=216)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 7E-G - accuracy / efficiency / fidelity

    Per subject, averaged over the responsive regions (double+ & tdTomato+ & c-Fos+ present);
    t-test Acute-TRAP vs each condition, BH. **Accuracy = double+/tdTomato+ = overlap_over_561;
    Efficiency = double+/c-Fos+ = overlap_over_639** (manuscript definitions; the original
    notebook's E/F axis labels appear swapped -- verify).
    """)
    return


@app.cell
def _(Condition_color, Condition_figure_name, Conditions, FIG_OUT, REF, figure_key, plt, rejected_acronyms, sns, total_df):
    from statannotations.Annotator import Annotator as _Annotator

    def metric_plot(variable, ylabel, out_key):
        _t = total_df[(total_df.acronym.isin(rejected_acronyms))
                      & (total_df.Ex_561_Ch1_stitched_newcounts > 0)
                      & (total_df.Ex_639_Ch2_stitched_newcounts > 0)]
        _t = _t[['ID', 'Condition', variable]].groupby(['ID', 'Condition']).mean().reset_index()
        (_fig, _ax) = plt.subplots(1, 1, figsize=(1, 2))
        sns.stripplot(data=_t, x='Condition', y=variable, order=Conditions, ax=_ax, palette=Condition_color, alpha=0.25)
        sns.pointplot(data=_t, x='Condition', y=variable, order=Conditions, ax=_ax, palette=Condition_color,
                      markers="o", markersize=4, linestyle="none", linewidth=0.5)
        sns.despine(); _ax.set_ylim(0,); _ax.set_ylabel(ylabel); _ax.set_xticklabels(Condition_figure_name, rotation=-45)
        _pairs = [(REF, c) for c in Conditions if c != REF]
        _an = _Annotator(_ax, _pairs, data=_t, x='Condition', y=variable, order=Conditions)
        _an.configure(test='t-test_ind', text_format='simple', loc='outside', comparisons_correction="BH", correction_format="replace")
        _an.apply_and_annotate()
        _fig.savefig(FIG_OUT / f'{figure_key}{out_key}.png', bbox_inches='tight', dpi=216)
        _fig.savefig(FIG_OUT / f'{figure_key}{out_key}.pdf', bbox_inches='tight', dpi=216)

    metric_plot('overlap_over_Ex_561_Ch1_stitched', 'Accuracy', 'E')     # double+/tdTomato+
    metric_plot('overlap_over_Ex_639_Ch2_stitched', 'Efficiency', 'F')   # double+/c-Fos+
    metric_plot('F1_Score', 'TRAP fidelity score', 'G')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure 7I - fidelity by Figure-5 cluster""")
    return


@app.cell
def _(Condition_color, Conditions, FIG_OUT, REF, embedding_df, figure_key, plt, sns, total_df):
    from statannotations.Annotator import Annotator as _Annotator
    _df = total_df.merge(embedding_df[['label']], left_on='acronym', right_index=True)
    _regs = embedding_df.index.values
    _t = _df[(_df.acronym.isin(_regs)) & (_df.Ex_561_Ch1_stitched_newcounts > 0) & (_df.Ex_639_Ch2_stitched_newcounts > 0)]
    _t = _t[['ID', 'Condition', 'label', 'F1_Score']].groupby(['ID', 'Condition', 'label']).mean().reset_index()
    _labels = sorted(_t.label.unique())
    (_fig, _ax) = plt.subplots(1, 1, figsize=(2.5, 1))
    sns.stripplot(data=_t, hue='Condition', x='label', y='F1_Score', dodge=True, order=_labels,
                  hue_order=Conditions, ax=_ax, palette=Condition_color, alpha=0.25, size=2)
    sns.pointplot(data=_t, hue='Condition', x='label', y='F1_Score', order=_labels, dodge=0.8 - 0.8 / len(Conditions),
                  hue_order=Conditions, ax=_ax, palette=Condition_color, markers="o", markersize=4, linestyle="none", linewidth=0.5)
    sns.despine(); _ax.set_ylabel('TRAP fidelity score'); _ax.set_xlabel('Cluster')
    _ax.set_xticklabels([f"{int(r) + 1}" for r in _labels])
    _pairs = [((c, REF), (c, a)) for c in _labels for a in Conditions if a != REF]
    _an = _Annotator(_ax, _pairs, data=_t, hue='Condition', x='label', y='F1_Score', hue_order=Conditions, order=_labels)
    _an.configure(test='t-test_ind', text_format='star', loc='outside', comparisons_correction="BH", correction_format="replace")
    _an.apply_and_annotate()
    if _ax.get_legend() is not None:
        _ax.get_legend().remove()
    _fig.savefig(FIG_OUT / f'{figure_key}I.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}I.pdf', bbox_inches='tight', dpi=216)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure 7J - fidelity in selected regions + Table 8""")
    return


@app.cell
def _(Conditions, FIG_OUT, GROUP, REF, atlas_df, Condition_color, figure_key, np, pd, plt, sns, total_df):
    from scipy.stats import ttest_ind as _ttest_ind
    from statsmodels.stats.multitest import multipletests as _mt

    _sorted = ['A24a (IL)', 'Au1', 'IMD', 'RLi', 'PrEW', 'A24 (Cg)', 'Cl', 'A32 (PrL)', 'VTA', 'AcbC', 'AcbSh',
               'La', 'IGL', 'IPAC', 'PrG', 'Ce']
    _regs = [r for r in _sorted if r in set(total_df.acronym)]
    _clean = atlas_df.set_index('acronym').loc[_regs, 'cleaned_acronym'].values
    _t = total_df[total_df.acronym.isin(_regs)]

    (_fig, _ax) = plt.subplots(1, 1, figsize=(max(2, len(_regs) / 3.2), 1.2))
    sns.stripplot(data=_t, hue='Condition', x='acronym', y='F1_Score', dodge=True, order=_regs,
                  hue_order=Conditions, ax=_ax, palette=Condition_color, alpha=0.25, size=2)
    sns.pointplot(data=_t, hue='Condition', x='acronym', y='F1_Score', order=_regs, dodge=0.8 - 0.8 / len(Conditions),
                  hue_order=Conditions, ax=_ax, palette=Condition_color, markers="o", markersize=4, linestyle="none", linewidth=0.5)
    sns.despine(); _ax.set_ylabel('TRAP fidelity score'); _ax.set_xlabel(''); _ax.set_xticklabels(_clean, rotation=-45)
    if _ax.get_legend() is not None:
        _ax.get_legend().remove()
    _fig.savefig(FIG_OUT / f'{figure_key}J.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}J.pdf', bbox_inches='tight', dpi=216)

    # Table 8 - per-region fidelity t-test, Acute-TRAP vs each condition, BH per comparison
    _others = [c for c in Conditions if c != REF]
    _reg_all = total_df.acronym.unique()
    _tbl = pd.DataFrame({'acronym': _reg_all})
    _tbl['name'] = [atlas_df.loc[atlas_df.acronym == a, 'name'].values[0] if (atlas_df.acronym == a).any() else a for a in _reg_all]
    for _c in _others:
        _ps, _ts, _ds = [], [], []
        for _a in _reg_all:
            _x = total_df[(total_df.acronym == _a) & (total_df.Condition == REF)]['F1_Score'].values
            _y = total_df[(total_df.acronym == _a) & (total_df.Condition == _c)]['F1_Score'].values
            _s, _p = _ttest_ind(_x, _y)
            _ps.append(1.0 if np.isnan(_p) else _p); _ts.append(_s); _ds.append(len(_x) + len(_y) - 2)
        _q = _mt(np.nan_to_num(_ps, nan=1.0), method='fdr_bh')[1]
        _tbl[f'AcuteTRAP_vs_{_c}_corrected_p'] = _q
        _tbl[f'AcuteTRAP_vs_{_c}_t'] = _ts
        _tbl[f'AcuteTRAP_vs_{_c}_dof'] = _ds
    _tbl.to_excel(GROUP / 'Table8_TRAP_fidelity.xlsx', index=False)
    print("Table 8 saved:", _tbl.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure 7H - representative double+ beta maps (guarded)""")
    return


@app.cell
def _(Conditions, FIG_OUT, GROUP, atlas_img, contour_img, figure_key, np, overlap_contour, pd, plt, set_transparency):
    # 7H: MEAN double+ (overlap) density per condition, averaged over subjects from the overlap
    # heatmap zarr (not a regression). fnamelist.npy gives the zarr row order.
    import dask.array as _da
    _zarr = GROUP / "overlap_heatmap_array"
    _fn = GROUP / "fnamelist.npy"
    if _zarr.exists() and _fn.exists():
        _fnamelist = np.load(_fn, allow_pickle=True)
        _meta = pd.read_csv(GROUP / "OPTRAP_meta.csv").set_index('fname').loc[_fnamelist].reset_index()
        _arr = _da.from_zarr(str(_zarr))
        _zplanes = [84, 104, 117, 153, 186, 220]
        _imy, _imx = slice(25, 425), slice(50, 600)
        _vmax = 30   # double+ cell density; adjust to match the published scale
        for _cond in Conditions:
            _hm = np.nanmean(_arr[(_meta.Condition == _cond).values], axis=0).reshape(atlas_img.shape)
            _hm = np.asarray(_hm); _hm[np.isnan(_hm)] = 0
            (_fig, _axs) = plt.subplots(1, len(_zplanes), figsize=(3 * len(_zplanes), 3), sharey=True)
            _fig.subplots_adjust(wspace=0.25, hspace=0.3)
            for _i, _ax in enumerate(_axs):
                (__, _ov) = overlap_contour(_hm[_zplanes[_i], _imy, _imx], contour_img[_zplanes[_i], _imy, _imx],
                                            cmin=0, cmax=_vmax, colormap=plt.cm.Reds)
                _tr = set_transparency(_ov, (atlas_img == 0)[_zplanes[_i], _imy, _imx])
                _ax.imshow(_tr); _ax.axis('off'); _ax.set_ylabel(_cond, color='black')
            _fig.savefig(FIG_OUT / f'{figure_key}H_{_cond}.png', bbox_inches='tight', dpi=512)
            _fig.savefig(FIG_OUT / f'{figure_key}H_{_cond}.pdf', bbox_inches='tight', dpi=512)
    else:
        print("overlap_heatmap_array / fnamelist.npy missing -> 7H skipped (copy the overlap zarr into the deposit).")
    return


if __name__ == "__main__":
    app.run()
