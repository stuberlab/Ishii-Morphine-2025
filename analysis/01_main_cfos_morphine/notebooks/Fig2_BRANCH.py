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
    # Figure 2 — BRANCH hierarchical statistical testing

    Main whole-brain c-Fos dataset. Per-region negative-binomial GLM + likelihood-ratio
    test for the effect of drug condition, p-values corrected with **BRANCH** (Simes
    aggregation up the Unified-atlas ontology, then hierarchical sFDR via TreeBH).

    Pipeline: GLM (here) → write `Figure2_C_glm_stat_df.csv` → **R TreeBH**
    (`Fig2_TreeBH.R`) → `TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv` → read back here for the
    sunburst (2C–G), post-hoc Table 3, discovery heatmap, and the condition correlation
    matrix (2H). The sunburst also needs `Acute_Morphine_betas.npy` from the Figure 1
    regression. Panels 2I–J (scatter plots) are produced in the clustering notebook, not here.
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import tifffile
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns

    from brain_vis import overlap_contour, set_transparency, get_subregions
    return (
        Path,
        datetime,
        get_subregions,
        np,
        os,
        pd,
        pickle,
        plt,
        sns,
        tifffile,
    )


@app.cell
def _(Path, os):
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    GROUP = DATA_ROOT / "01_main_cfos_morphine"
    ATLAS = DATA_ROOT / "shared" / "atlas"
    FIG_OUT = Path("../figures")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    figure_key = "Figure2"
    assert GROUP.exists(), f"DATA_ROOT not found: {DATA_ROOT}. Set OPIOID_DATA_ROOT."
    return ATLAS, FIG_OUT, GROUP, figure_key


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
    mo.md(r"""
    ## Load data (same dataset as Figure 1)
    """)
    return


@app.cell
def _(ATLAS, GROUP, datetime, get_subregions, pd, pickle, tifffile):
    metadf = pd.read_csv(GROUP / "OP_meta.csv", index_col=False)
    metadf = metadf[metadf.ID != 'A7'].reset_index(drop=True)

    atlas_df = pd.read_csv(
        ATLAS / "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv",
        index_col=False)
    contour_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v2.9_contour_map.tif")
    atlas_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v4.0.tif")
    with open(ATLAS / "curated_acronym.pickle", "rb") as _h:
        curated_acronyms = pickle.load(_h)
    with open(ATLAS / "ancestor_curated_acronym.pickle", "rb") as _h:
        ancestor_curated_acronyms = pickle.load(_h)

    Conditions = ['Saline', 'Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine', 'Chronic_Morphine_21', 'Withdrawal_Morphine_21']
    Condition_figure_name = ['Saline', 'Acute', 'Chronic', 'Early WD', 'Re-exposure', 'Late WD']
    metadf = metadf[metadf.Condition.isin(Conditions)]
    metadf['age'] = [(datetime.strptime(_p, '%m/%d/%Y') - datetime.strptime(_d, '%m/%d/%Y')).days
                     for _p, _d in metadf.loc[:, ['Date_Perfusion', 'DOB']].values]

    pivot_heatmap_df = pd.read_csv(GROUP / "OP_cFos_heatmap.csv", index_col=0)
    pivot_heatmap_df = pivot_heatmap_df[metadf['ID'].values]

    merge_df = pd.read_csv(GROUP / "OP_cFos_full_result.csv", index_col=0)
    merge_df = merge_df[merge_df.Condition.isin(Conditions)]
    merge_df = merge_df[merge_df.ID.isin(metadf.ID.values)]

    # drop HB + CBL subtrees
    _rm_ids = atlas_df[(atlas_df.acronym == 'HB') | (atlas_df.acronym == 'CBL')]['id'].values
    _rm = pd.concat([get_subregions(atlas_df, _i, return_original=True) for _i in _rm_ids], axis=0)
    _sub_atlas = atlas_df.set_index(['id']).drop(_rm['id'].values)
    merge_df = merge_df[merge_df.acronym.isin(_sub_atlas.acronym.unique())]

    # sex / condition dummies + constant
    merge_df['Sex'] = merge_df['Sex'].astype(pd.CategoricalDtype(categories=['F', 'M'], ordered=False))
    merge_df['Condition'] = merge_df['Condition'].astype(pd.CategoricalDtype(categories=Conditions, ordered=True))
    _cond_d = pd.get_dummies(merge_df['Condition'])
    _sex_d = pd.get_dummies(merge_df['Sex']).loc[:, ['F']].rename(columns={'F': 'Sex_d'})
    merge_df_d = pd.concat([merge_df, _cond_d, _sex_d], axis=1)
    merge_df_d['constant'] = 1

    atlasmeta = merge_df_d.reset_index().loc[
        merge_df_d.reset_index().ID == 'A1', ['id', 'parent_id', 'acronym', 'name', 'parent_acronym']]

    sub_conditions = list(Conditions)
    sub_pivot_df = merge_df_d.pivot(columns='ID', index='acronym', values='density')[metadf.ID.values]
    return (
        Condition_figure_name,
        Conditions,
        ancestor_curated_acronyms,
        atlas_df,
        atlas_img,
        atlasmeta,
        contour_img,
        curated_acronyms,
        merge_df_d,
        metadf,
        pivot_heatmap_df,
        sub_conditions,
        sub_pivot_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2B — region GLM + likelihood-ratio test

    Per region: negative-binomial GLM (5 drug conditions + sex + BW + age + constant) vs. a
    reduced model with the **drug-condition** terms removed; chi-square LRT p-value. Written
    out for the BRANCH (R TreeBH) step.
    """)
    return


@app.cell
def _(Conditions, GROUP, atlasmeta, figure_key, get_subregions, merge_df_d, np, pd):
    import statsmodels.api as _sm
    from sklearn.preprocessing import MinMaxScaler as _MinMaxScaler
    from scipy.stats import chi2 as _chi2

    def _lrt(_full, _red):
        _ll_diff = -2 * (_red.llf - _full.llf)
        _ddf = _full.df_model - _red.df_model
        return _chi2.sf(_ll_diff, _ddf)

    _metacols = list(Conditions[1:]) + ['constant', 'Sex_d', 'BW', 'Age']
    _cont = ['BW', 'Age']
    _family = _sm.families.NegativeBinomial()
    _rows = []
    for _acr in merge_df_d.acronym.unique():
        _exog = merge_df_d.loc[merge_df_d.acronym == _acr, _metacols].copy()
        _exog[_cont] = _MinMaxScaler().fit_transform(_exog[_cont])
        _endog = merge_df_d.loc[merge_df_d.acronym == _acr, 'density']
        try:
            _full = _sm.GLM(_endog, np.asarray(_exog.astype('float64')), family=_family).fit()
            _red = _sm.GLM(_endog, _exog.astype('float64').drop(columns=Conditions[1:]), family=_family).fit()
            _pv = _lrt(_full, _red)
        except Exception:
            _pv = np.nan
        _rows.append({'acronym': _acr, 'pvalue': _pv})

    glm_stat_df = pd.DataFrame(_rows)
    glm_stat_df = pd.merge(get_subregions(atlasmeta, 8, return_original=True), glm_stat_df, on='acronym').rename(columns={'name': 'Name'})
    _out = GROUP / f'{figure_key}_C_glm_stat_df.csv'
    glm_stat_df.to_csv(_out, index=False)
    print("Wrote GLM p-values for BRANCH ->", _out.name)
    return (glm_stat_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### BRANCH (TreeBH) step

    Run `Fig2_TreeBH.R` on the GLM CSV above to produce
    `TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv` (Simes aggregation = BRANCH). That file is in
    the deposit, so the cells below run directly; re-run R only if the GLM changes.
    """)
    return


@app.cell
def _(GROUP, curated_acronyms, pd):
    _f = GROUP / "TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv"
    if _f.exists():
        TreeFDRS_df = pd.read_csv(_f, index_col=False)
        rejected_acronyms = TreeFDRS_df[(TreeFDRS_df.acronym.isin(curated_acronyms)) & (TreeFDRS_df.rejected == True)].acronym.values
        print(len(rejected_acronyms), "BRANCH-rejected regions")
    else:
        TreeFDRS_df = None
        rejected_acronyms = []
        print("BRANCH output not found:", _f.name, "-> run Fig2_TreeBH.R first.")
    return TreeFDRS_df, rejected_acronyms


@app.cell
def _(TreeFDRS_df, atlas_df, ancestor_curated_acronyms, get_subregions, np):
    # proportion of leaf nodes rejected per major subtree (diagnostic)
    if TreeFDRS_df is not None:
        for _anc in np.unique(ancestor_curated_acronyms):
            _aid = atlas_df[atlas_df.acronym == _anc].id.values[0]
            _sub = get_subregions(atlas_df, _aid, return_original=True)
            _leaves = np.setdiff1d(_sub['acronym'], _sub['parent_acronym'])
            _rej = TreeFDRS_df[(TreeFDRS_df.acronym.isin(_leaves)) & (TreeFDRS_df.rejected == True)].acronym.values
            print(f"  {_anc}: {len(_rej)} / {len(_leaves)} leaf nodes rejected")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table 3 — post-hoc Chronic vs. Early-WD in BRANCH-rejected regions
    """)
    return


@app.cell
def _(GROUP, atlas_df, merge_df_d, np, pd, rejected_acronyms):
    if len(rejected_acronyms) > 0:
        import scipy.stats as _stats
        from statsmodels.stats.multitest import fdrcorrection as _fdrcorrection
        _res, _pvals = [], []
        for _acr in rejected_acronyms:
            _t = merge_df_d[merge_df_d.acronym == _acr]
            _name = atlas_df[atlas_df.acronym == _acr].name.values[0]
            _c = _t[_t.Condition == 'Chronic_Morphine'].normalized_density.values
            _w = _t[_t.Condition == 'Withdrawal_Morphine'].normalized_density.values
            _tv, _pv = _stats.ttest_ind(_c, _w)
            _res.append({'acronym': _acr, 'name': _name, 'pvalue': _pv, 'tvalue': _tv, 'df': len(_c) + len(_w) - 2})
            _pvals.append(_pv)
        _rej, _pc = _fdrcorrection(_pvals, alpha=0.05)
        for _i, _r in enumerate(_res):
            _r['corrected'] = _pc[_i]
        results_df = pd.DataFrame(_res)
        results_df.to_excel(GROUP / 'Table 3.xlsx', index=False)
        print("Table 3 saved:", len(results_df), "regions")
    else:
        results_df = None
        print("Table 3 skipped — no BRANCH-rejected regions.")
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2C–G — sunburst + region maps (interactive)

    Sunburst of BRANCH results across the ontology, linked to the acute-morphine β map.
    `run_app` builds the Dash app (purple = rejected, grey = not, white = not tested); launch
    with `fig2_sunburst_app.run()` and click a region to view its coronal maps (D–G).
    Needs `Acute_Morphine_betas.npy` from the Figure 1 regression.
    """)
    return


@app.cell
def _(FIG_OUT, GROUP, TreeFDRS_df, atlas_df, atlas_img, figure_key, np, pd, plt):
    _betas_f = GROUP / "Acute_Morphine_betas.npy"
    if TreeFDRS_df is not None and _betas_f.exists():
        from brain_vis import sunburst_app as _sunburst_app
        _betacoef = np.load(_betas_f)
        _df = pd.merge(TreeFDRS_df[['acronym', 'rejected', 'p.val']], atlas_df, on='acronym', how='inner')
        fig2_sunburst_app = _sunburst_app.run_app(
            _df, _betacoef, atlas_img, str(FIG_OUT / f'{figure_key}_sunburst'),
            data_variable='rejected', colormap=plt.cm.coolwarm, data_cmin=-20, data_cmax=20)
        print("Sunburst built. Launch with: fig2_sunburst_app.run()")
    else:
        fig2_sunburst_app = None
        print("Sunburst skipped — needs TreeFDRS output and Acute_Morphine_betas.npy (run Fig1 first).")
    return (fig2_sunburst_app,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Discovery + per-condition density heatmap
    """)
    return


@app.cell
def _(
    Condition_figure_name,
    Conditions,
    FIG_OUT,
    TreeFDRS_df,
    ancestor_curated_acronyms,
    curated_acronyms,
    figure_key,
    metadf,
    np,
    sub_conditions,
    sub_pivot_df,
    plt,
    sns,
):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    if TreeFDRS_df is not None:
        _a = (TreeFDRS_df.loc[TreeFDRS_df.acronym.isin(curated_acronyms), ['acronym', 'rejected']]
              .astype('str').replace('nan', 0).replace('False', -1).replace('True', 1)
              .rename(columns={'rejected': 'TreeBH'}))
        _hm = _a.set_index('acronym')

        _cmap = ListedColormap(['white', 'gray', 'magenta'])
        _norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], _cmap.N)
        _vmin, _vmax = 0, 2000
        _nrow = len(sub_conditions) + 1
        _fig, _axs = plt.subplots(_nrow, 1, figsize=(5, len(sub_conditions) * 0.4 + 1), sharex=True,
                                  gridspec_kw={"height_ratios": [0.25] + [1] * len(sub_conditions)})
        _theatmapdf = _hm.loc[curated_acronyms, ['TreeBH']]
        _anc = [ancestor_curated_acronyms[i] for i, f in enumerate(curated_acronyms) if f in _theatmapdf.index]
        _u, _ix = np.unique(_anc, return_index=True)
        _labels = [_anc[i] for i in sorted(_ix)]
        _counts = [int(np.sum(np.array(_anc) == l)) for l in _labels]
        _borders = [0] + list(np.cumsum(_counts))
        _x = np.array(_borders)
        _ticks = (_x[1:] + _x[:-1]) / 2

        sns.heatmap(_theatmapdf.T, cbar=False, ax=_axs[0], cmap=_cmap, norm=_norm)
        _axs[0].set_xticks(_ticks); _axs[0].set_xticklabels(_labels, rotation=0); _axs[0].set_xlabel('')
        [_axs[0].axvline(b, color='white', lw=1, ls=':') for b in _borders[1:-1]]
        _axs[0].set_yticks([0.5]); _axs[0].set_yticklabels(['Discoveries'], rotation=0)

        for _idx, _cond in enumerate(sub_conditions):
            _subj = metadf[metadf.Condition == _cond].ID.values
            if _idx == len(sub_conditions) - 1:
                _cax = _fig.add_axes([_axs[_idx + 1].get_position().x1 + 0.01, _axs[_idx + 1].get_position().y0 + 0.2, 0.02, 0.5])
                sns.heatmap(sub_pivot_df.loc[curated_acronyms, _subj].T, cbar_ax=_cax, ax=_axs[_idx + 1], vmin=_vmin, vmax=_vmax)
                _cax.set_ylabel('c-Fos+ cell density\n(cells/mm3)', rotation=270, labelpad=10, fontsize=10)
            else:
                sns.heatmap(sub_pivot_df.loc[curated_acronyms, _subj].T, cbar=False, ax=_axs[_idx + 1], vmin=_vmin, vmax=_vmax)
            _axs[_idx + 1].set_xticks(_ticks); _axs[_idx + 1].set_xticklabels(_labels, rotation=-45); _axs[_idx + 1].set_xlabel('')
            [_axs[_idx + 1].axvline(b, color='yellow', lw=1.5, ls=':') for b in _borders[1:-1]]
            _axs[_idx + 1].set_yticks([len(_subj) // 2 + 0.5])
            _axs[_idx + 1].set_yticklabels([Condition_figure_name[Conditions.index(_cond)]], rotation=0)
            _axs[_idx + 1].set_ylabel('')
        _fig.savefig(FIG_OUT / f'{figure_key}_discovery_heatmap.png', bbox_inches='tight', dpi=216)
        _fig.savefig(FIG_OUT / f'{figure_key}_discovery_heatmap.pdf', bbox_inches='tight', dpi=216)
    else:
        print("Discovery heatmap skipped — BRANCH output not found.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 2H — correlation matrix of mean c-Fos density across conditions

    Pairwise Pearson correlation between condition-mean density vectors over the curated
    (~233) regions (saline excluded).
    """)
    return


@app.cell
def _(Condition_figure_name, Conditions, FIG_OUT, curated_acronyms, figure_key, metadf, np, pivot_heatmap_df, plt, sns):
    _meta = metadf[metadf.Condition != 'Saline']
    _data = pivot_heatmap_df.loc[curated_acronyms, _meta.ID].values        # regions x subjects
    _labels = _meta.set_index('ID').loc[pivot_heatmap_df.loc[curated_acronyms, _meta.ID].columns, 'Condition'].values
    _conds = [c for c in Conditions if c in np.unique(_labels)]
    _means = {c: np.mean(_data[:, np.where(_labels == c)[0]], axis=1) for c in _conds}
    _n = len(_conds)
    _corr = np.zeros((_n, _n))
    for _i in range(_n):
        for _j in range(_n):
            _corr[_i, _j] = np.corrcoef(_means[_conds[_i]], _means[_conds[_j]])[0, 1]

    _fig, _ax = plt.subplots(1, 1, figsize=(5, 4))
    _names = [Condition_figure_name[Conditions.index(c)] for c in _conds]
    sns.heatmap(_corr, annot=True, xticklabels=_names, yticklabels=_names, cmap="coolwarm", vmin=0, vmax=1, ax=_ax)
    _ax.set_xlabel("Condition"); _ax.set_ylabel("Condition")
    _ax.set_xticklabels(_names, rotation=-45); _ax.set_yticklabels(_names, rotation=0)
    _fig.savefig(FIG_OUT / f'{figure_key}H_correlation_matrix.png', dpi=216, bbox_inches='tight')
    _fig.savefig(FIG_OUT / f'{figure_key}H_correlation_matrix.pdf', dpi=216, bbox_inches='tight')
    return


if __name__ == "__main__":
    app.run()
