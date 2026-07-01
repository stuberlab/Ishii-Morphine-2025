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
    # Figure S4 — Sex differences in brain-wide c-Fos activation (related to Figure 2)

    Same main whole-brain c-Fos dataset as Figures 1/2. Three parts:

    - **S4A** — voxel-wise regression weight for the *sex* covariate (Sex_d beta map).
      Uses `Sex_d_betas.npy`, an intermediate produced by the Figure 1 regression
      (`Fig1_S2_whole_brain_cFos`) and stored in the Figshare deposit.
    - **Region GLM** — per-region negative-binomial GLM with a likelihood-ratio test of
      the full model vs. a reduced model with *sex* removed. Writes per-region p-values
      for the BRANCH (TreeBH) step.
    - **BRANCH-dependent (S4C sunburst, S4D swarmplot)** — read the TreeBH output
      (`TreeFDRS_pvalue_Figure1_Sex_glm.csv`). These cells are **guarded**: they run only
      once that R-generated file is present in the data folder.
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

    # brainvis package (install from GitHub; see requirements.txt)
    from brain_vis import overlap_contour, set_transparency, get_subregions
    return (
        Path,
        datetime,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paths

    `DATA_ROOT` points at the downloaded Figshare deposit (set `OPIOID_DATA_ROOT` or edit below).
    """)
    return


@app.cell
def _(Path, os):
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    GROUP = DATA_ROOT / "01_main_cfos_morphine"
    ATLAS = DATA_ROOT / "shared" / "atlas"
    FIG_OUT = Path("../figures")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    figure_key = "FigureS4"
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
    ## Load data (same dataset as Figure 1/2)

    Region table with sex/condition dummies and a constant, HB/CBL subtrees removed.
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

    Conditions = ['Saline', 'Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine', 'Chronic_Morphine_21', 'Withdrawal_Morphine_21']
    metadf = metadf[metadf.Condition.isin(Conditions)]
    metadf['age'] = [(datetime.strptime(_p, '%m/%d/%Y') - datetime.strptime(_d, '%m/%d/%Y')).days
                     for _p, _d in metadf.loc[:, ['Date_Perfusion', 'DOB']].values]

    merge_df = pd.read_csv(GROUP / "OP_cFos_full_result.csv", index_col=0)
    merge_df = merge_df[merge_df.Condition.isin(Conditions)]
    merge_df = merge_df[merge_df.ID.isin(metadf.ID.values)]

    # drop HB + CBL subtrees (poor registration, out of scope)
    _rm_ids = atlas_df[(atlas_df.acronym == 'HB') | (atlas_df.acronym == 'CBL')]['id'].values
    _rm = pd.concat([get_subregions(atlas_df, _i, return_original=True) for _i in _rm_ids], axis=0)
    _sub_atlas = atlas_df.set_index(['id']).drop(_rm['id'].values)
    merge_df = merge_df[merge_df.acronym.isin(_sub_atlas.acronym.unique())]

    # sex / condition dummies + constant
    merge_df['Sex'] = merge_df['Sex'].astype(pd.CategoricalDtype(categories=['F', 'M'], ordered=False))
    merge_df['Condition'] = merge_df['Condition'].astype(pd.CategoricalDtype(categories=Conditions, ordered=True))
    _cond_d = pd.get_dummies(merge_df['Condition'])
    _sex_d = pd.get_dummies(merge_df['Sex']).loc[:, ['F']].rename(columns={'F': 'Sex_d'})  # female = 1
    merge_df_d = pd.concat([merge_df, _cond_d, _sex_d], axis=1)
    merge_df_d['constant'] = 1

    atlasmeta = merge_df_d.reset_index().loc[
        merge_df_d.reset_index().ID == 'A1', ['id', 'parent_id', 'acronym', 'name', 'parent_acronym']]

    print("Animals per group:")
    print(metadf.groupby('Condition').size())
    return Conditions, atlas_df, atlas_img, atlasmeta, contour_img, curated_acronyms, merge_df_d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## S4A — Sex-covariate beta map

    Voxel-wise regression weight for the sex covariate (`Sex_d_betas.npy` from the Figure 1
    regression). Positive = higher modeled c-Fos+ in females.
    """)
    return


@app.cell
def _(FIG_OUT, GROUP, atlas_img, contour_img, figure_key, np, overlap_contour, plt, set_transparency):
    curated_zplanes = [84, 104, 117, 153, 186, 220]
    _imy = slice(25, 425)
    _imx = slice(50, 600)
    _theatmap = np.load(GROUP / "Sex_d_betas.npy")   # produced by Fig1_S2_whole_brain_cFos regression
    (__, _ov) = overlap_contour(_theatmap, contour_img, cmin=-15, cmax=15, outputpath=None, overlap_black=True)
    (_fig, _axs) = plt.subplots(1, len(curated_zplanes), figsize=(3 * len(curated_zplanes), 3), sharey=True)
    _fig.subplots_adjust(wspace=0.25, hspace=0.3)
    for (_i, _ax) in enumerate(_axs):
        _tr = set_transparency(_ov[curated_zplanes[_i], :, :], (atlas_img == 0)[curated_zplanes[_i], :, :])
        _ax.imshow(_tr[_imy, _imx])
        _ax.axis('off')
        _ax.set_ylabel('Sex', color='black')
    _fig.savefig(FIG_OUT / f'{figure_key}A_Sex_d_betacoef.png', bbox_inches='tight', dpi=1024)
    _fig.savefig(FIG_OUT / f'{figure_key}A_Sex_d_betacoef.pdf', bbox_inches='tight', dpi=1024)
    return (curated_zplanes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Region GLM + likelihood-ratio test (input to BRANCH)

    Per region: negative-binomial GLM (drug conditions + sex + BW + age + constant) vs. a
    reduced model with *sex* removed; chi-square LRT p-value. Saved for the R TreeBH step.
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
            _red = _sm.GLM(_endog, _exog.astype('float64').drop(columns=['Sex_d']), family=_family).fit()
            _pv = _lrt(_full, _red)
        except Exception:
            _pv = np.nan
        _rows.append({'acronym': _acr, 'pvalue': _pv})

    glm_stat_df = pd.DataFrame(_rows)
    glm_stat_df = pd.merge(get_subregions(atlasmeta, 8, return_original=True), glm_stat_df, on='acronym').rename(columns={'name': 'Name'})
    _outpath = GROUP / f'{figure_key}_Sex_glm_stat_df.csv'
    glm_stat_df.to_csv(_outpath, index=False)
    print("Wrote per-region sex GLM p-values for BRANCH ->", _outpath.name)
    return (glm_stat_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### BRANCH (TreeBH) step — run externally

    Run the TreeBH R script on the GLM p-values above to produce
    `TreeFDRS_pvalue_Figure1_Sex_glm.csv` in the data folder. The cells below read that
    output; until it exists they print a notice and skip.
    """)
    return


@app.cell
def _(GROUP, curated_acronyms, pd):
    # Guarded read of the BRANCH (TreeBH) output.
    _f = GROUP / "TreeFDRS_pvalue_Figure1_Sex_glm.csv"
    if _f.exists():
        TreeFDRS_df = pd.read_csv(_f, index_col=False)
        rejected_acronyms = TreeFDRS_df[(TreeFDRS_df.acronym.isin(curated_acronyms)) & (TreeFDRS_df.rejected == True)].acronym.values
        print(len(rejected_acronyms), "BRANCH-rejected sex regions:", list(rejected_acronyms))
    else:
        TreeFDRS_df = None
        rejected_acronyms = []
        print("BRANCH output not found:", _f.name, "-> run the R TreeBH step (S4C/S4D skipped).")
    return TreeFDRS_df, rejected_acronyms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## S4D — c-Fos+ cell density by sex (BRANCH-rejected + curated regions)
    """)
    return


@app.cell
def _(FIG_OUT, curated_acronyms, figure_key, merge_df_d, np, plt, rejected_acronyms, sns):
    if len(rejected_acronyms) > 0:
        _curated_texts = (
            ["VTA", "VTg", "PT", "MD", "PVT", "AcbC", "AcbSh", "LS", "O", "ASt", "La",
             "CPce", "CPc", "Ce", "ST", "BL", "MBO", "DP", "Pl", "DS", "CPre", "STr",
             "PrL", "DI", "AI", "IL", "Cg", "SPTg", "Cl"]
            + ["A24a (IL)", "Au1", "RLi", "PrEW", "A24 (Cg)", "Cl", "A32 (PrL)",
               "VTA", "AcbC", "AcbSh", "La", "IPAC", "PrG", "Ce", "ST"]
        )
        _curated_set = list(np.unique(np.char.strip(np.array(_curated_texts, dtype=str))))
        _order = [f for f in list(rejected_acronyms) + _curated_set if f in curated_acronyms]
        (_fig, _ax) = plt.subplots(1, 1, figsize=(10, 2))
        sns.swarmplot(
            data=merge_df_d[merge_df_d.acronym.isin(list(rejected_acronyms) + _curated_set)],
            x="acronym", hue="Sex", y="density", order=_order, size=1, dodge=True, ax=_ax)
        sns.despine()
        _fig.savefig(FIG_OUT / f'{figure_key}D_Sex_d_swarmplot.png', bbox_inches='tight', dpi=216)
        _fig.savefig(FIG_OUT / f'{figure_key}D_Sex_d_swarmplot.pdf', bbox_inches='tight', dpi=216)
    else:
        print("Swarmplot skipped — no BRANCH-rejected regions (run R TreeBH first).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## S4C — Sunburst of sex statistical testing across the ontology

    Interactive Dash app (purple = rejected, grey = not rejected, white = not tested).
    `run_app` builds the app; launch it with `sex_sunburst_app.run()` to explore/export.
    """)
    return


@app.cell
def _(FIG_OUT, GROUP, TreeFDRS_df, atlas_df, atlas_img, figure_key, np, pd, plt):
    if TreeFDRS_df is not None:
        from brain_vis import sunburst_app as _sunburst_app   # imports dash/plotly lazily
        _theatmap = np.load(GROUP / "Sex_d_betas.npy")
        _df = pd.merge(TreeFDRS_df[['acronym', 'rejected', 'p.val']], atlas_df, on='acronym', how='inner')
        sex_sunburst_app = _sunburst_app.run_app(
            _df, _theatmap, atlas_img, str(FIG_OUT / f'{figure_key}_sunburst'),
            data_variable='rejected', colormap=plt.cm.coolwarm,
            data_cmin=-20, data_cmax=20, uniformtext_minsize=6, uniformtext_mode='hide')
        print("Sunburst Dash app built. Launch with: sex_sunburst_app.run()")
    else:
        sex_sunburst_app = None
        print("Sunburst skipped — BRANCH output not found.")
    return (sex_sunburst_app,)


if __name__ == "__main__":
    app.run()
