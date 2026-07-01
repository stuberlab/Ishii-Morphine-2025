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
    # Figures 1 & S2 — Whole-brain c-Fos response to morphine and withdrawal

    Generates main-text panels **1B, 1C, 1D** and supplements **S2** (density maps) and
    **S5** (lateralization) from the main whole-brain c-Fos dataset (43 animals, 5–6 drug conditions).

    **Data:** download the Figshare deposit and point `OPIOID_DATA_ROOT` at it (see the
    path cell below). All inputs resolve from there via the Figshare folder layout.
    """)
    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import tifffile
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    import dask.array as da

    # brainvis package (install from GitHub; see requirements.txt):
    #   pip install git+https://github.com/kenjp1223/brainvis
    from brain_vis import overlap_contour, set_transparency, get_subregions

    return (
        Path,
        da,
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

    `DATA_ROOT` points at the downloaded **Figshare deposit**, whose folder layout mirrors
    this repo's data-source groups. Set the `OPIOID_DATA_ROOT` environment variable, or edit
    the default below. Rendered panels are written to the repo (`../figures`).
    """)
    return


@app.cell
def _(Path, os):
    # >>> Set this to the downloaded Figshare deposit (or export OPIOID_DATA_ROOT) <<<
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()

    GROUP = DATA_ROOT / "01_main_cfos_morphine"   # this dataset (raw + processed)
    ATLAS = DATA_ROOT / "shared" / "atlas"        # shared atlas / ontology

    FIG_OUT = Path("../figures")                  # rendered panels -> repo
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    figure_key = "Figure1"
    assert GROUP.exists(), f"DATA_ROOT not found: {DATA_ROOT}. Set OPIOID_DATA_ROOT."
    return ATLAS, FIG_OUT, GROUP, figure_key


@app.cell
def _(plt):
    # Plot style: vector text editable in Illustrator, transparent backgrounds
    plt.rcParams.update({
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.facecolor': 'none',
        'legend.edgecolor': 'none',
        'text.color': 'black',
        'font.family': 'Arial',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load metadata, atlas, and ontology
    """)
    return


@app.cell
def _(ATLAS, GROUP, pd, pickle, tifffile):
    # Animal metadata (drop A7 = missing data)
    metadf = pd.read_csv(GROUP / "OP_meta.csv", index_col=False)
    metadf = metadf[metadf.ID != 'A7'].reset_index(drop=True)

    # Unified (Kim) mouse-brain atlas: ontology table, contour map, label volume
    atlas_df = pd.read_csv(
        ATLAS / "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv",
        index_col=False)
    contour_img = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v2.9_contour_map.tif")
    atlas_img   = tifffile.imread(ATLAS / "Kim_ref_adult_FP-label_v4.0.tif")

    # Curated region lists
    with open(ATLAS / "curated_acronym.pickle", "rb") as handle:
        curated_acronyms = pickle.load(handle)
    with open(ATLAS / "ancestor_curated_acronym.pickle", "rb") as handle:
        ancestor_curated_acronyms = pickle.load(handle)

    # files flagged as fully processed
    fnames = [f for f in metadf.fname.values if 'DONE' in f]
    return atlas_df, atlas_img, contour_img, fnames, metadf


@app.cell
def _(metadf):
    # Drug conditions (morphine-related groups) and display labels/colors
    Conditions = ['Saline', 'Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine', 'Chronic_Morphine_21', 'Withdrawal_Morphine_21']
    Condition_figure_name = ['Saline', 'Acute', 'Chronic', 'Early WD', 'Re-exposure', 'Late WD']
    Condition_color = ['gray', 'lime', 'orange', 'cyan', 'blue', 'purple']
    metadf_1 = metadf[metadf.Condition.isin(Conditions)]
    return Condition_color, Condition_figure_name, Conditions, metadf_1


@app.cell
def _(Conditions, GROUP, fnames, metadf_1, pd):
    # Region-level results (long) and per-animal region-density heatmap (wide)
    pivot_heatmap_df = pd.read_csv(GROUP / 'OP_cFos_heatmap.csv', index_col=0)
    pivot_heatmap_df = pivot_heatmap_df[metadf_1[metadf_1.Condition.isin(Conditions)]['ID'].values]
    merge_df = pd.read_csv(GROUP / 'OP_cFos_full_result.csv', index_col=0)
    merge_df = merge_df[merge_df.Condition.isin(Conditions)]
    merge_df = merge_df[merge_df.fname.isin(fnames)]
    assert (merge_df.fname.unique() == fnames).all()
    assert (pivot_heatmap_df.columns == metadf_1.ID.unique()).all()
    # sanity checks
    print('Animals per group:')
    print(metadf_1.groupby('Condition').size())
    return (merge_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remove the hindbrain (HB) and cerebellum (CBL) subtrees: poor registration quality and
    outside the scope of this analysis.
    """)
    return


@app.cell
def _(atlas_df, get_subregions, merge_df, pd):
    unique_ancestor_curated_acronyms = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL', 'TH', 'HY', 'MB']
    ancestor_names = [atlas_df.loc[atlas_df.acronym == f, 'name'].values[0] for f in unique_ancestor_curated_acronyms]
    ancestor_idxs = [atlas_df.loc[atlas_df.acronym == f, 'id'].values[0] for f in unique_ancestor_curated_acronyms]
    remove_ancestor_ids = atlas_df[(atlas_df.acronym == 'HB') | (atlas_df.acronym == 'CBL')]['id'].values
    remove_df = pd.concat([get_subregions(atlas_df, _idx, return_original=True) for _idx in remove_ancestor_ids], axis=0)
    sub_atlas_df = atlas_df.set_index(['id']).drop(remove_df['id'].values)
    merge_df_1 = merge_df[merge_df.acronym.isin(sub_atlas_df.acronym.unique())]
    return (merge_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the design matrix for the GLM

    Experiment variables = 5 drug conditions (saline as baseline); covariates = body weight,
    age, sex. Staining batch could not be included because some conditions coincide with batch.
    """)
    return


@app.cell
def _(datetime, merge_df_1, metadf_1):
    # age in days; per-region atlas metadata
    metadf_1['age'] = [(datetime.strptime(pday, '%m/%d/%Y') - datetime.strptime(dob, '%m/%d/%Y')).days for (pday, dob) in metadf_1.loc[:, ['Date_Perfusion', 'DOB']].values]
    atlasmeta = merge_df_1.reset_index().loc[merge_df_1.reset_index().ID == 'A1', ['id', 'parent_id', 'acronym', 'name', 'parent_acronym']]
    return


@app.cell
def _(Conditions, merge_df_1, pd):
    # categorical encodings -> dummy columns
    sex_category = pd.CategoricalDtype(categories=['F', 'M'], ordered=False)
    condition_category = pd.CategoricalDtype(categories=Conditions, ordered=True)
    merge_df_1['Sex'] = merge_df_1['Sex'].astype(sex_category)
    merge_df_1['Condition'] = merge_df_1['Condition'].astype(condition_category)
    condition_dummies = pd.get_dummies(merge_df_1['Condition'])
    sex_dummies = pd.get_dummies(merge_df_1['Sex']).loc[:, ['F']].rename(columns={'F': 'Sex_d'})
    merge_df_2 = pd.concat([merge_df_1, condition_dummies, sex_dummies], axis=1)  # female = 1
    return (merge_df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1B — Whole-brain c-Fos+ cell counts per condition
    """)
    return


@app.cell
def _(
    Condition_color,
    Condition_figure_name,
    Conditions,
    FIG_OUT,
    figure_key,
    merge_df_2,
    plt,
    sns,
):
    pannel_key = 'B'
    (_fig, _axs) = plt.subplots(1, 1, figsize=(1, 2))
    merge_df_2.Condition = merge_df_2.Condition.astype('str')
    tdata = merge_df_2.loc[merge_df_2.parent_acronym == 'grey', ['ID', 'Condition', 'newcounts']].groupby(['ID', 'Condition']).sum().reset_index().dropna()
    sns.stripplot(data=tdata, y='Condition', x='newcounts', order=Conditions, ax=_axs, palette=Condition_color, alpha=0.25)
    sns.pointplot(data=tdata, y='Condition', x='newcounts', order=Conditions, ax=_axs, palette=Condition_color, markers='o', markersize=4, linestyle='none', linewidth=0.5)
    sns.despine()
    _axs.set_xlabel('# of whole brain\nc-Fos+ cells')
    _axs.set_yticklabels(Condition_figure_name, rotation=0)
    _axs.set_xlim(0)
    _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key}.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key}.pdf', bbox_inches='tight', dpi=216)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear regression (predictive activation maps)

    Voxel-wise OLS of the c-Fos+ cell-count map on experiment + non-experiment variables.
    The condition beta coefficients (relative to saline) are the condition-specific
    **predictive activation scores** visualized in 1C–1D. The spatial heatmap is stored as a
    zarr array; row order follows `OP_cFos_fnamelist.npy`.
    """)
    return


@app.cell
def _(Conditions, GROUP, merge_df_2, metadf_1, np):
    # variables, one row per animal (region 'CH' = whole grey-matter root)
    variable_df = merge_df_2.loc[merge_df_2.acronym == 'CH', ['fname', 'Condition', 'BW', 'Age', 'Sex_d'] + Conditions]
    variable_df = variable_df.set_index('fname')
    fnamelist = np.load(GROUP / 'OP_cFos_fnamelist.npy')
    # order rows to match the spatial heatmap
    variable_df = variable_df.loc[fnamelist, :]
    metadf_2 = metadf_1.set_index('fname').loc[fnamelist, :]
    return metadf_2, variable_df


@app.cell
def _(atlas_df, atlas_img, np):
    # voxels that belong to included brain regions (exclude background / removed subtrees)
    brain_voxels = np.where(np.isin(atlas_img.flatten(), atlas_df['id'].values))[0]
    return (brain_voxels,)


@app.cell
def _(metadf_2, np, variable_df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=True)
    # standardize continuous covariates
    variable_df[['BW', 'Age']] = scaler.fit_transform(variable_df[['BW', 'Age']])
    variable_df_1 = variable_df.drop(columns='Saline')
    variable_df_1['constant'] = 1
    # saline = baseline -> drop; add intercept; order rows to metadf
    variable_df_1 = variable_df_1.loc[metadf_2.index]
    variable_df_1 = variable_df_1.drop(columns='Condition')
    variables = np.array(variable_df_1.astype('float'))
    return variable_df_1, variables


@app.cell
def _(GROUP, brain_voxels, da):
    # load the spatial c-Fos heatmap (zarr) and keep only brain voxels
    heatmap_da = da.from_zarr(str(GROUP / "OP_cFos_heatmap_array"), mode="r")
    brain_heatmap = heatmap_da[:, brain_voxels].compute()
    return brain_heatmap, heatmap_da


@app.cell
def _(brain_heatmap, variables):
    import statsmodels.api as sm

    # voxel-wise OLS: (n_animals x n_voxels) on (n_animals x n_variables)
    models = sm.OLS(brain_heatmap, variables).fit()
    return (models,)


@app.cell
def _(GROUP, atlas_img, brain_voxels, models, np, variable_df_1):
    for (variable_idx, _variable_name) in enumerate(variable_df_1.columns):
        temp_img = np.zeros_like(atlas_img.flatten(), dtype='float64')
        temp_img[brain_voxels] = models.params[variable_idx, :]
        np.save(GROUP / f'{_variable_name}_betas.npy', np.reshape(temp_img, atlas_img.shape))
    print('Saved beta maps for:', list(variable_df_1.columns))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1C — Predictive activation maps, single coronal plane

    Beta-coefficient maps (cmin/cmax = ±15) overlaid on the atlas contour, one panel per
    variable at a representative z-plane.
    """)
    return


@app.cell
def _(
    FIG_OUT,
    GROUP,
    atlas_img,
    contour_img,
    figure_key,
    np,
    overlap_contour,
    plt,
    set_transparency,
    variable_df_1,
):
    pannel_key_1 = 'C'
    imy_slice = slice(50, 400)
    imx_slice = slice(75, 575)
    zplace = 125
    (_fig, _axs) = plt.subplots(2, int(np.ceil(len(variable_df_1.columns) / 2)), figsize=(20, np.ceil(len(variable_df_1.columns) / 2)))
    for (cidx, _variable_name) in enumerate(variable_df_1.columns):
        _ax = _axs.flatten()[cidx]
        beta_array = np.load(GROUP / f'{_variable_name}_betas.npy')
        (base_image_color, _overlayed_image) = overlap_contour(beta_array, contour_img, cmin=-15, cmax=15, outputpath=False)
        _trans_img = set_transparency(_overlayed_image[zplace, :, :], (atlas_img == 0)[zplace, :, :])
        _ax.imshow(_trans_img[imy_slice, imx_slice])
        _ax.axis('off')
        _ax.set_title(f'{_variable_name}_z={zplace}')
    _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key_1}.png', dpi=216, bbox_inches='tight')
    _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key_1}.pdf', dpi=216, bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 1D — Predictive activation maps across coronal planes

    Per drug condition (relative to saline), beta maps along the A–P axis at curated
    z-planes. Saline (baseline, no beta) is shown as the mean raw density for reference.
    """)
    return


@app.cell
def _(
    Conditions,
    FIG_OUT,
    GROUP,
    atlas_img,
    contour_img,
    figure_key,
    np,
    overlap_contour,
    plt,
    set_transparency,
):
    pannel_key_2 = 'D'
    curated_zplanes = [84, 104, 117, 153, 186, 220]
    imy_slice_1 = slice(25, 425)
    imx_slice_1 = slice(50, 600)
    for _condition in Conditions[1:] + ['constant']:
        _theatmap = np.load(GROUP / f'{_condition}_betas.npy')
        (__, _overlayed_image) = overlap_contour(_theatmap, contour_img, cmin=-15, cmax=15, outputpath=None)
        (_fig, _axs) = plt.subplots(1, len(curated_zplanes), figsize=(3 * len(curated_zplanes), 3), sharey=True)
        _fig.subplots_adjust(wspace=0.25, hspace=0.3)
        for (_idx, _ax) in enumerate(_axs):
            _trans_img = set_transparency(_overlayed_image[curated_zplanes[_idx], :, :], (atlas_img == 0)[curated_zplanes[_idx], :, :])
            _ax.imshow(_trans_img[imy_slice_1, imx_slice_1])
            _ax.axis('off')
            _ax.set_ylabel(_condition, color='black')
        _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key_2}_{_condition}_betacoef.png', bbox_inches='tight', dpi=1024)
        _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key_2}_{_condition}_betacoef.pdf', bbox_inches='tight', dpi=1024)
    return curated_zplanes, imx_slice_1, imy_slice_1, pannel_key_2


@app.cell
def _(
    FIG_OUT,
    atlas_img,
    contour_img,
    curated_zplanes,
    figure_key,
    heatmap_da,
    imx_slice_1,
    imy_slice_1,
    metadf_2,
    np,
    overlap_contour,
    pannel_key_2,
    plt,
    set_transparency,
):
    _condition = 'Saline'
    _theatmap = np.mean(heatmap_da[metadf_2.Condition == 'Saline', :], axis=0).reshape(atlas_img.shape)
    (_fig, _axs) = plt.subplots(1, len(curated_zplanes), figsize=(3 * len(curated_zplanes), 3), sharey=True)
    _fig.subplots_adjust(wspace=0.25, hspace=0.3)
    for (_idx, _ax) in enumerate(_axs):
        (__, _overlayed_image) = overlap_contour(_theatmap, contour_img, cmin=0, cmax=10, outputpath=None, colormap=plt.cm.viridis)
        _trans_img = set_transparency(_overlayed_image[curated_zplanes[_idx], :, :], (atlas_img == 0)[curated_zplanes[_idx], :, :])
        _ax.imshow(_trans_img[imy_slice_1, imx_slice_1])
        _ax.axis('off')
        _ax.set_ylabel(_condition, color='black')
    _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key_2}_{_condition}_mean.png', bbox_inches='tight', dpi=1024)
    _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key_2}_{_condition}_mean.pdf', bbox_inches='tight', dpi=1024)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure S2 — Brain-wide c-Fos+ cell density across conditions (related to Figure 1)

    Representative 50-µm coronal planes of mean c-Fos+ cell density (per mm³) for each
    condition (hot colormap, 0–15), at the same curated z-planes as 1D.
    """)
    return


@app.cell
def _(
    Conditions,
    FIG_OUT,
    atlas_img,
    contour_img,
    curated_zplanes,
    heatmap_da,
    metadf_2,
    np,
    overlap_contour,
    plt,
    set_transparency,
):
    supp_key = 'FigureS2'
    imy_slice_2 = slice(25, 425)
    imx_slice_2 = slice(50, 600)
    for _condition in Conditions:
        _theatmap = np.nanmean(heatmap_da[metadf_2.Condition == _condition], axis=0).reshape(atlas_img.shape).compute()
        (__, _overlayed_image) = overlap_contour(_theatmap, contour_img, cmin=0, cmax=15, outputpath=None, colormap=plt.cm.hot, overlap_black=False)
        (_fig, _axs) = plt.subplots(1, len(curated_zplanes), figsize=(3 * len(curated_zplanes), 3), sharey=True)
        _fig.subplots_adjust(wspace=0.25, hspace=0.3)
        for (_idx, _ax) in enumerate(_axs):
            _trans_img = set_transparency(_overlayed_image[curated_zplanes[_idx], :, :], (atlas_img == 0)[curated_zplanes[_idx], :, :])
            _ax.imshow(_trans_img[imy_slice_2, imx_slice_2])
            _ax.axis('off')
            _ax.set_ylabel(_condition, color='black')
        _fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=15), cmap=plt.cm.hot), ax=_axs, fraction=0.03, pad=0.02)
        _fig.savefig(FIG_OUT / f'{supp_key}_{_condition}_rawcount.png', bbox_inches='tight', dpi=1024)
        _fig.savefig(FIG_OUT / f'{supp_key}_{_condition}_rawcount.pdf', bbox_inches='tight', dpi=1024)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure S5 — Voxel-wise lateralization analysis (related to Figure 1)

    Voxel-wise **paired** t-tests between the left and right hemispheres of the spatial
    c-Fos+ cell-count data, per condition, with Benjamini–Hochberg FDR correction. As a
    **positive control**, saline-vs-condition (two-sample) t-tests are also computed.
    Significant voxels (adjusted p < 0.05) are rendered as t-value maps on one hemisphere.
    """)
    return


@app.cell
def _(atlas_img, np):
    # left / right hemisphere masks (the x-axis splits the hemispheres)
    _z, _y, _x = atlas_img.shape
    print("atlas_img shape (z, y, x) =", atlas_img.shape)
    left_hemi_mask = np.zeros_like(atlas_img, dtype=bool)
    left_hemi_mask[:, :, :_x // 2] = True
    right_hemi_mask = np.zeros_like(atlas_img, dtype=bool)
    right_hemi_mask[:, :, _x // 2:] = True
    return left_hemi_mask, right_hemi_mask


@app.cell
def _(Conditions, atlas_img, da, heatmap_da, left_hemi_mask, metadf_2, np, right_hemi_mask):
    # paired left-vs-right t-test per condition (right hemisphere flipped to align)
    from scipy.stats import ttest_rel as _ttest_rel
    from statsmodels.stats.multitest import multipletests as _multipletests

    _z, _y, _x = atlas_img.shape
    _n_voxels = heatmap_da.shape[1] // 2
    all_p_value_map_inverted = np.full((len(Conditions), _n_voxels), np.nan, dtype=np.float32)
    all_subset_t_value_map = np.zeros((len(Conditions), _n_voxels), dtype=np.float32)

    for _cidx, _cond in enumerate(Conditions):
        _cmask = metadf_2.Condition == _cond
        _left = heatmap_da[_cmask][:, left_hemi_mask.flatten()]
        _right = heatmap_da[_cmask][:, right_hemi_mask.flatten()]
        # reshape + flip the right hemisphere along x so it aligns with the left
        _right = _right.reshape([_right.shape[0], _z, _y, _x // 2])
        _right = da.flip(_right, axis=3).reshape(_right.shape[0], -1)

        _res = _ttest_rel(_left, _right, axis=0)
        _t, _p = _res.statistic, _res.pvalue

        _corr = np.full_like(_p, np.nan)
        _nn = ~np.isnan(_p)
        if _nn.any():
            _corr[_nn] = _multipletests(_p[_nn], method='fdr_bh')[1]
        _sig = _corr < 0.05
        _inv = np.where(_t < 0, -_corr, _corr)
        _inv[~_sig] = np.nan
        all_p_value_map_inverted[_cidx, :] = _inv
        all_subset_t_value_map[_cidx, :] = np.where(_sig, _t, 0)
    return (all_subset_t_value_map,)


@app.cell
def _(Conditions, atlas_img, brain_voxels, heatmap_da, metadf_2, np):
    # positive control: saline vs each condition (two-sample), FDR-corrected
    from scipy.stats import ttest_ind as _ttest_ind
    from statsmodels.stats.multitest import multipletests as _multipletests

    _nvox_full = int(np.prod(atlas_img.shape))
    all_atlas_subset_t_value_map = np.zeros((len(Conditions), _nvox_full))
    for _cidx, _cond in enumerate(Conditions):
        _b = heatmap_da[metadf_2.Condition == _cond][:, brain_voxels]
        _a = heatmap_da[metadf_2.Condition == 'Saline'][:, brain_voxels]
        _res = _ttest_ind(_a, _b, axis=0)
        _t, _p = _res.statistic, _res.pvalue
        _corr = np.full_like(_p, np.nan)
        _nn = ~np.isnan(_p)
        if _nn.any():
            _corr[_nn] = _multipletests(_p[_nn], method='fdr_bh')[1]
        _sig = _corr < 0.05
        _tmap = np.zeros(_nvox_full)
        _tmap[brain_voxels] = np.where(_sig, _t, 0)
        all_atlas_subset_t_value_map[_cidx, :] = _tmap

    print("Positive control (saline vs condition) — # significant voxels:")
    _nz = np.sum(all_atlas_subset_t_value_map != 0, axis=1)
    for _cidx, _cond in enumerate(Conditions):
        print(f"  {_cond}: {int(_nz[_cidx])} / {len(brain_voxels)}")
    return


@app.cell
def _(
    Conditions,
    FIG_OUT,
    all_subset_t_value_map,
    atlas_img,
    contour_img,
    curated_zplanes,
    overlap_contour,
    plt,
    set_transparency,
):
    # t-value maps (significant L/R voxels) on the left hemisphere
    _z, _y, _x = atlas_img.shape
    _imy = slice(25, 425)
    _imx = slice(50, 600)
    for _cidx, _cond in enumerate(Conditions):
        _theatmap = all_subset_t_value_map[_cidx, :].reshape(_z, _y, _x // 2)
        _overlayed = overlap_contour(_theatmap, contour_img[:, :, :_x // 2],
                                     cmin=-1, cmax=1, colormap=plt.cm.coolwarm, outputpath=None)[1]
        _atlas_slice = (atlas_img == 0)[:, :, :_x // 2]
        _fig, _axs = plt.subplots(1, len(curated_zplanes), figsize=(3 * len(curated_zplanes), 3), sharey=True)
        _fig.subplots_adjust(wspace=0.02, hspace=0.02)
        for _idx, _ax in enumerate(_axs):
            _trans = set_transparency(_overlayed[curated_zplanes[_idx], :, :], _atlas_slice[curated_zplanes[_idx], :, :])
            _ax.imshow(_trans[_imy, _imx])
            _ax.axis('off')
        _fig.savefig(FIG_OUT / f'Lateralization_analysis_tvalmap_{_cond}.png', bbox_inches='tight', dpi=216)
        _fig.savefig(FIG_OUT / f'Lateralization_analysis_tvalmap_{_cond}.pdf', bbox_inches='tight', dpi=216)
        plt.close(_fig)
    return


@app.cell
def _(Conditions, all_subset_t_value_map, brain_voxels, np):
    # proportion of voxels with a significant left/right difference (manuscript text)
    _lgr = np.sum(all_subset_t_value_map > 0, axis=1)
    _rgl = np.sum(all_subset_t_value_map < 0, axis=1)
    _half = len(brain_voxels) // 2
    print("Lateralization (left>right / right>left / % non-significant):")
    for _i, _c in enumerate(Conditions):
        _ns = (_half - (_lgr[_i] + _rgl[_i])) / _half
        print(f"  {_c}: L>R={int(_lgr[_i])}, R>L={int(_rgl[_i])}, non-sig={100 * _ns:.4f}%")
    return



if __name__ == "__main__":
    app.run()
