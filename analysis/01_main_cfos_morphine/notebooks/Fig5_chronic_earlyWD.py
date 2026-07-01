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
    # Figure 5 — Chronic morphine vs. early withdrawal

    Main whole-brain c-Fos dataset. Clusters the BRANCH-rejected regions (from Figure 2) by
    their Chronic / Early-WD density profiles and shows the spatial difference maps.

    - **5B** difference map (Chronic − Early-WD β)
    - **5C** UMAP of regions colored by HDBSCAN cluster; **5D** clustered density heatmap
      (+ HDBSCAN linkage tree); **5E** per-cluster scaled density; **5F** per-region post-hoc
    - **5G–J** predictive-activity maps in PFC, central amygdala (Ce), accumbens (Acb), VTA

    Inputs from the deposit: `OP_cFos_full_result.csv`, `OP_meta.csv`, the Figure-1 β maps
    (`{condition}_betas.npy`), and the Figure-2 BRANCH output
    (`TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv`). No zarr needed.
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
    GROUP = DATA_ROOT / "01_main_cfos_morphine"
    ATLAS = DATA_ROOT / "shared" / "atlas"
    FIG_OUT = Path("../figures")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    figure_key = "Figure5"
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
    mo.md(r"""## Load data""")
    return


@app.cell
def _(ATLAS, GROUP, get_subregions, pd, pickle, tifffile):
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
    Condition_figure_name = ['Saline', 'Acute', 'Chronic', 'Early WD', 'Re-exposure', 'Late WD']
    Condition_color = ['gray', 'lime', 'orange', 'cyan', 'blue', 'purple']
    metadf = metadf[metadf.Condition.isin(Conditions)]

    merge_df = pd.read_csv(GROUP / "OP_cFos_full_result.csv", index_col=0)
    merge_df = merge_df[merge_df.Condition.isin(Conditions)]
    merge_df = merge_df[merge_df.ID.isin(metadf.ID.values)]
    _rm_ids = atlas_df[(atlas_df.acronym == 'HB') | (atlas_df.acronym == 'CBL')]['id'].values
    _rm = pd.concat([get_subregions(atlas_df, _i, return_original=True) for _i in _rm_ids], axis=0)
    _sub_atlas = atlas_df.set_index(['id']).drop(_rm['id'].values)
    merge_df = merge_df[merge_df.acronym.isin(_sub_atlas.acronym.unique())]

    sub_pivot_df = merge_df.pivot(columns='ID', index='acronym', values='density')[metadf.ID.values]
    effect_size_df = (merge_df[['acronym', 'Condition', 'density']]
                      .groupby(['acronym', 'Condition']).mean().reset_index()
                      .pivot(index='acronym', columns='Condition', values='density'))
    return (
        Condition_color,
        Condition_figure_name,
        Conditions,
        atlas_df,
        atlas_img,
        contour_img,
        curated_acronyms,
        effect_size_df,
        merge_df,
        metadf,
        sub_pivot_df,
    )


@app.cell
def _(GROUP, atlas_df, curated_acronyms, pd):
    # BRANCH-rejected regions from Figure 2
    _f = GROUP / "TreeFDRS_pvalue_Figure2_C_glm_stat_df.csv"
    TreeFDRS_df = pd.read_csv(_f, index_col=False).merge(atlas_df[['acronym', 'cleaned_acronym']], on='acronym')
    rejected_acronyms = TreeFDRS_df[(TreeFDRS_df.acronym.isin(curated_acronyms)) & (TreeFDRS_df.rejected == True)].acronym.values
    print(len(rejected_acronyms), "BRANCH-rejected regions")
    return (rejected_acronyms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure 5B — difference map (Chronic − Early-WD)""")
    return


@app.cell
def _(FIG_OUT, GROUP, atlas_img, contour_img, figure_key, np, overlap_contour, plt, set_transparency):
    _curated_zplanes = [84, 104, 117, 153, 186, 220]
    _imy, _imx = slice(25, 425), slice(50, 600)
    _theatmap = np.load(GROUP / "Chronic_Morphine_betas.npy") - np.load(GROUP / "Withdrawal_Morphine_betas.npy")
    (__, _ov) = overlap_contour(_theatmap, contour_img, cmin=-15, cmax=15, outputpath=None)
    (_fig, _axs) = plt.subplots(1, len(_curated_zplanes), figsize=(3 * len(_curated_zplanes), 3), sharey=True)
    _fig.subplots_adjust(wspace=0.25, hspace=0.3)
    for (_i, _ax) in enumerate(_axs):
        _tr = set_transparency(_ov[_curated_zplanes[_i], :, :], (atlas_img == 0)[_curated_zplanes[_i], :, :])
        _ax.imshow(_tr[_imy, _imx]); _ax.axis('off')
    _fig.savefig(FIG_OUT / f'{figure_key}B.png', bbox_inches='tight', dpi=1024)
    _fig.savefig(FIG_OUT / f'{figure_key}B.pdf', bbox_inches='tight', dpi=1024)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 5C–E — UMAP / HDBSCAN clustering of rejected regions

    Cluster the BRANCH-rejected regions by their per-subject Chronic / Early-WD density
    (log1p, z-scored). UMAP (n_neighbors=3, min_dist=0.2, seed 42) → HDBSCAN.
    """)
    return


@app.cell
def _(metadf, np, pd, rejected_acronyms, sub_pivot_df):
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    _ids = metadf[metadf.Condition.isin(['Chronic_Morphine', 'Withdrawal_Morphine'])]['ID']
    umap_data = sub_pivot_df.loc[rejected_acronyms, _ids]
    _scaled = _StandardScaler().fit_transform(np.log1p(umap_data).to_numpy().T)
    umap_input = pd.DataFrame(_scaled.T, columns=umap_data.columns, index=umap_data.index)  # regions x subjects
    return (umap_input,)


@app.cell
def _(GROUP, figure_key, np, pd, rejected_acronyms, umap_input):
    import umap as _umap
    import hdbscan as _hdbscan
    from scipy.cluster.hierarchy import leaves_list as _leaves_list

    mapper = _umap.UMAP(n_components=2, n_neighbors=3, min_dist=0.2, random_state=42).fit(umap_input)
    embedding = mapper.embedding_
    clusterer = _hdbscan.HDBSCAN(min_cluster_size=3).fit(embedding)

    _labels = clusterer.single_linkage_tree_.get_clusters(2, min_cluster_size=2)
    _cdf = clusterer.single_linkage_tree_.to_pandas()
    _Z = np.column_stack([_cdf['left_child'], _cdf['right_child'], _cdf['distance'], _cdf['size']])
    _order = _labels[_leaves_list(_Z)]
    _uniq = []
    for _l in _order:
        if _l not in _uniq:
            _uniq.append(_l)
    _map, _new = {}, 1
    for _l in _uniq:
        if _l >= 0:
            _map[_l] = _new; _new += 1
        else:
            _map[_l] = -1
    relabeled_labels = [_map[_l] for _l in _labels]

    embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=umap_input.index)
    embedding_df['label'] = relabeled_labels
    embedding_df['rejected'] = [f in rejected_acronyms for f in embedding_df.index]
    embedding_df.to_csv(GROUP / f'{figure_key}E_umap_embedding.csv', index=True)
    print("clusters:", sorted(set(relabeled_labels)))
    return clusterer, embedding, embedding_df


@app.cell
def _(FIG_OUT, clusterer, figure_key, plt):
    # 5D inset — HDBSCAN single-linkage tree
    (_fig, _ax) = plt.subplots(1, 1, figsize=(4, 2))
    clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    _fig.savefig(FIG_OUT / f'{figure_key}D_hdbscan_tree.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}D_hdbscan_tree.pdf', bbox_inches='tight', dpi=216)
    return


@app.cell
def _(FIG_OUT, atlas_df, embedding, embedding_df, figure_key, plt, rejected_acronyms, umap_input):
    from adjustText import adjust_text as _adjust_text
    _clustered = (embedding_df['label'] >= 0)
    _rej = (embedding_df['rejected'] == True)
    (_fig, _ax) = plt.subplots(1, 1, figsize=(4.5, 4.5))
    _sc = _ax.scatter(embedding[_clustered & _rej, 0], embedding[_clustered & _rej, 1],
                      c=embedding_df['label'][_clustered & _rej], s=60, edgecolor='k', cmap='Set1')
    _ax.legend(*_sc.legend_elements(), title='Cluster')
    _curated = set(['VTA', 'VTg', 'PT', 'MD', 'PVT', 'AcbC', 'AcbSh', 'LS', 'O', 'ASt', 'La', 'CPce', 'CPc',
                    'Ce', 'ST', 'BL', 'MBO', 'DP', 'Pl', 'DS', 'CPre', 'STr', 'PrL', 'DI', 'AI', 'IL', 'Cg',
                    'SPTg', 'Cl', 'A24a (IL)', 'Au1', 'RLi', 'PrEW', 'A24 (Cg)', 'A32 (PrL)', 'IPAC', 'PrG'])
    _texts = []
    for _i, _acr in enumerate(umap_input.index):
        _clean = atlas_df[atlas_df.acronym == _acr].cleaned_acronym.values[0]
        if _clean in _curated and _acr in rejected_acronyms:
            _texts.append(_ax.text(embedding[_i, 0], embedding[_i, 1], _clean, fontsize=10, ha='right'))
    _adjust_text(_texts, ax=_ax, expand=(2, 2), force_text=(0.25, 0.25), arrowprops=dict(color='gray', lw=1, alpha=0.75))
    _ax.set_ylabel('UMAP1', fontsize=12); _ax.set_xlabel('UMAP2', fontsize=12)
    _fig.savefig(FIG_OUT / f'{figure_key}C.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}C.pdf', bbox_inches='tight', dpi=216)
    return


@app.cell
def _(atlas_df, effect_size_df, embedding_df, rejected_acronyms, umap_input):
    # join cluster labels + Chronic-minus-EarlyWD delta onto the scaled region x subject table
    _delta = ((effect_size_df['Chronic_Morphine'] - effect_size_df['Withdrawal_Morphine'])
              .rename('dcw').to_frame()
              .join(atlas_df.set_index('acronym')[['cleaned_acronym']]))
    data_full = _delta.loc[rejected_acronyms].join(umap_input.join(embedding_df))
    return (data_full,)


@app.cell
def _(Condition_figure_name, FIG_OUT, data_full, figure_key, metadf, np, plt, sns, umap_input):
    # 5D — clustered scaled-density heatmap (Chronic, Early-WD)
    _sub_conditions = ['Chronic_Morphine', 'Withdrawal_Morphine']
    _sorted = data_full[data_full.label >= 0].sort_values(by=['label', 'dcw'], ascending=[True, False])
    _acr = _sorted.index
    _labels, _ix = np.unique(_sorted['label'], return_index=True)
    _ylabels = [_sorted['label'][i] for i in sorted(_ix)[::-1]]
    _counts = [int(np.sum(np.array(_sorted['label']) == y)) for y in _ylabels][::-1]
    _borders = [0] + list(np.cumsum(_counts))
    _x = np.array(_borders); _ticks = (_x[1:] + _x[:-1]) / 2
    (_fig, _axs) = plt.subplots(len(_sub_conditions), 1, figsize=(7, len(_sub_conditions) * 1.0), sharex=True)
    for (_idx, _cond) in enumerate(_sub_conditions):
        _subj = metadf[metadf.Condition == _cond].ID.values
        if _idx == len(_sub_conditions) - 1:
            _cax = _fig.add_axes([_axs[_idx].get_position().x1 + 0.01, _axs[_idx].get_position().y0 + 0.2, 0.05, 0.5])
            sns.heatmap(_sorted.loc[_acr, _subj].T, cbar_ax=_cax, ax=_axs[_idx], vmin=0, vmax=2)
            _cax.set_ylabel('Scaled density', rotation=270, labelpad=10, fontsize=10)
        else:
            sns.heatmap(_sorted.loc[_acr, _subj].T, cbar=False, ax=_axs[_idx], vmin=0, vmax=2)
        _axs[_idx].set_xticks(_ticks); _axs[_idx].set_xticklabels(_ylabels[::-1]); _axs[_idx].set_xlabel('')
        [_axs[_idx].axvline(b, color='yellow', lw=1, ls=':') for b in _borders[1:-1]]
        _axs[_idx].set_yticks(np.array(range(len(_subj))) + 0.5); _axs[_idx].set_yticklabels([])
        _axs[_idx].set_ylabel(Condition_figure_name[2:4][_idx])
    _fig.savefig(FIG_OUT / f'{figure_key}D_clustered_scaled_heatmap.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}D_clustered_scaled_heatmap.pdf', bbox_inches='tight')
    return


@app.cell
def _(Condition_color, Conditions, FIG_OUT, data_full, figure_key, metadf, np, plt, sns, umap_input):
    # 5E — per-cluster scaled density (Chronic vs Early-WD)
    from statannotations.Annotator import Annotator as _Annotator
    _stack = (data_full[list(umap_input.columns) + ['label']].reset_index()
              .melt(id_vars=['acronym', 'label'], var_name='ID', value_name='scaled_density'))
    _stack = _stack.merge(metadf[['ID', 'Condition']], on='ID')
    _stack = _stack[_stack.label >= 0]
    _order = np.unique(_stack.label.unique())
    _sub = _stack[_stack.Condition.isin(['Chronic_Morphine', 'Withdrawal_Morphine'])]
    print("N per cluster/condition:"); print(_sub.groupby(['label', 'Condition']).size())
    (_fig, _ax) = plt.subplots(1, 1, figsize=(2, 1.5))
    sns.stripplot(data=_sub, hue='Condition', y='scaled_density', x='label', dodge=True, order=_order,
                  hue_order=Conditions[2:4], ax=_ax, palette=Condition_color[2:4], alpha=0.2, size=1)
    sns.pointplot(data=_sub, hue='Condition', y='scaled_density', x='label', order=_order,
                  dodge=0.6 - 0.6 / 4, errorbar=("ci", 95), hue_order=Conditions[2:4], ax=_ax,
                  palette=Condition_color[2:4], markers="o", markersize=5, linestyle="none", linewidth=1)
    sns.despine(); _ax.set_xlabel('Cluster'); _ax.set_ylabel('Scaled density', fontsize=12)
    if _ax.get_legend() is not None:
        _ax.get_legend().remove()
    _pairs = [((c, "Chronic_Morphine"), (c, "Withdrawal_Morphine")) for c in _order]
    _an = _Annotator(_ax, _pairs, data=_sub, y='scaled_density', x='label', order=_order,
                     hue='Condition', hue_order=['Chronic_Morphine', 'Withdrawal_Morphine'])
    _an.configure(test='t-test_ind', text_format='star', loc='outside')
    _an.configure(comparisons_correction="BH", correction_format="replace")
    _an.apply_and_annotate()
    _fig.savefig(FIG_OUT / f'{figure_key}E.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}E.pdf', bbox_inches='tight')
    return


@app.cell
def _(Condition_color, Conditions, FIG_OUT, atlas_df, data_full, figure_key, merge_df, metadf, plt, rejected_acronyms, sns, umap_input):
    # 5F — per-region post-hoc (Chronic vs Early-WD) in selected regions
    from statannotations.Annotator import Annotator as _Annotator
    _scaled_long = (data_full[list(umap_input.columns)].stack().reset_index()
                    .rename(columns={'level_1': 'ID', 0: 'scaled_density'}))
    _smdf = merge_df[merge_df.Condition.isin(['Chronic_Morphine', 'Withdrawal_Morphine'])].merge(
        _scaled_long, on=['acronym', 'ID'])
    _smdf.Condition = _smdf.Condition.astype('str')
    _sorted = ['A24a (IL)', 'Au1', 'IMD', 'RLi', 'PrEW', 'A24 (Cg)', 'Cl', 'A32 (PrL)', 'VTA', 'AcbC',
               'AcbSh', 'La', 'IGL', 'IPAC', 'PrG', 'Ce']
    _rest = [f for f in rejected_acronyms if f not in _sorted]
    _regs = [r for r in _sorted + _rest if r in set(merge_df.acronym)]
    _clean = atlas_df.set_index('acronym').loc[_regs, 'cleaned_acronym'].values
    _tdata = _smdf[_smdf.acronym.isin(_regs)].copy()
    _tdata.scaled_density = _tdata.scaled_density.astype('float64')
    (_fig, _ax) = plt.subplots(1, 1, figsize=(len(_regs) // 3.2, 1.5))
    sns.stripplot(data=_tdata, hue='Condition', y='scaled_density', x='acronym', dodge=True, order=_regs,
                  hue_order=Conditions[2:4], ax=_ax, palette=Condition_color[2:4], alpha=0.25, size=2)
    sns.pointplot(data=_tdata, hue='Condition', y='scaled_density', x='acronym', order=_regs,
                  dodge=0.5 - 0.5 / len(_regs), hue_order=Conditions[2:4], ax=_ax, palette=Condition_color[2:4],
                  markers="o", markersize=3, linestyle="none", linewidth=0.5)
    sns.despine(); _ax.set_xticks(range(len(_clean))); _ax.set_xticklabels(_clean, fontsize=11, rotation=-45)
    if _ax.get_legend() is not None:
        _ax.get_legend().remove()
    _ax.set_xlabel(''); _ax.set_ylabel('Scaled density', fontsize=12)
    _pairs = [((c, "Chronic_Morphine"), (c, "Withdrawal_Morphine")) for c in _regs]
    _an = _Annotator(_ax, _pairs, data=_tdata, y='scaled_density', x='acronym', hue='Condition',
                     hue_order=Conditions[2:4], order=_regs)
    _an.configure(test='t-test_ind', text_format='star', loc='outside')
    _an.configure(comparisons_correction="BH", correction_format="replace")
    _an.apply_and_annotate()
    _fig.savefig(FIG_OUT / f'{figure_key}F.png', bbox_inches='tight', dpi=216)
    _fig.savefig(FIG_OUT / f'{figure_key}F.pdf', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 5G–J — predictive-activity maps in target regions

    Chronic / Early-WD β maps zoomed to each target region across its A–P extent.
    """)
    return


@app.cell
def _(FIG_OUT, GROUP, atlas_df, atlas_img, contour_img, figure_key, get_subregions, np, overlap_contour, plt, set_transparency):
    def plot_region_maps(target_site_acronym, hemi, pannel_key):
        _tid = atlas_df.loc[atlas_df.acronym == target_site_acronym, 'id'].values[0]
        _subids = get_subregions(atlas_df, _tid, return_original=True)['id'].values
        _zs = np.concatenate([np.where(atlas_img == ID)[0] for ID in _subids])
        _zu = np.unique(_zs).astype('uint16')
        _half = atlas_img.shape[2] // 2

        def _get_slice(z, window=60):
            _hs = slice(0, _half) if hemi == 'left' else slice(_half, atlas_img.shape[2])
            _ys, _xs = np.array([]), np.array([])
            for ID in _subids:
                _y, _x = np.where(atlas_img[z, :, _hs] == ID)
                _xs = np.concatenate([_xs, _x]); _ys = np.concatenate([_ys, _y])
            _yc, _xc = int(np.mean(_ys)), int(np.mean(_xs))
            if hemi == 'left':
                return slice(_yc - window, _yc + window), slice(_xc - window, _xc + window)
            elif hemi == 'right':
                return slice(_yc - window, _yc + window), slice(_xc - window + _half, _xc + window + _half)
            return slice(_yc - window, _yc + window), slice(_half - window, _half + window)

        for _cond in ['Withdrawal_Morphine', 'Chronic_Morphine', 'empty']:
            _theatmap = np.zeros(atlas_img.shape) if _cond == 'empty' else np.load(GROUP / f'{_cond}_betas.npy')
            (__, _ov) = overlap_contour(_theatmap, contour_img, cmin=-15, cmax=15, outputpath=None)
            _zsel = _zu[::5]
            (_fig, _axs) = plt.subplots(1, len(_zsel), figsize=(3 * len(_zsel), 3), sharey=True)
            _fig.subplots_adjust(wspace=0.25, hspace=0.3)
            _axs = np.atleast_1d(_axs)
            for _i, _zp in enumerate(_zsel):
                _ysl, _xsl = _get_slice(_zp)
                _tr = set_transparency(_ov[_zp, :, :], (atlas_img == 0)[_zp, :, :])
                _axs[_i].imshow(_tr[_ysl, _xsl]); _axs[_i].axis('off'); _axs[_i].set_ylabel(_cond, color='black')
            _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key}_{_cond}.png', bbox_inches='tight', dpi=512)
            _fig.savefig(FIG_OUT / f'{figure_key}{pannel_key}_{_cond}.pdf', bbox_inches='tight', dpi=512)
            plt.close(_fig)
    return (plot_region_maps,)


@app.cell
def _(plot_region_maps):
    # G = PFC (IL), H = central amygdala (Ce), I = accumbens (Acb), J = VTA
    for _target, _hemi, _key in [('A24a (IL)', 'center', 'G'), ('Ce', 'right', 'H'),
                                 ('Acb', 'right', 'I'), ('VTA', 'right', 'J')]:
        plot_region_maps(_target, _hemi, _key)
    return


if __name__ == "__main__":
    app.run()
