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
    # Figure S1 — Behavioral characterization

    Separate cohort (n up to 20), 6 conditions across days: open-field (OF) and linear-track
    (LT) behavior. Minimum for the panels + table:

    - **S1C** open-field quantification (per-variable lineplots, paired t-test vs saline + Bonferroni).
    - **S1E** linear-track quantification (saline-normalized heatmaps by variable group).
    - **S1F** behavior-based condition-similarity clustermap.
    - **Table 2** all paired-t-test statistics -> `FigureS1_all_statistics.xlsx`
      (sheet `All_stats_combined`).

    Inputs (deposit `08_behavior/`): `Opioid_revision_meta.csv`,
    `Opioid_revision_OF_results.csv`, `Opioid_revision_LT_merge_df_260310.csv`
    (the pre-merged linear-track table; the one-time raw-xlsx merge is not reproduced here).
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    return Path, np, os, pd, plt, sns


@app.cell
def _(Path, os):
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    GROUP = DATA_ROOT / "08_behavior"
    FIG_OUT = Path("../figures")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    assert GROUP.exists(), f"DATA_ROOT not found: {DATA_ROOT}. Set OPIOID_DATA_ROOT."
    return FIG_OUT, GROUP


@app.cell
def _(plt):
    plt.rcParams.update({
        'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.edgecolor': 'black',
        'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
        'legend.facecolor': 'none', 'legend.edgecolor': 'none', 'text.color': 'black',
        'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    return


@app.cell
def _(pd):
    # ---- config ----
    Conditions = ['Saline', 'Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine',
                  'ReExposure_Morphine', 'Late_Withdrawal_Morphine']
    Condition_clean_names = ['Saline', 'Acute', 'Chronic', 'Early W.D.', 'Re-Exposure', 'Late W.D.']
    cond_label = dict(zip(Conditions, Condition_clean_names))

    OF_variables = ['Distance_moved_cm', 'Velocity_cm_s', 'Center_zone_frequency',
                    'Center_zone_duration_s', 'Center_zone_latency_s', 'Immobile_frequency', 'Immobile_duration']
    OF_labels = ['Distance moved (cm)', 'Velocity (cm/s)', 'Center zone frequency',
                 'Center zone duration (s)', 'Center zone latency (s)', 'Number of immobile bouts',
                 'Total immobile duration (s)']
    of_label = dict(zip(OF_variables, OF_labels))

    LT_variable_sets = {
        'Basic': ['Total Distance (cm)', 'Max Speed (cm/s)', 'Zone Edge', 'Zone Middle', 'Zone Center'],
        'Behavior': ['Walk Speed_Average Speed (cm/s)', 'Walk Speed_Peak Speed (cm/s)',
                     'Walk_Number of Bouts', 'Walk_Total Time Spent (s)', 'Rear_Number of Bouts',
                     'Rear_Total Time Spent (s)', 'Groom_Number of Bouts', 'Groom_Total Time Spent (s)'],
        'Kinematic': ['Average Forepaws Gait (cm)', 'Average Hindpaw Gait (cm)',
                      'Average Forepaw Stride (cm)', 'Average Hindpaw Stride (cm)',
                      'Average Tailbase Height (cm)', 'Avergae Midtail Height (cm)'],
    }

    def p_to_sig(p):
        if pd.isna(p):
            return ""
        return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
    return (
        Conditions,
        Condition_clean_names,
        LT_variable_sets,
        OF_variables,
        cond_label,
        of_label,
        p_to_sig,
    )


@app.cell
def _(GROUP, pd):
    metadf = pd.read_csv(GROUP / "Opioid_revision_meta.csv", index_col=False)
    metadf = metadf[metadf.Usable == True]

    of_df = pd.read_csv(GROUP / "Opioid_revision_OF_results.csv", index_col=False)
    of_df = of_df[of_df.ID.isin(metadf.ID)]

    lt_df = pd.read_csv(GROUP / "Opioid_revision_LT_merge_df_260310.csv")
    lt_df = lt_df[lt_df.ID.isin(metadf.ID)]
    print("OF:", of_df.shape, "| LT:", lt_df.shape)
    return lt_df, metadf, of_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Statistics — paired t-test vs saline + Bonferroni (Table 2)""")
    return


@app.cell
def _(Conditions, GROUP, LT_variable_sets, OF_variables, cond_label, lt_df, np, of_df, of_label, p_to_sig, pd):
    from scipy.stats import ttest_rel as _ttest_rel

    def paired_ttest_vs_saline(data, variables, panel, variable_group):
        _out = []
        for _v in variables:
            if _v not in data.columns:
                continue
            _d = data[['ID', 'Condition', _v]].copy()
            _d[_v] = pd.to_numeric(_d[_v], errors='coerce')
            _d = _d.dropna(subset=[_v])
            _d = _d.groupby(['ID', 'Condition'], observed=True, as_index=False)[_v].mean()
            _wide = _d.pivot(index='ID', columns='Condition', values=_v)
            _rows = []
            for _c in Conditions:
                if _c == 'Saline' or _c not in _wide.columns or 'Saline' not in _wide.columns:
                    continue
                _p = _wide[['Saline', _c]].dropna()
                if len(_p) >= 2:
                    _t, _pr = _ttest_rel(_p[_c].astype(float), _p['Saline'].astype(float), nan_policy='omit')
                    _dfree = len(_p) - 1
                else:
                    _t, _pr, _dfree = np.nan, np.nan, np.nan
                _rows.append({'panel': panel, 'variable_group': variable_group, 'variable': _v,
                              'variable_label': of_label.get(_v, _v),
                              'comparison': f"{cond_label.get(_c, _c)} vs Saline", 'condition': _c,
                              'n_pairs': len(_p), 'df': _dfree, 't_value': _t, 'p_raw': _pr})
            _vdf = pd.DataFrame(_rows)
            if len(_vdf):
                _n = _vdf['p_raw'].notna().sum()
                _vdf['p_adjusted_bonferroni'] = np.minimum(_vdf['p_raw'] * _n, 1.0)
                _vdf['significance'] = _vdf['p_adjusted_bonferroni'].apply(p_to_sig)
            _out.append(_vdf)
        return pd.concat(_out, ignore_index=True) if _out else pd.DataFrame()

    _stats = [paired_ttest_vs_saline(of_df, OF_variables, 'S1C', 'Open field')]
    for _grp, _vars in LT_variable_sets.items():
        _stats.append(paired_ttest_vs_saline(lt_df, _vars, 'S1E', f'Linear track - {_grp}'))
    stats_all = pd.concat(_stats, ignore_index=True)
    with pd.ExcelWriter(GROUP / "FigureS1_all_statistics.xlsx") as _w:
        stats_all.to_excel(_w, sheet_name='All_stats_combined', index=False)
    print("Table 2 saved:", stats_all.shape, "| significant:", int((stats_all.p_adjusted_bonferroni < 0.05).sum()))
    return (stats_all,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure S1C — open-field quantification (lineplots)""")
    return


@app.cell
def _(Condition_clean_names, Conditions, FIG_OUT, OF_variables, np, of_df, of_label, plt, sns, stats_all):
    for _v in OF_variables:
        (_fig, _ax) = plt.subplots(1, 1, figsize=(2, 2))
        sns.lineplot(data=of_df, x='Condition', y=_v, units='ID', estimator=None, alpha=0.3, linewidth=1, ax=_ax, color='gray')
        sns.lineplot(data=of_df, x='Condition', y=_v, err_style=None, linewidth=2, ax=_ax, color='black')
        _ax.set_ylim(0,); sns.despine(); _ax.set_xlabel('')
        _ax.set_xticks(range(len(Conditions))); _ax.set_xticklabels(Condition_clean_names, rotation=-45)
        _ax.set_ylabel(of_label[_v])
        for _, _r in stats_all[(stats_all.panel == 'S1C') & (stats_all.variable == _v)].iterrows():
            _xi = Conditions.index(_r['condition'])
            _ax.text(_xi, _ax.get_ylim()[1] * 1.01, _r['significance'], ha='center', va='center', fontsize=8)
        _fig.savefig(FIG_OUT / f'FigureS1C_OF_{_v}.png', bbox_inches='tight', dpi=300)
        _fig.savefig(FIG_OUT / f'FigureS1C_OF_{_v}.pdf', bbox_inches='tight', dpi=300)
        plt.close(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure S1E — linear-track quantification (saline-normalized heatmaps)""")
    return


@app.cell
def _(Condition_clean_names, Conditions, FIG_OUT, LT_variable_sets, lt_df, np, plt, sns, stats_all):
    for _grp, _vars in LT_variable_sets.items():
        _valid = [v for v in _vars if v in lt_df.columns]
        _mean = lt_df.loc[:, ['Condition'] + _valid].groupby('Condition', observed=True).mean(numeric_only=True)
        _mean = _mean.loc[[c for c in Conditions if c in _mean.index]]
        _norm = _mean.apply(lambda x: (x - x.loc['Saline']) / x.loc['Saline'], axis=0)
        (_fig, _ax) = plt.subplots(1, 1, figsize=(len(_valid) * 2 / 3, 2))
        sns.heatmap(_norm, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=1, ax=_ax, square=True)
        _ax.set_xticklabels(_valid, rotation=-45, ha='left')
        _ax.set_yticklabels([Condition_clean_names[Conditions.index(c)] for c in _norm.index], rotation=0)
        for _, _r in stats_all[(stats_all.variable_group == f'Linear track - {_grp}')].iterrows():
            if _r['variable'] in _norm.columns and _r['condition'] in list(_norm.index):
                _ri = list(_norm.index).index(_r['condition']) + 0.5
                _ci = list(_norm.columns).index(_r['variable']) + 0.5
                _ax.text(_ci, _ri, _r['significance'], ha='center', va='center', fontsize=7)
        _fig.savefig(FIG_OUT / f'FigureS1E_LT_{_grp}_heatmap.png', bbox_inches='tight', dpi=300)
        _fig.savefig(FIG_OUT / f'FigureS1E_LT_{_grp}_heatmap.pdf', bbox_inches='tight', dpi=300)
        plt.close(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure S1F — behavior-based condition-similarity clustermap""")
    return


@app.cell
def _(Conditions, FIG_OUT, LT_variable_sets, OF_variables, lt_df, np, of_df, pd, sns):
    from scipy.cluster.hierarchy import linkage as _linkage
    from scipy.spatial.distance import pdist as _pdist

    _of_mean = of_df.loc[:, ['Condition'] + [v for v in OF_variables if v in of_df.columns]].groupby('Condition', observed=True).mean(numeric_only=True)
    _all_lt = sum(LT_variable_sets.values(), [])
    _lt_mean = lt_df.loc[:, ['Condition'] + [v for v in _all_lt if v in lt_df.columns]].groupby('Condition', observed=True).mean(numeric_only=True)
    _beh = _of_mean.join(_lt_mean)
    _beh = _beh.loc[:, ~_beh.columns.duplicated()]

    _delta = _beh.subtract(_beh.loc['Saline'], axis=1).drop(index='Saline')
    _z = _delta.apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    _corr = _z.T.corr(method='pearson')
    _link = _linkage(_pdist(_corr), method='average')
    _g = sns.clustermap(_corr.T, row_linkage=_link, col_linkage=_link, cmap='coolwarm',
                        vmin=-1, vmax=1, annot=True, figsize=(5, 5), cbar_kws={"label": "Correlation"})
    _g.savefig(FIG_OUT / 'FigureS1F_behavior_similarity_clustermap.png', bbox_inches='tight', dpi=300)
    _g.savefig(FIG_OUT / 'FigureS1F_behavior_similarity_clustermap.pdf', bbox_inches='tight', dpi=300)
    print("clustermap conditions:", list(_corr.index))
    return


if __name__ == "__main__":
    app.run()
