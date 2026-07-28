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
    # Statistical analysis
    """)
    return


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd

    # image processing packages
    import tifffile as tiff
    import skimage as ski
    from difflib import SequenceMatcher
    import matplotlib.pyplot as plt
    import seaborn as sns

    return np, os, pd, plt, sns


@app.cell
def _(plt):
    #from adjustText import adjust_text
    # Set matplotlib parameters for white text on transparent background
    #important for text to be detected when importing saved figures into illustrator
    plt.rcParams.update({'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'legend.facecolor': 'none', 'legend.edgecolor': 'none', 'text.color': 'black', 'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})  # Transparent figure background  # Transparent axes background  # White axes edge color  # White axis labels  # White tick labels  # Transparent legend background  # Transparent legend edgecolor  # White text color
    return


@app.cell
def _(os):
    from pathlib import Path
    DATA_ROOT = Path(os.environ.get('OPIOID_DATA_ROOT', '/path/to/Figshare_deposit')).expanduser()
    # Paths -> Figshare deposit (set OPIOID_DATA_ROOT or edit)
    GROUP = DATA_ROOT / '07_nac_histology_hcr'
    metapath = str(GROUP)
    analysis_resultpath = str(GROUP)
    analysis_figurepath = '../figures/Figure8_NAc_ISH'
    os.makedirs(analysis_figurepath, exist_ok=True)
    return analysis_figurepath, analysis_resultpath, metapath


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## start analysis by section
    """)
    return


@app.cell
def _(metapath, os, pd):
    # read a meta file for sections
    section_metadf = pd.read_excel(os.path.join(metapath,"NAc_ISH_meta_v3.xlsx"),engine='openpyxl')
    section_metadf = section_metadf[section_metadf.Usable]
    section_metadf = section_metadf.rename(columns = {'Section':'section'})
    #filter_list = section_metadf[['ID','Check']].rename(columns = {'Check':'section'})
    #filter_list = [tuple(f) for f in filter_list.values]
    return (section_metadf,)


@app.cell
def _(analysis_resultpath, os, pd):
    # read the dataframe
    total_merge_cell_df = pd.read_csv(os.path.join(analysis_resultpath,'total_merge_cell_df.csv'),index_col = 0)

    # drop the background
    total_merge_cell_df = total_merge_cell_df.drop(0)

    # drop the conditions that are not in the analysis plan
    Conditions = ['Saline','Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']
    total_merge_cell_df = total_merge_cell_df[total_merge_cell_df.Condition.isin(Conditions)]
    return Conditions, total_merge_cell_df


@app.cell
def _(section_metadf, total_merge_cell_df):
    # subset the dataframe to the ones that are usable
    total_merge_cell_df.section = total_merge_cell_df.section.astype('int')
    total_merge_cell_df_1 = section_metadf.set_index(['ID', 'section'])[['Usable']].join(total_merge_cell_df.set_index(['ID', 'section'])).reset_index()
    return (total_merge_cell_df_1,)


@app.cell
def _():
    # update the Kim_z column 
    #total_merge_cell_df.drop(columns = 'Kim_z').merge(section_metadf[['ID','section','Kim_z']], on = ['ID','section'], how = 'left').to_csv(os.path.join(analysis_resultpath,'total_merge_cell_df.csv'),index = False)
    return


@app.cell
def _():
    # read the dataframe
    #total_merge_cell_df = pd.read_csv(os.path.join(analysis_resultpath,'total_merge_cell_df.csv'),index_col = False)
    return


@app.cell
def _():
    # count threshold
    count_threshold = 5 # dots
    return (count_threshold,)


@app.cell
def _(count_threshold, total_merge_cell_df_1):
    genes = [f.replace('_FFT_intensity', '') for f in total_merge_cell_df_1.columns if '_FFT_intensity' in f]
    _gene = 'Fos'
    for _gene in genes:
        total_merge_cell_df_1[f'{_gene}_cells'] = total_merge_cell_df_1[f'{_gene}_regressed_counts'] > count_threshold
        total_merge_cell_df_1[f'{_gene}_sum_raw_intensity'] = total_merge_cell_df_1[f'{_gene}_raw_intensity'] * total_merge_cell_df_1[f'{_gene}_counts']
    return


@app.cell
def _(total_merge_cell_df_1):
    core_cell_df = total_merge_cell_df_1[(total_merge_cell_df_1.Core_cells == True) & (total_merge_cell_df_1.Shell_cells == False)]
    shell_cell_df = total_merge_cell_df_1[(total_merge_cell_df_1.Core_cells == False) & (total_merge_cell_df_1.Shell_cells == True)]
    return core_cell_df, shell_cell_df


@app.cell
def _(core_cell_df, pd, shell_cell_df):
    # select the dataframe to analyze
    _tcell_df = core_cell_df
    for (_idx, _tcell_df) in enumerate([core_cell_df, shell_cell_df]):
        _t_group_cell_df = _tcell_df.groupby(['Condition', 'ID', 'section']).mean().reset_index()
        if _idx == 0:
            group_cell_df = _t_group_cell_df
        else:
            group_cell_df = pd.concat([group_cell_df, _t_group_cell_df], axis=0)
    return (group_cell_df,)


@app.cell
def _(analysis_resultpath, group_cell_df, os):
    # subset the dataframe to the ones that are usable
    group_cell_df.section = group_cell_df.section.astype('int')
    #group_cell_df = section_metadf.set_index(['ID','section'])[['Usable']].join(group_cell_df.set_index(['ID','section'])).reset_index()
    # write the result
    group_cell_df.to_csv(os.path.join(analysis_resultpath,'group_cell_df.csv'),index = False)
    return


@app.cell
def _(Conditions, analysis_figurepath, group_cell_df, os, plt, sns):
    _filtered_data = group_cell_df[group_cell_df.Core_cells == 0]
    (_fig, _ax) = plt.subplots(figsize=(1.0, 1.5))
    sns.stripplot(data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions, dodge=False, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.pointplot(data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions, ax=_ax, linewidth=0.7, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.despine()
    _ax.set_xticklabels(['Saline', 'Acute', 'Chronic', 'Early WD'], rotation=-45)
    _ax.set_ylabel('Proportion of Fos+ cells')
    _ax.set_title('AcbSh')
    from statannotations.Annotator import Annotator
    _pairs = [('Saline', c) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
    _annotator = Annotator(_ax, _pairs, data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions)
    _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    _annotator.configure(comparisons_correction='BH', correction_format='replace')
    _annotator.apply_and_annotate()
    _fig.savefig(os.path.join(analysis_figurepath, 'Shell_cells_Fos_cells.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, 'Shell_cells_Fos_cells.pdf'), dpi=300, bbox_inches='tight')
    return (Annotator,)


@app.cell
def _(Annotator, Conditions, group_cell_df, plt, sns):
    _filtered_data = group_cell_df[group_cell_df.Core_cells == 0]
    (_fig, _ax) = plt.subplots(figsize=(1.0, 1.5))
    sns.stripplot(data=_filtered_data, x='Condition', y='Kcnip1_cells', order=Conditions, dodge=False, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.pointplot(data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions, ax=_ax, linewidth=0.7, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.despine()
    _ax.set_xticklabels(['Saline', 'Acute', 'Chronic', 'Early WD'], rotation=-45)
    _ax.set_ylabel('Proportion of Kcnip1+ cells')
    _ax.set_title('AcbSh')
    _pairs = [('Saline', c) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
    _annotator = Annotator(_ax, _pairs, data=_filtered_data, x='Condition', y='Kcnip1_cells', order=Conditions)
    _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    _annotator.configure(comparisons_correction='BH', correction_format='replace')
    _annotator.apply_and_annotate()
    return


@app.cell
def _(Annotator, Conditions, analysis_figurepath, group_cell_df, os, plt, sns):
    _filtered_data = group_cell_df[group_cell_df.Core_cells == 1]
    (_fig, _ax) = plt.subplots(figsize=(1.0, 1.5))
    sns.stripplot(data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions, dodge=False, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.pointplot(data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions, ax=_ax, linewidth=0.7, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.despine()
    _ax.set_xticklabels(['Saline', 'Acute', 'Chronic', 'Early WD'], rotation=-45)
    _ax.set_ylabel('Proportion of Fos+ cells')
    _ax.set_title('AcbC')
    _pairs = [('Saline', c) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
    _annotator = Annotator(_ax, _pairs, data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions)
    _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    _annotator.configure(comparisons_correction='BH', correction_format='replace')
    _annotator.apply_and_annotate()
    _fig.savefig(os.path.join(analysis_figurepath, 'Core_cells_Fos_cells.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, 'Core_cells_Fos_cells.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(Conditions, analysis_resultpath, group_cell_df, os, pd):
    import scipy.stats as stats
    # statistics
    ttest_df = pd.DataFrame(columns=['Subregion', 'condition', 'pvalue', 'tvalue'])
    for core_idx in range(2):
        for _condition in Conditions:
            if _condition == 'Saline':
                continue
            a = group_cell_df[(group_cell_df.Condition == 'Saline') & (group_cell_df.Core_cells == core_idx)].Fos_cells.values
            b = group_cell_df[(group_cell_df.Condition == _condition) & (group_cell_df.Core_cells == core_idx)].Fos_cells.values
            (t, p) = stats.ttest_ind(a, b)
            p = p * (len(Conditions) - 1)
            if core_idx == 0:  # bonferroni correction
                ttest_df = ttest_df.append({'Subregion': 'Shell', 'condition': _condition, 'pvalue': p, 'tvalue': t}, ignore_index=True)
                print(f'Shell:{_condition} vs. Saline: t = {t:.2f}, p = {p:.3f}')
            else:
                ttest_df = ttest_df.append({'Subregion': 'Core', 'condition': _condition, 'pvalue': p, 'tvalue': t}, ignore_index=True)
                print(f'Core:{_condition} vs. Saline: t = {t:.2f}, p = {p:.3f}')
    ttest_df.to_csv(os.path.join(analysis_resultpath, 'ttest_result_df.csv'), index=False)
    ttest_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot the results from Fos cells different Kim_z positions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To analyze the anterior-posteiror difference of Fos cell distribution, first group z = 88,90,92,94,96 to anterior, z = 98,100,102,104 as posterior
    """)
    return


@app.cell
def _(group_cell_df, total_merge_cell_df_1):
    def _assign_region(kim_z):
        if kim_z in [88, 90, 92, 94, 96]:
            return 0
        elif kim_z in [98, 100, 102, 104, 106]:
            return 1
        else:
            return 'Unknown'
    total_merge_cell_df_1['Region_idx'] = total_merge_cell_df_1['Kim_z'].apply(_assign_region)
    group_cell_df['Region_idx'] = group_cell_df['Kim_z'].apply(_assign_region)
    return


@app.cell
def _(Annotator, Conditions, analysis_figurepath, group_cell_df, os, plt, sns):
    _filtered_data = group_cell_df[group_cell_df.Core_cells == 1]
    _filtered_data = _filtered_data[_filtered_data.Region_idx != 'Unknown']
    _order = ['anterior', 'posterior']
    (_fig, _ax) = plt.subplots(figsize=(2, 2))
    sns.stripplot(data=_filtered_data, x='Region_idx', hue='Condition', y=f'Fos_cells', hue_order=Conditions, dodge=True, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.pointplot(data=_filtered_data, hue='Condition', x='Region_idx', y=f'Fos_cells', dodge=0.8 - 0.8 / 4, hue_order=Conditions, ax=_ax, linewidth=0, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.despine()
    _ax.set_xlabel('Region')
    _ax.set_xticklabels(['Anterior', 'Posterior'])
    _ax.set_ylabel('Proportion of Fos+ cells')
    _ax.set_title('Fos+ Cells in Core')
    _ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    _pairs = [((0, 'Saline'), (0, c)) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']] + [((1, 'Saline'), (1, c)) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
    _annotator = Annotator(_ax, _pairs, data=_filtered_data, hue='Condition', x='Region_idx', y=f'Fos_cells', hue_order=Conditions)
    _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    _annotator.configure(comparisons_correction='BH', correction_format='replace')
    _annotator.apply_and_annotate()
    _fig.savefig(os.path.join(analysis_figurepath, 'Core_cells_Fos_cells_by_Kim_z.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, 'Core_cells_Fos_cells_by_Kim_z.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(Annotator, Conditions, analysis_figurepath, group_cell_df, os, plt, sns):
    _filtered_data = group_cell_df[group_cell_df.Core_cells == 0]
    _filtered_data = _filtered_data[_filtered_data.Region_idx != 'Unknown']
    _order = ['anterior', 'posterior']
    (_fig, _ax) = plt.subplots(figsize=(2, 2))
    sns.stripplot(data=_filtered_data, x='Region_idx', hue='Condition', y=f'Fos_cells', hue_order=Conditions, dodge=True, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.pointplot(data=_filtered_data, hue='Condition', x='Region_idx', y=f'Fos_cells', dodge=0.8 - 0.8 / 4, hue_order=Conditions, ax=_ax, linewidth=0, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.despine()
    _ax.set_xlabel('Region')
    _ax.set_xticklabels(['Anterior', 'Posterior'])
    _ax.set_ylabel('Proportion of Fos+ cells')
    _ax.set_title('Fos+ Cells in Shell')
    _ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    _pairs = [((0, 'Saline'), (0, c)) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']] + [((1, 'Saline'), (1, c)) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
    _annotator = Annotator(_ax, _pairs, data=_filtered_data, hue='Condition', x='Region_idx', y=f'Fos_cells', hue_order=Conditions)
    _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    _annotator.configure(comparisons_correction='BH', correction_format='replace')
    _annotator.apply_and_annotate()
    _fig.savefig(os.path.join(analysis_figurepath, 'Shell_cells_Fos_cells_by_Kim_z.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, 'Shell_cells_Fos_cells_by_Kim_z.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the overlap between Calcr and Kcnip1
    """)
    return


@app.cell
def _(np, pd, section_metadf):
    ZtoAP = pd.DataFrame(np.unique(section_metadf[['AP','Kim_z']].values,axis = 0),columns = ['AP','Kim_z']).set_index('Kim_z')
    return (ZtoAP,)


@app.cell
def _(ZtoAP, analysis_figurepath, group_cell_df, np, os, plt, sns):
    _filtered_data = group_cell_df[(group_cell_df['Kim_z'] >= 88) & (group_cell_df['Kim_z'] <= 106)]
    AP_text = [f'+{ZtoAP.loc[f, :].values[0]}' for f in np.sort(_filtered_data.Kim_z.unique())]
    _filtered_data['Region'] = _filtered_data.apply(lambda row: 'Core' if row['Core_cells'] else 'Shell' if row['Shell_cells'] else 'Unknown', axis=1)
    _filtered_data = _filtered_data[_filtered_data['Region'] != 'Unknown']
    genes_1 = ['Kcnip1', 'Calcr', 'Oxtr']
    (_fig, _axs) = plt.subplots(len(genes_1), 1, figsize=(2, len(genes_1) * 1.2), sharex=True)
    for (i, _gene) in enumerate(genes_1):
        _ax = _axs[i]
        sns.pointplot(data=_filtered_data, x='Kim_z', y=f'{_gene}_cells', hue='Region', dodge=True, ax=_ax, errwidth=0.5, linewidth=1, markersize=2, markers='o', palette=['orange', 'purple'])
        _ax.set_xlabel('from Bregma (mm)')
        _ax.set_ylabel(f'{_gene} Counts')
        sns.despine()
        _ax.set_ylim(0)
    plt.tight_layout()
    _ax.set_xticklabels(AP_text)
    _fig.savefig(os.path.join(analysis_figurepath, 'Core_Shell_Gene_Distribution.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, 'Core_Shell_Gene_Distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    return (genes_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## subset to Fos+ cells and see the overlap of Drd1/Drd2 cells
    """)
    return


@app.cell
def _(core_cell_df, count_threshold, genes_1, pd, shell_cell_df):
    _tcell_df = core_cell_df
    for (_idx, _tcell_df) in enumerate([core_cell_df, shell_cell_df]):
        for _gene in genes_1:
            if _gene == 'Fos':
                continue
            _tcell_df.loc[:, f'Fos_{_gene}_cells'] = False
            _tcell_df.loc[_tcell_df.Fos_cells & (_tcell_df[f'{_gene}_regressed_counts'] > count_threshold), f'Fos_{_gene}_cells'] = True
        _t_group_cell_df = _tcell_df.groupby(['Condition', 'ID', 'section']).mean().reset_index()
        if _idx == 0:
            Fos_group_cell_df = _t_group_cell_df
        else:
            Fos_group_cell_df = pd.concat([Fos_group_cell_df, _t_group_cell_df], axis=0)
    return (Fos_group_cell_df,)


@app.cell
def _(Fos_group_cell_df):
    # Group Kim_z into anterior and posterior regions
    def _assign_region(kim_z):
        if kim_z in [88, 90, 92, 94, 96]:
            return 0
        elif kim_z in [98, 100, 102, 104, 106]:
            return 1
        else:
            return 'Unknown'
    # Apply the function to create a new column
    Fos_group_cell_df['Region_idx'] = Fos_group_cell_df['Kim_z'].apply(_assign_region)
    return


@app.cell
def _(
    Annotator,
    Conditions,
    Fos_group_cell_df,
    analysis_figurepath,
    os,
    plt,
    sns,
):
    _filtered_data = Fos_group_cell_df[Fos_group_cell_df.Core_cells == 0]
    _gene = 'Kcnip1'
    for _gene in ['Kcnip1', 'Calcr', 'Oxtr']:
        (_fig, _ax) = plt.subplots(figsize=(1.0, 1.5))
        sns.stripplot(data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions, dodge=False, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
        sns.pointplot(data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions, ax=_ax, linewidth=0.7, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
        sns.despine()
        _ax.set_xticklabels(['Saline', 'Acute', 'Chronic', 'Early WD'], rotation=-45)
        _ax.set_ylabel('Proportion of Fos+ cells')
        _ax.set_title('AcbSh')
        _pairs = [('Saline', c) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
        _annotator = Annotator(_ax, _pairs, data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions)
        _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        _annotator.configure(comparisons_correction='BH', correction_format='replace')
        _annotator.apply_and_annotate()
        _fig.savefig(os.path.join(analysis_figurepath, f'Shell_cells_Fos_{_gene}_cells.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(os.path.join(analysis_figurepath, f'Shell_cells_Fos_{_gene}_cells.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(Conditions):
    Conditions
    return


@app.cell
def _(
    Annotator,
    Conditions,
    Fos_group_cell_df,
    analysis_figurepath,
    os,
    plt,
    sns,
):
    _filtered_data = Fos_group_cell_df[Fos_group_cell_df.Core_cells == 1]
    _gene = 'Kcnip1'
    for _gene in ['Kcnip1', 'Calcr', 'Oxtr']:
        (_fig, _ax) = plt.subplots(figsize=(1.0, 1.5))
        sns.stripplot(data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions, dodge=False, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
        sns.pointplot(data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions, ax=_ax, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
        sns.despine()
        _ax.set_xticklabels(['Saline', 'Acute', 'Chronic', 'Early WD'], rotation=-45)
        _ax.set_ylabel('Proportion of Fos+ cells')
        _ax.set_title('AcbC')
        _pairs = [('Saline', c) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
        _annotator = Annotator(_ax, _pairs, data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions)
        _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        _annotator.configure(comparisons_correction='BH', correction_format='replace')
        _annotator.apply_and_annotate()
        _fig.savefig(os.path.join(analysis_figurepath, f'Core_cells_Fos_{_gene}_cells.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(os.path.join(analysis_figurepath, f'Core_cells_Fos_{_gene}_cells.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(Annotator, Conditions, Fos_group_cell_df, plt, sns):
    _filtered_data = Fos_group_cell_df[Fos_group_cell_df.Core_cells == 0]
    _filtered_data = _filtered_data[_filtered_data.Region_idx != 'Unknown']
    _gene = 'Kcnip1'
    for _gene in ['Kcnip1', 'Calcr', 'Oxtr']:
        (_fig, _ax) = plt.subplots(figsize=(2, 1.5))
        sns.stripplot(data=_filtered_data, x='Region_idx', hue='Condition', y=f'Fos_{_gene}_cells', hue_order=Conditions, dodge=True, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
        sns.pointplot(data=_filtered_data, hue='Condition', x='Region_idx', y=f'Fos_{_gene}_cells', dodge=0.8 - 0.8 / 4, hue_order=Conditions, ax=_ax, linewidth=0, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
        _ax.legend_.remove()
        sns.despine()
        _ax.set_xticklabels(['Anterior', 'Posterior'], rotation=-45)
        _ax.set_ylabel('Proportion of Fos+ cells')
        _ax.set_title('AcbSh')
        _pairs = [((0, 'Saline'), (0, c)) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']] + [((1, 'Saline'), (1, c)) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
        _annotator = Annotator(_ax, _pairs, data=_filtered_data, hue='Condition', x='Region_idx', y=f'Fos_{_gene}_cells', hue_order=Conditions)
        _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        _annotator.configure(comparisons_correction='BH', correction_format='replace')
        _annotator.apply_and_annotate()
    return


@app.cell
def _(Annotator, Conditions, Fos_group_cell_df, plt, sns):
    _filtered_data = Fos_group_cell_df[Fos_group_cell_df.Core_cells == 1]
    _filtered_data = _filtered_data[_filtered_data.Region_idx != 'Unknown']
    _gene = 'Kcnip1'
    for _gene in ['Kcnip1', 'Calcr', 'Oxtr']:
        (_fig, _ax) = plt.subplots(figsize=(2.5, 1.5))
        sns.stripplot(data=_filtered_data, x='Region_idx', hue='Condition', y=f'Fos_{_gene}_cells', hue_order=Conditions, dodge=True, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
        sns.pointplot(data=_filtered_data, hue='Condition', x='Region_idx', y=f'Fos_{_gene}_cells', dodge=0.8 - 0.8 / 4, hue_order=Conditions, ax=_ax, linewidth=0, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
        _ax.legend_.remove()
        sns.despine()
        _ax.set_xticklabels(['Anterior', 'Posterior'], rotation=-45)
        _ax.set_ylabel('Proportion of Fos+ cells')
        _ax.set_title('AcbC')
        _pairs = [((0, 'Saline'), (0, c)) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']] + [((1, 'Saline'), (1, c)) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
        _annotator = Annotator(_ax, _pairs, data=_filtered_data, hue='Condition', x='Region_idx', y=f'Fos_{_gene}_cells', hue_order=Conditions)
        _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        _annotator.configure(comparisons_correction='BH', correction_format='replace')
        _annotator.apply_and_annotate()
    return


@app.cell
def _(
    Annotator,
    Conditions,
    Fos_group_cell_df,
    analysis_figurepath,
    os,
    plt,
    sns,
):
    _filtered_data = Fos_group_cell_df[Fos_group_cell_df.Core_cells == 1]
    _gene = 'Kcnip1'
    for _gene in ['Kcnip1', 'Calcr', 'Oxtr']:
        (_fig, _ax) = plt.subplots(figsize=(1.25, 1.5))
        sns.stripplot(data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions, dodge=False, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
        sns.pointplot(data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions, ax=_ax, linewidth=0.7, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
        sns.despine()
        _ax.set_xticklabels(['Saline', 'Acute', 'Chronic', 'Early WD'], rotation=-45)
        _ax.set_ylabel('Proportion of Fos+ cells')
        _ax.set_title('AcbC')
        _pairs = [('Saline', c) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
        _annotator = Annotator(_ax, _pairs, data=_filtered_data, x='Condition', y=f'Fos_{_gene}_cells', order=Conditions)
        _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        _annotator.configure(comparisons_correction='BH', correction_format='replace')
        _annotator.apply_and_annotate()
        _fig.savefig(os.path.join(analysis_figurepath, f'Core_cells_Fos_{_gene}_cells.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(os.path.join(analysis_figurepath, f'Core_cells_Fos_{_gene}_cells.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(Annotator, Conditions, analysis_figurepath, group_cell_df, os, plt, sns):
    _filtered_data = group_cell_df[group_cell_df.Core_cells == 1]
    (_fig, _ax) = plt.subplots(figsize=(1.25, 1.5))
    sns.stripplot(data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions, dodge=False, ax=_ax, alpha=0.5, size=2, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.pointplot(data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions, ax=_ax, linewidth=0.7, errwidth=1, markersize=5, palette=['gray', 'lime', 'orange', 'cyan'])
    sns.despine()
    _ax.set_xticklabels(['Saline', 'Acute', 'Chronic', 'Early WD'], rotation=-45)
    _ax.set_ylabel('Proportion of Fos+ cells')
    _ax.set_title('AcbC')
    _pairs = [('Saline', c) for c in ['Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine']]
    _annotator = Annotator(_ax, _pairs, data=_filtered_data, x='Condition', y='Fos_cells', order=Conditions)
    _annotator.configure(test='t-test_ind', text_format='star', loc='outside')
    _annotator.configure(comparisons_correction='BH', correction_format='replace')
    _annotator.apply_and_annotate()
    _fig.savefig(os.path.join(analysis_figurepath, 'Core_cells_Fos_cells.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, 'Core_cells_Fos_cells.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(
    Conditions,
    Fos_group_cell_df,
    analysis_figurepath,
    genes_1,
    os,
    plt,
    sns,
):
    for _condition in Conditions:
        if _condition == 'Saline':
            continue
        tconditions = ['Saline', _condition]
        (_fig, _axs) = plt.subplots(len(genes_1) - 1, 2, figsize=(4, 1.0 * (len(genes_1) - 1)), sharex=True)
        for ridx in range(2):
            region = ['Shell', 'Core'][ridx]
            tdf = Fos_group_cell_df[(Fos_group_cell_df.Core_cells == ridx) & Fos_group_cell_df.Condition.isin(tconditions)]
            for (gidx, _gene) in enumerate([f for f in genes_1 if f != 'Fos']):
                _ax = _axs[gidx, ridx]
                sns.swarmplot(data=tdf, y='Condition', x=f'Fos_{_gene}_cells', order=tconditions, dodge=True, ax=_ax, edgecolor='k', linewidth=0.5)
                sns.barplot(data=tdf, y='Condition', x=f'Fos_{_gene}_cells', order=tconditions, ax=_ax, edgecolor='k', linewidth=0.5, errwidth=1)
                sns.despine()
                _ax.set_yticks([0.5])
                _axs[gidx, 0].set_yticklabels([_gene], rotation=-0)
                _ax.set_xlabel('')
                _ax.set_ylabel('')
            _axs[0, ridx].set_title(region)
        _fig.text(0.5, 0.02, 'Proportion of Fos+ GeneX+ cells', ha='center', fontsize=12)
        _fig.savefig(os.path.join(analysis_figurepath, f'Core-Shell_Fos_subset_{tconditions[1]}_cells.pdf'), bbox_inches='tight', dpi=216)
        _fig.savefig(os.path.join(analysis_figurepath, f'Core-Shell_Fos_subset_{tconditions[1]}_cells.png'), bbox_inches='tight', dpi=216)
    return


if __name__ == "__main__":
    app.run()
