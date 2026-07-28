import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import statsmodels.stats.api as sms
    import statsmodels.stats.anova as smsanova

    return np, os, pd, plt, sns


@app.cell
def _(plt):
    from adjustText import adjust_text
    # Set matplotlib parameters for white text on transparent background
    #important for text to be detected when importing saved figures into illustrator
    plt.rcParams.update({'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'legend.facecolor': 'none', 'legend.edgecolor': 'none', 'text.color': 'black', 'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})  # Transparent figure background  # Transparent axes background  # White axes edge color  # White axis labels  # White tick labels  # Transparent legend background  # Transparent legend edgecolor  # White text color
    return


@app.cell
def _(os, pd):
    from pathlib import Path
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    GROUP = DATA_ROOT / "07_nac_histology_hcr"
    assert GROUP.exists(), f"DATA_ROOT not found: {DATA_ROOT}. Set OPIOID_DATA_ROOT."

    metapath = str(GROUP)                 # meta (analysis_master.xlsx, Opioid_pPDH_meta.csv) in group folder
    analysis_resultpath = str(GROUP)      # processed master_df / stat CSVs here
    analysis_figurepath = "../figures"    # rendered panels -> repo
    os.makedirs(analysis_figurepath, exist_ok=True)

    # raw QuPath per-cell detection output (large); set OPIOID_QUPATH_ROOT, else expect it under the group
    rawdatapath = os.environ.get("OPIOID_QUPATH_ROOT", str(GROUP))  # QuPath tree lives directly in the group folder
    cell_quantification_roundness_path = os.path.join(rawdatapath, "cell_quantification_roundness")

    metadf = pd.read_excel(os.path.join(metapath, "analysis_master.xlsx"))
    animal_metadf = pd.read_csv(os.path.join(metapath, "Opioid_pPDH_meta.csv"))
    return (
        analysis_figurepath,
        analysis_resultpath,
        cell_quantification_roundness_path,
        metadf,
        metapath,
        rawdatapath,
    )


@app.cell
def _():
    # channel names
    channel_dict = {'FITC':'MOR',
                    'Cy3':'PPD',
                    'Cy5':'c-Fos'}

    # Define all 8 type mappings (A, B, C) -> type name
    cell_type_map = {
        (True,  True,  True ): 'MOR+ PPD+ c-Fos+',
        (True,  True,  False): 'MOR+ PPD+ c-Fos-',
        (True,  False, True ): 'MOR+ PPD- c-Fos+',
        (True,  False, False): 'MOR+ PPD- c-Fos-',
        (False, True,  True ): 'MOR- PPD+ c-Fos+',
        (False, True,  False): 'MOR- PPD+ c-Fos-',
        (False, False, True ): 'MOR- PPD- c-Fos+',
        (False, False, False): 'MOR- PPD- c-Fos-',
    }

    condition_colors = ['gray','green']
    channels = list(channel_dict.keys())  # ['FITC','Cy3','Cy5'] (was undefined in the original)
    return cell_type_map, channel_dict, channels, condition_colors


@app.cell
def _(cell_quantification_roundness_path, metadf, os, pd):
    # set an ROI name 
    ROI_names = ['AcbC', 'AcbSh']
    for _ROI_name in ROI_names:
        ROI_path = os.path.join(cell_quantification_roundness_path, _ROI_name)
        _files = os.listdir(ROI_path)  # set the path to the ROI folder
        for (_idx, fname) in enumerate(_files):
            imgdate = fname.split('.vsi')[0]  # get the list of files in the ROI folder
            annotation_id = fname.split('__')[-1].replace('.csv', '')
            section = fname.split('Cy5_')[1][:2]
            section = int(section)  # create a new section meta data file
            true_imgfname = imgdate + '.vsi'
            true_fname = f'{imgdate}_{section}'
            if len(metadf[metadf['img_fname'] == true_fname]) != 0:
                section_meta = metadf[metadf['img_fname'] == true_fname]  # this is a two digit number (e.g. 02), change to single digit
                (subject, zposition) = section_meta[['subject', 'zposition']].values[0]
            else:
                print('no metadata found for', true_fname)
                subject = 'unknown'
                zposition = 'unknown'  #print("processing",true_fname)
            temp_section_meta = pd.DataFrame({'fname': fname, 'true_imgfname': true_imgfname, 'img_fname': true_fname, 'section_id': section, 'subject': subject, 'zposition': zposition, 'annotation_id': annotation_id, 'region': _ROI_name}, index=[0])
            if (_idx == 0) & (_ROI_name == ROI_names[0]):  # use the true_fname to get the section metadata
                total_section_meta = temp_section_meta
            else:
                total_section_meta = pd.concat([total_section_meta, temp_section_meta])  # create a new section meta data file
    return ROI_names, ROI_path, fname, total_section_meta


@app.cell
def _():
    # write the section meta data
    from datetime import datetime
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = '20260513_001335'
    #total_section_meta.to_csv(os.path.join(metapath, f"section_meta_{timestamp}.csv"),index=False)
    return (timestamp,)


@app.cell
def _(ROI_path, fname, os, pd):
    pd.read_csv(os.path.join(ROI_path, fname), on_bad_lines='skip')
    return


@app.cell
def _(ROI_path, cell_type_map, fname, os, pd):
    # ── cell classification (mirrors quantify_cell_df logic) ──────────────────
    def classify_cells(df, Cy3_threshold=9000, FITC_threshold='auto', Cy5_threshold='auto'):
        if FITC_threshold != None:
            df['FITC_pos'] = (df['FITC_round_pos'] == True) & (df['FITC_thr_pos'] == True)  # FITC (MOR+)
        elif FITC_threshold == 'auto':
            FITC_threshold = df['FITC_mean_corr'].mean() + df['FITC_mean_corr'].std() * 2
            df['FITC_pos'] = (df['FITC_round_pos'] == True) & (df['FITC_mean_corr'] > FITC_threshold)
        else:
            df['FITC_pos'] = (df['FITC_round_pos'] == True) & (df['FITC_mean_corr'] > FITC_threshold)
        if Cy5_threshold != None:
            df['Cy5_pos'] = (df['Cy5_round_pos'] == True) & (df['Cy5_thr_pos'] == True)
        elif Cy5_threshold == 'auto':
            Cy5_threshold = df['Cy5_mean_corr'].mean() + df['Cy5_mean_corr'].std() * 2  # Cy5 (c-Fos+)
            df['Cy5_pos'] = (df['Cy5_round_pos'] == True) & (df['Cy5_mean_corr'] > Cy5_threshold)
        else:
            df['Cy5_pos'] = (df['Cy5_round_pos'] == True) & (df['Cy5_mean_corr'] > Cy5_threshold)
        if Cy3_threshold == 'auto':
            Cy3_threshold = df['Cy3_meas'].mean() + df['Cy3_meas'].std() * 4
        elif df['Cy3_meas'].mean() > Cy3_threshold:
            print('The mean Cy3 signal is greater than the threshold')
            Cy3_threshold = df['Cy3_meas'].mean() + df['Cy3_meas'].std() * 4
        df['Cy3_pos'] = df['Cy3_meas'] > Cy3_threshold  # Cy3 (PPD+)
        return df

    def quantify_cell_df(cell_df_path, Cy3_threshold=9000, FITC_threshold='auto', Cy5_threshold='auto'):
        cell_df = pd.read_csv(os.path.join(ROI_path, fname), on_bad_lines='skip')
        cell_df = pd.read_csv(os.path.join(ROI_path, fname), header=cell_df.shape[0] + 1)
        if FITC_threshold != None:
            cell_df['FITC_pos'] = (cell_df['FITC_round_pos'] == True) & (cell_df['FITC_thr_pos'] == True)
        elif FITC_threshold == 'auto':
            FITC_threshold = cell_df['FITC_mean_corr'].mean() + cell_df['FITC_mean_corr'].std() * 2
    # read the associated dataframe
            cell_df['FITC_pos'] = (cell_df['FITC_round_pos'] == True) & (cell_df['FITC_mean_corr'] > FITC_threshold)
        else:  # error_bad_lines=False (pandas <1.3) or on_bad_lines='warn' (pandas ≥1.3)
            cell_df['FITC_pos'] = (cell_df['FITC_round_pos'] == True) & (cell_df['FITC_mean_corr'] > FITC_threshold)
        if Cy5_threshold != None:  #print(df.shape)        # (rows, cols) → tells you which header count won
            cell_df['Cy5_pos'] = (cell_df['Cy5_round_pos'] == True) & (cell_df['Cy5_thr_pos'] == True)  #print(df.columns.tolist())
        elif Cy5_threshold == 'auto':
            Cy5_threshold = cell_df['Cy5_mean_corr'].mean() + cell_df['Cy5_mean_corr'].std() * 2
            cell_df['Cy5_pos'] = (cell_df['Cy5_round_pos'] == True) & (cell_df['Cy5_mean_corr'] > Cy5_threshold)  # identify cells that are both round and threshold positive for FITC
        else:
            cell_df['Cy5_pos'] = (cell_df['Cy5_round_pos'] == True) & (cell_df['Cy5_mean_corr'] > Cy5_threshold)
        if Cy3_threshold == 'auto':
            Cy3_threshold = cell_df['Cy3_meas'].mean() + cell_df['Cy3_meas'].std() * 2
        elif cell_df['Cy3_meas'].mean() > Cy3_threshold:
            print('The mean Cy3 signal is greater than the threshold')
            Cy3_threshold = cell_df['Cy3_meas'].mean() + cell_df['Cy3_meas'].std() * 2
        cell_df['Cy3_pos'] = cell_df['Cy3_meas'] > Cy3_threshold
        cols = ['FITC_pos', 'Cy3_pos', 'Cy5_pos']
        cell_df['cell_type'] = cell_df[cols].apply(tuple, axis=1).map(cell_type_map)
        mean_columns = ['centroid_x_um', 'centroid_y_um', 'FITC_mean_corr', 'Cy3_meas', 'Cy5_mean_corr']
        counts = cell_df['cell_type'].value_counts().reindex(cell_type_map.values(), fill_value=0).reset_index()
        counts.columns = ['cell_type', 'count']
        counts['percentage'] = (counts['count'] / len(cell_df) * 100).round(2)
        counts = counts.set_index('cell_type')  # Cy3 threshold logic
        means = cell_df.groupby('cell_type')[mean_columns].mean()
        data_df = counts.join(means)
        wildcard_groups = {'MOR+': cell_df['FITC_pos'], 'PPD+': cell_df['Cy3_pos'], 'c-Fos+': cell_df['Cy5_pos'], 'MOR+ PPD+': cell_df['FITC_pos'] & cell_df['Cy3_pos'], 'MOR- PPD+': ~cell_df['FITC_pos'] & cell_df['Cy3_pos'], 'MOR+ PPD-': cell_df['FITC_pos'] & ~cell_df['Cy3_pos'], 'MOR+ c-Fos+': cell_df['FITC_pos'] & cell_df['Cy5_pos'], 'MOR+ c-Fos-': cell_df['FITC_pos'] & ~cell_df['Cy5_pos'], 'MOR- c-Fos+': ~cell_df['FITC_pos'] & cell_df['Cy5_pos'], 'PPD+ c-Fos+': cell_df['Cy3_pos'] & cell_df['Cy5_pos'], 'PPD+ c-Fos-': cell_df['Cy3_pos'] & ~cell_df['Cy5_pos'], 'PPD- c-Fos+': ~cell_df['Cy3_pos'] & cell_df['Cy5_pos'], 'all cells': pd.Series(True, index=cell_df.index)}
        wildcard_rows = []
        for (label, mask) in wildcard_groups.items():
            _subset = cell_df[mask]
            n = len(_subset)
            _row = {'count': n, 'percentage': round(n / len(cell_df) * 100, 2)}
            _row.update(_subset[mean_columns].mean().to_dict())
            wildcard_rows.append(pd.Series(_row, name=label))  # Create cell_type column (8 specific types)
        wildcard_df = pd.DataFrame(wildcard_rows)
        wildcard_df['count'] = wildcard_df['count'].astype(int)
        data_df = pd.concat([data_df, wildcard_df]).reset_index().rename(columns={'index': 'cell_type'})
        return data_df  # --- Quantify 8 specific types ---  # --- Wildcard groupings ---  # Single markers (one True, any any)  # Pairs (two True, any)  # Combine

    return classify_cells, quantify_cell_df


@app.cell
def _(
    ROI_names,
    cell_quantification_roundness_path,
    os,
    pd,
    total_section_meta,
):
    master_df = pd.DataFrame()
    for _ROI_name in ROI_names:
        ROI_path_1 = os.path.join(cell_quantification_roundness_path, _ROI_name)
        _files = [_f for _f in os.listdir(ROI_path_1) if _f.endswith('.csv')]
        for fname_1 in _files:
            if _ROI_name != total_section_meta[total_section_meta.fname == fname_1]['region'].values[0]:
                print(fname_1, _ROI_name)
    return


@app.cell
def _(ROI_names, cell_quantification_roundness_path, os, pd, quantify_cell_df):
    master_df_1 = pd.DataFrame()
    for _ROI_name in ROI_names:
        ROI_path_2 = os.path.join(cell_quantification_roundness_path, _ROI_name)
        _files = [_f for _f in os.listdir(ROI_path_2) if _f.endswith('.csv')]
        for (_idx, fname_2) in enumerate(_files):
            data_df = quantify_cell_df(os.path.join(ROI_path_2, fname_2), Cy3_threshold='auto', FITC_threshold='auto', Cy5_threshold='auto')
            data_df['fname'] = fname_2
            data_df['region'] = _ROI_name
            master_df_1 = pd.concat([master_df_1, data_df], axis=0)
    return (master_df_1,)


@app.cell
def _(analysis_resultpath, master_df_1, os, timestamp):
    master_df_1.to_csv(os.path.join(analysis_resultpath, f'master_df_{timestamp}.csv'), index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    go back to Qupath to annotate the un annotated section
    """)
    return


@app.cell
def _(metapath, os, pd):
    total_section_meta_1 = pd.read_csv(os.path.join(metapath, 'section_meta_20260513_001335.csv'))
    return (total_section_meta_1,)


@app.cell
def _(total_section_meta_1):
    if len(total_section_meta_1[total_section_meta_1.zposition == 'unknown'].fname.unique()) > 0:
        print(total_section_meta_1[total_section_meta_1.zposition == 'unknown'].fname.unique())
    else:
        print('no unannotated section')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clean up the data
    """)
    return


@app.cell
def _(analysis_resultpath, metapath, os, pd):
    total_section_meta_2 = pd.read_csv(os.path.join(metapath, 'section_meta_20260513_001335.csv'))
    subject_df = pd.read_csv(os.path.join(metapath, 'Opioid_pPDH_meta.csv'))
    master_df_2 = pd.read_csv(os.path.join(analysis_resultpath, f'master_df_20260513_001335.csv'))
    return master_df_2, subject_df, total_section_meta_2


@app.cell
def _(master_df_2, subject_df, total_section_meta_2):
    master_df_3 = master_df_2.merge(total_section_meta_2, left_on=['fname', 'region'], right_on=['fname', 'region'], how='left')
    master_df_3 = master_df_3.merge(subject_df, left_on='subject', right_on='ID', how='left')
    return (master_df_3,)


@app.cell
def _():
    def remove_outliers(df, column, method='iqr', threshold=1.5, groupby=None):
        """
        Remove outliers from a column, optionally within groups.

        Parameters
        ----------
        df        : DataFrame
        column    : column to check for outliers
        method    : 'iqr' or 'zscore'
        threshold : IQR multiplier (default 1.5) or Z-score cutoff (default 3)
        groupby   : list of columns to compute outliers within (e.g. ['region', 'Condition'])
        """
        if groupby:
            mask = df.groupby(groupby)[column].transform(lambda x: _outlier_mask(x, method, threshold))
        else:
            mask = _outlier_mask(df[column], method, threshold)

        n_removed = (~mask).sum()
        print(f"Removed {n_removed} outliers ({n_removed/len(df)*100:.1f}%) from '{column}'")
        return df[mask].copy()


    def _outlier_mask(s, method, threshold):
        """Returns a boolean mask — True = keep."""
        if method == 'iqr':
            q1, q3 = s.quantile(0.05), s.quantile(0.95)
            iqr = q3 - q1
            return s.between(q1 - threshold * iqr, q3 + threshold * iqr)
        elif method == 'zscore':
            return (s - s.mean()).abs() / s.std() < threshold
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

    return (remove_outliers,)


@app.cell
def _(pd):
    def bin_zposition(df, n_bins=6):
        """
        Bin zposition into AP axis bins, ordered largest to smallest (anterior → posterior).

        Parameters
        ----------
        df     : DataFrame with a 'zposition' column
        n_bins : number of bins (default 6)
        """
        df = df.copy()
        df['AP_bin'] = pd.cut(df['zposition'], bins=n_bins)
        _bin_order = sorted(df['AP_bin'].unique(), reverse=True)
        bin_labels = [f'{b.left:.2f}–{b.right:.2f}' for b in _bin_order]
        df['AP_label'] = pd.Categorical(df['AP_bin'].map({b: label for (b, label) in zip(_bin_order, bin_labels)}), categories=bin_labels, ordered=True)
        print(f"Binned zposition ({df['zposition'].min():.2f}–{df['zposition'].max():.2f}) into {n_bins} bins:")
        print(df.groupby('AP_label', observed=True).size().to_string())
        return (df, _bin_order, bin_labels)

    return (bin_zposition,)


@app.cell
def _(master_df_3):
    # make a cut off for the zposition, only use zpo 1.4-0.7
    master_df_4 = master_df_3[(master_df_3.zposition >= 0.7) & (master_df_3.zposition <= 1.4)]
    return (master_df_4,)


@app.cell
def _(bin_zposition, master_df_4):
    n_bins = 3
    (master_df_5, _bin_order, bin_labels) = bin_zposition(master_df_4, n_bins=n_bins)
    return (master_df_5,)


@app.cell
def _(total_section_meta_2):
    total_section_meta_3 = total_section_meta_2[(total_section_meta_2.zposition >= 0.7) & (total_section_meta_2.zposition <= 1.4)]
    return (total_section_meta_3,)


@app.cell
def _(bin_zposition, total_section_meta_3):
    n_bins_1 = 3
    (total_section_meta_4, _bin_order, bin_labels_1) = bin_zposition(total_section_meta_3, n_bins=n_bins_1)
    return bin_labels_1, n_bins_1


@app.cell
def _(channel_dict, channels, master_df_5, pd, remove_outliers):
    filtered_master_df = pd.DataFrame({'cell_type': []})
    for (cidx, channel) in enumerate(channels):
        channel_name = channel_dict[channel]
        target_cell_types = [_f for _f in master_df_5.cell_type.unique() if channel_name + '+' in _f]
        print(target_cell_types)
        column_name = ['FITC_mean_corr', 'Cy3_meas', 'Cy5_mean_corr'][cidx]
        for _target_cell_type in target_cell_types:
            if _target_cell_type in filtered_master_df['cell_type'].unique():
                tfiltered_master_df = remove_outliers(filtered_master_df[filtered_master_df.cell_type == _target_cell_type], column=column_name)
            else:
                tfiltered_master_df = remove_outliers(master_df_5[master_df_5.cell_type == _target_cell_type], column=column_name)
            filtered_master_df = pd.concat([filtered_master_df, tfiltered_master_df], axis=0)
    return (filtered_master_df,)


@app.cell
def _(analysis_resultpath, filtered_master_df, master_df_5, os, timestamp):
    # write the results
    master_df_5.to_csv(os.path.join(analysis_resultpath, f'master_df_with_metadata.csv'), index=False)
    filtered_master_df.to_csv(os.path.join(analysis_resultpath, f'filtered_master_df_{timestamp}.csv'), index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting
    """)
    return


@app.cell
def _(analysis_resultpath, os, pd, timestamp):
    # read the results
    master_d = pd.read_csv(os.path.join(analysis_resultpath, f'master_df_with_metadata.csv'), index_col=False)
    filtered_master_df_1 = pd.read_csv(os.path.join(analysis_resultpath, f'filtered_master_df_{timestamp}.csv'), index_col=False)
    return (filtered_master_df_1,)


@app.cell
def _(bin_labels_1, filtered_master_df_1):
    sec = filtered_master_df_1.drop_duplicates('img_fname')  # one row per section

    def section_count_text(condition, label=None):
        sub = sec[sec.Condition == condition]
        parts = [f"{label or condition}, n = {sub['subject'].nunique()} mice"]
        parts = parts + [f"{b}: n = {sub[sub.AP_label == b]['img_fname'].nunique()} sections" for b in bin_labels_1]
        return ', '.join(parts) + '.'
    text = '\n'.join((section_count_text(c, l) for (c, l) in [('Saline', 'Saline'), ('Acute_Morphine', 'Acute Morphine')]))
    print(text)
    return


@app.cell
def _(master_df_5):
    filtered_master_df_2 = master_df_5
    return (filtered_master_df_2,)


@app.cell
def _():
    from statannotations.Annotator import Annotator

    return (Annotator,)


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    analysis_resultpath,
    bin_labels_1,
    condition_colors,
    filtered_master_df_2,
    n_bins_1,
    os,
    pd,
    plt,
    sns,
):
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests
    _target_cell_type = 'c-Fos+'

    def _fmt_p(p):
        if p < 0.01:
            (m, e) = f'{p:.2e}'.split('e')
            return f'{m} x 10{int(e)}'
        return f'{p:.2f}'
    (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
    _ttest_results = []
    for (_idx, _region) in enumerate(ROI_names):
        _ax = _axs[_idx]
        _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
        sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
        sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
        sns.despine()
        _ax.set_title(_region)
        _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
        _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
        _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
        _annotator.configure(test='t-test_ind', comparisons_correction='Benjamini-Hochberg', text_format='star', loc='inside', verbose=2)
        _annotator.apply_and_annotate()
        _region_rows = []
        for _f in bin_labels_1:
            _sal = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Saline'), 'count'].dropna()
            _mor = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Acute_Morphine'), 'count'].dropna()
            (_t_stat, _p_raw) = ttest_ind(_sal, _mor)
            _df_ = len(_sal) + len(_mor) - 2
            _region_rows.append({'cell_type': _target_cell_type, 'region': _region, 'AP_bin': _f, 'n_saline': len(_sal), 'n_morphine': len(_mor), 'mean_saline': _sal.mean(), 'mean_morphine': _mor.mean(), 't_stat': _t_stat, 'df': _df_, 'pvalue_raw': _p_raw})
        (_rej, _p_corr, _, _) = multipletests([_r['pvalue_raw'] for _r in _region_rows], alpha=0.05, method='fdr_bh')
        for (_r, _pc, _rj) in zip(_region_rows, _p_corr, _rej):
            _r['pvalue_bh'] = _pc
            _r['significant'] = bool(_rj)
        _ttest_results.extend(_region_rows)
        _ax.set_ylabel(f'{_target_cell_type} cells')
    _fig.savefig(os.path.join(analysis_figurepath, 'c-Fos_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, 'c-Fos_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
    _ttest_df = pd.DataFrame(_ttest_results)
    out_csv = os.path.join(analysis_resultpath, 'c-Fos_count_by_region_and_condition_ttest.csv')
    _ttest_df.to_csv(out_csv, index=False)
    report_lines = []
    for _region in ROI_names:
        sub = _ttest_df[_ttest_df.region == _region]
        comps = ' '.join((f'{_r.AP_bin}: P = {_fmt_p(_r.pvalue_bh)}, t({_r.df}) = {_r.t_stat:.2f}.' for _r in sub.itertuples()))
        report_lines.append(f'{_region}, Saline vs. Acute Morphine — {comps}')
    _report = '\n'.join(report_lines)
    with open(os.path.join(analysis_resultpath, 'c-Fos_count_ttest_report.txt'), 'w') as _fh:
        _fh.write(_report)
    print(_report)
    _ttest_df
    return multipletests, ttest_ind


@app.cell
def _():
    '''target_cell_type = 'c-Fos+'

    fig,axs = plt.subplots(1,2,figsize = (5,1.5),sharex = True,sharey = True)

    for idx,region in enumerate(ROI_names):
        ax = axs[idx]
        data = filtered_master_df[(filtered_master_df.cell_type == target_cell_type)&(filtered_master_df.region == region)]
        sns.stripplot(data = data,\
            x = 'AP_label',y = 'count',hue = 'Condition', order = bin_labels,\
                dodge = True,palette = condition_colors,alpha=.25,ax = ax,legend = False)

        sns.pointplot(data = data,\
            x = 'AP_label',y = 'count',hue = 'Condition', dodge=.8 - .8 / 2,order = bin_labels,palette = condition_colors,
            markers="o", markersize=4, linestyle="none",errwidth = .8,ax = ax)
        sns.despine()
        ax.set_title(region)
        ax.set_xticklabels(['anterior'] + [''] * (n_bins-2) +['posterior'])

        pairs = [((f,'Saline'),(f,'Acute_Morphine')) for f in bin_labels]
        annotator = Annotator(ax,pairs = pairs, data = data,

        x='AP_label', y='count', hue='Condition', order=bin_labels)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
        annotator.apply_and_annotate()
        ax.set_ylabel(f'{target_cell_type} cells')
    fig.savefig(os.path.join(analysis_figurepath, f"c-Fos_count_by_region_and_condition.png"),dpi = 300,bbox_inches = 'tight')
    fig.savefig(os.path.join(analysis_figurepath, f"c-Fos_count_by_region_and_condition.pdf"),dpi = 300,bbox_inches = 'tight')'''
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    bin_labels_1,
    condition_colors,
    filtered_master_df_2,
    n_bins_1,
    os,
    plt,
    sns,
):
    _target_cell_type = 'MOR+'
    (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
    for (_idx, _region) in enumerate(ROI_names):
        _ax = _axs[_idx]
        _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
        sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
        sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
        sns.despine()
        _ax.set_title(_region)
        _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
        _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
        _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
        _annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
        _annotator.apply_and_annotate()
        _ax.set_ylabel(f'{_target_cell_type} cells')
    _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type[:-1]}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type[:-1]}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    analysis_resultpath,
    bin_labels_1,
    condition_colors,
    filtered_master_df_2,
    multipletests,
    n_bins_1,
    os,
    pd,
    plt,
    sns,
    ttest_ind,
):
    _target_cell_type = 'MOR+'
    _name = _target_cell_type[:-1]

    def _fmt_p(p):
        if p < 0.01:
            (m, e) = f'{p:.2e}'.split('e')
            return f'{m} x 10{int(e)}'
        return f'{p:.2f}'
    (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
    _ttest_results = []
    for (_idx, _region) in enumerate(ROI_names):
        _ax = _axs[_idx]
        _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
        sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
        sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
        sns.despine()
        _ax.set_title(_region)
        _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
        _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
        _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
        _annotator.configure(test='t-test_ind', comparisons_correction='Benjamini-Hochberg', text_format='star', loc='inside', verbose=2)
        _annotator.apply_and_annotate()
        _region_rows = []
        for _f in bin_labels_1:
            _sal = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Saline'), 'count'].dropna()
            _mor = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Acute_Morphine'), 'count'].dropna()
            (_t_stat, _p_raw) = ttest_ind(_sal, _mor)
            _df_ = len(_sal) + len(_mor) - 2
            _region_rows.append({'cell_type': _target_cell_type, 'region': _region, 'AP_bin': _f, 'n_saline': len(_sal), 'n_morphine': len(_mor), 'mean_saline': _sal.mean(), 'mean_morphine': _mor.mean(), 't_stat': _t_stat, 'df': _df_, 'pvalue_raw': _p_raw})
        (_rej, _p_corr, _, _) = multipletests([_r['pvalue_raw'] for _r in _region_rows], alpha=0.05, method='fdr_bh')
        for (_r, _pc, _rj) in zip(_region_rows, _p_corr, _rej):
            _r['pvalue_bh'] = _pc
            _r['significant'] = bool(_rj)
        _ttest_results.extend(_region_rows)
        _ax.set_ylabel(f'{_target_cell_type} cells')
    _fig.savefig(os.path.join(analysis_figurepath, f'{_name}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, f'{_name}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
    _ttest_df = pd.DataFrame(_ttest_results)
    _ttest_df.to_csv(os.path.join(analysis_resultpath, f'{_name}_count_by_region_and_condition_ttest.csv'), index=False)
    _report = '\n'.join((f'{_region}, Saline vs. Acute Morphine — ' + ' '.join((f'{_r.AP_bin}: P = {_fmt_p(_r.pvalue_bh)}, t({_r.df}) = {_r.t_stat:.2f}.' for _r in _ttest_df[_ttest_df.region == _region].itertuples())) for _region in ROI_names))
    with open(os.path.join(analysis_resultpath, f'{_name}_count_ttest_report.txt'), 'w') as _fh:
        _fh.write(_report)
    print(_report)
    _ttest_df
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    bin_labels_1,
    condition_colors,
    filtered_master_df_2,
    n_bins_1,
    os,
    plt,
    sns,
):
    _target_cell_type = 'PPD+'
    (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
    for (_idx, _region) in enumerate(ROI_names):
        _ax = _axs[_idx]
        _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
        sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
        sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
        sns.despine()
        _ax.set_title(_region)
        _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
        _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
        _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
        _annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
        _annotator.apply_and_annotate()
        _ax.set_ylabel(f'{_target_cell_type} cells')
    _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type[:-1]}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type[:-1]}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    analysis_resultpath,
    bin_labels_1,
    condition_colors,
    filtered_master_df_2,
    multipletests,
    n_bins_1,
    os,
    pd,
    plt,
    sns,
    ttest_ind,
):
    _target_cell_type = 'PPD+'
    _name = _target_cell_type[:-1]

    def _fmt_p(p):
        if p < 0.01:
            (m, e) = f'{p:.2e}'.split('e')
            return f'{m} x 10{int(e)}'
        return f'{p:.2f}'
    (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
    _ttest_results = []
    for (_idx, _region) in enumerate(ROI_names):
        _ax = _axs[_idx]
        _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
        sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
        sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
        sns.despine()
        _ax.set_title(_region)
        _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
        _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
        _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
        _annotator.configure(test='t-test_ind', comparisons_correction='Benjamini-Hochberg', text_format='star', loc='inside', verbose=2)
        _annotator.apply_and_annotate()
        _region_rows = []
        for _f in bin_labels_1:
            _sal = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Saline'), 'count'].dropna()
            _mor = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Acute_Morphine'), 'count'].dropna()
            (_t_stat, _p_raw) = ttest_ind(_sal, _mor)
            _df_ = len(_sal) + len(_mor) - 2
            _region_rows.append({'cell_type': _target_cell_type, 'region': _region, 'AP_bin': _f, 'n_saline': len(_sal), 'n_morphine': len(_mor), 'mean_saline': _sal.mean(), 'mean_morphine': _mor.mean(), 't_stat': _t_stat, 'df': _df_, 'pvalue_raw': _p_raw})
        (_rej, _p_corr, _, _) = multipletests([_r['pvalue_raw'] for _r in _region_rows], alpha=0.05, method='fdr_bh')
        for (_r, _pc, _rj) in zip(_region_rows, _p_corr, _rej):
            _r['pvalue_bh'] = _pc
            _r['significant'] = bool(_rj)
        _ttest_results.extend(_region_rows)
        _ax.set_ylabel(f'{_target_cell_type} cells')
    _fig.savefig(os.path.join(analysis_figurepath, f'{_name}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
    _fig.savefig(os.path.join(analysis_figurepath, f'{_name}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
    _ttest_df = pd.DataFrame(_ttest_results)
    _ttest_df.to_csv(os.path.join(analysis_resultpath, f'{_name}_count_by_region_and_condition_ttest.csv'), index=False)
    _report = '\n'.join((f'{_region}, Saline vs. Acute Morphine, ' + ' '.join((f'{_r.AP_bin}: P = {_fmt_p(_r.pvalue_bh)}, t({_r.df}) = {_r.t_stat:.2f}.' for _r in _ttest_df[_ttest_df.region == _region].itertuples())) for _region in ROI_names))
    with open(os.path.join(analysis_resultpath, f'{_name}_count_ttest_report.txt'), 'w') as _fh:
        _fh.write(_report)
    print(_report)
    _ttest_df
    return


@app.cell
def _(filtered_master_df_2):
    filtered_master_df_2['cell_type'].unique()
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    bin_labels_1,
    cell_type_map,
    condition_colors,
    filtered_master_df_2,
    n_bins_1,
    os,
    plt,
    sns,
):
    _target_cell_type = 'PPD+'
    for _target_cell_type in cell_type_map.values():
        (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
        for (_idx, _region) in enumerate(ROI_names):
            _ax = _axs[_idx]
            _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
            sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
            sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
            sns.despine()
            _ax.set_title(_region)
            _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
            _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
            _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
            _annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
            _annotator.apply_and_annotate()
            _ax.set_ylabel(f'{_target_cell_type} cells')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    analysis_resultpath,
    bin_labels_1,
    cell_type_map,
    condition_colors,
    filtered_master_df_2,
    multipletests,
    n_bins_1,
    os,
    pd,
    plt,
    sns,
    ttest_ind,
):
    def _fmt_p(p):
        if p < 0.01:
            (m, e) = f'{p:.2e}'.split('e')
            return f'{m} x 10{int(e)}'
        return f'{p:.2f}'
    _all_results = []
    _report_blocks = []
    for _target_cell_type in cell_type_map.values():
        _name = _target_cell_type.replace(' ', '_')
        (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
        for (_idx, _region) in enumerate(ROI_names):
            _ax = _axs[_idx]
            _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
            sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
            sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
            sns.despine()
            _ax.set_title(_region)
            _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
            _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
            _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
            _annotator.configure(test='t-test_ind', comparisons_correction='Benjamini-Hochberg', text_format='star', loc='inside', verbose=2)
            _annotator.apply_and_annotate()
            _region_rows = []
            for _f in bin_labels_1:
                _sal = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Saline'), 'count'].dropna()
                _mor = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Acute_Morphine'), 'count'].dropna()
                (_t_stat, _p_raw) = ttest_ind(_sal, _mor)
                _df_ = len(_sal) + len(_mor) - 2
                _region_rows.append({'cell_type': _target_cell_type, 'region': _region, 'AP_bin': _f, 'n_saline': len(_sal), 'n_morphine': len(_mor), 'mean_saline': _sal.mean(), 'mean_morphine': _mor.mean(), 't_stat': _t_stat, 'df': _df_, 'pvalue_raw': _p_raw})
            (_rej, _p_corr, _, _) = multipletests([_r['pvalue_raw'] for _r in _region_rows], alpha=0.05, method='fdr_bh')
            for (_r, _pc, _rj) in zip(_region_rows, _p_corr, _rej):
                _r['pvalue_bh'] = _pc
                _r['significant'] = bool(_rj)
            _all_results.extend(_region_rows)
            _ax.set_ylabel(f'{_target_cell_type} cells')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_name}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_name}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
        _rows = [_r for _r in _all_results if _r['cell_type'] == _target_cell_type]
        _block = f'=== {_target_cell_type} ===\n' + '\n'.join((f'{_region}, Saline vs. Acute Morphine, ' + ' '.join((f"{_r['AP_bin']}: P = {_fmt_p(_r['pvalue_bh'])}, t({_r['df']}) = {_r['t_stat']:.2f}." for _r in _rows if _r['region'] == _region)) for _region in ROI_names))
        _report_blocks.append(_block)
        print(_block, '\n')
    _ttest_df = pd.DataFrame(_all_results)
    _ttest_df.to_csv(os.path.join(analysis_resultpath, 'count_by_region_and_condition_ttest_ALL.csv'), index=False)
    with open(os.path.join(analysis_resultpath, 'count_ttest_report_ALL.txt'), 'w') as _fh:
        _fh.write('\n\n'.join(_report_blocks))
    _ttest_df
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    bin_labels_1,
    cell_type_map,
    condition_colors,
    filtered_master_df_2,
    n_bins_1,
    os,
    plt,
    sns,
):
    _target_cell_type = 'PPD+'
    for _target_cell_type in cell_type_map.values():
        (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
        for (_idx, _region) in enumerate(ROI_names):
            _ax = _axs[_idx]
            _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
            sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
            sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
            sns.despine()
            _ax.set_title(_region)
            _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
            _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
            _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
            _annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
            _annotator.apply_and_annotate()
            _ax.set_ylabel(f'{_target_cell_type} cells')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(filtered_master_df_2):
    filtered_master_df_2.cell_type.unique()
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    bin_labels_1,
    condition_colors,
    filtered_master_df_2,
    n_bins_1,
    os,
    plt,
    sns,
):
    for _target_cell_type in ['MOR+ PPD+', 'MOR- PPD+', 'MOR+ PPD-', 'MOR+ c-Fos+', 'MOR+ c-Fos-', 'MOR- c-Fos+', 'PPD+ c-Fos+', 'PPD+ c-Fos-', 'PPD- c-Fos+', 'all cells']:
        print('Plotting', _target_cell_type)
        (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
        for (_idx, _region) in enumerate(ROI_names):
            _ax = _axs[_idx]
            _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
            sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
            sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
            sns.despine()
            _ax.set_title(_region)
            _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
            _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
            _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
            _annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
            _annotator.apply_and_annotate()
            _ax.set_ylabel(f'{_target_cell_type} cells')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_target_cell_type}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell
def _(
    Annotator,
    ROI_names,
    analysis_figurepath,
    analysis_resultpath,
    bin_labels_1,
    condition_colors,
    filtered_master_df_2,
    multipletests,
    n_bins_1,
    os,
    pd,
    plt,
    sns,
    ttest_ind,
):
    def _fmt_p(p):
        if p < 0.01:
            (m, e) = f'{p:.2e}'.split('e')
            return f'{m} x 10{int(e)}'
        return f'{p:.2f}'
    _all_results = []
    _report_blocks = []
    for _target_cell_type in ['MOR+ PPD+', 'MOR- PPD+', 'MOR+ PPD-', 'MOR+ c-Fos+', 'MOR+ c-Fos-', 'MOR- c-Fos+', 'PPD+ c-Fos+', 'PPD+ c-Fos-', 'PPD- c-Fos+', 'all cells']:
        print('Plotting', _target_cell_type)
        _name = _target_cell_type.replace(' ', '_')
        (_fig, _axs) = plt.subplots(1, 2, figsize=(5, 1.5), sharex=True, sharey=True)
        for (_idx, _region) in enumerate(ROI_names):
            _ax = _axs[_idx]
            _data = filtered_master_df_2[(filtered_master_df_2.cell_type == _target_cell_type) & (filtered_master_df_2.region == _region)]
            sns.stripplot(data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1, dodge=True, palette=condition_colors, alpha=0.25, ax=_ax, legend=False)
            sns.pointplot(data=_data, x='AP_label', y='count', hue='Condition', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
            sns.despine()
            _ax.set_title(_region)
            _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
            _pairs = [((_f, 'Saline'), (_f, 'Acute_Morphine')) for _f in bin_labels_1]
            _annotator = Annotator(_ax, pairs=_pairs, data=_data, x='AP_label', y='count', hue='Condition', order=bin_labels_1)
            _annotator.configure(test='t-test_ind', comparisons_correction='Benjamini-Hochberg', text_format='star', loc='inside', verbose=2)
            _annotator.apply_and_annotate()
            _region_rows = []
            for _f in bin_labels_1:
                _sal = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Saline'), 'count'].dropna()
                _mor = _data.loc[(_data.AP_label == _f) & (_data.Condition == 'Acute_Morphine'), 'count'].dropna()
                (_t_stat, _p_raw) = ttest_ind(_sal, _mor)
                _df_ = len(_sal) + len(_mor) - 2
                _region_rows.append({'cell_type': _target_cell_type, 'region': _region, 'AP_bin': _f, 'n_saline': len(_sal), 'n_morphine': len(_mor), 'mean_saline': _sal.mean(), 'mean_morphine': _mor.mean(), 't_stat': _t_stat, 'df': _df_, 'pvalue_raw': _p_raw})
            (_rej, _p_corr, _, _) = multipletests([_r['pvalue_raw'] for _r in _region_rows], alpha=0.05, method='fdr_bh')
            for (_r, _pc, _rj) in zip(_region_rows, _p_corr, _rej):
                _r['pvalue_bh'] = _pc
                _r['significant'] = bool(_rj)
            _all_results.extend(_region_rows)
            _ax.set_ylabel(f'{_target_cell_type} cells')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_name}_count_by_region_and_condition.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(os.path.join(analysis_figurepath, f'{_name}_count_by_region_and_condition.pdf'), dpi=300, bbox_inches='tight')
        _rows = [_r for _r in _all_results if _r['cell_type'] == _target_cell_type]
        _block = f'=== {_target_cell_type} ===\n' + '\n'.join((f'{_region}, Saline vs. Acute Morphine, ' + ' '.join((f"{_r['AP_bin']}: P = {_fmt_p(_r['pvalue_bh'])}, t({_r['df']}) = {_r['t_stat']:.2f}." for _r in _rows if _r['region'] == _region)) for _region in ROI_names))
        _report_blocks.append(_block)
        print(_block, '\n')
    _ttest_df = pd.DataFrame(_all_results)
    _ttest_df.to_csv(os.path.join(analysis_resultpath, 'wildcard_count_by_region_and_condition_ttest_ALL.csv'), index=False)
    with open(os.path.join(analysis_resultpath, 'wildcard_count_ttest_report_ALL.txt'), 'w') as _fh:
        _fh.write('\n\n'.join(_report_blocks))
    _ttest_df
    return


@app.cell
def _(
    bin_labels_1,
    condition_colors,
    filtered_master_df_2,
    n_bins_1,
    plt,
    sns,
):
    _data = filtered_master_df_2[(filtered_master_df_2.cell_type == 'MOR+ PPD+') | (filtered_master_df_2.cell_type == 'MOR+ PPD-') & (filtered_master_df_2.region == 'AcbSh') & (filtered_master_df_2.Condition == 'Acute_Morphine')]
    (_fig, _ax) = plt.subplots(1, 1, figsize=(5, 1.5))
    sns.stripplot(data=_data, x='AP_label', y='count', hue='cell_type', order=bin_labels_1, palette=condition_colors, alpha=0.25, ax=_ax, legend=False, dodge=True)
    sns.pointplot(data=_data, x='AP_label', y='count', hue='cell_type', dodge=0.8 - 0.8 / 2, order=bin_labels_1, palette=condition_colors, markers='o', markersize=4, linestyle='none', errwidth=0.8, ax=_ax)
    sns.despine()
    _ax.set_title('MOR+ PPD+ vs. MOR+ PPD-')
    _ax.set_xticklabels(['anterior'] + [''] * (n_bins_1 - 2) + ['posterior'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load representative images
    """)
    return


@app.cell
def _(master_df_5, np):
    _sorted_index = np.argsort(master_df_5[(master_df_5.cell_type == 'MOR+ PPD+') & (master_df_5.Condition == 'Acute_Morphine') & (master_df_5.AP_label == '0.70–0.93')]['count']).index
    _center_indexes = _sorted_index[len(_sorted_index) // 2 - 2:len(_sorted_index) // 2 + 2]
    print('Use these sections for representative images', master_df_5.loc[_center_indexes, ['fname', 'subject', 'section_id', 'region', 'Condition', 'zposition']])
    return


@app.cell
def _(master_df_5, np):
    _sorted_index = np.argsort(master_df_5[(master_df_5.cell_type == 'MOR+ PPD+') & (master_df_5.Condition == 'Acute_Morphine')]['count']).index
    _center_indexes = _sorted_index[len(_sorted_index) // 2 - 2:len(_sorted_index) // 2 + 2]
    print('Use these sections for representative images', master_df_5.loc[_center_indexes, ['fname', 'subject', 'section_id', 'region', 'Condition', 'zposition']])
    return


@app.cell
def _(master_df_5, np):
    _sorted_index = np.argsort(master_df_5[(master_df_5.cell_type == 'MOR+ PPD+') & (master_df_5.Condition == 'Saline')]['count']).index
    _center_indexes = _sorted_index[len(_sorted_index) // 2 - 2:len(_sorted_index) // 2 + 2]
    print('Use these sections for representative images', master_df_5.loc[_center_indexes, ['fname', 'subject', 'section_id', 'region', 'Condition', 'zposition']])
    return


@app.cell
def _(master_df_5, np):
    _sorted_index = np.argsort(master_df_5[(master_df_5.cell_type == 'MOR+ PPD+') & (master_df_5.Condition == 'Saline') & (master_df_5.AP_label == '0.70–0.93')]['count']).index
    _center_indexes = _sorted_index[len(_sorted_index) // 2 - 2:len(_sorted_index) // 2 + 2]
    print('Use these sections for representative images', master_df_5.loc[_center_indexes, ['fname', 'subject', 'section_id', 'region', 'Condition', 'zposition']])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spatial analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The annotations for each section was aligned, only annotations from sections with both hemisphere was used.
    """)
    return


@app.cell
def _(analysis_resultpath, metapath, os, pd):
    total_section_meta_5 = pd.read_csv(os.path.join(metapath, 'section_meta_20260513_001335.csv'))
    subject_df_1 = pd.read_csv(os.path.join(metapath, 'Opioid_pPDH_meta.csv'))
    master_df_6 = pd.read_csv(os.path.join(analysis_resultpath, f'master_df_with_metadata.csv'))
    return (master_df_6,)


@app.cell
def _(analysis_resultpath, os, pd):
    master_annotation_df = pd.read_csv(os.path.join(analysis_resultpath, 'master_annotation_log.csv'))
    master_annotation_df.head()
    return (master_annotation_df,)


@app.cell
def _(master_df_6):
    master_df_6.columns
    return


@app.cell
def _(master_annotation_df, master_df_6):
    # merge the rotation meta data
    master_df_7 = master_df_6.merge(master_annotation_df, left_on='annotation_id', right_on='annotation_id', how='left')
    return (master_df_7,)


@app.cell
def _(os, rawdatapath):
    template_dir = os.path.join(rawdatapath, 'template')
    return (template_dir,)


@app.cell
def _(os, rawdatapath):
    rotated_cell_path = os.path.join(rawdatapath, "aligned_cells")
    return (rotated_cell_path,)


@app.cell
def _(
    ROI_names,
    classify_cells,
    master_df_7,
    np,
    os,
    pd,
    rotated_cell_path,
    template_dir,
):
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator

    def load_template_mask(ROI_name, AP_label):
        _tag = AP_label.replace(' ', '_')
        prob = np.load(os.path.join(template_dir, f'tmpl_{_ROI_name}_{_tag}_prob.npy'))
        return prob >= 0.5
    (X_MIN, X_MAX) = (-1000, 1000)
    (Y_MIN, Y_MAX) = (-1000, 1000)
    GRID_NX = 200
    GRID_NY = 200
    SIGMA_UM = 200
    x_edges = np.linspace(X_MIN, X_MAX, GRID_NX + 1)
    y_edges = np.linspace(Y_MIN, Y_MAX, GRID_NY + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    sigma_pix = SIGMA_UM / ((X_MAX - X_MIN) / GRID_NX)

    def cells_to_grid(xy):
        if len(xy) == 0:
            return np.zeros((GRID_NY, GRID_NX))
        (h, _, _) = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
        return h.T

    def load_template_mask(ROI_name, AP_label):
        _tag = AP_label.replace(' ', '_')
        prob = np.load(os.path.join(template_dir, f'tmpl_{_ROI_name}_{_tag}_prob.npy'))
        return prob >= 0.5

    def to_masked_grid(xy):
        g = gaussian_filter(cells_to_grid(xy), sigma=sigma_pix)
        return g * mask
    heatmaps = {}
    for _ROI_name in ROI_names:
        print(_ROI_name)
        heatmaps[_ROI_name] = {}
        cell_path = os.path.join(rotated_cell_path, _ROI_name)
        for AP_label in master_df_7['AP_label'].dropna().unique():
            print(' ', AP_label)
            mask = load_template_mask(_ROI_name, AP_label)
            heatmaps[_ROI_name][AP_label] = {}
            for condition in ['Saline', 'Acute_Morphine']:
                print('   ', condition)
                _subset = master_df_7[(master_df_7['cell_type'] == 'all cells') & (master_df_7.quality_flag == 'GOOD') & (master_df_7['class'] == _ROI_name) & (master_df_7['AP_label'] == AP_label) & (master_df_7['Condition'] == condition) & master_df_7['cell_csv'].notna()]
                cfos_list = []
                ppd_list = []
                mor_list = []
                total_list = []
                section_ids = []
                mor_ppd_pos_list = []
                mor_ppd_neg_list = []
                mor_c_fos_pos_list = []
                mor_c_fos_neg_list = []
                for (_, _row) in _subset.iterrows():
                    _fp = os.path.join(cell_path, _row['cell_csv'])
                    if not os.path.exists(_fp):
                        continue
                    try:
                        df = pd.read_csv(_fp, comment='#')
                    except Exception:
                        continue
                    if df.empty:
                        continue
                    df = classify_cells(df, Cy3_threshold='auto', FITC_threshold='auto', Cy5_threshold='auto')
                    xy_all = df[['x_aligned_um', 'y_aligned_um']].dropna().values
                    xy_cfos = df.loc[df['Cy5_pos'], ['x_aligned_um', 'y_aligned_um']].dropna().values
                    xy_ppd = df.loc[df['Cy3_pos'], ['x_aligned_um', 'y_aligned_um']].dropna().values
                    xy_mor = df.loc[df['FITC_pos'], ['x_aligned_um', 'y_aligned_um']].dropna().values
                    xy_mor_ppd_pos = df.loc[df['FITC_pos'] & df['Cy3_pos'], ['x_aligned_um', 'y_aligned_um']].dropna().values
                    xy_mor_ppd_neg = df.loc[df['FITC_pos'] & ~df['Cy3_pos'], ['x_aligned_um', 'y_aligned_um']].dropna().values
                    xy_mor_c_fos_pos = df.loc[df['FITC_pos'] & df['Cy5_pos'], ['x_aligned_um', 'y_aligned_um']].dropna().values
                    xy_mor_c_fos_neg = df.loc[df['FITC_pos'] & ~df['Cy5_pos'], ['x_aligned_um', 'y_aligned_um']].dropna().values
                    mor_ppd_pos_list.append(to_masked_grid(xy_mor_ppd_pos))
                    mor_ppd_neg_list.append(to_masked_grid(xy_mor_ppd_neg))
                    mor_c_fos_pos_list.append(to_masked_grid(xy_mor_c_fos_pos))
                    mor_c_fos_neg_list.append(to_masked_grid(xy_mor_c_fos_neg))
                    cfos_list.append(to_masked_grid(xy_cfos))
                    ppd_list.append(to_masked_grid(xy_ppd))
                    mor_list.append(to_masked_grid(xy_mor))
                    total_list.append(to_masked_grid(xy_all))
                    section_ids.append(_row['annotation_id'])
                n = len(section_ids)
                empty = np.zeros((0, GRID_NY, GRID_NX))
                heatmaps[_ROI_name][AP_label][condition] = {'c-Fos': np.stack(cfos_list, axis=0) if n else empty, 'PPD': np.stack(ppd_list, axis=0) if n else empty, 'MOR': np.stack(mor_list, axis=0) if n else empty, 'total': np.stack(total_list, axis=0) if n else empty, 'section_ids': section_ids, 'MOR+PPD+': np.stack(mor_ppd_pos_list, axis=0) if n else empty, 'MOR+PPD-': np.stack(mor_ppd_neg_list, axis=0) if n else empty, 'MOR+c-Fos+': np.stack(mor_c_fos_pos_list, axis=0) if n else empty, 'MOR+c-Fos-': np.stack(mor_c_fos_neg_list, axis=0) if n else empty}
                print(f'      n={n}')
    return GRID_NX, GRID_NY, condition, heatmaps


@app.cell
def _(os, rawdatapath):
    import glob
    from matplotlib.path import Path as MplPath
    BASE = rawdatapath
    poly_dir = os.path.join(BASE, 'aligned_polygons')
    template_dir_1 = os.path.join(BASE, 'template')
    os.makedirs(template_dir_1, exist_ok=True)
    # ── config ────────────────────────────────────────────────────────────────
    ROI_names_1 = ['AcbC', 'AcbSh']
    (X_MIN_1, X_MAX_1) = (-1000, 1000)
    (Y_MIN_1, Y_MAX_1) = (-1000, 1000)
    # use same grid as heatmaps so mask can be applied directly (no interpolation)
    GRID_N = 200
    return (
        GRID_N,
        MplPath,
        ROI_names_1,
        X_MAX_1,
        X_MIN_1,
        Y_MAX_1,
        Y_MIN_1,
        poly_dir,
        template_dir_1,
    )


@app.cell
def _(
    GRID_N,
    MplPath,
    ROI_names_1,
    X_MAX_1,
    X_MIN_1,
    Y_MAX_1,
    Y_MIN_1,
    master_df_7,
    np,
    os,
    pd,
    poly_dir,
    template_dir_1,
):
    xs = np.linspace(X_MIN_1, X_MAX_1, GRID_N)
    ys = np.linspace(Y_MIN_1, Y_MAX_1, GRID_N)
    (xx, yy) = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    AP_labels = master_df_7['AP_label'].dropna().unique()
    for _ROI_name in ROI_names_1:
        for AP_label_1 in AP_labels:
            _subset = master_df_7[(master_df_7['quality_flag'] == 'GOOD') & (master_df_7['class'] == _ROI_name) & (master_df_7['AP_label'] == AP_label_1)]
            counts = np.zeros(GRID_N * GRID_N, dtype=float)
            n_used = 0
            for (_, _row) in _subset.iterrows():
                aid = str(_row['annotation_id'])
                _fp = os.path.join(poly_dir, _ROI_name, f'{aid}.csv')
                if not os.path.exists(_fp):
                    continue
                poly = pd.read_csv(_fp).values.astype(float)
                if len(poly) < 3:
                    continue
                inside = MplPath(poly).contains_points(pts)
                counts = counts + inside.astype(float)
                n_used = n_used + 1
            if n_used == 0:
                print(f'  {_ROI_name} {AP_label_1}: no sections, skipping')
                continue
            prob = (counts / n_used).reshape(GRID_N, GRID_N)
            _tag = AP_label_1.replace(' ', '_')
            np.save(os.path.join(template_dir_1, f'tmpl_{_ROI_name}_{_tag}_prob.npy'), prob)
            np.save(os.path.join(template_dir_1, f'tmpl_{_ROI_name}_{_tag}_xs.npy'), xs)
            np.save(os.path.join(template_dir_1, f'tmpl_{_ROI_name}_{_tag}_ys.npy'), ys)
            print(f'  {_ROI_name} {AP_label_1}: n={n_used}  coverage={prob.max():.2f}')
    return (AP_label_1,)


@app.cell
def _(np):
    def remove_outlier_sections(data, n_std=2.0, verbose=True):
        """
        data : (n_sections, GRID_NY, GRID_NX)
        Returns filtered array with outlier sections removed.
        """
        if _data.shape[0] < 3:
            return _data
        totals = _data.sum(axis=(1, 2))
        (mu, sd) = (totals.mean(), totals.std())
        keep = np.abs(totals - mu) <= n_std * sd
        if verbose:
            n_removed = (~keep).sum()
            if n_removed:
                print(f'    removed {n_removed}/{len(totals)} outlier sections (totals: {totals[~keep].round(3)})')
        return _data[keep]

    return (remove_outlier_sections,)


@app.cell
def _(AP_label_1, condition, heatmaps):
    heatmaps['AcbSh'][AP_label_1][condition].keys()
    return


@app.cell
def _(
    GRID_NX,
    GRID_NY,
    X_MAX_1,
    X_MIN_1,
    Y_MAX_1,
    Y_MIN_1,
    analysis_figurepath,
    heatmaps,
    master_df_7,
    np,
    os,
    plt,
    rawdatapath,
    remove_outlier_sections,
):
    template_dir_2 = os.path.join(rawdatapath, 'template')
    for (midx, marker) in enumerate(['c-Fos', 'PPD', 'MOR', 'MOR+PPD+', 'MOR+c-Fos+']):
        vmax = [0.02, 0.001, 0.2, 0.0001, 0.01][midx]
        cmap = [plt.cm.Purples, plt.cm.RdPu, plt.cm.YlGn, plt.cm.Blues, plt.cm.Oranges][midx]
        for AP_label_2 in master_df_7['AP_label'].dropna().unique():
            _tag = AP_label_2.replace(' ', '_')
            sh_prob = np.load(os.path.join(template_dir_2, f'tmpl_AcbSh_{_tag}_prob.npy'))
            sh_xs = np.load(os.path.join(template_dir_2, f'tmpl_AcbSh_{_tag}_xs.npy'))
            sh_ys = np.load(os.path.join(template_dir_2, f'tmpl_AcbSh_{_tag}_ys.npy'))
            c_prob = np.load(os.path.join(template_dir_2, f'tmpl_AcbC_{_tag}_prob.npy'))
            c_xs = np.load(os.path.join(template_dir_2, f'tmpl_AcbC_{_tag}_xs.npy'))
            c_ys = np.load(os.path.join(template_dir_2, f'tmpl_AcbC_{_tag}_ys.npy'))
            (_fig, _axs) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)
            extent = [X_MIN_1, X_MAX_1, Y_MIN_1, Y_MAX_1]
            for (_ax, condition_1) in zip(_axs, ['Saline', 'Acute_Morphine']):
                sh_data = remove_outlier_sections(heatmaps['AcbSh'][AP_label_2][condition_1][marker])
                c_data = remove_outlier_sections(heatmaps['AcbC'][AP_label_2][condition_1][marker])
                n_sh = sh_data.shape[0]
                n_c = c_data.shape[0]
                hm = np.mean(sh_data, axis=0) if n_sh > 0 else np.zeros((GRID_NY, GRID_NX))
                hm2 = np.mean(c_data, axis=0) if n_c > 0 else np.zeros((GRID_NY, GRID_NX))
                hm = hm + hm2
                hm[hm == 0] = np.nan
                im = _ax.imshow(hm, vmax=vmax, cmap=cmap, origin='lower', extent=extent, aspect='equal')
                _ax.contour(sh_xs, sh_ys, sh_prob, levels=[0.5], colors='black', linewidths=1, linestyles='dotted')
                _ax.contour(c_xs, c_ys, c_prob, levels=[0.5], colors='black', linewidths=1, linestyles='dotted')
                _ax.set_title(f'{condition_1}  (AcbSh n={n_sh}, AcbC n={n_c})')
                _ax.set_xlabel('x (µm)')
            _axs[0].set_ylabel('y (µm)')
            _fig.suptitle(f'AcbSh + AcbC  ·  {marker}  ·  {AP_label_2}')
            for _ax in _axs:
                _ax.spines['top'].set_visible(False)
                _ax.spines['right'].set_visible(False)
                _ax.spines['bottom'].set_visible(False)
                _ax.spines['left'].set_visible(False)
            _fig.subplots_adjust(right=0.82)
            cbar_ax = _fig.add_axes([0.85, 0.15, 0.02, 0.7])
            _fig.colorbar(im, cax=cbar_ax, label=f'{marker}+ cell density')
            plt.tight_layout()
            plt.show()
            _fig.savefig(os.path.join(analysis_figurepath, f'{marker}_removed_outlier_heatmap_{AP_label_2}.png'), dpi=300)
            _fig.savefig(os.path.join(analysis_figurepath, f'{marker}_removed_outlier_heatmap_{AP_label_2}.pdf'), dpi=300)
    return


if __name__ == "__main__":
    app.run()
