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
    import numpy as np
    import pandas as pd
    import shutil
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    import tifffile
    import tifffile as tiff
    from datetime import datetime

    # All brain-visualization / region helpers now come from the brain_vis package
    # (github.com/kenjp1223/brainvis), which supersedes the old vendored
    # `contour_visualization`, `contour_visualization2`, and `create_mask_for_region`.
    from brain_vis import overlap_contour, set_transparency, get_subregions

    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list

    return (
        datetime,
        flatten_extend,
        get_subregions,
        np,
        os,
        overlap_contour,
        pd,
        pickle,
        plt,
        set_transparency,
        shutil,
        sns,
        tiff,
        tifffile,
    )


@app.cell
def _(os):
    from pathlib import Path

    # ---- Figshare deposit layout (mirror of the reproducibility folder) ----
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    GROUP = DATA_ROOT / "01_main_cfos_morphine"
    ATLAS = DATA_ROOT / "shared" / "atlas"
    FIG_OUT = Path("../figures")
    assert GROUP.exists(), f"DATA_ROOT not found: {DATA_ROOT}. Set OPIOID_DATA_ROOT."

    # External semi-NMF factorization outputs (params.mat, params.pkl, factors.zarr) produced by
    # the `fos` package (modified Linderman-lab semi-NMF; run separately). Set OPIOID_FACTOR_RESULTS
    # or place them in GROUP/spatial_clustering_results.
    FACTOR_DIR = Path(
        os.environ.get("OPIOID_FACTOR_RESULTS", str(GROUP / "spatial_clustering_results"))
    ).expanduser()

    ATLAS_INFO_CSV = str(
        ATLAS / "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv"
    )
    return ATLAS, ATLAS_INFO_CSV, DATA_ROOT, FACTOR_DIR, FIG_OUT, GROUP


@app.cell
def _(atlas_img, np):
    # subset to target region
    # make the slice move along the z position

    def get_slice(z,target_site_subids,hemi = 'left',window = 60):
        ys = np.array([])
        xs = np.array([])
        if hemi == 'left':
            hemi_slice = slice(0,atlas_img.shape[2]//2)
        else:
            hemi_slice = slice(atlas_img.shape[2]//2,atlas_img.shape[2])
        for ID in target_site_subids:
            y_,x_ = np.where(atlas_img[z,:,hemi_slice] == ID)
            xs = np.concatenate([xs,x_])
            ys = np.concatenate([ys,y_])
        # find the center of mass of the OFC
        y_center = int(np.mean(ys))
        x_center = int(np.mean(xs))
        # set the slice to the center of mass of the target site
        if hemi == 'left':
            yslice = slice(y_center-window,y_center+window)
            xslice = slice(x_center-window,x_center+window)
        elif hemi == 'right':
            yslice = slice(y_center-window,y_center+window)
            xslice = slice(x_center-window + atlas_img.shape[2]//2,x_center+window + atlas_img.shape[2]//2)        
        elif hemi == 'center':
            yslice = slice(y_center-window,y_center+window)
            xslice = slice(atlas_img.shape[2]//2-window,window + atlas_img.shape[2]//2)

        return yslice,xslice

    return (get_slice,)


@app.cell
def _():
    # set figure key
    figure_key = 'Figure6'
    return (figure_key,)


@app.cell
def _(ATLAS, ATLAS_INFO_CSV, FIG_OUT, GROUP, figure_key, os, pd, tifffile):
    # ---- paths (Figshare deposit) ----
    metapath = str(GROUP)              # meta lives in the group folder
    analysis_resultpath = str(GROUP)   # result CSVs / intermediates in the group folder
    analysis_figurepath = os.path.join(str(FIG_OUT), figure_key)
    os.makedirs(analysis_figurepath, exist_ok=True)

    # load meta info of the files
    metadf = pd.read_csv(os.path.join(metapath, "OP_meta.csv"), index_col=False)
    # temporary drop of A7 due to missing data
    metadf = metadf[metadf.ID != 'A7'].reset_index(drop=True)
    # load brain atlas to register
    atlas_df = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    metacolumns = ['id','acronym','parent_acronym','parent_id','structure_order']
    contour_img = tifffile.imread(os.path.join(str(ATLAS), "Kim_ref_adult_FP-label_v2.9_contour_map.tif"))
    # retrieve list of files
    fnames = [f for f in metadf.fname.values if 'DONE' in f]
    return (
        analysis_figurepath,
        analysis_resultpath,
        atlas_df,
        contour_img,
        metadf,
        metapath,
    )


@app.cell
def _(metadf):
    #Conditions = metadf.Condition.unique()
    # Use only morphine related groups
    Conditions = ['Saline', 'Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine', 'Chronic_Morphine_21', 'Withdrawal_Morphine_21']
    print(Conditions)
    # subset the meta dataframe
    metadf_1 = metadf[metadf.Condition.isin(Conditions)]
    return (metadf_1,)


@app.cell
def _(plt):
    from adjustText import adjust_text

    # Set matplotlib parameters for white text on transparent background
    plt.rcParams.update({
        'figure.facecolor': 'none',  # Transparent figure background
        'axes.facecolor': 'none',    # Transparent axes background
        'axes.edgecolor': 'black',   # White axes edge color
        'axes.labelcolor': 'black',  # White axis labels
        'xtick.color': 'black',      # White tick labels
        'ytick.color': 'black',      # White tick labels
        'legend.facecolor': 'none',  # Transparent legend background
        'legend.edgecolor': 'none',  # Transparent legend edgecolor
        'text.color': 'black',       # White text color
        'font.family':'Arial',
        'pdf.fonttype':42,
        'ps.fonttype':42,
   
    })
    #important for text to be detected when importing saved figures into illustrator
    return (adjust_text,)


@app.cell
def _(ATLAS, os, pickle):
    with open(os.path.join(str(ATLAS), 'curated_acronym.pickle'), 'rb') as handle:
        curated_acronyms = pickle.load(handle)

    with open(os.path.join(str(ATLAS), 'ancestor_curated_acronym.pickle'), 'rb') as handle:
        ancestor_curated_acronyms = pickle.load(handle,)
    return


@app.cell
def _(ATLAS, analysis_resultpath, metadf_1, os, pd, pickle):
    default_depth = 4
    # set heatmap variables
    vmin = -5
    vmax = 10
    Conditions_1 = ['Saline', 'Acute_Morphine', 'Chronic_Morphine', 'Withdrawal_Morphine', 'Chronic_Morphine_21', 'Withdrawal_Morphine_21']
    #Conditions = metadf.Condition.unique()
    # Use only morphine related groups
    print(Conditions_1)
    Condition_figure_name = ['Saline', 'Acute', 'Chronic', 'Early WD', 'Re-exposure', 'Late WD']
    #Condition_figure_name = ['Saline','Acute','Chronic 1 day','W.D. 1 day','Chronic 21 days','W.D. 21 days'] # changed this to betterones
    Condition_color = ['gray', 'lime', 'orange', 'cyan', 'blue', 'purple']
    metadf_2 = metadf_1[metadf_1.Condition.isin(Conditions_1)]
    # subset the meta dataframe
    pivot_heatmap_df = pd.read_csv(os.path.join(analysis_resultpath, 'long_pivoted_heatmap_df_with_normalized_density.csv'), index_col=0)
    pivot_heatmap_df = pivot_heatmap_df[metadf_2[metadf_2.Condition.isin(Conditions_1)]['ID'].values]
    # load and subset dataframes
    merge_df = pd.read_csv(os.path.join(analysis_resultpath, 'Ex_639_Ch2_stitched_long_merge_Annotated_counts_with_leaf_with_density_with_normalized_density.csv'), index_col=0)
    merge_df = merge_df[merge_df.Condition.isin(Conditions_1)]
    with open(os.path.join(str(ATLAS), 'curated_acronym.pickle'), 'rb') as handle_1:
        curated_acronyms_1 = pickle.load(handle_1)
    # Load the acronyms for plotting
    with open(os.path.join(str(ATLAS), 'ancestor_curated_acronym.pickle'), 'rb') as handle_1:
        ancestor_curated_acronyms_1 = pickle.load(handle_1)
    return (
        Condition_color,
        Condition_figure_name,
        Conditions_1,
        curated_acronyms_1,
        merge_df,
        metadf_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    remove HB and CBL from the list of ancestores
    """)
    return


@app.cell
def _(atlas_df):
    # Update the ancestor curated acronyms so it matches the tree devisions
    unique_ancestor_curated_acronyms = ['Isocortex','OLF','HPF','CTXsp','STR','PAL','TH','HY','MB',]
    #unique_ancestor_curated_acronyms = ['Isocortex','OLF','HPF','CTXsp','STR','PAL','TH','HY','MB','HB','CBL']

    # get a list of idx for the ancestors
    ancestor_names = [atlas_df.loc[atlas_df.acronym == f,'name'].values[0] for f in unique_ancestor_curated_acronyms]
    ancestor_idxs = [atlas_df.loc[atlas_df.acronym == f,'id'].values[0] for f in unique_ancestor_curated_acronyms]
    '''
    curated_acronyms = []
    ancestor_curated_acronyms = []
    for idx,i in enumerate(ancestor_idxs):
        create_mask_for_region.get_subregions(atlas_df,i,return_original = True)
        tdf = create_mask_for_region.get_subregions(atlas_df,i,return_original = True)
        curated_acronyms += list(tdf[tdf.Curated_list].acronym)
        ancestor_curated_acronyms += [unique_ancestor_curated_acronyms[idx]]*tdf[tdf.Curated_list].shape[0]

    # save the new list of acronyms
    with open(os.path.join(analysis_resultpath,f'curated_acronym.pickle'), 'wb') as handle:
        pickle.dump(curated_acronyms,handle,)

    with open(os.path.join(analysis_resultpath,f'ancestor_curated_acronym.pickle'), 'wb') as handle:
        pickle.dump(ancestor_curated_acronyms,handle,)'''
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    remove CBL and MB subtree from the atlas file and the merge_df
    """)
    return


@app.cell
def _(atlas_df, create_mask_for_region, merge_df, pd):
    # remove CBL and MB subtree from the data. These regions had bad registration quality and low interest
    remove_ancestor_ids = atlas_df[(atlas_df.acronym == 'HB') | (atlas_df.acronym == 'CBL')]['id'].values
    remove_df = pd.concat([create_mask_for_region.get_subregions(atlas_df, idx, return_original=True) for idx in remove_ancestor_ids], axis=0)
    sub_atlas_df = atlas_df.set_index(['id']).drop(remove_df['id'].values)
    merge_df_1 = merge_df[merge_df.acronym.isin(sub_atlas_df.acronym.unique())]
    return (merge_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    set up the metacolumns to be compatible for GLM
    """)
    return


@app.cell
def _(datetime, merge_df_1, metadf_2):
    # load meta info of the files
    metadf_2['age'] = [(datetime.strptime(pday, '%m/%d/%Y') - datetime.strptime(dob, '%m/%d/%Y')).days for (pday, dob) in metadf_2.loc[:, ['Date_Perfusion', 'DOB']].values]
    #metaexog = metadf[['Condition','Sex','BW','age','Staining_Batch']]
    #metacolumns = ['Saline','Acute_Morphine','Chronic_Morphine','Sex_d','Batch_d']
    atlasmeta = merge_df_1.reset_index().loc[merge_df_1.reset_index().ID == 'A1', ['id', 'parent_id', 'acronym', 'name', 'parent_acronym']]
    return


@app.cell
def _(Conditions_1, merge_df_1, pd):
    # change categorical values to dummy chategories dtypes
    sex_category = pd.CategoricalDtype(categories=['F', 'M'], ordered=False)
    condition_category = pd.CategoricalDtype(categories=Conditions_1, ordered=True)
    batch_category = pd.CategoricalDtype(categories=[1, 2, 3, 4], ordered=False)
    merge_df_1['Sex'] = merge_df_1['Sex'].astype(sex_category)
    merge_df_1['Condition'] = merge_df_1['Condition'].astype(condition_category)
    merge_df_1['Staining_Batch'] = merge_df_1['Staining_Batch'].astype(batch_category)
    condition_dummies = pd.get_dummies(merge_df_1['Condition'])
    sex_dummies = pd.get_dummies(merge_df_1['Sex']).loc[:, ['F']].rename(columns={'F': 'Sex_d'})
    batch_dummies = pd.get_dummies(merge_df_1['Staining_Batch'])
    # create dummy cats
    batch_dummies.columns = [f'Batch_{c}_d' for c in range(4)]
    merge_df_2 = pd.concat([merge_df_1, condition_dummies, sex_dummies, batch_dummies], axis=1)  # female 1
    return (merge_df_2,)


@app.cell
def _():
    # add flags for conditions
    #merge_df = pd.merge(merge_df,metadf[['ID','Acute_flag','Chronic_flag','Spontaneous_flag']],left_on = 'ID',right_on = 'ID')
    return


@app.cell
def _(merge_df_2):
    # calculate effect size raw
    raw_effect_size_df = merge_df_2[['acronym', 'Condition', 'density']].groupby(['acronym', 'Condition']).mean().reset_index().pivot(columns='Condition', index='acronym', values='density')
    return


@app.cell
def _():
    # (Original wrote clean_curated_acronyms.csv into the data folder; it already ships in
    #  shared/atlas, so the export is disabled here to keep the deposit read-only.)
    # atlas_df.loc[atlas_df.acronym.isin(curated_acronyms_1), ['acronym', 'name']].to_csv(
    #     os.path.join(metapath, 'clean_curated_acronyms.csv'), index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Prepare heatmaps into a dictionary format
    """)
    return


@app.cell
def _(ATLAS, os, tifffile):
    # read an annotated atlas file
    atlas_img = tifffile.imread(os.path.join(str(ATLAS), "Kim_ref_adult_FP-label_v4.0.tif"))
    return (atlas_img,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Semi-nmf
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The semi-nnmf analysis was conducted using a modified code from the Lindermann lab
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The main codes are in the folder below
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    C:\Users\stuberadmin\fos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results are saved in this iteration of factorization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Semi-NMF factorization outputs (`params.mat`, `params.pkl`, `factors.zarr`) are read from
    `FACTOR_DIR` (env `OPIOID_FACTOR_RESULTS`, else `GROUP/spatial_clustering_results`).
    They are produced separately by the `fos` package (modified Linderman-lab semi-NMF).
    """)
    return


@app.cell
def _(FACTOR_DIR):
    factor_resultspath = str(FACTOR_DIR)
    return (factor_resultspath,)


@app.cell
def _(factor_resultspath, os):
    from scipy.io import loadmat
    # read the params matrics
    # read params.mat and params.pkl
    params_mat = os.path.join(factor_resultspath,'params.mat')
    params_pkl = os.path.join(factor_resultspath,'params.pkl')
    # read the params.mat file
    params = loadmat(params_mat)
    return (params,)


@app.cell
def _(params):
    # read factors and loadings from the semi-nnmf result
    factors = params['factors']
    count_loadings = params['count_loadings']
    return


@app.cell
def _(metadf_2, params, pd):
    # convert the loadings into a dataframe
    count_loadings_df = pd.DataFrame(params['count_loadings'], columns=[f'factor_{idx}' for idx in range(params['count_loadings'].shape[1])])
    # merge with metadf
    count_loadings_df = count_loadings_df.join(metadf_2.reset_index()['Condition'])
    # convert the loadings dataframe to stacked
    stack_count_loadings_df = count_loadings_df.set_index('Condition').stack().reset_index().rename(columns={'level_1': 'factor', 0: 'weights'})
    return (stack_count_loadings_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # spatial factor
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The factors are stored in a zarr array
    """)
    return


@app.cell
def _(factor_resultspath, os):
    import dask.array as da
    # read the zarr file as a dask.array
    factors_da = da.from_zarr(os.path.join(factor_resultspath, "factors.zarr"))
    factors_da
    return (factors_da,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Statistical analysis to specify factors to focus
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To specify the factors to analyze, we conduct a one-way anova for the loadings in each factor, followed by a multiple comparison correction
    """)
    return


@app.cell
def _(analysis_resultpath, os, pd, stack_count_loadings_df):
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    results = []
    factors_1 = stack_count_loadings_df['factor'].unique()
    # Create a results DataFrame to store ANOVA results
    raw_p_values = []
    for factor in factors_1:
    # Get unique factors
        factor_data = stack_count_loadings_df[stack_count_loadings_df['factor'] == factor]
        conditions = [c for c in factor_data['Condition'].unique() if c != 'Saline']
    # Store raw p-values for multiple testing correction
        groups = [factor_data[factor_data['Condition'] == 'Saline']['weights'].values]
        for condition in conditions:
    # For each factor, perform ANOVA
            groups.append(factor_data[factor_data['Condition'] == condition]['weights'].values)
        (f_stat, p_value) = stats.f_oneway(*groups)  # Subset data for current factor
        raw_p_values.append(p_value)
        results.append({'factor': factor, 'f_statistic': f_stat, 'p_value': p_value})
    results_df = pd.DataFrame(results)  # Get all conditions except Saline
    (reject, pvals_corrected, _, _) = multipletests(raw_p_values, method='fdr_bh')
    results_df['corrected_p_value'] = pvals_corrected
    for (idx, row) in results_df.iterrows():  # Perform one-way ANOVA
        if row['corrected_p_value'] < 0.05:
            factor = row['factor']
            factor_data = stack_count_loadings_df[stack_count_loadings_df['factor'] == factor]
            conditions = [c for c in factor_data['Condition'].unique() if c != 'Saline']
            significant_comparisons = []  # Perform ANOVA
            pvals = []
            for (cidx, condition) in enumerate(conditions):
                saline_data = factor_data[factor_data['Condition'] == 'Saline']['weights'].values
                condition_data = factor_data[factor_data['Condition'] == condition]['weights'].values  # Store results with placeholder for corrected p-value
                (t_stat, p_val) = stats.ttest_ind(saline_data, condition_data)
                degree_of_freedom = len(saline_data) + len(condition_data) - 2
                results_df.loc[idx, f'p-value: Saline vs. {condition}'] = p_val
                results_df.loc[idx, f't-stat: Saline vs. {condition}'] = t_stat
                results_df.loc[idx, f'degree of freedom: Saline vs. {condition}'] = degree_of_freedom  #'significant_comparisons': []  # Will be filled after multiple testing correction
    print('\nANOVA Results:')
    print(results_df)
    # Convert results to DataFrame
    # Perform multiple testing correction on ANOVA p-values
    # Add corrected p-values to results DataFrame
    # Now perform post-hoc tests only for factors with significant corrected p-values
    # Display results
    # write the results file
    results_df.to_csv(os.path.join(analysis_resultpath, 'factor_anova_stat_df.csv'), index=False)  # Using corrected p-value for decision  # Perform t-test between Saline and current condition  # Update the significant comparisons in the results DataFrame  #results_df.at[idx, 'significant_comparisons'] = significant_comparisons
    return (pd,)


@app.cell
def _(analysis_resultpath, os, pd):
    results_df_1 = pd.read_csv(os.path.join(analysis_resultpath, 'factor_anova_stat_df.csv'))
    return (results_df_1,)


@app.cell
def _(results_df_1):
    # get the rejected factors
    rejected_factor_idx = results_df_1.loc[results_df_1.corrected_p_value < 0.05, 'factor'].index
    print(rejected_factor_idx)
    return (rejected_factor_idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure G, I and supplemental figure D, F, H
    """)
    return


@app.cell
def _(
    Condition_color,
    Condition_figure_name,
    Conditions_1,
    analysis_figurepath,
    os,
    plt,
    results_df_1,
    sns,
    stack_count_loadings_df,
):
    tfigurepath = os.path.join(analysis_figurepath, 'factors_by_condition')
    if not os.path.exists(tfigurepath):
        os.mkdir(tfigurepath)
    factors_2 = results_df_1['factor'].unique()
    # Create directory for plots if it doesn't exist

    def get_stars(p_value):
        """Return the appropriate number of stars based on p-value"""
        if p_value < 0.0001:
    # Get unique factors from results_df
            return '****'
        elif p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        return ''
    for factor_1 in factors_2:
        (fig, axs) = plt.subplots(1, 1, figsize=(2, 1.5))
        factor_data_1 = stack_count_loadings_df[stack_count_loadings_df.factor == factor_1]
        sns.stripplot(data=factor_data_1, y='weights', x='Condition', ax=axs, palette=Condition_color, order=Conditions_1, alpha=0.5, size=2)
        sns.pointplot(data=factor_data_1, y='weights', x='Condition', palette=Condition_color, order=Conditions_1, ax=axs, markers='o', markersize=5, linestyle='none', linewidth=0.5)
        axs.set_ylim(-15000000.0, 15000000.0)
        axs.axhline(0, ls=':', color='gray', alpha=0.3)
        axs.set_xticklabels(Condition_figure_name, rotation=-45)  # Create figure
        significant_comps = results_df_1[results_df_1['factor'] == factor_1]['significant_comparisons'].iloc[0]
        if significant_comps:
            for comp in significant_comps:  # Get the data for current factor
                condition_1 = comp.split('-')[1].split(' ')[0]
                p_value_1 = float(comp.split('p=')[1].split(')')[0])
                x_pos = Conditions_1.index(condition_1)  # Create strip plot
                stars = get_stars(p_value_1)
                if stars:
                    axs.text(x_pos, axs.get_ylim()[1] * 0.95, stars, ha='center', va='bottom', fontsize=12)
        sns.despine()  # Create point plot  # Set y-axis limits  # Add horizontal line at y=0  # Set x-axis labels  # Get significant comparisons for this factor  # Add stars for significant comparisons  # If there are any significant comparisons  # Extract condition name and p-value from the comparison string  # Gets the condition name from "Saline-Condition (p=...)"  # Gets the p-value from "Saline-Condition (p=0.xxxx)"  # Find the x position of this condition  # Get the appropriate number of stars  # Add stars above the condition  # Only add if there are stars to show  # Remove spines  # Save the figure  #fig.savefig(os.path.join(tfigurepath, f'{factor}.png'), dpi=216, bbox_inches='tight')  #fig.savefig(os.path.join(tfigurepath, f'{factor}.pdf'), dpi=216, bbox_inches='tight')
    return (condition_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fiugre F and supplemental figure B
    """)
    return


@app.cell
def _(Conditions_1, stack_count_loadings_df):
    ave_stack_count_loadings_df = stack_count_loadings_df.groupby(['Condition', 'factor']).mean().reset_index().pivot(columns='factor', index='Condition', values='weights')
    ave_stack_count_loadings_df = ave_stack_count_loadings_df.loc[Conditions_1, [f'factor_{idx}' for idx in range(22)]]
    return (ave_stack_count_loadings_df,)


@app.cell
def _():
    pannel_key = 'F'
    figure_key_1 = 'Figure6'
    return figure_key_1, pannel_key


@app.cell
def _(
    Condition_figure_name,
    analysis_figurepath,
    ave_stack_count_loadings_df,
    figure_key_1,
    np,
    os,
    pannel_key,
    params,
    plt,
    rejected_factor_idx,
    sns,
):
    vmax_1 = abs(params['count_loadings']).max()
    (fig_1, axs_1) = plt.subplots(1, 1, figsize=(4, 6))
    sns.heatmap(data=ave_stack_count_loadings_df, cmap='coolwarm', vmin=-vmax_1 // 3, vmax=vmax_1 // 3, ax=axs_1)
    axs_1.set_xticks(np.array(range(22)) + 0.5)
    # make a list for the xticks labels
    # for idx in range(22) not in rejected_factor_idx, make it to '', otherwise use the factor name
    xticks_labels = ['' for idx in range(22)]
    for idx_1 in rejected_factor_idx:
        xticks_labels[idx_1] = f'{idx_1}'
    axs_1.set_xticklabels(xticks_labels, rotation=0)
    axs_1.set_yticklabels(Condition_figure_name, rotation=90)
    fig_1.savefig(os.path.join(analysis_figurepath, f'{figure_key_1}{pannel_key}.png'), dpi=216, bbox_inches='tight')
    fig_1.savefig(os.path.join(analysis_figurepath, f'{figure_key_1}{pannel_key}.pdf'), dpi=216, bbox_inches='tight')
    return (vmax_1,)


@app.cell
def _():
    pannel_key_1 = 'F'
    figure_key_2 = 'Figure6-supplemental figure 1'
    return figure_key_2, pannel_key_1


@app.cell
def _(Conditions_1, metadf_2, np):
    idx_drugs = np.array([np.where(np.array(Conditions_1) == drug)[0][0] for drug in metadf_2.Condition.values])
    perm = np.argsort(idx_drugs)
    bounds = np.cumsum(np.bincount(idx_drugs)) - 0.5
    yticks = [np.cumsum(np.bincount(idx_drugs))[fidx] - f / 2 - 0.5 for (fidx, f) in enumerate(np.bincount(idx_drugs))]
    best_num_factors = 22
    return best_num_factors, bounds, perm, yticks


@app.cell
def _(
    Condition_figure_name,
    analysis_figurepath,
    best_num_factors,
    bounds,
    figure_key_2,
    np,
    os,
    pannel_key_1,
    params,
    perm,
    plt,
    vmax_1,
    yticks,
):
    loadings = params['count_loadings']
    (fig_2, axs_2) = plt.subplots(1, 1, figsize=(4, 6))
    axs_2.imshow(loadings[perm], vmin=-vmax_1, vmax=vmax_1, cmap='coolwarm', aspect='auto', interpolation='none')
    for bound in bounds:
        axs_2.axhline(bound, color='k')
    _ = axs_2.set_xticks(np.arange(best_num_factors))
    axs_2.set_xlabel('factor')
    axs_2.set_ylabel('mouse')
    axs_2.set_title('count loadings (sorted by drug group)')
    axs_2.set_yticks(yticks)
    axs_2.set_yticklabels(Condition_figure_name, rotation=90, va='center')
    #plt.colorbar()
    fig_2.savefig(os.path.join(analysis_figurepath, f'{figure_key_2}{pannel_key_1}.png'), dpi=300, bbox_inches='tight')
    fig_2.savefig(os.path.join(analysis_figurepath, f'{figure_key_2}{pannel_key_1}.pdf'), dpi=300, bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5-supplemental figure 1 C
    """)
    return


@app.cell
def _(
    Condition_figure_name,
    Conditions_1,
    analysis_figurepath,
    metadf_2,
    np,
    os,
    params,
    plt,
    sns,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, cross_val_predict
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import KFold
    import warnings
    warnings.filterwarnings('ignore')

    def normalize_weights(weights):
    # normalize weights and factors
        U = weights
        U = U - U.mean()
        U = U / U.std()
        return U

    def downstream_task(W, drugs, **kwargs):
        """
        kwargs
        ------
        cv: pass in the same CV splitter for both the grid search and the confusion matrix
        """
        W_norm = W
    #     W_norm = normalize_weights(W)
        parameters = {'C': 10 ** np.linspace(-15, 15, num=31)}
        lr = LogisticRegression()
        gridsearch = GridSearchCV(lr, parameters, **kwargs)
        gridsearch.fit(W_norm, drugs)
        drug_clf_acc = gridsearch.best_score_
        classifier = gridsearch.best_estimator_
        drugs_pred = cross_val_predict(classifier, W_norm, y=drugs, **kwargs)
        confusion_mat = confusion_matrix(drugs, drugs_pred)
        return (classifier, drug_clf_acc, confusion_mat)
    pannel_key_2 = 'C'
    figure_key_3 = 'Figure6-supplemental figure 1'
    features = np.hstack([normalize_weights(params['count_loadings'])])
    cv = KFold(shuffle=True, random_state=0)
    (clf, acc, cmat) = downstream_task(features, metadf_2.Condition.values, cv=cv)
    (fig_3, axs_3) = plt.subplots(1, 1, figsize=(4, 3.25))
    sns.heatmap(cmat, ax=axs_3, cmap='viridis')
    axs_3.set_title('count loadings (acc={:.2f})'.format(acc))
    axs_3.set_ylabel('true drug')
    axs_3.set_xlabel('predicted drug')
    axs_3.set_xticks(np.array(range(len(Conditions_1))) + 0.5, Condition_figure_name, rotation=-45, ha='left')
    axs_3.set_yticks(np.array(range(len(Conditions_1))) + 0.5, Condition_figure_name)
    fig_3.savefig(os.path.join(analysis_figurepath, f'{figure_key_3}{pannel_key_2}.png'), dpi=216, bbox_inches='tight')
    #plt.colorbar(axs.images[0], ax=axs)
    fig_3.savefig(os.path.join(analysis_figurepath, f'{figure_key_3}{pannel_key_2}.pdf'), dpi=216, bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure H, J, supplemental figure 1 A, E, G, I
    """)
    return


@app.cell
def _():
    pannel_key_3 = 'H'
    figure_key_4 = 'Figure6'
    return figure_key_4, pannel_key_3


@app.cell
def _():
    # pre selected zplanes
    curated_zplanes = [84,104,117,153,186,220]
    return (curated_zplanes,)


@app.function
def set_transparency_1(rgba_img, mask):
    """
    Applies a transparency mask to an existing RGBA image.

    Parameters:
    - rgba_img: np.ndarray of shape (H, W, 4), dtype uint8
        The input RGBA image.
    - mask: np.ndarray of shape (H, W), dtype bool
        Boolean mask where True means the pixel should be transparent.

    Returns:
    - np.ndarray of shape (H, W, 4), modified RGBA image.
    """
    if rgba_img.shape[-1] != 4:
        raise ValueError('Input image must be RGBA (shape must be H x W x 4).')
    if rgba_img.shape[:2] != mask.shape:
        raise ValueError('Mask shape must match image height and width.')
    result = rgba_img.copy()
    result[mask, 3] = 0
    return result  # Copy to avoid modifying the original  # Set alpha to 0 (transparent) where mask is True


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    condition_1,
    contour_img,
    curated_zplanes,
    factors_da,
    figure_key_4,
    os,
    pannel_key_3,
    plt,
):
    import brain_vis as cv2  # brain_vis supersedes contour_visualization2
    # plot every 10 zplanes
    # slice for visualization
    imy_slice = slice(25, 425)
    imx_slice = slice(50, 600)
    for factor_idx in range(22):
        theatmap = factors_da[factor_idx, :].reshape(atlas_img.shape)
        (fig_4, axs_4) = plt.subplots(1, len(curated_zplanes), figsize=(3 * len(curated_zplanes), 3), sharey=True)
        fig_4.subplots_adjust(wspace=0.25, hspace=0.3)
        for (idx_2, ax) in enumerate(axs_4):
            formatted_idx = f'{curated_zplanes[idx_2]:04}'
            (__, overlayed_image) = cv2.overlap_contour(theatmap, contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.viridis)
            trans_img = set_transparency_1(overlayed_image[curated_zplanes[idx_2], :, :], (atlas_img == 0)[curated_zplanes[idx_2], :, :])
            ax.imshow(trans_img[imy_slice, imx_slice])
            ax.axis('off')
            ax.set_title('')
            ax.set_ylabel(condition_1, color='black')
        fig_4.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_3}_factor_{factor_idx}.png'), bbox_inches='tight', dpi=1024)
        fig_4.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_3}_factor_{factor_idx}.svg'), bbox_inches='tight', dpi=1024)
    return (cv2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of NAc
    """)
    return


@app.cell
def _():
    import brain_vis as create_mask_for_region  # brain_vis supersedes create_mask_for_region

    return (create_mask_for_region,)


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_1 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of NAc distribution
    """)
    return


@app.cell
def _(atlas_df_1, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym = 'Acb'
    target_site_id = atlas_df_1.loc[atlas_df_1.acronym == target_site_acronym, 'id'].values[0]
    target_site_subids = create_mask_for_region.get_subregions(atlas_df_1, target_site_id, return_original=True)['id'].values
    target_site_subacronyms = create_mask_for_region.get_subregions(atlas_df_1, target_site_id, return_original=True)['acronym'].values
    return target_site_acronym, target_site_subacronyms, target_site_subids


@app.cell
def _(atlas_img, np, target_site_subids):
    # collect all the z positions where there is the target site
    zs = np.array([])
    for ID in target_site_subids:
        z_,y_,x_ = np.where(atlas_img == ID)
        zs = np.concatenate([zs,z_])
    # find the center of mass of the VTA
    z_unique = np.unique(zs).astype('uint16')
    z_center = int(np.mean(zs))
    return (z_unique,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap = np.reshape(factors_da,[22] + list(atlas_img.shape)).compute()
    return (subset_heatmap,)


@app.cell
def _(atlas_img, np, subset_heatmap, target_site_subids, z_unique):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    spatial_cell_count_for_subregion = np.array([[[np.nanmean(subset_heatmap[f,z,np.where(atlas_img[z,:,:] == target_site_subid)[0],np.where(atlas_img[z,:,:] == target_site_subid)[1]]) for z in z_unique] for target_site_subid in target_site_subids] for f in range(22)])
    # append spatial_c
    return (spatial_cell_count_for_subregion,)


@app.cell
def _(np, spatial_cell_count_for_subregion):
    # calculate cell count for full region
    spatial_cell_count_for_full = np.nansum(spatial_cell_count_for_subregion,axis = 1)
    return (spatial_cell_count_for_full,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full,
    spatial_cell_count_for_subregion,
    target_site_acronym,
):
    np.save(os.path.join(analysis_resultpath,f'spatial_factors_for_subregion_sum_{target_site_acronym}.npy'),spatial_cell_count_for_full,)
    np.save(os.path.join(analysis_resultpath,f'spatial_factors_for_subregion_{target_site_acronym}.npy'),spatial_cell_count_for_subregion,)
    return


@app.cell
def _(np, z_unique):
    # split the Acumbens into anterior middle posterior
    z_anterior,z_medial,z_posterior = np.array_split(z_unique, 3)
    return


@app.cell
def _():
    # subset to NAc region
    xslice = slice(325,325+120)
    yslice = slice(240,240+120)
    return xslice, yslice


@app.cell
def _():
    pannel_key_4 = 'L'
    return (pannel_key_4,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_4,
    plt,
    sns,
    spatial_cell_count_for_subregion,
    target_site_subacronyms,
):
    for (idx_3, sub_acronym) in enumerate(target_site_subacronyms):
        sub_factors = [1, 2]
        tarray = spatial_cell_count_for_subregion[sub_factors, idx_3, :]  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        tarray = tarray[:, ~np.isnan(tarray[0, :])]
        (fig_5, axs_5) = plt.subplots(1, 1, figsize=(3, 0.3))
        sns.heatmap(tarray, ax=axs_5, cmap='viridis', vmin=0, vmax=5e-05)
        axs_5.set_title(sub_acronym)
        axs_5.set_yticks(np.array(range(len(sub_factors))) + 0.5)
        axs_5.set_yticklabels(sub_factors, rotation=0)
        axs_5.set_xticks([0, tarray.shape[1]])
        axs_5.set_xticklabels(['anterior', 'posterior'])
        axs_5.set_ylabel('factor')
        axs_5.set_xlabel('z position')
        fig_5.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_4}_{sub_acronym}.png'), bbox_inches='tight', dpi=512)
        fig_5.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_4}_{sub_acronym}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    contour_img,
    cv2,
    factors_da,
    figure_key_4,
    os,
    plt,
    rejected_factor_idx,
    xslice,
    yslice,
    z_unique,
):
    # plot the spatial distribution of starter cells for each subject
    pannel_key_5 = 'K'
    for (cidx_1, rejected_factor) in enumerate(rejected_factor_idx):
        (fig_6, axs_6) = plt.subplots(1, len(z_unique[::5]), figsize=(len(z_unique[::5]) * 1.0, 1), sharex=True, sharey=True)
        theatmap_1 = factors_da[rejected_factor, :].reshape(atlas_img.shape).compute()
        (__, overlayed_image_1) = cv2.overlap_contour(theatmap_1[:, :, ::-1], contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.viridis)
        for (idx_4, curated_zplane) in enumerate(z_unique[::5]):
            ax_1 = axs_6[idx_4]
            ax_1.imshow(overlayed_image_1[curated_zplane, yslice, xslice])
            ax_1.set_xticks([])
            ax_1.set_yticks([])
            if idx_4 == 0:
                ax_1.set_ylabel(f'factor:{rejected_factor}', color='black')
            else:  # Remove x ticks
                ax_1.set_ylabel('', color='black')  # Remove y ticks
        fig_6.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_5}_factor_{rejected_factor}.png'), bbox_inches='tight', dpi=512)  #ax.axis('off')
        fig_6.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_5}_factor_{rejected_factor}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of Ce
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_2 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of Ce distribution
    """)
    return


@app.cell
def _(atlas_df_2, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_1 = 'Ce'
    target_site_id_1 = atlas_df_2.loc[atlas_df_2.acronym == target_site_acronym_1, 'id'].values[0]
    target_site_subids_1 = create_mask_for_region.get_subregions(atlas_df_2, target_site_id_1, return_original=True)['id'].values
    target_site_subacronyms_1 = create_mask_for_region.get_subregions(atlas_df_2, target_site_id_1, return_original=True)['acronym'].values
    return (
        target_site_acronym_1,
        target_site_subacronyms_1,
        target_site_subids_1,
    )


@app.cell
def _(atlas_img, np, target_site_subids_1):
    # collect all the z positions where there is the target site
    zs_1 = np.array([])
    for ID_1 in target_site_subids_1:
        (z__1, y__1, x__1) = np.where(atlas_img == ID_1)
        zs_1 = np.concatenate([zs_1, z__1])
    # find the center of mass of the VTA
    z_unique_1 = np.unique(zs_1).astype('uint16')
    z_center_1 = int(np.mean(zs_1))
    return (z_unique_1,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_1 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_1,)


@app.cell
def _(atlas_img, np, subset_heatmap_1, target_site_subids_1, z_unique_1):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_1 = np.array([[[np.nanmean(subset_heatmap_1[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_1] for target_site_subid in target_site_subids_1] for f in range(22)])
    return (spatial_cell_count_for_subregion_1,)


@app.cell
def _(np, spatial_cell_count_for_subregion_1):
    # calculate cell count for full region
    spatial_cell_count_for_full_1 = np.nansum(spatial_cell_count_for_subregion_1, axis=1)
    return (spatial_cell_count_for_full_1,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_1,
    spatial_cell_count_for_subregion_1,
    target_site_acronym_1,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_1}.npy'), spatial_cell_count_for_full_1)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_1}.npy'), spatial_cell_count_for_subregion_1)
    return


@app.cell
def _(np, z_unique_1):
    # split the Acumbens into anterior middle posterior
    (z_anterior_1, z_medial_1, z_posterior_1) = np.array_split(z_unique_1, 3)
    return


@app.cell
def _():
    # subset to Ce region
    xslice_1 = slice(420, 420 + 120)
    yslice_1 = slice(245, 245 + 120)
    return xslice_1, yslice_1


@app.cell
def _():
    pannel_key_6 = 'N'
    return (pannel_key_6,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_6,
    plt,
    sns,
    spatial_cell_count_for_subregion_1,
    target_site_subacronyms_1,
):
    for (idx_5, sub_acronym_1) in enumerate(target_site_subacronyms_1):
        sub_factors_1 = [1, 2]
        tarray_1 = spatial_cell_count_for_subregion_1[sub_factors_1, idx_5, :]  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        tarray_1 = tarray_1[:, ~np.isnan(tarray_1[0, :])]
        (fig_7, axs_7) = plt.subplots(1, 1, figsize=(3, 0.3))
        sns.heatmap(tarray_1, ax=axs_7, cmap='viridis', vmin=0, vmax=5e-05)
        axs_7.set_title(sub_acronym_1)
        axs_7.set_yticks(np.array(range(len(sub_factors_1))) + 0.5)
        axs_7.set_yticklabels(sub_factors_1, rotation=0)
        axs_7.set_xticks([0, tarray_1.shape[1]])
        axs_7.set_xticklabels(['anterior', 'posterior'])
        axs_7.set_ylabel('factor')
        axs_7.set_xlabel('z position')
        fig_7.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_6}_{sub_acronym_1}.png'), bbox_inches='tight', dpi=512)
        fig_7.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_6}_{sub_acronym_1}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    contour_img,
    cv2,
    factors_da,
    figure_key_4,
    os,
    plt,
    rejected_factor_idx,
    xslice_1,
    yslice_1,
    z_unique_1,
):
    # plot the spatial distribution of starter cells for each subject
    pannel_key_7 = 'M'
    for (cidx_2, rejected_factor_1) in enumerate(rejected_factor_idx):
        (fig_8, axs_8) = plt.subplots(1, len(z_unique_1[::5]), figsize=(len(z_unique_1[::5]) * 1.0, 1), sharex=True, sharey=True)
        theatmap_2 = factors_da[rejected_factor_1, :].reshape(atlas_img.shape).compute()
        (__, overlayed_image_2) = cv2.overlap_contour(theatmap_2[:, :, ::-1], contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.viridis)
        for (idx_6, curated_zplane_1) in enumerate(z_unique_1[::5]):
            ax_2 = axs_8[idx_6]
            ax_2.imshow(overlayed_image_2[curated_zplane_1, yslice_1, xslice_1])
            ax_2.set_xticks([])
            ax_2.set_yticks([])
            if idx_6 == 0:
                ax_2.set_ylabel(f'factor:{rejected_factor_1}', color='black')
            else:  # Remove x ticks
                ax_2.set_ylabel('', color='black')  # Remove y ticks
        fig_8.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_7}_factor_{rejected_factor_1}.png'), bbox_inches='tight', dpi=512)  #ax.axis('off')
        fig_8.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_7}_factor_{rejected_factor_1}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of BLA
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_3 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of Ce distribution
    """)
    return


@app.cell
def _(atlas_df_3, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_2 = 'BLA'
    target_site_id_2 = atlas_df_3.loc[atlas_df_3.acronym == target_site_acronym_2, 'id'].values[0]
    target_site_subids_2 = create_mask_for_region.get_subregions(atlas_df_3, target_site_id_2, return_original=True)['id'].values
    target_site_subacronyms_2 = create_mask_for_region.get_subregions(atlas_df_3, target_site_id_2, return_original=True)['acronym'].values
    return (
        target_site_acronym_2,
        target_site_subacronyms_2,
        target_site_subids_2,
    )


@app.cell
def _(atlas_img, np, target_site_subids_2):
    # collect all the z positions where there is the target site
    zs_2 = np.array([])
    for ID_2 in target_site_subids_2:
        (z__2, y__2, x__2) = np.where(atlas_img == ID_2)
        zs_2 = np.concatenate([zs_2, z__2])
    # find the center of mass of the VTA
    z_unique_2 = np.unique(zs_2).astype('uint16')
    z_center_2 = int(np.mean(zs_2))
    return (z_unique_2,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_2 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_2,)


@app.cell
def _(atlas_img, np, subset_heatmap_2, target_site_subids_2, z_unique_2):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_2 = np.array([[[np.nanmean(subset_heatmap_2[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_2] for target_site_subid in target_site_subids_2] for f in range(22)])
    return (spatial_cell_count_for_subregion_2,)


@app.cell
def _(np, spatial_cell_count_for_subregion_2):
    # calculate cell count for full region
    spatial_cell_count_for_full_2 = np.nansum(spatial_cell_count_for_subregion_2, axis=1)
    return (spatial_cell_count_for_full_2,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_2,
    spatial_cell_count_for_subregion_2,
    target_site_acronym_2,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_2}.npy'), spatial_cell_count_for_full_2)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_2}.npy'), spatial_cell_count_for_subregion_2)
    return


@app.cell
def _(np, z_unique_2):
    # split the Acumbens into anterior middle posterior
    (z_anterior_2, z_medial_2, z_posterior_2) = np.array_split(z_unique_2, 3)
    return


@app.cell
def _():
    pannel_key_8 = 'N'
    return (pannel_key_8,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_8,
    plt,
    sns,
    spatial_cell_count_for_subregion_2,
    target_site_subacronyms_2,
):
    for (idx_7, sub_acronym_2) in enumerate(target_site_subacronyms_2):
        sub_factors_2 = [1, 2]
        tarray_2 = spatial_cell_count_for_subregion_2[sub_factors_2, idx_7, :]  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        tarray_2 = tarray_2[:, ~np.isnan(tarray_2[0, :])]
        (fig_9, axs_9) = plt.subplots(1, 1, figsize=(3, 0.3))
        sns.heatmap(tarray_2, ax=axs_9, cmap='viridis', vmin=0, vmax=5e-05)
        axs_9.set_title(sub_acronym_2)
        axs_9.set_yticks(np.array(range(len(sub_factors_2))) + 0.5)
        axs_9.set_yticklabels(sub_factors_2, rotation=0)
        axs_9.set_xticks([0, tarray_2.shape[1]])
        axs_9.set_xticklabels(['anterior', 'posterior'])
        axs_9.set_ylabel('factor')
        axs_9.set_xlabel('z position')
        fig_9.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_8}_{sub_acronym_2}.png'), bbox_inches='tight', dpi=512)
        fig_9.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_8}_{sub_acronym_2}.pdf'), bbox_inches='tight', dpi=512)
    return (sub_acronym_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of La
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_4 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of Ce distribution
    """)
    return


@app.cell
def _(atlas_df_4, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_3 = 'La'
    target_site_id_3 = atlas_df_4.loc[atlas_df_4.acronym == target_site_acronym_3, 'id'].values[0]
    target_site_subids_3 = create_mask_for_region.get_subregions(atlas_df_4, target_site_id_3, return_original=True)['id'].values
    target_site_subacronyms_3 = create_mask_for_region.get_subregions(atlas_df_4, target_site_id_3, return_original=True)['acronym'].values
    return target_site_acronym_3, target_site_subids_3


@app.cell
def _(atlas_img, np, target_site_subids_3):
    # collect all the z positions where there is the target site
    zs_3 = np.array([])
    for ID_3 in target_site_subids_3:
        (z__3, y__3, x__3) = np.where(atlas_img == ID_3)
        zs_3 = np.concatenate([zs_3, z__3])
    # find the center of mass of the VTA
    z_unique_3 = np.unique(zs_3).astype('uint16')
    z_center_3 = int(np.mean(zs_3))
    return (z_unique_3,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_3 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_3,)


@app.cell
def _(atlas_img, np, subset_heatmap_3, target_site_subids_3, z_unique_3):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_3 = np.array([[[np.nanmean(subset_heatmap_3[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_3] for target_site_subid in target_site_subids_3] for f in range(22)])
    return (spatial_cell_count_for_subregion_3,)


@app.cell
def _(np, spatial_cell_count_for_subregion_3):
    # calculate cell count for full region
    spatial_cell_count_for_full_3 = np.nansum(spatial_cell_count_for_subregion_3, axis=1)
    return (spatial_cell_count_for_full_3,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_3,
    spatial_cell_count_for_subregion_3,
    target_site_acronym_3,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_3}.npy'), spatial_cell_count_for_full_3)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_3}.npy'), spatial_cell_count_for_subregion_3)
    return


@app.cell
def _(np, z_unique_3):
    # split the Acumbens into anterior middle posterior
    (z_anterior_3, z_medial_3, z_posterior_3) = np.array_split(z_unique_3, 3)
    return


@app.cell
def _():
    pannel_key_9 = 'N'
    return (pannel_key_9,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_9,
    plt,
    sns,
    spatial_cell_count_for_full_3,
    sub_acronym_2,
    target_site_acronym_3,
):
    sub_factors_3 = [1, 2]
    #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
    tarray_3 = spatial_cell_count_for_full_3[sub_factors_3, :]
    tarray_3 = tarray_3[:, ~np.isnan(tarray_3[0, :])]
    (fig_10, axs_10) = plt.subplots(1, 1, figsize=(3, 0.3))
    sns.heatmap(tarray_3, ax=axs_10, cmap='viridis', vmin=0, vmax=5e-05)
    axs_10.set_title(sub_acronym_2)
    axs_10.set_yticks(np.array(range(len(sub_factors_3))) + 0.5)
    axs_10.set_yticklabels(sub_factors_3, rotation=0)
    axs_10.set_xticks([0, tarray_3.shape[1]])
    axs_10.set_xticklabels(['anterior', 'posterior'])
    axs_10.set_ylabel('factor')
    axs_10.set_xlabel('z position')
    fig_10.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_9}_{target_site_acronym_3}.png'), bbox_inches='tight', dpi=512)
    fig_10.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_9}_{target_site_acronym_3}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of VeP
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_5 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of VeP distribution
    """)
    return


@app.cell
def _(atlas_df_5, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_4 = 'VeP'
    target_site_id_4 = atlas_df_5.loc[atlas_df_5.acronym == target_site_acronym_4, 'id'].values[0]
    target_site_subids_4 = create_mask_for_region.get_subregions(atlas_df_5, target_site_id_4, return_original=True)['id'].values
    target_site_subacronyms_4 = create_mask_for_region.get_subregions(atlas_df_5, target_site_id_4, return_original=True)['acronym'].values
    return (
        target_site_acronym_4,
        target_site_subacronyms_4,
        target_site_subids_4,
    )


@app.cell
def _(atlas_img, np, target_site_subids_4):
    # collect all the z positions where there is the target site
    zs_4 = np.array([])
    for ID_4 in target_site_subids_4:
        (z__4, y__4, x__4) = np.where(atlas_img == ID_4)
        zs_4 = np.concatenate([zs_4, z__4])
    # find the center of mass of the VTA
    z_unique_4 = np.unique(zs_4).astype('uint16')
    z_center_4 = int(np.mean(zs_4))
    return


@app.cell
def _(np):
    z_unique_5 = np.arange(102, 131, 1)  # manually pick the z planes for VeP
    return (z_unique_5,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_4 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_4,)


@app.cell
def _(atlas_img, np, subset_heatmap_4, target_site_subids_4, z_unique_5):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_4 = np.array([[[np.nanmean(subset_heatmap_4[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_5] for target_site_subid in target_site_subids_4] for f in range(22)])
    return (spatial_cell_count_for_subregion_4,)


@app.cell
def _(np, spatial_cell_count_for_subregion_4):
    # calculate cell count for full region
    spatial_cell_count_for_full_4 = np.nansum(spatial_cell_count_for_subregion_4, axis=1)
    return (spatial_cell_count_for_full_4,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_4,
    spatial_cell_count_for_subregion_4,
    target_site_acronym_4,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_4}.npy'), spatial_cell_count_for_full_4)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_4}.npy'), spatial_cell_count_for_subregion_4)
    return


@app.cell
def _(np, z_unique_5):
    # split the Acumbens into anterior middle posterior
    (z_anterior_4, z_medial_4, z_posterior_4) = np.array_split(z_unique_5, 3)
    return


@app.cell
def _():
    # subset to target region
    xslice_2 = slice(325, 325 + 120)
    yslice_2 = slice(255, 255 + 120)
    return xslice_2, yslice_2


@app.cell
def _():
    pannel_key_10 = 'P'
    return (pannel_key_10,)


@app.cell
def _(spatial_cell_count_for_full_4):
    spatial_cell_count_for_full_4.shape
    return


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_10,
    plt,
    sns,
    spatial_cell_count_for_full_4,
    sub_acronym_2,
    target_site_acronym_4,
):
    sub_factors_4 = [1, 2]
    #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
    tarray_4 = spatial_cell_count_for_full_4[sub_factors_4, :]
    tarray_4 = tarray_4[:, ~np.isnan(tarray_4[0, :])]
    (fig_11, axs_11) = plt.subplots(1, 1, figsize=(3, 0.3))
    sns.heatmap(tarray_4, ax=axs_11, cmap='viridis', vmin=0, vmax=5e-05)
    axs_11.set_title(sub_acronym_2)
    axs_11.set_yticks(np.array(range(len(sub_factors_4))) + 0.5)
    axs_11.set_yticklabels(sub_factors_4, rotation=0)
    axs_11.set_xticks([0, tarray_4.shape[1]])
    axs_11.set_xticklabels(['anterior', 'posterior'])
    axs_11.set_ylabel('factor')
    axs_11.set_xlabel('z position')
    fig_11.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_10}_{target_site_acronym_4}.png'), bbox_inches='tight', dpi=512)
    fig_11.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_10}_{target_site_acronym_4}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_10,
    plt,
    sns,
    spatial_cell_count_for_subregion_4,
    target_site_subacronyms_4,
):
    for (idx_8, sub_acronym_3) in enumerate(target_site_subacronyms_4):
        sub_factors_5 = [1, 2]
        tarray_5 = spatial_cell_count_for_subregion_4[sub_factors_5, idx_8, :]  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        tarray_5 = tarray_5[:, ~np.isnan(tarray_5[0, :])]
        (fig_12, axs_12) = plt.subplots(1, 1, figsize=(3, 0.3))
        sns.heatmap(tarray_5, ax=axs_12, cmap='viridis', vmin=0, vmax=5e-05)
        axs_12.set_title(sub_acronym_3)
        axs_12.set_yticks(np.array(range(len(sub_factors_5))) + 0.5)
        axs_12.set_yticklabels(sub_factors_5, rotation=0)
        axs_12.set_xticks([0, tarray_5.shape[1]])
        axs_12.set_xticklabels(['anterior', 'posterior'])
        axs_12.set_ylabel('factor')
        axs_12.set_xlabel('z position')
        fig_12.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_10}_{sub_acronym_3}.png'), bbox_inches='tight', dpi=512)
        fig_12.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_10}_{sub_acronym_3}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    contour_img,
    cv2,
    factors_da,
    figure_key_4,
    os,
    plt,
    rejected_factor_idx,
    xslice_2,
    yslice_2,
    z_unique_5,
):
    # plot the spatial distribution of starter cells for each subject
    pannel_key_11 = 'O'
    for (cidx_3, rejected_factor_2) in enumerate(rejected_factor_idx):
        (fig_13, axs_13) = plt.subplots(1, len(z_unique_5[::5]), figsize=(len(z_unique_5[::5]) * 1.0, 1), sharex=True, sharey=True)
        theatmap_3 = factors_da[rejected_factor_2, :].reshape(atlas_img.shape).compute()
        (__, overlayed_image_3) = cv2.overlap_contour(theatmap_3[:, :, ::-1], contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.viridis)
        for (idx_9, curated_zplane_2) in enumerate(z_unique_5[::5]):
            ax_3 = axs_13[idx_9]
            ax_3.imshow(overlayed_image_3[curated_zplane_2, yslice_2, xslice_2])
            ax_3.set_xticks([])
            ax_3.set_yticks([])
            if idx_9 == 0:
                ax_3.set_ylabel(f'factor:{rejected_factor_2}', color='black')
            else:  # Remove x ticks
                ax_3.set_ylabel('', color='black')  # Remove y ticks
        fig_13.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_11}_factor_{rejected_factor_2}.png'), bbox_inches='tight', dpi=512)  #ax.axis('off')
        fig_13.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_11}_factor_{rejected_factor_2}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of VTA
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_6 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of NAc distribution
    """)
    return


@app.cell
def _(atlas_df_6, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_5 = 'VTA'
    target_site_id_5 = atlas_df_6.loc[atlas_df_6.acronym == target_site_acronym_5, 'id'].values[0]
    target_site_subids_5 = create_mask_for_region.get_subregions(atlas_df_6, target_site_id_5, return_original=True)['id'].values
    target_site_subacronyms_5 = create_mask_for_region.get_subregions(atlas_df_6, target_site_id_5, return_original=True)['acronym'].values
    return (
        target_site_acronym_5,
        target_site_subacronyms_5,
        target_site_subids_5,
    )


@app.cell
def _(atlas_img, np, target_site_subids_5):
    # collect all the z positions where there is the target site
    zs_5 = np.array([])
    for ID_5 in target_site_subids_5:
        (z__5, y__5, x__5) = np.where(atlas_img == ID_5)
        zs_5 = np.concatenate([zs_5, z__5])
    # find the center of mass of the VTA
    z_unique_6 = np.unique(zs_5).astype('uint16')
    z_center_5 = int(np.mean(zs_5))
    return (z_unique_6,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_5 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_5,)


@app.cell
def _(atlas_img, np, subset_heatmap_5, target_site_subids_5, z_unique_6):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_5 = np.array([[[np.nanmean(subset_heatmap_5[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_6] for target_site_subid in target_site_subids_5] for f in range(22)])
    return (spatial_cell_count_for_subregion_5,)


@app.cell
def _(np, spatial_cell_count_for_subregion_5):
    # calculate cell count for full region
    spatial_cell_count_for_full_5 = np.nansum(spatial_cell_count_for_subregion_5, axis=1)
    return (spatial_cell_count_for_full_5,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_5,
    spatial_cell_count_for_subregion_5,
    target_site_acronym_5,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_5}.npy'), spatial_cell_count_for_full_5)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_5}.npy'), spatial_cell_count_for_subregion_5)
    return


@app.cell
def _(np, z_unique_6):
    # split the Acumbens into anterior middle posterior
    (z_anterior_5, z_medial_5, z_posterior_5) = np.array_split(z_unique_6, 3)
    return


@app.cell
def _():
    # subset to NAc region
    xslice_3 = slice(325, 325 + 120)
    yslice_3 = slice(240, 240 + 120)
    return xslice_3, yslice_3


@app.cell
def _():
    pannel_key_12 = 'L+'
    return (pannel_key_12,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_12,
    plt,
    sns,
    spatial_cell_count_for_subregion_5,
    target_site_subacronyms_5,
):
    for (idx_10, sub_acronym_4) in enumerate(target_site_subacronyms_5):
        sub_factors_6 = [1, 2]
        tarray_6 = spatial_cell_count_for_subregion_5[sub_factors_6, idx_10, :]  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        tarray_6 = tarray_6[:, ~np.isnan(tarray_6[0, :])]
        (fig_14, axs_14) = plt.subplots(1, 1, figsize=(3, 0.3))
        sns.heatmap(tarray_6, ax=axs_14, cmap='viridis', vmin=0, vmax=0.0001)
        axs_14.set_title(sub_acronym_4)
        axs_14.set_yticks(np.array(range(len(sub_factors_6))) + 0.5)
        axs_14.set_yticklabels(sub_factors_6, rotation=0)
        axs_14.set_xticks([0, tarray_6.shape[1]])
        axs_14.set_xticklabels(['anterior', 'posterior'])
        axs_14.set_ylabel('factor')
        axs_14.set_xlabel('z position')
        fig_14.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_12}_{sub_acronym_4}.png'), bbox_inches='tight', dpi=512)
        fig_14.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_12}_{sub_acronym_4}.pdf'), bbox_inches='tight', dpi=512)
    return (sub_acronym_4,)


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    contour_img,
    cv2,
    factors_da,
    figure_key_4,
    os,
    plt,
    rejected_factor_idx,
    xslice_3,
    yslice_3,
    z_unique_6,
):
    # plot the spatial distribution of starter cells for each subject
    pannel_key_13 = 'K+'
    for (cidx_4, rejected_factor_3) in enumerate(rejected_factor_idx):
        (fig_15, axs_15) = plt.subplots(1, len(z_unique_6[::5]), figsize=(len(z_unique_6[::5]) * 1.0, 1), sharex=True, sharey=True)
        theatmap_4 = factors_da[rejected_factor_3, :].reshape(atlas_img.shape).compute()
        (__, overlayed_image_4) = cv2.overlap_contour(theatmap_4[:, :, ::-1], contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.viridis)
        for (idx_11, curated_zplane_3) in enumerate(z_unique_6[::5]):
            ax_4 = axs_15[idx_11]
            ax_4.imshow(overlayed_image_4[curated_zplane_3, yslice_3, xslice_3])
            ax_4.set_xticks([])
            ax_4.set_yticks([])
            if idx_11 == 0:
                ax_4.set_ylabel(f'factor:{rejected_factor_3}', color='black')
            else:
                ax_4.set_ylabel('', color='black')  # Remove x ticks
        fig_15.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_13}_factor_{rejected_factor_3}.png'), bbox_inches='tight', dpi=512)  # Remove y ticks
        fig_15.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_13}_factor_{rejected_factor_3}.pdf'), bbox_inches='tight', dpi=512)  #ax.axis('off')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of Ce
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of OFC
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_7 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of OFC distribution
    """)
    return


@app.cell
def _(atlas_df_7, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_6 = 'O'
    target_site_id_6 = atlas_df_7.loc[atlas_df_7.acronym == target_site_acronym_6, 'id'].values[0]
    target_site_subids_6 = create_mask_for_region.get_subregions(atlas_df_7, target_site_id_6, return_original=True)['id'].values
    target_site_subacronyms_6 = create_mask_for_region.get_subregions(atlas_df_7, target_site_id_6, return_original=True)['acronym'].values
    return target_site_acronym_6, target_site_subids_6


@app.cell
def _(atlas_img, np, target_site_subids_6):
    # collect all the z positions where there is the target site
    zs_6 = np.array([])
    for ID_6 in target_site_subids_6:
        (z__6, y__6, x__6) = np.where(atlas_img == ID_6)
        zs_6 = np.concatenate([zs_6, z__6])
    # find the center of mass of the VTA
    z_unique_7 = np.unique(zs_6).astype('uint16')
    z_center_6 = int(np.mean(zs_6))
    return (z_unique_7,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_6 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_6,)


@app.cell
def _(atlas_img, np, subset_heatmap_6, target_site_subids_6, z_unique_7):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_6 = np.array([[[np.nanmean(subset_heatmap_6[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_7] for target_site_subid in target_site_subids_6] for f in range(22)])
    return (spatial_cell_count_for_subregion_6,)


@app.cell
def _(np, spatial_cell_count_for_subregion_6):
    # calculate cell count for full region
    spatial_cell_count_for_full_6 = np.nansum(spatial_cell_count_for_subregion_6, axis=1)
    return (spatial_cell_count_for_full_6,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_6,
    spatial_cell_count_for_subregion_6,
    target_site_acronym_6,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_6}.npy'), spatial_cell_count_for_full_6)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_6}.npy'), spatial_cell_count_for_subregion_6)
    return


@app.cell
def _(np, z_unique_7):
    # split the Acumbens into anterior middle posterior
    (z_anterior_6, z_medial_6, z_posterior_6) = np.array_split(z_unique_7, 3)
    return


@app.cell
def _():
    # subset to target region
    xslice_4 = slice(325, 325 + 120)
    yslice_4 = slice(160, 160 + 120)
    return xslice_4, yslice_4


@app.cell
def _():
    pannel_key_14 = 'R'
    return (pannel_key_14,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_14,
    plt,
    sns,
    spatial_cell_count_for_full_6,
    sub_acronym_4,
    target_site_acronym_6,
):
    sub_factors_7 = [1, 2]
    #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
    tarray_7 = spatial_cell_count_for_full_6[sub_factors_7, :]
    tarray_7 = tarray_7[:, ~np.isnan(tarray_7[0, :])]
    (fig_16, axs_16) = plt.subplots(1, 1, figsize=(3, 0.3))
    sns.heatmap(tarray_7, ax=axs_16, cmap='viridis', vmin=0, vmax=0.0005)
    axs_16.set_title(sub_acronym_4)
    axs_16.set_yticks(np.array(range(len(sub_factors_7))) + 0.5)
    axs_16.set_yticklabels(sub_factors_7, rotation=0)
    axs_16.set_xticks([0, tarray_7.shape[1]])
    axs_16.set_xticklabels(['anterior', 'posterior'])
    axs_16.set_ylabel('factor')
    axs_16.set_xlabel('z position')
    fig_16.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_14}_{target_site_acronym_6}.png'), bbox_inches='tight', dpi=512)
    fig_16.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_14}_{target_site_acronym_6}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    contour_img,
    cv2,
    factors_da,
    figure_key_4,
    os,
    plt,
    rejected_factor_idx,
    xslice_4,
    yslice_4,
    z_unique_7,
):
    # plot the spatial distribution of starter cells for each subject
    pannel_key_15 = 'Q'
    for (cidx_5, rejected_factor_4) in enumerate(rejected_factor_idx):
        (fig_17, axs_17) = plt.subplots(1, len(z_unique_7[::5]), figsize=(len(z_unique_7[::5]) * 1.0, 1), sharex=True, sharey=True)
        theatmap_5 = factors_da[rejected_factor_4, :].reshape(atlas_img.shape).compute()
        (__, overlayed_image_5) = cv2.overlap_contour(theatmap_5[:, :, ::-1], contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.viridis)
        for (idx_12, curated_zplane_4) in enumerate(z_unique_7[::5]):
            ax_5 = axs_17[idx_12]
            ax_5.imshow(overlayed_image_5[curated_zplane_4, yslice_4, xslice_4])
            ax_5.set_xticks([])
            ax_5.set_yticks([])
            if idx_12 == 0:
                ax_5.set_ylabel(f'factor:{rejected_factor_4}', color='black')
            else:  # Remove x ticks
                ax_5.set_ylabel('', color='black')  # Remove y ticks
        fig_17.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_15}_factor_{rejected_factor_4}.png'), bbox_inches='tight', dpi=512)  #ax.axis('off')
        fig_17.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_15}_factor_{rejected_factor_4}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _(
    analysis_figurepath,
    analysis_resultpath,
    atlas_df_7,
    atlas_img,
    create_mask_for_region,
    factors_da,
    figure_key_4,
    np,
    os,
    plt,
    sns,
):
    # target site can be experiment specific
    #target_site_acronym = 'VO'
    for target_site_acronym_7 in ['MO', 'LO', 'VO']:
        target_site_id_7 = atlas_df_7.loc[atlas_df_7.acronym == target_site_acronym_7, 'id'].values[0]
        target_site_subids_7 = create_mask_for_region.get_subregions(atlas_df_7, target_site_id_7, return_original=True)['id'].values
        target_site_subacronyms_7 = create_mask_for_region.get_subregions(atlas_df_7, target_site_id_7, return_original=True)['acronym'].values
        zs_7 = np.array([])  # collect all the z positions where there is the target site
        for ID_7 in target_site_subids_7:
            (z__7, y__7, x__7) = np.where(atlas_img == ID_7)
            zs_7 = np.concatenate([zs_7, z__7])
        z_unique_8 = np.unique(zs_7).astype('uint16')
        z_center_7 = int(np.mean(zs_7))  # find the center of mass of the VTA
        subset_heatmap_7 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
        spatial_cell_count_for_subregion_7 = np.array([[[np.nanmean(subset_heatmap_7[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_8] for target_site_subid in target_site_subids_7] for f in range(22)])
        spatial_cell_count_for_full_7 = np.nansum(spatial_cell_count_for_subregion_7, axis=1)
        np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_7}.npy'), spatial_cell_count_for_full_7)  # calculate the spatial distribution of c-Fos+ cells for each subject
        np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_7}.npy'), spatial_cell_count_for_subregion_7)
        (z_anterior_7, z_medial_7, z_posterior_7) = np.array_split(z_unique_8, 3)  # append spatial_c
        xslice_5 = slice(325, 325 + 120)  # calculate cell count for full region
        yslice_5 = slice(255, 255 + 120)
        pannel_key_16 = 'P'
        sub_factors_8 = [1, 2]
        tarray_8 = spatial_cell_count_for_full_7[sub_factors_8, :]  # split the Acumbens into anterior middle posterior
        tarray_8 = tarray_8[:, ~np.isnan(tarray_8[0, :])]
        (fig_18, axs_18) = plt.subplots(1, 1, figsize=(3, 0.3))  # subset to target region
        sns.heatmap(tarray_8, ax=axs_18, cmap='viridis', vmin=0, vmax=0.0005)
        axs_18.set_title(target_site_acronym_7)
        axs_18.set_yticks(np.array(range(len(sub_factors_8))) + 0.5)
        axs_18.set_yticklabels(sub_factors_8, rotation=0)
        axs_18.set_xticks([0, tarray_8.shape[1]])  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        axs_18.set_xticklabels(['anterior', 'posterior'])
        axs_18.set_ylabel('factor')
        axs_18.set_xlabel('z position')
        fig_18.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_16}_{target_site_acronym_7}.png'), bbox_inches='tight', dpi=512)
        fig_18.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_16}_{target_site_acronym_7}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of Insular
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_8 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of Insular distribution
    """)
    return


@app.cell
def _(atlas_df_8, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_8 = 'Insular'
    # for this case the target site is the insular cortex and is defined by acronym AI, GI and DI
    # get the ids for these regions and combine them as target_site_ids, 
    target_site_ids = atlas_df_8.loc[atlas_df_8.acronym.isin(['AI', 'GI', 'DI']), 'id'].values
    # then get target_site_subids for all sub regions of target_site_ids
    target_site_subids_8 = [create_mask_for_region.get_subregions(atlas_df_8, target_site_id, return_original=True)['id'].values for target_site_id in target_site_ids]
    return target_site_acronym_8, target_site_subids_8


@app.cell
def _(np, target_site_subids_8):
    # flatten target_site_subids, a list, into one np array
    target_site_subids_9 = np.concatenate(target_site_subids_8).astype('int')
    return (target_site_subids_9,)


@app.cell
def _(atlas_img, np, target_site_subids_9):
    # collect all the z positions where there is the target site
    zs_8 = np.array([])
    for ID_8 in target_site_subids_9:
        (z__8, y__8, x__8) = np.where(atlas_img == ID_8)
        zs_8 = np.concatenate([zs_8, z__8])
    # find the center of mass of the VTA
    z_unique_9 = np.unique(zs_8).astype('uint16')
    z_center_8 = int(np.mean(zs_8))
    return (z_unique_9,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_8 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_8,)


@app.cell
def _(atlas_img, np, subset_heatmap_8, target_site_subids_9, z_unique_9):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_8 = np.array([[[np.nanmean(subset_heatmap_8[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_9] for target_site_subid in target_site_subids_9] for f in range(22)])
    return (spatial_cell_count_for_subregion_8,)


@app.cell
def _(np, spatial_cell_count_for_subregion_8):
    # calculate cell count for full region
    spatial_cell_count_for_full_8 = np.nansum(spatial_cell_count_for_subregion_8, axis=1)
    return (spatial_cell_count_for_full_8,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_8,
    spatial_cell_count_for_subregion_8,
    target_site_acronym_8,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_8}.npy'), spatial_cell_count_for_full_8)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_8}.npy'), spatial_cell_count_for_subregion_8)
    return


@app.cell
def _(np, z_unique_9):
    # split the Acumbens into anterior middle posterior
    (z_anterior_8, z_medial_8, z_posterior_8) = np.array_split(z_unique_9, 3)
    return


@app.cell
def _():
    pannel_key_17 = 'T'
    return (pannel_key_17,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_17,
    plt,
    sns,
    spatial_cell_count_for_full_8,
    target_site_acronym_8,
):
    sub_factors_9 = [1, 2]
    #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
    tarray_9 = spatial_cell_count_for_full_8[sub_factors_9, :]
    tarray_9 = tarray_9[:, ~np.isnan(tarray_9[0, :])]
    (fig_19, axs_19) = plt.subplots(1, 1, figsize=(3, 0.3))
    sns.heatmap(tarray_9, ax=axs_19, cmap='viridis', vmin=0, vmax=0.0005)
    axs_19.set_title(target_site_acronym_8)
    axs_19.set_yticks(np.array(range(len(sub_factors_9))) + 0.5)
    axs_19.set_yticklabels(sub_factors_9, rotation=0)
    axs_19.set_xticks([0, tarray_9.shape[1]])
    axs_19.set_xticklabels(['anterior', 'posterior'])
    axs_19.set_ylabel('factor')
    axs_19.set_xlabel('z position')
    fig_19.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_17}_{target_site_acronym_8}.png'), bbox_inches='tight', dpi=512)
    fig_19.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_17}_{target_site_acronym_8}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _(
    analysis_figurepath,
    analysis_resultpath,
    atlas_df_8,
    atlas_img,
    create_mask_for_region,
    factors_da,
    figure_key_4,
    np,
    os,
    plt,
    sns,
):
    # target site can be experiment specific
    for target_site_acronym_9 in ['AI', 'GI', 'DI']:
        target_site_id_8 = atlas_df_8.loc[atlas_df_8.acronym == target_site_acronym_9, 'id'].values[0]
        target_site_subids_10 = create_mask_for_region.get_subregions(atlas_df_8, target_site_id_8, return_original=True)['id'].values
        target_site_subacronyms_8 = create_mask_for_region.get_subregions(atlas_df_8, target_site_id_8, return_original=True)['acronym'].values
        zs_9 = np.array([])  # collect all the z positions where there is the target site
        for ID_9 in target_site_subids_10:
            (z__9, y__9, x__9) = np.where(atlas_img == ID_9)
            zs_9 = np.concatenate([zs_9, z__9])
        z_unique_10 = np.unique(zs_9).astype('uint16')
        z_center_9 = int(np.mean(zs_9))  # find the center of mass of the VTA
        subset_heatmap_9 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
        spatial_cell_count_for_subregion_9 = np.array([[[np.nanmean(subset_heatmap_9[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_10] for target_site_subid in target_site_subids_10] for f in range(22)])
        spatial_cell_count_for_full_9 = np.nansum(spatial_cell_count_for_subregion_9, axis=1)
        np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_9}.npy'), spatial_cell_count_for_full_9)  # calculate the spatial distribution of c-Fos+ cells for each subject
        np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_9}.npy'), spatial_cell_count_for_subregion_9)
        (z_anterior_9, z_medial_9, z_posterior_9) = np.array_split(z_unique_10, 3)  # append spatial_c
        xslice_6 = slice(325, 325 + 120)  # calculate cell count for full region
        yslice_6 = slice(255, 255 + 120)
        pannel_key_18 = 'P'
        sub_factors_10 = [1, 2]
        tarray_10 = spatial_cell_count_for_full_9[sub_factors_10, :]  # split the Acumbens into anterior middle posterior
        tarray_10 = tarray_10[:, ~np.isnan(tarray_10[0, :])]
        (fig_20, axs_20) = plt.subplots(1, 1, figsize=(3, 0.3))  # subset to target region
        sns.heatmap(tarray_10, ax=axs_20, cmap='viridis', vmin=0, vmax=0.0005)
        axs_20.set_title(target_site_acronym_9)
        axs_20.set_yticks(np.array(range(len(sub_factors_10))) + 0.5)
        axs_20.set_yticklabels(sub_factors_10, rotation=0)
        axs_20.set_xticks([0, tarray_10.shape[1]])  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        axs_20.set_xticklabels(['anterior', 'posterior'])
        axs_20.set_ylabel('factor')
        axs_20.set_xlabel('z position')
        fig_20.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_18}_{target_site_acronym_9}.png'), bbox_inches='tight', dpi=512)
        fig_20.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_18}_{target_site_acronym_9}.pdf'), bbox_inches='tight', dpi=512)
    return target_site_subids_10, z_unique_10


@app.cell
def _(
    atlas_img,
    contour_img,
    cv2,
    factors_da,
    get_slice,
    plt,
    rejected_factor_idx,
    target_site_subids_10,
    z_unique_10,
):
    # plot the spatial distribution of starter cells for each subject
    pannel_key_19 = 'S'
    for (cidx_6, rejected_factor_5) in enumerate(rejected_factor_idx):
        (fig_21, axs_21) = plt.subplots(1, len(z_unique_10[::5]), figsize=(len(z_unique_10[::5]) * 1.0, 1), sharex=True, sharey=True)
        theatmap_6 = factors_da[rejected_factor_5, :].reshape(atlas_img.shape).compute()
        (__, overlayed_image_6) = cv2.overlap_contour(theatmap_6[:, :, ::-1], contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.coolwarm)
        for (idx_13, curated_zplane_5) in enumerate(z_unique_10[::5]):
            ax_6 = axs_21[idx_13]
            (yslice_7, xslice_7) = get_slice(curated_zplane_5, target_site_subids_10, hemi='right')
            ax_6.imshow(overlayed_image_6[curated_zplane_5, yslice_7, xslice_7])
            ax_6.set_xticks([])
            ax_6.set_yticks([])
            if idx_13 == 0:
                ax_6.set_ylabel(f'factor:{rejected_factor_5}', color='black')
            else:  # Remove x ticks
                ax_6.set_ylabel('', color='black')  # Remove y ticks  #ax.axis('off')  #fig.savefig(os.path.join(analysis_figurepath,f'{figure_key}{pannel_key}_factor_{rejected_factor}.png'),bbox_inches='tight',dpi = 512)  #fig.savefig(os.path.join(analysis_figurepath,f'{figure_key}{pannel_key}_factor_{rejected_factor}.pdf'),bbox_inches='tight',dpi = 512)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of PVT
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_9 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of NAc distribution
    """)
    return


@app.cell
def _(atlas_df_9, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_10 = 'PVT'
    target_site_id_9 = atlas_df_9.loc[atlas_df_9.acronym == target_site_acronym_10, 'id'].values[0]
    target_site_subids_11 = create_mask_for_region.get_subregions(atlas_df_9, target_site_id_9, return_original=True)['id'].values
    target_site_subacronyms_9 = create_mask_for_region.get_subregions(atlas_df_9, target_site_id_9, return_original=True)['acronym'].values
    return (
        target_site_acronym_10,
        target_site_subacronyms_9,
        target_site_subids_11,
    )


@app.cell
def _(atlas_img, np, target_site_subids_11):
    # collect all the z positions where there is the target site
    zs_10 = np.array([])
    for ID_10 in target_site_subids_11:
        (z__10, y__10, x__10) = np.where(atlas_img == ID_10)
        zs_10 = np.concatenate([zs_10, z__10])
    # find the center of mass of the VTA
    z_unique_11 = np.unique(zs_10).astype('uint16')
    z_center_10 = int(np.mean(zs_10))
    return (z_unique_11,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_10 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_10,)


@app.cell
def _(atlas_img, np, subset_heatmap_10, target_site_subids_11, z_unique_11):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_10 = np.array([[[np.nanmean(subset_heatmap_10[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_11] for target_site_subid in target_site_subids_11] for f in range(22)])
    return (spatial_cell_count_for_subregion_10,)


@app.cell
def _(np, spatial_cell_count_for_subregion_10):
    # calculate cell count for full region
    spatial_cell_count_for_full_10 = np.nansum(spatial_cell_count_for_subregion_10, axis=1)
    return (spatial_cell_count_for_full_10,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_10,
    spatial_cell_count_for_subregion_10,
    target_site_acronym_10,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_10}.npy'), spatial_cell_count_for_full_10)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_10}.npy'), spatial_cell_count_for_subregion_10)
    return


@app.cell
def _(np, z_unique_11):
    # split the Acumbens into anterior middle posterior
    (z_anterior_10, z_medial_10, z_posterior_10) = np.array_split(z_unique_11, 3)
    return


@app.cell
def _():
    # subset to NAc region
    xslice_8 = slice(325, 325 + 120)
    yslice_8 = slice(240, 240 + 120)
    return


@app.cell
def _():
    pannel_key_20 = 'J+'
    return (pannel_key_20,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_20,
    plt,
    sns,
    spatial_cell_count_for_subregion_10,
    target_site_subacronyms_9,
):
    for (idx_14, sub_acronym_5) in enumerate(target_site_subacronyms_9):
        sub_factors_11 = [1, 2]
        tarray_11 = spatial_cell_count_for_subregion_10[sub_factors_11, idx_14, :]  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        tarray_11 = tarray_11[:, ~np.isnan(tarray_11[0, :])]
        (fig_22, axs_22) = plt.subplots(1, 1, figsize=(3, 0.3))
        sns.heatmap(tarray_11, ax=axs_22, cmap='viridis', vmin=0, vmax=0.0005)
        axs_22.set_title(sub_acronym_5)
        axs_22.set_yticks(np.array(range(len(sub_factors_11))) + 0.5)
        axs_22.set_yticklabels(sub_factors_11, rotation=0)
        axs_22.set_xticks([0, tarray_11.shape[1]])
        axs_22.set_xticklabels(['anterior', 'posterior'])
        axs_22.set_ylabel('factor')
        axs_22.set_xlabel('z position')
        fig_22.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_20}_{sub_acronym_5}.png'), bbox_inches='tight', dpi=512)
        fig_22.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_20}_{sub_acronym_5}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _(atlas_df_9):
    atlas_df_9[atlas_df_9.acronym == 'PV'].name.values
    return


@app.cell
def _():
    rejected_factor_idx_1 = [0, 1, 2, 3, 16]
    return (rejected_factor_idx_1,)


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    contour_img,
    cv2,
    factors_da,
    figure_key_4,
    get_slice,
    os,
    plt,
    rejected_factor_idx_1,
    target_site_subids_11,
    z_unique_11,
):
    # plot the spatial distribution of starter cells for each subject
    pannel_key_21 = 'J+2'
    for (cidx_7, rejected_factor_6) in enumerate(rejected_factor_idx_1):
        (fig_23, axs_23) = plt.subplots(1, len(z_unique_11[::5]), figsize=(len(z_unique_11[::5]) * 1.0, 1), sharex=True, sharey=True)
        theatmap_7 = factors_da[rejected_factor_6, :].reshape(atlas_img.shape).compute()
        (__, overlayed_image_7) = cv2.overlap_contour(theatmap_7[:, :, ::-1], contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.viridis)
        for (idx_15, curated_zplane_6) in enumerate(z_unique_11[::5]):
            ax_7 = axs_23[idx_15]
            (yslice_9, xslice_9) = get_slice(curated_zplane_6, target_site_subids_11, hemi='center')
            ax_7.imshow(overlayed_image_7[curated_zplane_6, yslice_9, xslice_9])
            ax_7.set_xticks([])
            ax_7.set_yticks([])
            if idx_15 == 0:
                ax_7.set_ylabel(f'factor:{rejected_factor_6}', color='black')
            else:
                ax_7.set_ylabel('', color='black')
        fig_23.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_21}_factor_{rejected_factor_6}.png'), bbox_inches='tight', dpi=512)
        fig_23.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_21}_factor_{rejected_factor_6}.pdf'), bbox_inches='tight', dpi=512)  # Remove x ticks  # Remove y ticks  #ax.axis('off')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 5: spatial analysis of Cl
    """)
    return


@app.cell
def _(ATLAS_INFO_CSV, pd):
    atlas_df_10 = pd.read_csv(ATLAS_INFO_CSV, index_col=False)
    return (atlas_df_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    conduct analysis of NAc distribution
    """)
    return


@app.cell
def _(atlas_df_10, create_mask_for_region):
    # target site can be experiment specific
    target_site_acronym_11 = 'Cl'
    target_site_id_10 = atlas_df_10.loc[atlas_df_10.acronym == target_site_acronym_11, 'id'].values[0]
    target_site_subids_12 = create_mask_for_region.get_subregions(atlas_df_10, target_site_id_10, return_original=True)['id'].values
    target_site_subacronyms_10 = create_mask_for_region.get_subregions(atlas_df_10, target_site_id_10, return_original=True)['acronym'].values
    return (
        target_site_acronym_11,
        target_site_subacronyms_10,
        target_site_subids_12,
    )


@app.cell
def _(atlas_img, np, target_site_subids_12):
    # collect all the z positions where there is the target site
    zs_11 = np.array([])
    for ID_11 in target_site_subids_12:
        (z__11, y__11, x__11) = np.where(atlas_img == ID_11)
        zs_11 = np.concatenate([zs_11, z__11])
    # find the center of mass of the VTA
    z_unique_12 = np.unique(zs_11).astype('uint16')
    z_center_11 = int(np.mean(zs_11))
    return (z_unique_12,)


@app.cell
def _(atlas_img, factors_da, np):
    subset_heatmap_11 = np.reshape(factors_da, [22] + list(atlas_img.shape)).compute()
    return (subset_heatmap_11,)


@app.cell
def _(atlas_img, np, subset_heatmap_11, target_site_subids_12, z_unique_12):
    # calculate the spatial distribution of c-Fos+ cells for each subject
    # append spatial_c
    spatial_cell_count_for_subregion_11 = np.array([[[np.nanmean(subset_heatmap_11[f, z, np.where(atlas_img[z, :, :] == target_site_subid)[0], np.where(atlas_img[z, :, :] == target_site_subid)[1]]) for z in z_unique_12] for target_site_subid in target_site_subids_12] for f in range(22)])
    return (spatial_cell_count_for_subregion_11,)


@app.cell
def _(np, spatial_cell_count_for_subregion_11):
    # calculate cell count for full region
    spatial_cell_count_for_full_11 = np.nansum(spatial_cell_count_for_subregion_11, axis=1)
    return (spatial_cell_count_for_full_11,)


@app.cell
def _(
    analysis_resultpath,
    np,
    os,
    spatial_cell_count_for_full_11,
    spatial_cell_count_for_subregion_11,
    target_site_acronym_11,
):
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_sum_{target_site_acronym_11}.npy'), spatial_cell_count_for_full_11)
    np.save(os.path.join(analysis_resultpath, f'spatial_factors_for_subregion_{target_site_acronym_11}.npy'), spatial_cell_count_for_subregion_11)
    return


@app.cell
def _(np, z_unique_12):
    # split the Acumbens into anterior middle posterior
    (z_anterior_11, z_medial_11, z_posterior_11) = np.array_split(z_unique_12, 3)
    return


@app.cell
def _():
    # subset to NAc region
    xslice_10 = slice(325, 325 + 120)
    yslice_10 = slice(240, 240 + 120)
    return


@app.cell
def _(target_site_acronym_11):
    pannel_key_22 = f'{target_site_acronym_11}+'
    return (pannel_key_22,)


@app.cell
def _(
    analysis_figurepath,
    figure_key_4,
    np,
    os,
    pannel_key_22,
    plt,
    sns,
    spatial_cell_count_for_subregion_11,
    target_site_subacronyms_10,
):
    for (idx_16, sub_acronym_6) in enumerate(target_site_subacronyms_10):
        sub_factors_12 = [1, 2]
        tarray_12 = spatial_cell_count_for_subregion_11[sub_factors_12, idx_16, :]  #tarray = spatial_cell_count_for_subregion[rejected_factor_idx,idx,:]
        tarray_12 = tarray_12[:, ~np.isnan(tarray_12[0, :])]
        (fig_24, axs_24) = plt.subplots(1, 1, figsize=(3, 0.3))
        sns.heatmap(tarray_12, ax=axs_24, cmap='viridis', vmin=0, vmax=0.0001)
        axs_24.set_title(sub_acronym_6)
        axs_24.set_yticks(np.array(range(len(sub_factors_12))) + 0.5)
        axs_24.set_yticklabels(sub_factors_12, rotation=0)
        axs_24.set_xticks([0, tarray_12.shape[1]])
        axs_24.set_xticklabels(['anterior', 'posterior'])
        axs_24.set_ylabel('factor')
        axs_24.set_xlabel('z position')
        fig_24.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_22}_{sub_acronym_6}.png'), bbox_inches='tight', dpi=512)
        fig_24.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_22}_{sub_acronym_6}.pdf'), bbox_inches='tight', dpi=512)
    return


@app.cell
def _():
    rejected_factor_idx_2 = [0, 1, 2, 3, 16]
    return (rejected_factor_idx_2,)


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    contour_img,
    cv2,
    factors_da,
    figure_key_4,
    get_slice,
    os,
    plt,
    rejected_factor_idx_2,
    target_site_acronym_11,
    target_site_subids_12,
    z_unique_12,
):
    # plot the spatial distribution of starter cells for each subject
    pannel_key_23 = f'{target_site_acronym_11}+2'
    for (cidx_8, rejected_factor_7) in enumerate(rejected_factor_idx_2):
        (fig_25, axs_25) = plt.subplots(1, len(z_unique_12[::5]), figsize=(len(z_unique_12[::5]) * 1.0, 1), sharex=True, sharey=True)
        theatmap_8 = factors_da[rejected_factor_7, :].reshape(atlas_img.shape).compute()
        (__, overlayed_image_8) = cv2.overlap_contour(theatmap_8[:, :, ::-1], contour_img, cmin=0, cmax=0.00015, outputpath=None, colormap=plt.cm.viridis)
        for (idx_17, curated_zplane_7) in enumerate(z_unique_12[::5]):
            ax_8 = axs_25[idx_17]
            (yslice_11, xslice_11) = get_slice(curated_zplane_7, target_site_subids_12, hemi='right')
            ax_8.imshow(overlayed_image_8[curated_zplane_7, yslice_11, xslice_11])
            ax_8.set_xticks([])
            ax_8.set_yticks([])
            if idx_17 == 0:
                ax_8.set_ylabel(f'factor:{rejected_factor_7}', color='black')
            else:
                ax_8.set_ylabel('', color='black')
        fig_25.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_23}_factor_{rejected_factor_7}.png'), bbox_inches='tight', dpi=512)
        fig_25.savefig(os.path.join(analysis_figurepath, f'{figure_key_4}{pannel_key_23}_factor_{rejected_factor_7}.pdf'), bbox_inches='tight', dpi=512)  # Remove x ticks  # Remove y ticks  #ax.axis('off')
    return


if __name__ == "__main__":
    app.run()
