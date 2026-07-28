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
    # Figure 4: acute opioid c-Fos mapping

    Cleaned notebook for Figure 4 generation and statistical-table export.

    Notes:
    - Raw preprocessing, voxel-wise regression, and TreeBH/TreeFDR computation cells were removed.
    - TreeBH/TreeFDR is assumed to be run separately. This notebook only loads the existing TreeFDR output CSV files.
    - UMAP/HDBSCAN clustering parameters are kept from the original figure-generation notebook: `n_neighbors = 3`, `min_dist = 0.05`, `random_state = 12`, `HDBSCAN(min_cluster_size=2)`, and `get_clusters(1, min_cluster_size=2)`.
    - The clustering output is written to CSV so downstream statistics and figure tables use the same cluster assignment.
    """)
    return


@app.cell
def _():
    # ============================================================
    # Imports and plotting defaults
    # ============================================================
    import os
    import glob
    import pickle
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import dask.array as da
    import tifffile
    import matplotlib.pyplot as plt
    import seaborn as sns

    from adjustText import adjust_text
    from scipy import stats
    from scipy.cluster.hierarchy import leaves_list
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
    from sklearn.preprocessing import StandardScaler

    import umap.umap_
    import hdbscan

    from brain_vis.overlay import overlap_contour
    from brain_vis.rgba import set_transparency
    from brain_vis.regions import get_subregions

    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.facecolor": "none",
        "legend.edgecolor": "none",
        "text.color": "black",
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return (
        StandardScaler,
        adjust_text,
        anova_lm,
        da,
        get_subregions,
        glob,
        hdbscan,
        leaves_list,
        multipletests,
        np,
        ols,
        os,
        overlap_contour,
        pairwise_tukeyhsd,
        pd,
        pickle,
        plt,
        set_transparency,
        sns,
        stats,
        tifffile,
        umap,
    )


@app.cell
def _(os):
    # ============================================================
    # Paths, condition labels, and output folders
    # ============================================================
    from pathlib import Path

    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    GROUP = DATA_ROOT / "04_acute_opioids_4drugs"
    assert GROUP.exists(), f"DATA_ROOT not found: {DATA_ROOT}. Set OPIOID_DATA_ROOT."

    metapath = str(GROUP)                             # meta CSV in the group folder
    atlaspath = str(DATA_ROOT / "shared" / "atlas")   # shared atlas / ontology
    analysis_resultpath = str(GROUP)                  # processed results + precomputed stat CSVs
    FIGS7_STATS_DIR = str(GROUP)                       # stat-table CSVs read/written here
    analysis_figurepath = "../figures"                # rendered panels -> repo
    os.makedirs(analysis_figurepath, exist_ok=True)

    Conditions = [
        "Saline",
        "Acute_Morphine",
        "Acute_Fentanyl",
        "Acute_Buprenorphine",
        "Acute_Oxycodone",
    ]
    Condition_figure_name = ["Saline", "Morphine", "Fentanyl", "Buprenorphine", "Oxycodone"]
    condition_colors = ["gray", "lime", "magenta", "darkcyan", "orange"]
    condition_pretty = dict(zip(Conditions, Condition_figure_name))

    figure_key = "Figure4"
    channel = "Ex_639_Ch2_stitched"

    print(f"analysis_resultpath: {analysis_resultpath}")
    print(f"analysis_figurepath: {analysis_figurepath}")
    print(f"FIGS7_STATS_DIR: {FIGS7_STATS_DIR}")
    return (
        Condition_figure_name,
        Conditions,
        FIGS7_STATS_DIR,
        analysis_figurepath,
        analysis_resultpath,
        atlaspath,
        channel,
        condition_colors,
        condition_pretty,
        figure_key,
        metapath,
    )


@app.cell
def _(atlas_df, np, os, pd):
    # ============================================================
    # Helper functions
    # ============================================================
    def sem(x):
        x = pd.Series(x).dropna().astype(float)
        if len(x) <= 1:
            return np.nan
        return x.std(ddof=1) / np.sqrt(len(x))


    def p_to_sig_text(p):
        if pd.isna(p) or float(p) >= 0.05:
            return ""
        if float(p) < 0.001:
            return "***"
        if float(p) < 0.01:
            return "**"
        return "*"


    def fmt_p(p):
        if pd.isna(p):
            return "NA"
        p = float(p)
        if p == 0:
            return "< 1e-300"
        if p < 1e-3:
            base, exp = f"{p:.2e}".split("e")
            return f"{float(base):.2g} × 10^{int(exp)}"
        if p < 0.01:
            return f"{p:.3f}"
        return f"{p:.2f}"


    def fmt_float(x, ndigits=2):
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{ndigits}f}"


    def standardize_rejected_column(df):
        if "rejected" not in df.columns:
            raise ValueError("TreeFDR result table must contain a 'rejected' column.")
        if df["rejected"].dtype == bool:
            return df
        df = df.copy()
        df["rejected"] = df["rejected"].map(
            lambda x: str(x).strip().lower() in {"true", "t", "1", "yes"}
            if pd.notna(x) else False
        )
        return df


    def first_existing_path(paths):
        for p in paths:
            if p and os.path.exists(p):
                return p
        return None


    def safe_clean_acronym(acronym):
        try:
            vals = atlas_df.loc[atlas_df["acronym"] == acronym, "cleaned_acronym"].values
            return str(vals[0]) if len(vals) else str(acronym)
        except Exception:
            return str(acronym)


    def ensure_normalized_density(df):
        """Compute density / saline mean per region if normalized_density is not already present."""
        df = df.copy()
        if "normalized_density" not in df.columns:
            saline_mean = df.loc[df["Condition"].astype(str) == "Saline"].groupby("acronym")["density"].mean()
            df["normalized_density"] = df.apply(
                lambda r: r["density"] / saline_mean.loc[r["acronym"]]
                if r["acronym"] in saline_mean.index and saline_mean.loc[r["acronym"]] != 0
                else np.nan,
                axis=1,
            )
        return df

    return (
        first_existing_path,
        fmt_float,
        fmt_p,
        p_to_sig_text,
        safe_clean_acronym,
        sem,
        standardize_rejected_column,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load processed data and atlas

    This notebook starts from processed outputs: the zarr heatmap array, long-format annotated cell-density table, atlas images, and existing TreeFDR output CSVs.
    """)
    return


@app.cell
def _(Conditions, analysis_resultpath, channel, da, metapath, np, os, pd):
    # ============================================================
    # Load processed data
    # ============================================================
    metadf = pd.read_csv(os.path.join(metapath, "acute_drug_revision_full_list.csv"), index_col=False)
    heatmap_da = da.from_zarr(os.path.join(analysis_resultpath, f"{channel}_heatmap_array"))
    merge_df = pd.read_csv(
        os.path.join(analysis_resultpath, f"{channel}_long_merge_Annotated_counts_clean_with_density.csv"),
        index_col=False,
    )
    fname_list = np.load(os.path.join(analysis_resultpath, "fname_list.npy"))

    # Align metadata + heatmap rows to fname_list, then drop subjects absent from the meta
    # (Cocaine / excluded) and any Usable == False. The zarr holds all imaged subjects (52);
    # the analysis uses the 44 usable, non-Cocaine subjects.
    metadf = metadf.set_index("fname").reindex(fname_list).reset_index()
    metadf["Usable"] = metadf["Usable"].fillna(False).astype(bool)
    usable_index = metadf.index[metadf["Usable"]].to_numpy()
    metadf = metadf.loc[usable_index].reset_index(drop=True)
    heatmap_da = heatmap_da[usable_index]
    fname_list = np.asarray(fname_list)[usable_index]
    merge_df = merge_df.loc[merge_df["ID"].isin(metadf["ID"])].copy()

    # Normalize categorical columns and create dummy covariates used by precomputed analyses/plots.
    sex_category = pd.CategoricalDtype(categories=["F", "M"], ordered=False)
    condition_category = pd.CategoricalDtype(categories=Conditions, ordered=True)
    batch_category = pd.CategoricalDtype(categories=np.sort(merge_df["Staining_Batch"].unique()), ordered=False)

    merge_df["Sex"] = merge_df["Sex"].astype(sex_category)
    merge_df["Condition"] = merge_df["Condition"].astype(condition_category)
    merge_df["Staining_Batch"] = merge_df["Staining_Batch"].astype(batch_category)

    condition_dummies = pd.get_dummies(merge_df["Condition"])
    sex_dummies = pd.get_dummies(merge_df["Sex"]).loc[:, ["F"]].rename(columns={"F": "Sex_d"})
    batch_dummies = pd.get_dummies(merge_df["Staining_Batch"])
    batch_dummies.columns = [f"Batch_{c}_d" for c in range(len(np.sort(merge_df["Staining_Batch"].unique())))]
    merge_df = pd.concat([merge_df, condition_dummies, sex_dummies, batch_dummies], axis=1)

    print("Subjects per condition:")
    print(metadf.groupby("Condition").size())
    assert heatmap_da.shape[0] == merge_df["ID"].nunique(), "Mismatch between heatmap rows and subject count."
    return fname_list, heatmap_da, merge_df, metadf


@app.cell
def _(atlaspath, np, os, pd, pickle, tifffile):
    # ============================================================
    # Load atlas resources
    # ============================================================
    atlas_df = pd.read_csv(
        os.path.join(atlaspath, "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv"),
        index_col=False,
    )
    contour_img = tifffile.imread(os.path.join(atlaspath, "Kim_ref_adult_FP-label_v2.9_contour_map.tif"))
    atlas_img = tifffile.imread(os.path.join(atlaspath, "Kim_ref_adult_FP-label_v4.0.tif"))
    atlas_resolution = (20, 20, 50)

    with open(os.path.join(atlaspath, "curated_acronym.pickle"), "rb") as handle:
        curated_acronyms = pickle.load(handle)
    with open(os.path.join(atlaspath, "ancestor_curated_acronym.pickle"), "rb") as handle:
        ancestor_curated_acronyms = pickle.load(handle)

    ancestor_list, ancestor_idx = np.unique(ancestor_curated_acronyms, return_index=True)
    ancestor_list = list(np.array(ancestor_curated_acronyms)[np.sort(ancestor_idx)])

    curated_zplanes = [84, 104, 117, 153, 186, 220]
    imy_slice = slice(25, 425)
    imx_slice = slice(50, 600)

    print(f"Atlas image shape: {atlas_img.shape}")
    print(f"Number of curated acronyms: {len(curated_acronyms)}")
    return (
        atlas_df,
        atlas_img,
        contour_img,
        curated_acronyms,
        curated_zplanes,
        imx_slice,
        imy_slice,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load existing TreeFDR/TreeBH results

    TreeFDR/TreeBH is not recomputed in this notebook. Existing CSV outputs generated by the separate R workflow are loaded and standardized here.
    """)
    return


@app.cell
def _(
    FIGS7_STATS_DIR,
    analysis_resultpath,
    atlas_df,
    curated_acronyms,
    first_existing_path,
    get_subregions,
    glob,
    os,
    pd,
    safe_clean_acronym,
    standardize_rejected_column,
):
    # ============================================================
    # Load overall acute-opioid TreeFDR result to define regions for Figure 4 clustering
    # ============================================================
    def treefdr_overall_candidates():
        names = [
            "glm_stat_df_post_TreeBH.csv",
            "TreeFDRS_pvalue_AcuteOpioid_without_Cocaineglm.csv",
            "TreeFDRS_pvalue_AcuteOpioid_without_Cocaine_GLM.csv",
            "TreeFDRS_pvalue_AcuteOpioid_glm_without_Cocaine_GLM.csv",
        ]
        candidates = []
        for folder in [analysis_resultpath, FIGS7_STATS_DIR]:
            candidates.extend([os.path.join(folder, name) for name in names])
            candidates.extend(glob.glob(os.path.join(folder, "TreeFDRS*pvalue*AcuteOpioid*without*Cocaine*.csv")))
        return list(dict.fromkeys(candidates))


    def load_overall_treefdr():
        path = first_existing_path(treefdr_overall_candidates())
        if path is None:
            raise FileNotFoundError(
                "Could not find the overall acute-opioid TreeFDR output. "
                "Run the TreeFDR script first and place the result in analysis_resultpath."
            )
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^(Unnamed: 0|X)$")]
        df = standardize_rejected_column(df)
        if "Name" not in df.columns or "parent_acronym" not in df.columns:
            meta = get_subregions(atlas_df, 8, return_original=True).copy().rename(columns={"name": "Name"})
            meta_cols = [c for c in ["id", "parent_id", "acronym", "Name", "parent_acronym"] if c in meta.columns]
            df = df.merge(meta[meta_cols], on="acronym", how="left", suffixes=("", "_atlas"))
        df["source_treefdr_csv"] = path
        df["cleaned_acronym"] = df["acronym"].map(safe_clean_acronym)
        return df


    glm_stat_df = load_overall_treefdr()
    glm_stat_df.to_csv(os.path.join(analysis_resultpath, "glm_stat_df_post_TreeBH_loaded_for_Figure4.csv"), index=False)
    acute_rejected = glm_stat_df.loc[
        (glm_stat_df["acronym"].isin(curated_acronyms)) & (glm_stat_df["rejected"]), "acronym"
    ].dropna().unique()

    print(f"Loaded overall TreeFDR result: {glm_stat_df['source_treefdr_csv'].iloc[0]}")
    print(f"Curated rejected regions used for clustering: {len(acute_rejected)}")
    return (acute_rejected,)


@app.cell
def _(
    Conditions,
    FIGS7_STATS_DIR,
    analysis_resultpath,
    atlas_df,
    condition_pretty,
    curated_acronyms,
    first_existing_path,
    get_subregions,
    glob,
    merge_df,
    np,
    os,
    pd,
    safe_clean_acronym,
    sem,
    standardize_rejected_column,
):
    # ============================================================
    # Load existing per-condition TreeFDR tables for Table 7 export
    # ============================================================
    TREEFDR_Q = 0.01
    figure_s7_conditions = [c for c in Conditions if c in set(merge_df["Condition"].astype(str))]
    figure_s7_drug_conditions = [c for c in figure_s7_conditions if c != "Saline"]


    def treefdr_result_candidates(condition):
        exact_names = [
            f"TreeFDRS_pvalue_{condition}_glm_without_Cocaine_GLM.csv",
            f"TreeFDRS_pvalue_{condition}_glm_without_Cocaineglm.csv",
            f"TreeFDRS_pvalue_{condition}_without_Cocaine_GLM.csv",
            f"TreeFDRS_pvalue_{condition}.csv",
        ]
        candidates = []
        for folder in [analysis_resultpath, FIGS7_STATS_DIR]:
            candidates.extend([os.path.join(folder, name) for name in exact_names])
            candidates.extend(glob.glob(os.path.join(folder, f"TreeFDRS*{condition}*.csv")))
            candidates.extend(glob.glob(os.path.join(folder, f"*{condition}*TreeFDRS*.csv")))
        return list(dict.fromkeys(candidates))


    def glm_input_candidates(condition):
        names = [
            f"glm_stat_df_{condition}.csv",
            f"glm_stat_df_{condition}_without_Cocaine.csv",
            f"glm_stat_df_{condition}_glm_without_Cocaine_GLM.csv",
        ]
        candidates = []
        for folder in [analysis_resultpath, FIGS7_STATS_DIR]:
            candidates.extend([os.path.join(folder, name) for name in names])
            candidates.extend(glob.glob(os.path.join(folder, f"glm_stat_df*{condition}*.csv")))
        return list(dict.fromkeys(candidates))


    def condition_summary_table(condition):
        pair_conditions = ["Saline", condition]
        df = merge_df.loc[merge_df["Condition"].astype(str).isin(pair_conditions)].copy()
        df["Condition"] = df["Condition"].astype(str)
        rows = []
        for acronym in sorted(df["acronym"].dropna().unique()):
            sub = df.loc[df["acronym"] == acronym]
            sal = sub.loc[sub["Condition"] == "Saline", "density"].astype(float)
            drug = sub.loc[sub["Condition"] == condition, "density"].astype(float)
            mean_saline = sal.mean()
            mean_condition = drug.mean()
            rows.append({
                "acronym": acronym,
                "condition": condition,
                "comparison": f"{condition_pretty.get(condition, condition)} vs Saline",
                "n_saline": sal.notna().sum(),
                "n_condition": drug.notna().sum(),
                "mean_saline": mean_saline,
                "sem_saline": sem(sal),
                "mean_condition": mean_condition,
                "sem_condition": sem(drug),
                "delta_condition_minus_saline": mean_condition - mean_saline,
                "fraction_change_vs_saline": (mean_condition - mean_saline) / mean_saline
                if pd.notna(mean_saline) and mean_saline != 0 else np.nan,
            })
        return pd.DataFrame(rows)


    def load_existing_pairwise_treefdr(condition):
        tree_path = first_existing_path(treefdr_result_candidates(condition))
        if tree_path is None:
            expected = "\n  ".join(treefdr_result_candidates(condition)[:8])
            raise FileNotFoundError(
                f"Could not find existing TreeFDR result for {condition}.\n"
                f"Expected one of these files in analysis_resultpath:\n  {expected}\n"
                "Run TreeFDR_ttest_for_each_condition.R first."
            )
        tree_table = pd.read_csv(tree_path)
        tree_table = tree_table.loc[:, ~tree_table.columns.astype(str).str.match(r"^(Unnamed: 0|X)$")]
        tree_table = standardize_rejected_column(tree_table)
        if "p.val" not in tree_table.columns:
            if "p_val" in tree_table.columns:
                tree_table = tree_table.rename(columns={"p_val": "p.val"})
            elif "pvalue" in tree_table.columns:
                tree_table = tree_table.rename(columns={"pvalue": "p.val"})

        tree_meta = get_subregions(atlas_df, 8, return_original=True).copy().rename(columns={"name": "Name"})
        meta_cols = [c for c in ["id", "parent_id", "acronym", "Name", "parent_acronym"] if c in tree_meta.columns]
        tree_table = tree_table.merge(tree_meta[meta_cols], on="acronym", how="left", suffixes=("", "_atlas"))
        for col in ["id", "parent_id", "Name", "parent_acronym"]:
            atlas_col = f"{col}_atlas"
            if atlas_col in tree_table.columns:
                if col in tree_table.columns:
                    tree_table[col] = tree_table[col].where(tree_table[col].notna(), tree_table[atlas_col])
                else:
                    tree_table[col] = tree_table[atlas_col]
                tree_table = tree_table.drop(columns=[atlas_col])

        glm_input_path = first_existing_path(glm_input_candidates(condition))
        if glm_input_path is not None:
            glm_input = pd.read_csv(glm_input_path)
            glm_input = glm_input.loc[:, ~glm_input.columns.astype(str).str.match(r"^(Unnamed: 0|X)$")]
            if "pvalue" in glm_input.columns:
                glm_input = glm_input[["acronym", "pvalue"]].drop_duplicates("acronym").rename(
                    columns={"pvalue": "pvalue_leaf_glm_lrt"}
                )
                tree_table = tree_table.merge(glm_input, on="acronym", how="left")

        summary_df = condition_summary_table(condition)
        tree_table["condition"] = condition
        tree_table["comparison"] = f"{condition_pretty.get(condition, condition)} vs Saline"
        tree_table = tree_table.merge(summary_df, on=["acronym", "condition", "comparison"], how="left")
        tree_table["cleaned_acronym"] = tree_table["acronym"].map(safe_clean_acronym)
        tree_table["sig_text"] = tree_table["rejected"].map(lambda x: "*" if bool(x) else "")
        tree_table["source_treefdr_csv"] = tree_path
        if glm_input_path is not None:
            tree_table["source_glm_input_csv"] = glm_input_path
        return tree_table


    all_treefdr_tables = []
    for cond in figure_s7_drug_conditions:
        print(f"Loading existing TreeFDR results: {cond} vs Saline")
        cond_tree_table = load_existing_pairwise_treefdr(cond)
        cond_tree_table.to_csv(os.path.join(FIGS7_STATS_DIR, f"Figure4_TreeFDR_all_nodes_{cond}.csv"), index=False)
        cond_curated = cond_tree_table.loc[cond_tree_table["acronym"].isin(curated_acronyms)].copy()
        cond_curated.to_csv(os.path.join(FIGS7_STATS_DIR, f"Figure4_TreeFDR_curated_regions_{cond}.csv"), index=False)
        all_treefdr_tables.append(cond_tree_table)

    figure_s7_treefdr_all_nodes_df = pd.concat(all_treefdr_tables, ignore_index=True)
    figure_s7_treefdr_curated_df = figure_s7_treefdr_all_nodes_df.loc[
        figure_s7_treefdr_all_nodes_df["acronym"].isin(curated_acronyms)
    ].copy()

    figure_s7_treefdr_all_nodes_df.to_csv(os.path.join(FIGS7_STATS_DIR, "Figure4_TreeFDR_all_nodes_all_conditions.csv"), index=False)
    figure_s7_treefdr_curated_df.to_csv(os.path.join(FIGS7_STATS_DIR, "Figure4_TreeFDR_curated_regions_all_conditions.csv"), index=False)

    print("Saved loaded per-condition TreeFDR tables.")
    figure_s7_treefdr_curated_df.head()
    return TREEFDR_Q, figure_s7_drug_conditions, figure_s7_treefdr_curated_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4A: whole-brain c-Fos+ cell counts
    """)
    return


@app.cell
def _(
    Condition_figure_name,
    Conditions,
    analysis_figurepath,
    condition_colors,
    figure_key,
    merge_df,
    os,
    plt,
    sns,
):
    # ============================================================
    # Figure 4A: whole-brain c-Fos+ cell counts
    # ============================================================
    panel_key = "A"

    fig, ax = plt.subplots(1, 1, figsize=(1.25, 2.0))
    merge_df["Condition"] = merge_df["Condition"].astype(str)
    tdata = (
        merge_df.loc[merge_df["parent_acronym"] == "grey", ["ID", "Condition", "newcounts"]]
        .groupby(["ID", "Condition"])
        .sum()
        .reset_index()
        .dropna()
    )

    sns.stripplot(
        data=tdata,
        y="Condition",
        x="newcounts",
        order=Conditions,
        ax=ax,
        palette=condition_colors,
        alpha=0.25,
    )
    sns.pointplot(
        data=tdata,
        y="Condition",
        x="newcounts",
        order=Conditions,
        ax=ax,
        palette=condition_colors,
        markers="o",
        markersize=4,
        linestyle="none",
        linewidth=0.5,
    )
    sns.despine()
    ax.set_xlabel("# of whole brain\nc-Fos+ cells")
    ax.set_yticklabels(Condition_figure_name, rotation=0)
    ax.set_xlim(0, None)

    fig.savefig(os.path.join(analysis_figurepath, f"{figure_key}_{panel_key}.png"), bbox_inches="tight", dpi=216)
    fig.savefig(os.path.join(analysis_figurepath, f"{figure_key}_{panel_key}.pdf"), bbox_inches="tight", dpi=216)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4D: beta-coefficient maps

    The beta maps are loaded from precomputed `*_betas.npy` files. No voxel-wise regression is run here.
    """)
    return


@app.cell
def _(
    analysis_figurepath,
    atlas_img,
    betas_dict,
    contour_img,
    curated_zplanes,
    figure_key,
    imx_slice,
    imy_slice,
    np,
    os,
    overlap_contour,
    plt,
    set_transparency,
):
    # ============================================================
    # Figure 4D: precomputed GLM beta-coefficient maps
    panel_key_1 = 'D'
    condition_to_beta_name = {'Baseline': 'constant', 'Morphine': 'Acute_Morphine', 'Fentanyl': 'Acute_Fentanyl', 'Buprenorphine': 'Acute_Buprenorphine', 'Oxycodone': 'Acute_Oxycodone'}
    row_labels = list(condition_to_beta_name.keys())
    (fig_1, axs) = plt.subplots(len(row_labels), len(curated_zplanes), figsize=(3.0 * len(curated_zplanes), 2.2 * len(row_labels)), sharex=True, sharey=True)
    fig_1.subplots_adjust(wspace=0.05, hspace=0.05)
    for (ridx, row_label) in enumerate(row_labels):
        beta_name = condition_to_beta_name[row_label]
        beta_map = betas_dict[beta_name]
        (_, overlayed_image) = overlap_contour(beta_map, contour_img, cmin=-15, cmax=15, outputpath=None, overlap_black=True)
        for (zidx, zplane) in enumerate(curated_zplanes):
            ax_1 = axs[ridx, zidx]
            trans_img = set_transparency(overlayed_image[zplane, :, :], (atlas_img == 0)[zplane, :, :])
            ax_1.imshow(trans_img[imy_slice, imx_slice])
            ax_1.axis('off')
            if zidx == 0:
                ax_1.set_ylabel(row_label, rotation=90, ha='center', va='center')
    fig_1.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_1}.png'), bbox_inches='tight', dpi=216)
    fig_1.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_1}.pdf'), bbox_inches='tight', dpi=216)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4B/E/F clustering analysis

    This section keeps the original clustering settings. It also writes the UMAP/HDBSCAN cluster table so the exact cluster assignments used for the figure can be reused for Table 7.
    """)
    return


@app.cell
def _(
    Conditions,
    FIGS7_STATS_DIR,
    StandardScaler,
    acute_rejected,
    merge_df,
    metadf,
    np,
    os,
    pd,
):
    # ============================================================
    # UMAP/HDBSCAN clustering input
    # ============================================================
    sub_conditions = Conditions
    sub_IDs = metadf.loc[metadf["Condition"].astype(str).isin(sub_conditions), "ID"].values
    sub_merge_df = merge_df.loc[merge_df["Condition"].astype(str).isin(sub_conditions)].copy()
    sub_merge_df["Condition"] = sub_merge_df["Condition"].astype(str)

    sub_pivot_df = merge_df.pivot_table(columns="ID", index="acronym", values="density", aggfunc="mean")
    sub_pivot_df = sub_pivot_df.loc[:, [sid for sid in sub_IDs if sid in sub_pivot_df.columns]]

    umap_acronyms = [a for a in acute_rejected if a in sub_pivot_df.index]
    umap_data = sub_pivot_df.loc[umap_acronyms, :].dropna(axis=0, how="any")

    scale_key = True
    scaler = StandardScaler()
    if scale_key:
        data = np.log1p(umap_data)
        data = scaler.fit_transform(data.to_numpy().T)
        data = pd.DataFrame(data.T, columns=umap_data.columns, index=umap_data.index)
    else:
        data = umap_data.copy()

    data.to_csv(os.path.join(FIGS7_STATS_DIR, "Figure4_UMAP_input_log1p_standardized.csv"))
    print(f"UMAP input shape: {data.shape}")
    return data, scaler, sub_conditions, sub_merge_df


@app.cell
def _(
    FIGS7_STATS_DIR,
    acute_rejected,
    analysis_figurepath,
    data,
    figure_key,
    hdbscan,
    leaves_list,
    np,
    os,
    pd,
    plt,
    safe_clean_acronym,
    umap,
):
    # ============================================================
    # UMAP/HDBSCAN clustering — parameters intentionally unchanged
    n_neighbors = 3
    min_dist = 0.05
    umap_random_state = 12
    hdbscan_min_cluster_size = 2
    single_linkage_cut = 1
    print('Clustering parameters:')
    print(f'  UMAP n_neighbors={n_neighbors}, min_dist={min_dist}, random_state={umap_random_state}')
    print(f'  HDBSCAN min_cluster_size={hdbscan_min_cluster_size}; get_clusters({single_linkage_cut}, min_cluster_size={hdbscan_min_cluster_size})')
    mapper = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=umap_random_state).fit(data)
    embedding = mapper.embedding_
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
    clusterer.fit(embedding)
    labels = clusterer.single_linkage_tree_.get_clusters(single_linkage_cut, min_cluster_size=hdbscan_min_cluster_size)
    cluster_tree_df = clusterer.single_linkage_tree_.to_pandas()
    Z = np.column_stack([cluster_tree_df['left_child'], cluster_tree_df['right_child'], cluster_tree_df['distance'], cluster_tree_df['size']])
    leaf_order = leaves_list(Z)
    original_label_order = labels[leaf_order]
    unique_labels_ordered = []
    for label in original_label_order:
        if label not in unique_labels_ordered:
            unique_labels_ordered.append(label)
    label_mapping = {}
    new_label = 1
    # Reorder labels from left to right using the single-linkage dendrogram leaves.
    for old_label in unique_labels_ordered:
        if old_label >= 0:
            label_mapping[old_label] = new_label
            new_label = new_label + 1
        else:
            label_mapping[old_label] = -1
    relabeled_labels = [label_mapping[label] for label in labels]
    embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=data.index)
    embedding_df['label'] = relabeled_labels
    embedding_df['original_label'] = labels
    embedding_df['rejected'] = [a in acute_rejected for a in embedding_df.index]
    embedding_df['cleaned_acronym'] = embedding_df.index.map(safe_clean_acronym)
    embedding_df['n_neighbors'] = n_neighbors
    embedding_df['min_dist'] = min_dist
    embedding_df['umap_random_state'] = umap_random_state
    embedding_df['hdbscan_min_cluster_size'] = hdbscan_min_cluster_size
    embedding_df['single_linkage_cut'] = single_linkage_cut
    embedding_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_UMAP_HDBSCAN_cluster_membership.csv'), index=True)
    (fig_2, ax_2) = plt.subplots(1, 1, figsize=(4, 2))
    clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True, axis=ax_2)
    fig_2.savefig(os.path.join(analysis_figurepath, f'{figure_key}_HDBSCAN_single_linkage_tree.png'), bbox_inches='tight', dpi=216)
    fig_2.savefig(os.path.join(analysis_figurepath, f'{figure_key}_HDBSCAN_single_linkage_tree.pdf'), bbox_inches='tight', dpi=216)
    plt.show()
    print('Label mapping:', label_mapping)
    # Save the HDBSCAN single-linkage tree plot used to confirm cluster cut.
    print(embedding_df['label'].value_counts().sort_index())
    return embedding, embedding_df


@app.cell
def _(
    acute_rejected,
    adjust_text,
    analysis_figurepath,
    data,
    embedding,
    embedding_df,
    figure_key,
    np,
    os,
    plt,
    safe_clean_acronym,
):
    # ============================================================
    # Figure 4B: UMAP with HDBSCAN clusters
    panel_key_2 = 'B'
    clustered = embedding_df['label'] >= 0
    rejected = embedding_df['rejected'] == True
    (fig_3, ax_3) = plt.subplots(1, 1, figsize=(4.5, 4.5))
    scatter = ax_3.scatter(embedding[clustered & rejected, 0], embedding[clustered & rejected, 1], c=embedding_df.loc[clustered & rejected, 'label'], s=60, edgecolor='k', cmap='Set1')
    ax_3.legend(*scatter.legend_elements(), title='Cluster')
    curated_texts = ['VTA', 'VTg', 'PT', 'MD', 'PVT', 'AcbC', 'AcbSh', 'LS', 'O', 'ASt', 'La', 'CPce', 'CPc', 'Ce', 'ST', 'BL', 'MBO', 'DP', 'Pl', 'DS', 'CPre', 'STr', 'PrL', 'DI', 'AI', 'IL', 'Cg', 'SPTg', 'Cl'] + ['A24a (IL)', 'Au1', 'RLi ', 'PrEW', 'A24 (Cg)', 'Cl', 'A32 (PrL)', 'VTA', 'AcbC', 'AcbSh', 'La', 'IPAC', 'PrG', 'Ce', 'ST']
    curated_texts = np.unique(np.char.strip(np.array(curated_texts, dtype=str)))
    curated_set = set(curated_texts)
    texts = []
    for (i, acronym) in enumerate(data.index):
        clean_acronym = safe_clean_acronym(acronym)
        if clean_acronym not in curated_set:
            continue
        if acronym in acute_rejected:
            texts.append(ax_3.text(embedding[i, 0], embedding[i, 1], clean_acronym, fontsize=10, ha='right'))
    adjust_text(texts, ax=ax_3, expand=(2, 2), force_text=(0.25, 0.25), arrowprops=dict(color='gray', lw=1, alpha=0.75))
    ax_3.set_ylabel('UMAP1', fontsize=12)
    ax_3.set_xlabel('UMAP2', fontsize=12)
    fig_3.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_2}.png'), bbox_inches='tight', dpi=216)
    fig_3.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_2}.pdf'), bbox_inches='tight', dpi=216)
    plt.show()
    return (curated_set,)


@app.cell
def _(
    Conditions,
    FIGS7_STATS_DIR,
    acute_rejected,
    embedding_df,
    merge_df,
    os,
    pd,
    safe_clean_acronym,
    scaler,
    sub_merge_df,
):
    # ============================================================
    # Effect-size tables and cluster-labeled long table
    effect_size_df = merge_df[['acronym', 'Condition', 'density']].groupby(['acronym', 'Condition']).mean().reset_index().pivot(index='acronym', columns='Condition', values='density').reset_index()
    teffect_size_df = effect_size_df.set_index('acronym').loc[acute_rejected, Conditions]
    scale_key_1 = True
    if scale_key_1:
        scaled_effect_size_df = scaler.fit_transform(teffect_size_df.T)
        scaled_effect_size_df = pd.DataFrame(scaled_effect_size_df.T, columns=teffect_size_df.columns, index=teffect_size_df.index)
        scaled_effect_size_df = (scaled_effect_size_df - scaled_effect_size_df['Saline'].values[:, None]).reset_index()
    else:
        scaled_effect_size_df = teffect_size_df.reset_index()
    clustered_sub_merge_df = sub_merge_df.loc[sub_merge_df['acronym'].isin(embedding_df.index)].copy()
    clustered_sub_merge_df['label'] = clustered_sub_merge_df['acronym'].map(embedding_df['label'])
    clustered_sub_merge_df = clustered_sub_merge_df.loc[clustered_sub_merge_df['label'] >= 0].copy()
    clustered_sub_merge_df['cleaned_acronym'] = clustered_sub_merge_df['acronym'].map(safe_clean_acronym)
    clustered_sub_merge_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_clustered_region_subject_long.csv'), index=False)
    clustered_sub_merge_averaged_by_subject_df = clustered_sub_merge_df[['label', 'Condition', 'density', 'ID']].groupby(['label', 'Condition', 'ID']).mean().dropna(axis=0).reset_index()
    clustered_sub_merge_averaged_by_acronym_df = clustered_sub_merge_df[['label', 'Condition', 'density', 'acronym']].groupby(['label', 'Condition', 'acronym']).mean().dropna(axis=0).reset_index()
    clustered_sub_merge_averaged_by_subject_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_cluster_density_by_subject.csv'), index=False)
    # Long-format table with subject-level density and cluster labels. This is the canonical cluster table for Table 7.
    clustered_sub_merge_averaged_by_acronym_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_cluster_density_by_acronym.csv'), index=False)
    return (
        clustered_sub_merge_averaged_by_acronym_df,
        clustered_sub_merge_averaged_by_subject_df,
        clustered_sub_merge_df,
        effect_size_df,
        scale_key_1,
        scaled_effect_size_df,
    )


@app.cell
def _(
    Condition_figure_name,
    analysis_figurepath,
    effect_size_df,
    embedding_df,
    figure_key,
    os,
    plt,
    scale_key_1,
    scaled_effect_size_df,
    sub_conditions,
):
    # ============================================================
    # Figure 4E: UMAP colored by scaled effect size
    panel_key_3 = 'E'
    (vmin, vmax) = (0, 3) if scale_key_1 else (-5, 1500)
    (fig_4, axs_1) = plt.subplots(2, 3, figsize=(len(sub_conditions) * 0.7 * 3, 3 * 2))
    for (cidx, Condition) in enumerate(sub_conditions):
        tembedding_df = embedding_df.loc[embedding_df['label'] >= 0]
        ordered_acronym = effect_size_df.set_index('acronym').loc[tembedding_df.index, :].sort_values(by=Condition).index
        ax_4 = axs_1[cidx // 3, cidx % 3]
        scatter_1 = ax_4.scatter(tembedding_df.loc[ordered_acronym, 'UMAP1'], tembedding_df.loc[ordered_acronym, 'UMAP2'], c=scaled_effect_size_df.set_index('acronym').loc[ordered_acronym, Condition], s=40, alpha=0.7, vmin=vmin, vmax=vmax, cmap='viridis')
        if cidx == len(sub_conditions) - 1:
            colorbar = plt.colorbar(scatter_1, ax=axs_1, fraction=0.046, pad=0.04)
            colorbar.set_label('Scaled Effect Size', rotation=-90)
        ax_4.set_title(Condition_figure_name[cidx])
        ax_4.set_xlabel('UMAP1', fontsize=12)
        ax_4.set_ylabel('UMAP2', fontsize=12)
    for ax_4 in axs_1.flatten():
        ax_4.spines['top'].set_visible(False)
        ax_4.spines['right'].set_visible(False)
        ax_4.spines['left'].set_visible(False)
        ax_4.spines['bottom'].set_visible(False)
    fig_4.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_3}.tif'), bbox_inches='tight', dpi=216)
    fig_4.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_3}.pdf'), bbox_inches='tight', dpi=216)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4C and Table 7 cluster statistics
    """)
    return


@app.cell
def _(
    Conditions,
    FIGS7_STATS_DIR,
    anova_lm,
    clustered_sub_merge_averaged_by_subject_df,
    condition_pretty,
    ols,
    os,
    p_to_sig_text,
    pairwise_tukeyhsd,
    pd,
    sem,
):
    # ============================================================
    # Cluster-level ANOVA and Tukey HSD tables
    df_anova = clustered_sub_merge_averaged_by_subject_df.copy()
    df_anova['label'] = df_anova['label'].astype(int).astype('category')
    cluster_model = ols('density ~ C(Condition) + C(label) + C(Condition) * C(label)', data=df_anova).fit()
    figure_s7_cluster_anova_df = anova_lm(cluster_model, typ=2).reset_index().rename(columns={'index': 'term'})
    figure_s7_cluster_anova_df['df_resid'] = cluster_model.df_resid
    figure_s7_cluster_anova_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_cluster_two_way_ANOVA.csv'), index=False)
    posthoc_results = []
    labels_to_test = sorted([l for l in clustered_sub_merge_averaged_by_subject_df['label'].dropna().unique() if l >= 0])
    # Tukey HSD: condition comparisons within each cluster.
    for lab in labels_to_test:
        sub = clustered_sub_merge_averaged_by_subject_df.loc[(clustered_sub_merge_averaged_by_subject_df['label'] == lab) & clustered_sub_merge_averaged_by_subject_df['Condition'].isin(Conditions)].copy()
        if sub.shape[0] < 3 or sub['Condition'].nunique() < 2:
            continue
        tukey = pairwise_tukeyhsd(sub['density'], sub['Condition'], alpha=0.05)
        tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_df.insert(0, 'label', int(lab))
        counts = sub.groupby('Condition')['ID'].nunique()
        tukey_df['n_group1'] = tukey_df['group1'].map(counts)
        tukey_df['n_group2'] = tukey_df['group2'].map(counts)
        posthoc_results.append(tukey_df)
    posthoc_df = pd.concat(posthoc_results, ignore_index=True)
    posthoc_df['p-adj'] = posthoc_df['p-adj'].astype(float)
    posthoc_df['sig_text'] = posthoc_df['p-adj'].map(p_to_sig_text)
    posthoc_df['group1_pretty'] = posthoc_df['group1'].map(lambda x: condition_pretty.get(x, x))
    posthoc_df['group2_pretty'] = posthoc_df['group2'].map(lambda x: condition_pretty.get(x, x))
    posthoc_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_cluster_condition_TukeyHSD_all_pairs.csv'), index=False)
    condition_cluster_results = []
    for cond_1 in Conditions:
        sub = clustered_sub_merge_averaged_by_subject_df.loc[clustered_sub_merge_averaged_by_subject_df['Condition'] == cond_1].copy()
        if sub.shape[0] < 3 or sub['label'].nunique() < 2:
            continue
        sub['label_str'] = sub['label'].astype(int).astype(str)
        tukey = pairwise_tukeyhsd(sub['density'], sub['label_str'], alpha=0.05)
    # Tukey HSD: cluster-cluster comparisons within each condition.
        tdf = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        tdf.insert(0, 'Condition', cond_1)
        tdf['Condition_pretty'] = condition_pretty.get(cond_1, cond_1)
        counts = sub.groupby('label_str')['ID'].nunique()
        tdf['n_group1'] = tdf['group1'].map(counts)
        tdf['n_group2'] = tdf['group2'].map(counts)
        condition_cluster_results.append(tdf)
    figure_s7_condition_cluster_tukey_df = pd.concat(condition_cluster_results, ignore_index=True)
    figure_s7_condition_cluster_tukey_df['p-adj'] = figure_s7_condition_cluster_tukey_df['p-adj'].astype(float)
    figure_s7_condition_cluster_tukey_df['sig_text'] = figure_s7_condition_cluster_tukey_df['p-adj'].map(p_to_sig_text)
    figure_s7_condition_cluster_tukey_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_condition_cluster_TukeyHSD_all_pairs.csv'), index=False)
    cluster_density_summary_df = clustered_sub_merge_averaged_by_subject_df.groupby(['label', 'Condition'])['density'].agg(n='count', mean='mean', sem=sem).reset_index()
    cluster_density_summary_df['Condition_pretty'] = cluster_density_summary_df['Condition'].map(lambda x: condition_pretty.get(x, x))
    cluster_density_summary_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_cluster_density_summary.csv'), index=False)
    print(figure_s7_cluster_anova_df)
    return (
        figure_s7_cluster_anova_df,
        figure_s7_condition_cluster_tukey_df,
        posthoc_df,
    )


@app.cell
def _(
    Condition_figure_name,
    Conditions,
    analysis_figurepath,
    clustered_sub_merge_averaged_by_subject_df,
    figure_key,
    os,
    plt,
    posthoc_df,
    sns,
    sub_conditions,
):
    # ============================================================
    # Figure 4C: cluster x condition heatmap with Tukey-vs-saline labels
    panel_key_4 = 'C'
    tdf_1 = clustered_sub_merge_averaged_by_subject_df.drop(columns='ID').groupby(['label', 'Condition']).mean().reset_index().pivot(columns='Condition', index='label')
    tdf_1.columns = [f[1] for f in tdf_1.columns]
    (vmin_1, vmax_1) = (0, 1500)
    (fig_5, ax_5) = plt.subplots(1, 1, figsize=(2.2, 2.2))
    sns.heatmap(tdf_1.loc[:, sub_conditions].T, vmin=vmin_1, vmax=vmax_1, square=True, ax=ax_5, linewidths=0.5)
    ax_5.set_yticklabels([Condition_figure_name[Conditions.index(c)] for c in sub_conditions], rotation=0)
    if ax_5.collections:
        cbar = ax_5.collections[0].colorbar
        if cbar is not None:
            cbar.ax.set_ylabel('Scaled density', rotation=-90)
    sig_lookup = {}
    for (_, r) in posthoc_df.iterrows():
        (lab_1, g1, g2) = (r['label'], r['group1'], r['group2'])
        st = r.get('sig_text', '')
        if st:
            sig_lookup[lab_1, g1, g2] = st
            sig_lookup[lab_1, g2, g1] = st
    for (i_1, cond_2) in enumerate(sub_conditions):
        if cond_2 == 'Saline':
            continue
        for (j, lab_1) in enumerate(tdf_1.index):
            txt = sig_lookup.get((lab_1, 'Saline', cond_2), '')
            if txt:
                ax_5.text(j + 0.5, i_1 + 0.5, txt, ha='center', va='center', fontsize=8, color='white')
    fig_5.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_4}.png'), dpi=216, bbox_inches='tight')
    fig_5.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_4}.pdf'), dpi=216, bbox_inches='tight')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figure 4F and Table 7 region-level statistics
    """)
    return


@app.cell
def _(
    FIGS7_STATS_DIR,
    acute_rejected,
    clustered_sub_merge_averaged_by_acronym_df,
    clustered_sub_merge_df,
    condition_pretty,
    curated_set,
    embedding_df,
    figure_s7_drug_conditions,
    figure_s7_treefdr_curated_df,
    multipletests,
    np,
    os,
    p_to_sig_text,
    pd,
    safe_clean_acronym,
    sem,
    stats,
):
    # ============================================================
    # Region-level condition-vs-saline statistics for Figure 4F/Table 7
    all_clustered_rejected_acronyms = clustered_sub_merge_averaged_by_acronym_df.loc[(clustered_sub_merge_averaged_by_acronym_df['Condition'] == 'Acute_Morphine') & (clustered_sub_merge_averaged_by_acronym_df['label'] > 0)].sort_values(by=['label', 'density'], ascending=[True, False])['acronym'].drop_duplicates().tolist()
    # Figure 4F only displays a curated subset of labels for readability.
    # Table 7, however, should contain p-values for ALL clustered TreeFDR-rejected regions.
    # Therefore we define two region lists:
    #   1) sorted_acronyms: regions shown in panel F
    #   2) region_test_acronyms: all clustered TreeFDR-rejected regions tested for Table 7
    sorted_acronyms = [a for a in all_clustered_rejected_acronyms if safe_clean_acronym(a) in curated_set]
    # Region order: cluster order, then morphine scaled effect size within cluster.
    panel_f_shown_set = set(sorted_acronyms)
    region_test_acronyms = [a for a in all_clustered_rejected_acronyms if a in set(acute_rejected)]
    region_order_lookup = {a: i + 1 for (i, a) in enumerate(region_test_acronyms)}
    region_stat_rows = []
    for cond_3 in figure_s7_drug_conditions:
    # Regions shown in panel F: curated/annotated subset used for a legible heatmap.
        cond_rows = []
        for acronym_1 in region_test_acronyms:
            sub_1 = clustered_sub_merge_df.loc[(clustered_sub_merge_df['acronym'] == acronym_1) & clustered_sub_merge_df['Condition'].isin(['Saline', cond_3])]
    # Regions tested and exported in Table 7: all clustered rejected regions, not only those displayed in panel F.
            x = sub_1.loc[sub_1['Condition'] == 'Saline', 'density'].astype(float).dropna()
            y = sub_1.loc[sub_1['Condition'] == cond_3, 'density'].astype(float).dropna()
            if len(x) >= 2 and len(y) >= 2:
    # Match Figure 4 legend: independent t-tests versus saline, BH correction across regions within each opioid condition.
    # The BH correction is applied across ALL tested rejected regions within each opioid condition, including regions not shown in panel F.
                (t_stat, p_value) = stats.ttest_ind(y, x, equal_var=True, nan_policy='omit')
                df = len(x) + len(y) - 2
            else:
                (t_stat, p_value, df) = (np.nan, np.nan, np.nan)
            cond_rows.append({'figure_panel': '4F', 'analysis_section': 'Region-level condition-vs-saline t-test', 'acronym': acronym_1, 'cleaned_acronym': safe_clean_acronym(acronym_1), 'region_order_all_rejected': region_order_lookup.get(acronym_1, np.nan), 'shown_in_panel_F': acronym_1 in panel_f_shown_set, 'label': int(embedding_df.loc[acronym_1, 'label']) if acronym_1 in embedding_df.index else np.nan, 'condition': cond_3, 'condition_pretty': condition_pretty.get(cond_3, cond_3), 'comparison': f'{condition_pretty.get(cond_3, cond_3)} vs Saline', 'test': 'Student t-test', 'p_adjustment_scope': 'Benjamini-Hochberg across all clustered TreeFDR-rejected regions within condition', 'n_saline': len(x), 'n_condition': len(y), 'mean_saline': x.mean(), 'sem_saline': sem(x), 'mean_condition': y.mean(), 'sem_condition': sem(y), 'delta_condition_minus_saline': y.mean() - x.mean(), 't_stat': t_stat, 'df': df, 'p_value': p_value})
        cond_df = pd.DataFrame(cond_rows)
        valid = cond_df['p_value'].notna()
        cond_df['q_value_BH_within_condition'] = np.nan
        if valid.any():
            (_, qvals, _, _) = multipletests(cond_df.loc[valid, 'p_value'], method='fdr_bh')
            cond_df.loc[valid, 'q_value_BH_within_condition'] = qvals
        cond_df['sig_text'] = cond_df['q_value_BH_within_condition'].map(p_to_sig_text)
        region_stat_rows.append(cond_df)
    figure_s7_region_ttest_df = pd.concat(region_stat_rows, ignore_index=True)
    treefdr_cols = [c for c in ['condition', 'acronym', 'p.val', 'pvalue_leaf_glm_lrt', 'rejected', 'q_adj', 'source_treefdr_csv'] if c in figure_s7_treefdr_curated_df.columns]
    if set(['condition', 'acronym']).issubset(treefdr_cols):
        treefdr_for_region_tests = figure_s7_treefdr_curated_df[treefdr_cols].copy()
        treefdr_for_region_tests = treefdr_for_region_tests.rename(columns={'p.val': 'treefdr_node_p_value', 'pvalue_leaf_glm_lrt': 'treefdr_leaf_glm_lrt_p_value', 'rejected': 'treefdr_rejected_for_condition', 'q_adj': 'treefdr_q_adj'})
        figure_s7_region_ttest_df = figure_s7_region_ttest_df.merge(treefdr_for_region_tests, on=['condition', 'acronym'], how='left')
    figure_s7_region_ttest_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_region_condition_vs_saline_ttest_BH_all_rejected_regions.csv'), index=False)
    figure_s7_region_ttest_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_region_condition_vs_saline_ttest_BH.csv'), index=False)
    figure_s7_region_ttest_sig_df = figure_s7_region_ttest_df.loc[figure_s7_region_ttest_df['q_value_BH_within_condition'] < 0.05].copy()
    figure_s7_region_ttest_sig_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_region_condition_vs_saline_ttest_BH_significant_only.csv'), index=False)
    pd.DataFrame({'acronym': region_test_acronyms, 'cleaned_acronym': [safe_clean_acronym(a) for a in region_test_acronyms], 'label': [int(embedding_df.loc[a, 'label']) if a in embedding_df.index else np.nan for a in region_test_acronyms], 'shown_in_panel_F': [a in panel_f_shown_set for a in region_test_acronyms], 'region_order_all_rejected': [region_order_lookup.get(a, np.nan) for a in region_test_acronyms]}).to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_panelF_region_test_universe.csv'), index=False)
    print(f'Panel F displayed regions: {len(sorted_acronyms)}')
    print(f'Table 7 tested rejected regions: {len(region_test_acronyms)}')
    # Add corresponding per-condition TreeFDR information for these same regions, when available.
    # Keep the historical filename, but now it contains ALL tested rejected regions, not only panel-F-displayed regions.
    # Also write the list of regions shown in panel F and all tested regions for reproducibility.
    print(f'Region-level tests written: {len(figure_s7_region_ttest_df)} rows')
    return (
        figure_s7_region_ttest_df,
        figure_s7_region_ttest_sig_df,
        panel_f_shown_set,
        region_order_lookup,
        region_test_acronyms,
        sorted_acronyms,
    )


@app.cell
def _(
    Condition_figure_name,
    Conditions,
    analysis_figurepath,
    clustered_sub_merge_averaged_by_acronym_df,
    figure_key,
    figure_s7_region_ttest_df,
    np,
    os,
    p_to_sig_text,
    plt,
    safe_clean_acronym,
    scaled_effect_size_df,
    sns,
    sorted_acronyms,
):
    # ============================================================
    # Figure 4F: curated-region heatmap with BH-corrected significance labels
    panel_key_5 = 'F'
    plot_df = scaled_effect_size_df.set_index('acronym').loc[sorted_acronyms, :].T.reindex(Conditions)
    (nrow, ncol) = plot_df.shape
    annot = np.full((nrow, ncol), '', dtype=object)
    q_lookup = figure_s7_region_ttest_df.set_index(['condition', 'acronym'])['q_value_BH_within_condition'].to_dict()
    for (i_2, cond_4) in enumerate(Conditions):
        if cond_4 == 'Saline':
            continue
        for (j_1, acr) in enumerate(sorted_acronyms):
            annot[i_2, j_1] = p_to_sig_text(q_lookup.get((cond_4, acr), np.nan))
    (vmin_2, vmax_2) = (0, 3)
    w = max(6, 0.22 * ncol)
    h = max(3, 0.35 * nrow)
    cmap = sns.color_palette('rocket', as_cmap=True)
    (fig_6, ax_6) = plt.subplots(1, 1, figsize=(w, h))
    sns.heatmap(plot_df, vmin=vmin_2, vmax=vmax_2, square=True, ax=ax_6, linewidths=0.5, cmap=cmap, annot=annot, fmt='', annot_kws={'size': 8, 'color': 'white', 'weight': 'bold'})
    ax_6.set_yticklabels(Condition_figure_name, rotation=0)
    x_labels = [safe_clean_acronym(ac) for ac in sorted_acronyms]
    ax_6.set_xticks(np.arange(ncol) + 0.5)
    ax_6.set_xticklabels(x_labels, rotation=-45, ha='left')
    ax_6.tick_params(axis='x', which='major', labelbottom=True)
    plt.setp(ax_6.get_xticklabels(), fontsize=6)
    labs = clustered_sub_merge_averaged_by_acronym_df.drop_duplicates('acronym').set_index('acronym')['label'].reindex(sorted_acronyms)
    for k in range(1, len(labs)):
        if labs.iloc[k] != labs.iloc[k - 1]:
            ax_6.axvline(k, color='yellow', linewidth=2)
    fig_6.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_5}.png'), bbox_inches='tight', dpi=216)
    fig_6.savefig(os.path.join(analysis_figurepath, f'{figure_key}_{panel_key_5}.pdf'), bbox_inches='tight', dpi=216)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final single Table 7 export and legend statistics text

    This section writes one long-format Table 7 that contains all panel C pairwise-combination results and all panel F p-values for the full set of clustered TreeFDR-rejected regions.
    """)
    return


@app.cell
def _(
    Conditions,
    FIGS7_STATS_DIR,
    clustered_sub_merge_df,
    embedding_df,
    np,
    os,
    p_to_sig_text,
    pairwise_tukeyhsd,
    panel_f_shown_set,
    pd,
    region_order_lookup,
    region_test_acronyms,
    safe_clean_acronym,
):
    # ============================================================
    # Optional region-level Tukey HSD across all conditions
    region_tukey_results = []
    # Exported for completeness across ALL clustered TreeFDR-rejected regions, not only panel-F-displayed regions.
    for acronym_2 in region_test_acronyms:
        sub_2 = clustered_sub_merge_df.loc[(clustered_sub_merge_df['acronym'] == acronym_2) & clustered_sub_merge_df['Condition'].isin(Conditions)].copy()
        if sub_2.shape[0] < 3 or sub_2['Condition'].nunique() < 2:
            continue
        try:
            tukey_1 = pairwise_tukeyhsd(sub_2['density'], sub_2['Condition'], alpha=0.05)
            tdf_2 = pd.DataFrame(tukey_1.summary().data[1:], columns=tukey_1.summary().data[0])
            tdf_2.insert(0, 'figure_panel', '4F')
            tdf_2.insert(1, 'analysis_section', 'Region-level Tukey HSD across all conditions')
            tdf_2.insert(2, 'acronym', acronym_2)
            tdf_2.insert(3, 'cleaned_acronym', safe_clean_acronym(acronym_2))
            tdf_2.insert(4, 'region_order_all_rejected', region_order_lookup.get(acronym_2, np.nan))
            tdf_2.insert(5, 'shown_in_panel_F', acronym_2 in panel_f_shown_set)
            tdf_2.insert(6, 'label', int(embedding_df.loc[acronym_2, 'label']) if acronym_2 in embedding_df.index else np.nan)
            region_tukey_results.append(tdf_2)
        except Exception:
            pass
    if region_tukey_results:
        figure_s7_region_tukey_df = pd.concat(region_tukey_results, ignore_index=True)
        figure_s7_region_tukey_df['p-adj'] = figure_s7_region_tukey_df['p-adj'].astype(float)
        figure_s7_region_tukey_df['sig_text'] = figure_s7_region_tukey_df['p-adj'].map(p_to_sig_text)
        figure_s7_region_tukey_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_region_TukeyHSD_all_pairs_all_rejected_regions.csv'), index=False)
        figure_s7_region_tukey_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_region_TukeyHSD_all_pairs.csv'), index=False)
    else:
        figure_s7_region_tukey_df = pd.DataFrame()
    print(f'Region-level Tukey HSD rows written: {len(figure_s7_region_tukey_df)}')  # Historical filename now contains all tested rejected regions.
    return (figure_s7_region_tukey_df,)


@app.cell
def _(
    FIGS7_STATS_DIR,
    TREEFDR_Q,
    condition_pretty,
    embedding_df,
    figure_s7_cluster_anova_df,
    figure_s7_condition_cluster_tukey_df,
    figure_s7_drug_conditions,
    figure_s7_region_ttest_df,
    figure_s7_region_ttest_sig_df,
    figure_s7_region_tukey_df,
    figure_s7_treefdr_curated_df,
    fmt_float,
    fmt_p,
    np,
    os,
    p_to_sig_text,
    panel_f_shown_set,
    pd,
    posthoc_df,
    region_order_lookup,
    region_test_acronyms,
):
    # ============================================================
    # Single Table 7 export + copy/paste legend statistics text
    def _section_to_table_rows(df, defaults=None, rename=None):
    # Table 7 is exported as ONE long-format table. Use the columns
    # `figure_panel`, `analysis_section`, and `analysis_subsection` to filter sections.
    # It contains:
    #   - Panel C: two-way ANOVA results.
    #   - Panel C: all Tukey HSD condition-combination results within each cluster.
    #   - Panel C: all Tukey HSD cluster-combination results within each condition.
    #   - Panel F: condition-vs-saline t-test p-values/q-values for ALL clustered TreeFDR-rejected regions,
    #              with a flag indicating whether each region is shown in panel F.
    #   - Panel F: optional region-level Tukey HSD all-pair results across conditions for the same rejected regions.
    #   - Panel F: loaded per-condition TreeFDR results for the same rejected regions.
        defaults = defaults or {}
        rename = rename or {}
        if df is None or len(df) == 0:
            return []
        out = df.copy().rename(columns=rename)
        for (k, v) in defaults.items():
            if k not in out.columns:
                out[k] = v
        return out.to_dict('records')
    panel_c_anova = figure_s7_cluster_anova_df.copy()
    panel_c_anova = panel_c_anova.loc[panel_c_anova['term'] != 'Residual'].copy()
    panel_c_anova['figure_panel'] = '4C'
    # Panel C: ANOVA.
    panel_c_anova['analysis_section'] = 'Cluster-level two-way ANOVA'
    panel_c_anova['analysis_subsection'] = panel_c_anova['term'].map({'C(Condition)': 'Condition main effect', 'C(label)': 'Cluster main effect', 'C(Condition):C(label)': 'Condition × Cluster interaction'}).fillna(panel_c_anova['term'])
    panel_c_anova['test'] = 'Two-way ANOVA'
    panel_c_anova['statistic'] = 'F'
    panel_c_anova['statistic_value'] = panel_c_anova['F']
    panel_c_anova['df1'] = panel_c_anova['df']
    panel_c_anova['df2'] = panel_c_anova['df_resid']
    panel_c_anova['p_value'] = panel_c_anova['PR(>F)']
    panel_c_anova['p_adjusted'] = np.nan
    pval_for_sig = panel_c_anova['p_value']
    panel_c_anova['sig_text'] = pval_for_sig.map(p_to_sig_text)
    panel_c_condition_pairs = posthoc_df.copy()
    panel_c_condition_pairs['figure_panel'] = '4C'
    panel_c_condition_pairs['analysis_section'] = 'Cluster-level Tukey HSD'
    panel_c_condition_pairs['analysis_subsection'] = 'All condition combinations within each cluster'
    panel_c_condition_pairs['cluster'] = panel_c_condition_pairs['label'].astype(int)
    panel_c_condition_pairs['condition_group1'] = panel_c_condition_pairs['group1']
    panel_c_condition_pairs['condition_group2'] = panel_c_condition_pairs['group2']
    panel_c_condition_pairs['comparison'] = panel_c_condition_pairs['group1'].map(lambda x: condition_pretty.get(x, x)) + ' vs ' + panel_c_condition_pairs['group2'].map(lambda x: condition_pretty.get(x, x))
    # Panel C: all condition combinations within each cluster.
    panel_c_condition_pairs['test'] = 'Tukey HSD'
    panel_c_condition_pairs['mean_difference'] = panel_c_condition_pairs['meandiff']
    panel_c_condition_pairs['ci_lower'] = panel_c_condition_pairs['lower']
    panel_c_condition_pairs['ci_upper'] = panel_c_condition_pairs['upper']
    panel_c_condition_pairs['p_adjusted'] = panel_c_condition_pairs['p-adj']
    panel_c_condition_pairs['p_value'] = np.nan
    panel_c_cluster_pairs = figure_s7_condition_cluster_tukey_df.copy()
    panel_c_cluster_pairs['figure_panel'] = '4C'
    panel_c_cluster_pairs['analysis_section'] = 'Cluster-level Tukey HSD'
    panel_c_cluster_pairs['analysis_subsection'] = 'All cluster combinations within each condition'
    panel_c_cluster_pairs['condition'] = panel_c_cluster_pairs['Condition']
    panel_c_cluster_pairs['condition_pretty'] = panel_c_cluster_pairs['Condition_pretty']
    panel_c_cluster_pairs['cluster_group1'] = panel_c_cluster_pairs['group1']
    panel_c_cluster_pairs['cluster_group2'] = panel_c_cluster_pairs['group2']
    panel_c_cluster_pairs['comparison'] = 'Cluster ' + panel_c_cluster_pairs['group1'].astype(str) + ' vs Cluster ' + panel_c_cluster_pairs['group2'].astype(str)
    panel_c_cluster_pairs['test'] = 'Tukey HSD'
    panel_c_cluster_pairs['mean_difference'] = panel_c_cluster_pairs['meandiff']
    panel_c_cluster_pairs['ci_lower'] = panel_c_cluster_pairs['lower']
    panel_c_cluster_pairs['ci_upper'] = panel_c_cluster_pairs['upper']
    # Panel C: all cluster combinations within each condition.
    panel_c_cluster_pairs['p_adjusted'] = panel_c_cluster_pairs['p-adj']
    panel_c_cluster_pairs['p_value'] = np.nan
    panel_f_ttests = figure_s7_region_ttest_df.copy()
    panel_f_ttests['figure_panel'] = '4F'
    panel_f_ttests['analysis_section'] = 'Region-level condition-vs-saline t-test'
    panel_f_ttests['analysis_subsection'] = 'All clustered TreeFDR-rejected regions'
    panel_f_ttests['cluster'] = panel_f_ttests['label']
    panel_f_ttests['statistic'] = 't'
    panel_f_ttests['statistic_value'] = panel_f_ttests['t_stat']
    panel_f_ttests['df1'] = panel_f_ttests['df']
    panel_f_ttests['df2'] = np.nan
    panel_f_ttests['p_adjusted'] = panel_f_ttests['q_value_BH_within_condition']
    panel_f_ttests['mean_difference'] = panel_f_ttests['delta_condition_minus_saline']
    panel_f_region_tukey = figure_s7_region_tukey_df.copy()
    if len(panel_f_region_tukey) > 0:
        panel_f_region_tukey['figure_panel'] = '4F'
    # Panel F: all rejected-region condition-vs-saline p-values/q-values.
        panel_f_region_tukey['analysis_section'] = 'Region-level Tukey HSD'
        panel_f_region_tukey['analysis_subsection'] = 'All condition combinations within each clustered TreeFDR-rejected region'
        panel_f_region_tukey['cluster'] = panel_f_region_tukey['label']
        panel_f_region_tukey['condition_group1'] = panel_f_region_tukey['group1']
        panel_f_region_tukey['condition_group2'] = panel_f_region_tukey['group2']
        panel_f_region_tukey['comparison'] = panel_f_region_tukey['group1'].map(lambda x: condition_pretty.get(x, x)) + ' vs ' + panel_f_region_tukey['group2'].map(lambda x: condition_pretty.get(x, x))
        panel_f_region_tukey['test'] = 'Tukey HSD'
        panel_f_region_tukey['mean_difference'] = panel_f_region_tukey['meandiff']
        panel_f_region_tukey['ci_lower'] = panel_f_region_tukey['lower']
        panel_f_region_tukey['ci_upper'] = panel_f_region_tukey['upper']
        panel_f_region_tukey['p_adjusted'] = panel_f_region_tukey['p-adj']
        panel_f_region_tukey['p_value'] = np.nan
    # Panel F: optional region-level Tukey all-pair results for all rejected regions.
    treefdr_region_section = figure_s7_treefdr_curated_df.loc[figure_s7_treefdr_curated_df['acronym'].isin(region_test_acronyms)].copy()
    if len(treefdr_region_section) > 0:
        treefdr_region_section['figure_panel'] = '4F'
        treefdr_region_section['analysis_section'] = 'Loaded per-condition TreeFDR result'
        treefdr_region_section['analysis_subsection'] = 'TreeFDR values for all clustered rejected regions'
        treefdr_region_section['cluster'] = treefdr_region_section['acronym'].map(embedding_df['label']).astype('float')
        treefdr_region_section['region_order_all_rejected'] = treefdr_region_section['acronym'].map(region_order_lookup)
        treefdr_region_section['shown_in_panel_F'] = treefdr_region_section['acronym'].isin(panel_f_shown_set)
        treefdr_region_section['test'] = 'Loaded TreeFDR/TreeBH output'
        if 'p.val' in treefdr_region_section.columns:
            treefdr_region_section['p_value'] = treefdr_region_section['p.val']
        if 'rejected' in treefdr_region_section.columns:
            treefdr_region_section['treefdr_rejected_for_condition'] = treefdr_region_section['rejected']
        treefdr_region_section['sig_text'] = treefdr_region_section['treefdr_rejected_for_condition'].map(lambda x: '*' if bool(x) else '')
    preferred_cols = ['figure_panel', 'analysis_section', 'analysis_subsection', 'test', 'comparison', 'term', 'condition', 'condition_pretty', 'condition_group1', 'condition_group2', 'cluster', 'cluster_group1', 'cluster_group2', 'label', 'acronym', 'cleaned_acronym', 'region_order_all_rejected', 'shown_in_panel_F', 'n_saline', 'n_condition', 'n_group1', 'n_group2', 'mean_saline', 'sem_saline', 'mean_condition', 'sem_condition', 'mean_difference', 'statistic', 'statistic_value', 'df1', 'df2', 'df', 't_stat', 'p_value', 'p_adjusted', 'q_value_BH_within_condition', 'p_adjustment_scope', 'ci_lower', 'ci_upper', 'sig_text', 'reject', 'treefdr_node_p_value', 'treefdr_leaf_glm_lrt_p_value', 'treefdr_q_adj', 'treefdr_rejected_for_condition', 'source_treefdr_csv', 'source_glm_input_csv']
    sections = [panel_c_anova, panel_c_condition_pairs, panel_c_cluster_pairs, panel_f_ttests]
    if len(panel_f_region_tukey) > 0:
        sections.append(panel_f_region_tukey)
    if len(treefdr_region_section) > 0:
        sections.append(treefdr_region_section)
    # Panel F: loaded per-condition TreeFDR results for all rejected regions used in Table 7.
    all_cols = []
    for df_1 in sections:
        for c in df_1.columns:
            if c not in all_cols:
                all_cols.append(c)
    ordered_cols = [c for c in preferred_cols if c in all_cols] + [c for c in all_cols if c not in preferred_cols]
    table7_df = pd.concat([df.reindex(columns=ordered_cols) for df in sections], ignore_index=True)
    section_order = {'Cluster-level two-way ANOVA': 1, 'Cluster-level Tukey HSD': 2, 'Region-level condition-vs-saline t-test': 3, 'Region-level Tukey HSD': 4, 'Loaded per-condition TreeFDR result': 5}
    table7_df.insert(0, 'table', 'Table 7')
    table7_df.insert(1, 'table_title', 'Statistical summary for Figure 4')
    table7_df.insert(2, 'section_order', table7_df['analysis_section'].map(section_order).fillna(99).astype(int))
    table7_df = table7_df.sort_values(by=['section_order', 'figure_panel', 'analysis_section', 'analysis_subsection', 'cluster', 'region_order_all_rejected', 'condition', 'comparison'], kind='mergesort', na_position='last').reset_index(drop=True)
    table7_df.insert(3, 'row_id', np.arange(1, len(table7_df) + 1))
    table7_csv_path = os.path.join(FIGS7_STATS_DIR, 'Table7_Figure4_single_table.csv')
    table7_tsv_path = os.path.join(FIGS7_STATS_DIR, 'Table7_Figure4_single_table.tsv')
    table7_df.to_csv(table7_csv_path, index=False)
    table7_df.to_csv(table7_tsv_path, index=False, sep='\t')
    # Harmonize columns into a single long-format Table 7.
    table7_index_df = table7_df.groupby(['analysis_section', 'analysis_subsection'], dropna=False).size().reset_index(name='n_rows')
    table7_index_df.to_csv(os.path.join(FIGS7_STATS_DIR, 'Table7_Figure4_single_table_section_index.csv'), index=False)
    legend_lines = []
    legend_lines.append('Figure 4 statistical values')
    legend_lines.append('Generated by Opioid_Figure4_cleaned_final_Table7.ipynb')
    legend_lines.append('')
    legend_lines.append('Panel C. Cluster-level two-way ANOVA:')
    for (_, row) in panel_c_anova.iterrows():
        pretty_term = row['analysis_subsection']
        legend_lines.append(f"{pretty_term}: F({fmt_float(row['df1'], 0)}, {fmt_float(row['df2'], 0)}) = {fmt_float(row['statistic_value'], 2)}, P = {fmt_p(row['p_value'])}.")
    legend_lines.append('All Tukey HSD condition-combination results within each cluster and all cluster-combination results within each condition are provided in Table 7.')
    legend_lines.append('')
    legend_lines.append('Panel F. Region-level comparisons versus saline were performed using independent t-tests followed by Benjamini-Hochberg correction across all clustered TreeFDR-rejected regions within each opioid condition. Table 7 includes all tested rejected regions and indicates whether each region is displayed in panel F.')
    for cond_5 in figure_s7_drug_conditions:
        g = figure_s7_region_ttest_sig_df.loc[figure_s7_region_ttest_sig_df['condition'] == cond_5].copy()
        if len(g) == 0:
            legend_lines.append(f'{condition_pretty.get(cond_5, cond_5)}: no regions survived q < 0.05.')
            continue
        parts = []
        g = g.sort_values('region_order_all_rejected')
        for (_, row) in g.iterrows():
            parts.append(f"{row['cleaned_acronym']}: P = {fmt_p(row['p_value'])}, q = {fmt_p(row['q_value_BH_within_condition'])}, t({fmt_float(row['df'], 0)}) = {fmt_float(row['t_stat'], 2)}")
        legend_lines.append(f'{condition_pretty.get(cond_5, cond_5)} significant regions: ' + '; '.join(parts) + '.')
    legend_lines.append('')
    legend_lines.append(f'Per-condition post-TreeFDR GLM/LRT results used q = {TREEFDR_Q}. TreeFDR results for all clustered rejected regions are included in Table 7.')
    legend_text = '\n'.join(legend_lines)
    legend_text_path = os.path.join(FIGS7_STATS_DIR, 'Figure4_legend_statistical_values.txt')
    with open(legend_text_path, 'w', encoding='utf-8') as f:
    # Add row order for stable sorting and spreadsheet filtering.
        f.write(legend_text)
    manifest = pd.DataFrame([{'file': 'Table7_Figure4_single_table.csv', 'description': 'Single long-format Table 7 containing all Panel C combinations and Panel F rejected-region p-values.'}, {'file': 'Table7_Figure4_single_table.tsv', 'description': 'Tab-delimited copy of the single Table 7.'}, {'file': 'Table7_Figure4_single_table_section_index.csv', 'description': 'Row counts for each Table 7 section.'}, {'file': 'Figure4_region_condition_vs_saline_ttest_BH_all_rejected_regions.csv', 'description': 'Panel F region t-tests/q-values for all clustered TreeFDR-rejected regions.'}, {'file': 'Figure4_panelF_region_test_universe.csv', 'description': 'All tested rejected regions and whether they are shown in panel F.'}, {'file': 'Figure4_legend_statistical_values.txt', 'description': 'Copy/paste-ready statistical values for the figure legend.'}, {'file': 'Figure4_cluster_condition_TukeyHSD_all_pairs.csv', 'description': 'Panel C all condition combinations within each cluster.'}, {'file': 'Figure4_condition_cluster_TukeyHSD_all_pairs.csv', 'description': 'Panel C all cluster combinations within each condition.'}, {'file': 'Figure4_region_TukeyHSD_all_pairs_all_rejected_regions.csv', 'description': 'Optional region-level all-pair Tukey HSD for all rejected regions.'}])
    manifest.to_csv(os.path.join(FIGS7_STATS_DIR, 'Figure4_statistical_output_manifest.csv'), index=False)
    print('Saved single Table 7 to:')
    print(table7_csv_path)
    print(table7_tsv_path)
    print('\nTable 7 section index:')
    print(table7_index_df)
    print('\nLegend text preview:\n')
    # Keep a simpler Table 7 index for checking what each section contains.
    # Copy/paste-ready legend statistics text.
    print(legend_text[:3000])
    return


@app.cell
def _(FIGS7_STATS_DIR, np, pd):
    # ============================================================
    # Final Table 7 Excel export
    # Loads existing Figure 4 statistical results only.
    #
    # Table 7 contains:
    #   7A. Panel C: two-way ANOVA
    #   7B. Panel C: all condition-combination Tukey HSD results within each cluster
    #   7C. Panel C: all cluster-combination Tukey HSD results within each condition
    #   7D. Panel F: condition-vs-saline p-values for all TreeFDR-rejected / clustered regions
    #   7E. Panel F: corresponding per-condition TreeFDR statistics for those regions
    from pathlib import Path as _Path
    try:
        stats_dir = _Path(FIGS7_STATS_DIR)
    except NameError:
        stats_dir = _Path.cwd()
    out_xlsx = stats_dir / 'Table 7.xlsx'
    # Use existing stats directory from the notebook if defined
    out_csv = stats_dir / 'Table7_Figure4_single_table.csv'
    out_tsv = stats_dir / 'Table7_Figure4_single_table.tsv'

    def _load_first_existing(patterns, required=True):
        """Load the first matching CSV in stats_dir."""
        for pat in patterns:
            hits = sorted(stats_dir.glob(pat))
            if len(hits) > 0:
                print(f'Loaded: {hits[0].name}')
                return pd.read_csv(hits[0])
    # ------------------------------------------------------------
    # Helpers
        if required:
            raise FileNotFoundError(f'Could not find any of these files in {stats_dir}:\n' + '\n'.join(patterns))
        print('Optional file not found:', patterns)
        return pd.DataFrame()

    def _bool_series(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])

    def _add_section(df, section, panel, analysis):
        d = df.copy()
        d.insert(0, 'Table_section', section)
        d.insert(1, 'Figure_panel', panel)
        d.insert(2, 'Analysis', analysis)
        return d

    def _find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _sig_from_p(p):
        if pd.isna(p):
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return 'ns'
    anova_df = _load_first_existing(['Figure4_cluster_two_way_ANOVA.csv', '*cluster*ANOVA*.csv'])
    panelC_condition_pairs_df = _load_first_existing(['Figure4_cluster_condition_TukeyHSD_all_pairs.csv', '*cluster*condition*Tukey*all*pairs*.csv', '*Tukey*byCluster*.csv'])
    panelC_cluster_pairs_df = _load_first_existing(['Figure4_condition_cluster_TukeyHSD_all_pairs.csv', '*condition*cluster*Tukey*all*pairs*.csv', '*Tukey*byCondition*.csv'])
    region_ttest_df = _load_first_existing(['Figure4_region_condition_vs_saline_ttest_BH.csv', '*region*condition*saline*ttest*BH*.csv'])
    treefdr_df = _load_first_existing(['Figure4_TreeFDR_curated_regions_all_conditions.csv', '*TreeFDR*curated*all*conditions*.csv'])
    cluster_membership_df = _load_first_existing(['Figure4_UMAP_HDBSCAN_cluster_membership.csv', '*cluster*membership*.csv'], required=False)
    if not cluster_membership_df.empty and 'acronym' in cluster_membership_df.columns:
        panelF_acronyms = sorted(cluster_membership_df['acronym'].dropna().astype(str).unique())
        panelF_region_source = 'Figure4_UMAP_HDBSCAN_cluster_membership.csv'
    else:
        if 'rejected' not in treefdr_df.columns:
            raise ValueError("TreeFDR table must contain a 'rejected' column.")
        panelF_acronyms = sorted(treefdr_df.loc[_bool_series(treefdr_df['rejected']), 'acronym'].dropna().astype(str).unique())
        panelF_region_source = 'TreeFDR rejected regions'
    print(f'Panel F region set: {len(panelF_acronyms)} regions from {panelF_region_source}')
    shown_acronyms = set()
    for varname in ['panel_f_acronyms', 'shown_panel_f_acronyms', 'plot_acronyms', 'selected_acronyms', 'heatmap_acronyms']:
        if varname in globals():
            try:
    # Load existing outputs
                shown_acronyms = set([str(x) for x in globals()[varname]])
                print(f'Using shown_in_panel_F list from variable: {varname}')
                break
            except Exception:
                pass
    if len(shown_acronyms) == 0:
        print('No explicit panel-F display list found; shown_in_panel_F will be False for all rows.')
    table7_sections = []
    table7_sections.append(_add_section(anova_df, '7A. Panel C: two-way ANOVA of cluster-level scaled c-Fos density', 'Figure 4C', 'Two-way ANOVA'))
    table7_sections.append(_add_section(panelC_condition_pairs_df, '7B. Panel C: all opioid-condition pairwise comparisons within each cluster', 'Figure 4C', 'Tukey HSD: condition pairs within cluster'))
    table7_sections.append(_add_section(panelC_cluster_pairs_df, '7C. Panel C: all cluster pairwise comparisons within each condition', 'Figure 4C', 'Tukey HSD: cluster pairs within condition'))
    panelF_ttest = region_ttest_df.copy()
    if 'acronym' not in panelF_ttest.columns:
        raise ValueError("Region t-test table must contain an 'acronym' column.")
    panelF_ttest['acronym'] = panelF_ttest['acronym'].astype(str)
    panelF_ttest = panelF_ttest.loc[panelF_ttest['acronym'].isin(panelF_acronyms)].copy()
    panelF_ttest['shown_in_panel_F'] = panelF_ttest['acronym'].isin(shown_acronyms)
    if not cluster_membership_df.empty and 'acronym' in cluster_membership_df.columns:
        cluster_cols = [c for c in cluster_membership_df.columns if c in ['acronym', 'cleaned_acronym', 'cluster', 'HDBSCAN_cluster', 'umap_x', 'umap_y']]
        panelF_ttest = panelF_ttest.merge(cluster_membership_df[cluster_cols].drop_duplicates('acronym'), on='acronym', how='left', suffixes=('', '_cluster'))
    p_col = _find_col(panelF_ttest, ['p_value', 'pvalue', 'p', 'P', 'p.val'])
    q_col = _find_col(panelF_ttest, ['p_adj', 'p_adjusted', 'q_value', 'q', 'p_bh', 'BH_q', 'padj'])
    if q_col is not None:
        panelF_ttest['significance'] = panelF_ttest[q_col].apply(_sig_from_p)
    elif p_col is not None:
        panelF_ttest['significance'] = panelF_ttest[p_col].apply(_sig_from_p)
    table7_sections.append(_add_section(panelF_ttest, '7D. Panel F: condition-versus-saline tests for all TreeFDR-rejected clustered regions', 'Figure 4F', 'Independent t-test vs saline with Benjamini-Hochberg correction'))
    panelF_treefdr = treefdr_df.copy()
    if 'acronym' not in panelF_treefdr.columns:
        raise ValueError("TreeFDR table must contain an 'acronym' column.")
    panelF_treefdr['acronym'] = panelF_treefdr['acronym'].astype(str)
    panelF_treefdr = panelF_treefdr.loc[panelF_treefdr['acronym'].isin(panelF_acronyms)].copy()
    panelF_treefdr['shown_in_panel_F'] = panelF_treefdr['acronym'].isin(shown_acronyms)
    if not cluster_membership_df.empty and 'acronym' in cluster_membership_df.columns:
        cluster_cols = [c for c in cluster_membership_df.columns if c in ['acronym', 'cleaned_acronym', 'cluster', 'HDBSCAN_cluster', 'umap_x', 'umap_y']]
        panelF_treefdr = panelF_treefdr.merge(cluster_membership_df[cluster_cols].drop_duplicates('acronym'), on='acronym', how='left', suffixes=('', '_cluster'))
    # Define the Panel F region set
    # Prefer the clustered/rejected regions used for Figure 4 UMAP/heatmap.
    # Fallback: union of TreeFDR-rejected curated regions.
    table7_sections.append(_add_section(panelF_treefdr, '7E. Panel F: per-condition TreeFDR statistics for all regions included in Panel F analysis', 'Figure 4F', 'Pairwise GLM followed by TreeFDR/TreeBH-style correction; loaded from existing results'))
    all_cols_1 = []
    for d in table7_sections:
        for c_1 in d.columns:
            if c_1 not in all_cols_1:
                all_cols_1.append(c_1)
    table7_df_1 = pd.concat([d.reindex(columns=all_cols_1) for d in table7_sections], ignore_index=True)
    table7_df_1.to_csv(out_csv, index=False)
    table7_df_1.to_csv(out_tsv, index=False, sep='\t')
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({'bold': True, 'font_size': 14, 'bg_color': '#1F4E78', 'font_color': 'white', 'align': 'left', 'valign': 'vcenter'})
        note_fmt = workbook.add_format({'italic': True, 'font_size': 10, 'font_color': '#555555', 'text_wrap': True})
        section_fmt = workbook.add_format({'bold': True, 'font_size': 12, 'bg_color': '#D9EAF7', 'font_color': '#17365D', 'top': 1, 'bottom': 1, 'align': 'left'})
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#BFBFBF', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
        cell_fmt = workbook.add_format({'border': 1, 'valign': 'top'})
        num_fmt = workbook.add_format({'border': 1, 'valign': 'top', 'num_format': '0.0000'})
        sci_fmt = workbook.add_format({'border': 1, 'valign': 'top', 'num_format': '0.00E+00'})
        sig_fmt = workbook.add_format({'border': 1, 'valign': 'top', 'bg_color': '#FFF2CC'})
        sheet_name = 'Table 7'
    # Mark regions shown in Panel F, if a panel-F display list exists
        ws = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = ws
        ws.write(0, 0, 'Table 7. Statistical results for Figure 4', title_fmt)
        ws.merge_range(0, 0, 0, max(6, len(all_cols_1) - 1), 'Table 7. Statistical results for Figure 4', title_fmt)
        note = 'This table was generated from existing Figure 4 result files only. Panel C includes the two-way ANOVA and all Tukey HSD pairwise combinations. Panel F includes condition-versus-saline tests and TreeFDR statistics for all TreeFDR-rejected / clustered regions used in the Panel F analysis, not only the regions shown visually in the heatmap.'
        ws.merge_range(1, 0, 2, max(6, len(all_cols_1) - 1), note, note_fmt)
        start_row = 4
        for section_df in table7_sections:
            section_title = section_df['Table_section'].iloc[0]
            ws.merge_range(start_row, 0, start_row, max(6, len(all_cols_1) - 1), section_title, section_fmt)
            start_row = start_row + 1
            for (col_idx, col_name) in enumerate(all_cols_1):
                ws.write(start_row, col_idx, col_name, header_fmt)
            start_row = start_row + 1
            data_1 = section_df.reindex(columns=all_cols_1)
            for (r_idx, row_1) in enumerate(data_1.itertuples(index=False), start=start_row):
                for (c_idx, value) in enumerate(row_1):
                    col_name = all_cols_1[c_idx]
                    if pd.isna(value):
    # If no display list exists, leave as False rather than guessing.
                        ws.write_blank(r_idx, c_idx, None, cell_fmt)
                    elif isinstance(value, (int, float, np.integer, np.floating)):
                        if any((key in col_name.lower() for key in ['p', 'q', 'adj'])):
                            ws.write_number(r_idx, c_idx, float(value), sci_fmt)
                        else:
    # Prepare Panel C sections
                            ws.write_number(r_idx, c_idx, float(value), num_fmt)
                    else:
                        fmt = sig_fmt if col_name.lower() in ['significance', 'sig_text'] and str(value) not in ['', 'ns'] else cell_fmt
                        ws.write(r_idx, c_idx, str(value), fmt)
            start_row = start_row + (len(data_1) + 2)
        for (i_3, col) in enumerate(all_cols_1):
            max_len = len(str(col))
            sample_values = table7_df_1[col].dropna().astype(str).head(200).tolist() if col in table7_df_1.columns else []
            if sample_values:
                max_len = max(max_len, max((len(x) for x in sample_values)))
            width = min(max(max_len + 2, 10), 38)
            ws.set_column(i_3, i_3, width)
        ws.freeze_panes(5, 3)
        ws.autofilter(5, 0, max(5, start_row - 1), max(0, len(all_cols_1) - 1))
        for (i_3, col) in enumerate(all_cols_1):
            low = col.lower()
            if low in ['p', 'pvalue', 'p_value', 'p.val', 'p-adj', 'p_adj', 'p_adjusted', 'q', 'q_value'] or 'pvalue' in low or 'p_value' in low or ('p-adj' in low) or ('p_adj' in low) or ('q_value' in low):
                ws.conditional_format(6, i_3, max(6, start_row - 1), i_3, {'type': 'cell', 'criteria': '<', 'value': 0.05, 'format': workbook.add_format({'bg_color': '#FCE4D6'})})
        anova_df.to_excel(writer, sheet_name='7A_ANOVA', index=False)
        panelC_condition_pairs_df.to_excel(writer, sheet_name='7B_Tukey_condition_pairs', index=False)
        panelC_cluster_pairs_df.to_excel(writer, sheet_name='7C_Tukey_cluster_pairs', index=False)
        panelF_ttest.to_excel(writer, sheet_name='7D_PanelF_ttests', index=False)
        panelF_treefdr.to_excel(writer, sheet_name='7E_PanelF_TreeFDR', index=False)
        manifest_1 = pd.DataFrame({'Output': [str(out_xlsx), str(out_csv), str(out_tsv)], 'Description': ['Formatted Excel Table 7', 'Long-format Table 7 CSV', 'Long-format Table 7 TSV']})
        manifest_1.to_excel(writer, sheet_name='Manifest', index=False)
    print(f'Saved Excel Table 7: {out_xlsx}')
    print(f'Saved CSV: {out_csv}')
    print(f'Saved TSV: {out_tsv}')
    # Prepare Panel F region-level t-test section
    # Add cluster membership if available
    # Add significance if possible
    # Prepare Panel F TreeFDR section for the same regions
    # Combine into a single long-format Table 7
    # Union all columns while preserving section order
    # Save machine-readable versions
    # Write formatted Excel workbook
    print(table7_df_1.head(20).to_string())  # Single formatted sheet  # Header  # Rows  # Column widths  # Conditional formatting for p/q columns  # Also write clean source sheets for easier checking  # Manifest
    return


app._unparsable_cell(
    r"""
    pip install xlsxwriter
    """,
    name="_"
)


@app.cell
def _(Conditions, atlas_df, atlas_img, fname_list, heatmap_da, merge_df, np):
    # ============================================================
    # Voxel-wise GLM regression (restored from the original acute-drug analysis).
    # Covariates: BW, Age (standardized) + Sex; drug-condition dummies (Saline = baseline)
    # + constant. Produces the Figure 4D beta-coefficient maps in memory.
    # No files are written here (beta / adjusted-heatmap export is done separately).
    # ============================================================
    import statsmodels.api as _sm
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    # collect the per-subject variables (acronym == 'CH' = whole-brain root)
    _vdf = merge_df.loc[merge_df.acronym == 'CH', ['fname', 'Condition', 'BW', 'Age', 'Sex_d'] + Conditions].set_index('fname')
    _vdf = _vdf.loc[fname_list]                                  # order to the heatmap rows
    _vdf[['BW', 'Age']] = _StandardScaler(with_mean=True).fit_transform(_vdf[['BW', 'Age']])
    _vdf = _vdf.drop(columns='Saline')                          # treat Saline as baseline
    _vdf['constant'] = 1
    _vdf = _vdf.drop(columns='Condition')
    _variables = np.array(_vdf.astype('float'))

    # brain voxels only, then fit voxel-wise OLS
    _brain_voxels = np.where(np.isin(atlas_img.flatten(), atlas_df['id'].values))[0]
    _brain_heatmap = heatmap_da[:, _brain_voxels].compute()
    _models = _sm.OLS(_brain_heatmap, _variables).fit()

    # beta-coefficient map per variable
    betas_dict = {}
    for _idx, _name in enumerate(_vdf.columns):
        _img = np.zeros_like(atlas_img.flatten(), dtype='float64')
        _img[_brain_voxels] = _models.params[_idx, :]
        betas_dict[_name] = np.reshape(_img, atlas_img.shape)
    print("Computed beta maps for:", list(_vdf.columns))
    return (betas_dict,)


if __name__ == "__main__":
    app.run()
