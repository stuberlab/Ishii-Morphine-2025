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
    # Clean factor-region analysis

    This notebook separates the workflow into:

    1. **Configuration / helper functions**
    2. **Data preprocessing only**
       Computes and saves factor–gene and factor–cluster correlation tables.
    3. **Plotting only**
       Each plotting cell generates one plot for one region/factor pair.

    Default target regions are:

    ```python
    ["Acb", "Ce", "VTA"]
    ```

    Default factors are:

    ```python
    [0, 1, 2]
    ```

    The plotting cells read the saved result CSVs, so figure generation can be rerun without recomputing correlations.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import os

    # ============================================================
    # Config
    # ============================================================
    # Paths -> Figshare deposit (set OPIOID_DATA_ROOT or edit the default).
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    GROUP = DATA_ROOT / "06_spatial_gene_merfish"
    ATLAS_DIR = DATA_ROOT / "shared" / "atlas"
    assert GROUP.exists(), f"DATA_ROOT not found: {DATA_ROOT}. Set OPIOID_DATA_ROOT."

    DATA_DIR = GROUP
    FACTOR_DIR = GROUP / "factors"                 # semi-NMF factor{i}.npy (from Figure 6)
    OUTPUT_DIR = Path("../figures/Figure8")        # panels + result CSVs -> repo
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Atlas / metadata files (shared)
    ATLAS_IMAGE_PATH = ATLAS_DIR / "Kim_ref_adult_FP-label_v4.0.tif"
    ATLAS_INFO_PATH = ATLAS_DIR / "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv"
    CONTOUR_IMAGE_PATH = ATLAS_DIR / "Kim_ref_adult_FP-label_v2.9_contour_map.tif"

    # Allen-derived spatial data (ABI MERFISH -> Unified atlas)
    GENE_ZARR_PATH = str(GROUP / "gene_expression.zarr")
    GENE_METADATA_PATH = GROUP / "gene_df.csv"
    CLUSTER_ZARR_PATH = str(GROUP / "cluster_expression.zarr")
    CLUSTER_METADATA_PATH = GROUP / "cluster_details.csv"

    # Analysis targets
    TARGET_REGIONS = ["Acb", "Ce", "VTA"]
    FACTOR_INDICES = [0, 1, 2]

    # Results
    TOP_N = 25
    N_PERMUTATIONS = 100
    RNG_SEED = 42
    ALPHA = 0.05

    # Cluster filtering: keep only clusters present in the target region.
    MIN_CLUSTER_NONZERO_VOXELS_IN_REGION = 10
    MIN_CLUSTER_TOTAL_SIGNAL_IN_REGION = 10.0

    # Figure settings
    N_CLOSEUP_SLICES = 6
    CLOSEUP_MARGIN = 60
    HIGHLIGHT_GENES = ["Oprm1", "Oprd1", "Oprk1"]
    FIG_DPI = 1024

    # If your cluster zarr was created in a transposed / flattened orientation like in the dirty notebook,
    # keep this True. If it is already aligned as (cluster, z, y, x) / atlas_img.shape, set False.
    CLUSTER_NEEDS_AXIS_SWAP = True
    return (
        ALPHA,
        ATLAS_IMAGE_PATH,
        ATLAS_INFO_PATH,
        CLOSEUP_MARGIN,
        CLUSTER_METADATA_PATH,
        CLUSTER_NEEDS_AXIS_SWAP,
        CLUSTER_ZARR_PATH,
        CONTOUR_IMAGE_PATH,
        FACTOR_DIR,
        FACTOR_INDICES,
        GENE_METADATA_PATH,
        GENE_ZARR_PATH,
        HIGHLIGHT_GENES,
        MIN_CLUSTER_NONZERO_VOXELS_IN_REGION,
        MIN_CLUSTER_TOTAL_SIGNAL_IN_REGION,
        N_CLOSEUP_SLICES,
        N_PERMUTATIONS,
        OUTPUT_DIR,
        Path,
        RNG_SEED,
        TARGET_REGIONS,
        TOP_N,
    )


@app.cell
def _():
    # ============================================================
    # Imports
    import math
    import json
    import numpy as np
    import pandas as pd
    import dask.array as da
    import tifffile
    import matplotlib.pyplot as plt
    from statsmodels.stats.multitest import multipletests
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return da, json, multipletests, np, pd, plt, tifffile


@app.cell
def _(
    ALPHA,
    MIN_CLUSTER_NONZERO_VOXELS_IN_REGION,
    MIN_CLUSTER_TOTAL_SIGNAL_IN_REGION,
    OUTPUT_DIR,
    Path,
    multipletests,
    np,
    tifffile,
):
    def ensure_dir(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_optional_contour(contour_path):
        if contour_path is None:
            return None
        contour_path = Path(contour_path)
        if contour_path.exists():
            return tifffile.imread(contour_path)
        return None

    def get_descendant_ids(atlas_df, root_acronym, include_self=True):
        """Return all atlas ids under a region acronym."""
        if root_acronym not in set(atlas_df['acronym']):
            raise ValueError(f'Region acronym not found: {root_acronym}')
        root_id = int(atlas_df.loc[atlas_df['acronym'] == root_acronym, 'id'].iloc[0])
        parent_to_children = atlas_df.groupby('parent_id')['id'].apply(list).to_dict()
        ids = []
        stack = [root_id]
        while stack:
            current = stack.pop()
            ids.append(current)
            stack.extend(parent_to_children.get(current, []))
        if not include_self:
            ids = [_x for _x in ids if _x != root_id]
        return np.array(sorted(set(ids)), dtype=int)

    def average_hemispheres(img3d):
        """Average left hemisphere with mirrored right hemisphere."""
        half = img3d.shape[2] // 2
        left = img3d[:, :, :half]
        right = img3d[:, :, half:][:, :, ::-1]
        return (left + right) / 2.0

    def get_region_mask_and_bbox(hemi_atlas_img, region_ids, margin=20):
        _region_mask = np.isin(hemi_atlas_img, _region_ids)
        coords = np.argwhere(_region_mask)
        if coords.size == 0:
            raise ValueError('No voxels found for this region in the left hemisphere atlas.')
        (zmin, ymin, xmin) = coords.min(axis=0)
        (zmax, ymax, xmax) = coords.max(axis=0)
        ymin = max(0, ymin - margin)
        ymax = min(_region_mask.shape[1], ymax + margin + 1)
        xmin = max(0, xmin - margin)
        xmax = min(_region_mask.shape[2], xmax + margin + 1)
        return (_region_mask, (slice(ymin, ymax), slice(xmin, xmax)))

    def choose_region_slices(region_mask, n_slices=6):
        z_values = np.where(_region_mask.any(axis=(1, 2)))[0]
        if len(z_values) == 0:
            return np.array([], dtype=int)
        if len(z_values) <= n_slices:
            return z_values
        idx = np.linspace(0, len(z_values) - 1, n_slices).round().astype(int)
        return z_values[idx]

    def edge_from_mask(mask):
        """Simple edge extraction for overlay contours."""
        mask = mask.astype(bool)
        up = np.roll(mask, 1, axis=0)
        down = np.roll(mask, -1, axis=0)
        left = np.roll(mask, 1, axis=1)
        right = np.roll(mask, -1, axis=1)
        eroded = mask & up & down & left & right
        return mask & ~eroded

    def standardize_vector(x):
        _x = np.asarray(_x, dtype=float)
        _x = _x - np.nanmean(_x)
        sd = np.nanstd(_x, ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise ValueError('Target vector has zero variance.')
        return _x / sd

    def permutation_test_correlations(Y, x, n_perm=100, seed=42):
        """
        Pearson r between each row of Y and x.
        Uses voxel-shuffled x for empirical two-tailed permutation p-values.
        """
        rng = np.random.default_rng(seed)
        xz = standardize_vector(_x)
        Y = np.asarray(Y, dtype=float)
        Ym = Y.mean(axis=1, keepdims=True)
        Ysd = Y.std(axis=1, ddof=1)
        valid = np.isfinite(Ysd) & (Ysd > 0)
        observed = np.full(Y.shape[0], np.nan, dtype=float)
        pvals = np.full(Y.shape[0], np.nan, dtype=float)
        pvals_fdr = np.full(Y.shape[0], np.nan, dtype=float)
        if not valid.any():
            return (observed, pvals, pvals_fdr)
        Yz = (Y[valid] - Ym[valid]) / Ysd[valid, None]
        observed_valid = Yz @ xz / (Y.shape[1] - 1)
        observed[valid] = observed_valid
        perm_extreme = np.zeros(observed_valid.shape[0], dtype=int)
        for _ in range(n_perm):
            perm_xz = xz[rng.permutation(len(xz))]
            perm_r = Yz @ perm_xz / (Y.shape[1] - 1)
            perm_extreme = perm_extreme + (np.abs(perm_r) >= np.abs(observed_valid))
        pvals_valid = (perm_extreme + 1) / (n_perm + 1)
        pvals[valid] = pvals_valid
        (_, fdr_valid, _, _) = multipletests(pvals_valid, alpha=ALPHA, method='fdr_bh')
        pvals_fdr[valid] = fdr_valid
        return (observed, pvals, pvals_fdr)

    def sort_and_flag_results(df, r_col='r', fdr_col='p_perm_fdr', alpha=0.05):
        out = df.copy()
        out['abs_r'] = out[r_col].abs()
        out['significant'] = out[fdr_col] < alpha
        out = out.sort_values(['significant', 'abs_r'], ascending=[False, False]).reset_index(drop=True)
        out['rank_abs_r'] = np.arange(1, len(out) + 1)
        return out

    def top_significant(df, top_n=25):
        return df[df['significant']].sort_values('abs_r', ascending=False).head(top_n).copy()

    def feature_table_from_results(meta_df, label_col, r, p, p_fdr):
        meta = meta_df.copy().reset_index(drop=True)
        meta['r'] = r
        meta['p_perm'] = p
        meta['p_perm_fdr'] = p_fdr
        meta[label_col] = meta[label_col].astype(str)
        return sort_and_flag_results(meta, r_col='r', fdr_col='p_perm_fdr', alpha=ALPHA)

    def save_tables(full_df, top_df, out_prefix):
        out_prefix = Path(out_prefix)
        full_path = out_prefix.with_name(out_prefix.name + '_full_results.csv')
        top_path = out_prefix.with_name(out_prefix.name + '_top_results.csv')
        full_df.to_csv(full_path, index=False)
        top_df.to_csv(top_path, index=False)
        return (full_path, top_path)

    def prepare_gene_hemi_view(gene_zarr, atlas_shape):
        n_genes = gene_zarr.shape[0]
        return gene_zarr.reshape((n_genes,) + tuple(atlas_shape))[:, :, :, :atlas_shape[2] // 2].reshape((n_genes, -1))

    def prepare_cluster_hemi_view(cluster_zarr, atlas_shape, n_clusters, needs_axis_swap=True):
        if needs_axis_swap:
            cluster_flat = cluster_zarr.reshape((n_clusters,) + tuple(atlas_shape[::-1])).transpose(0, 3, 2, 1).reshape((n_clusters, -1))
        else:
            cluster_flat = cluster_zarr.reshape((n_clusters,) + tuple(atlas_shape)).reshape((n_clusters, -1))
        return cluster_flat.reshape((n_clusters,) + tuple(atlas_shape))[:, :, :, :atlas_shape[2] // 2].reshape((n_clusters, -1))

    def select_clusters_present_in_region(cluster_subset):
        nonzero_voxels = (cluster_subset > 0).sum(axis=1)
        total_signal = cluster_subset.sum(axis=1)
        keep = (nonzero_voxels >= MIN_CLUSTER_NONZERO_VOXELS_IN_REGION) & (total_signal >= MIN_CLUSTER_TOTAL_SIGNAL_IN_REGION)
        return (keep, nonzero_voxels, total_signal)

    def make_output_dirs():
        dirs = {'figure': ensure_dir(OUTPUT_DIR / 'figures'), 'closeup': ensure_dir(OUTPUT_DIR / 'figures' / 'factor_closeups'), 'gene_ecdf': ensure_dir(OUTPUT_DIR / 'figures' / 'gene_ecdf'), 'cluster_ecdf': ensure_dir(OUTPUT_DIR / 'figures' / 'cluster_ecdf'), 'gene_results': ensure_dir(OUTPUT_DIR / 'gene_results'), 'cluster_results': ensure_dir(OUTPUT_DIR / 'cluster_results'), 'summary': ensure_dir(OUTPUT_DIR / 'summaries')}
        return dirs

    return (
        average_hemispheres,
        choose_region_slices,
        ensure_dir,
        feature_table_from_results,
        get_descendant_ids,
        get_region_mask_and_bbox,
        load_optional_contour,
        make_output_dirs,
        permutation_test_correlations,
        prepare_cluster_hemi_view,
        prepare_gene_hemi_view,
        save_tables,
        select_clusters_present_in_region,
        top_significant,
    )


@app.cell
def _(
    ATLAS_IMAGE_PATH,
    ATLAS_INFO_PATH,
    CONTOUR_IMAGE_PATH,
    GENE_METADATA_PATH,
    GENE_ZARR_PATH,
    da,
    load_optional_contour,
    make_output_dirs,
    pd,
    prepare_gene_hemi_view,
    tifffile,
):
    # ============================================================
    # Load atlas, expression matrices, and metadata
    # ============================================================
    dirs = make_output_dirs()

    atlas_img = tifffile.imread(ATLAS_IMAGE_PATH)
    contour_img = load_optional_contour(CONTOUR_IMAGE_PATH)
    atlas_df = pd.read_csv(ATLAS_INFO_PATH)
    hemi_atlas_img = atlas_img[:, :, :atlas_img.shape[2] // 2]

    gene_meta = pd.read_csv(GENE_METADATA_PATH)
    if "gene_symbol" not in gene_meta.columns:
        raise ValueError("gene_df.csv must contain a 'gene_symbol' column.")

    gene_zarr = da.from_zarr(str(GENE_ZARR_PATH), mode="r")
    gene_hemi_view = prepare_gene_hemi_view(gene_zarr, atlas_img.shape)
    return (
        atlas_df,
        atlas_img,
        contour_img,
        dirs,
        gene_hemi_view,
        gene_meta,
        gene_zarr,
        hemi_atlas_img,
    )


@app.cell
def _(
    CLUSTER_METADATA_PATH,
    CLUSTER_NEEDS_AXIS_SWAP,
    CLUSTER_ZARR_PATH,
    OUTPUT_DIR,
    atlas_img,
    da,
    gene_meta,
    pd,
    prepare_cluster_hemi_view,
):
    cluster_details = pd.read_csv(CLUSTER_METADATA_PATH, index_col=0)
    cluster_zarr = da.from_zarr(str(CLUSTER_ZARR_PATH), mode="r")
    cluster_hemi_view = prepare_cluster_hemi_view(
        cluster_zarr,
        atlas_shape=atlas_img.shape,
        n_clusters=cluster_details.shape[0],
        needs_axis_swap=CLUSTER_NEEDS_AXIS_SWAP,
    )

    cluster_meta = cluster_details.copy().reset_index().rename(
        columns={cluster_details.index.name or "index": "cluster_alias"}
    )
    if "cluster_alias" not in cluster_meta.columns:
        cluster_meta["cluster_alias"] = cluster_meta.index.astype(str)
    if "cluster" not in cluster_meta.columns:
        cluster_meta["cluster"] = cluster_meta["cluster_alias"]

    print("Atlas shape:", atlas_img.shape)
    print("Genes:", gene_meta.shape[0])
    print("Clusters:", cluster_meta.shape[0])
    print("Output:", OUTPUT_DIR.resolve())
    return cluster_details, cluster_hemi_view, cluster_meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validate the gene matrix organization
    """)
    return


@app.cell
def _(gene_hemi_view, gene_meta, hemi_atlas_img, plt):
    plt.imshow(gene_hemi_view[gene_meta.gene_symbol == 'Slc17a6'].reshape(hemi_atlas_img.shape)[95,:,:])
    return


@app.cell
def _(cluster_details, cluster_hemi_view, hemi_atlas_img, plt):
    plt.imshow(cluster_hemi_view[cluster_details.cluster.str.contains('0964')].reshape(hemi_atlas_img.shape)[95,:,:])
    return


@app.cell
def _(atlas_img, gene_meta, gene_zarr, plt):
    # DEBUG LATER
    # The gene_hemi is in a wrong orientation
    plt.imshow(gene_zarr[gene_meta.gene_symbol == 'Slc17a6'].reshape(atlas_img.shape[::-1])[:,:,100].T)
    return


@app.cell
def _(
    ALPHA,
    CLOSEUP_MARGIN,
    FACTOR_DIR,
    FACTOR_INDICES,
    N_PERMUTATIONS,
    OUTPUT_DIR,
    RNG_SEED,
    TARGET_REGIONS,
    TOP_N,
    atlas_df,
    average_hemispheres,
    dirs,
    feature_table_from_results,
    gene_hemi_view,
    gene_meta,
    get_descendant_ids,
    get_region_mask_and_bbox,
    hemi_atlas_img,
    json,
    np,
    pd,
    permutation_test_correlations,
    save_tables,
    top_significant,
):
    # ============================================================
    # Data preprocessing only: GENE correlations
    all_gene_results = []
    # This cell computes and saves all gene correlation tables.
    # It does not generate figures.
    for _region_name in TARGET_REGIONS:
        print(f'\n=== Region: {_region_name} ===')
        _region_ids = get_descendant_ids(atlas_df, _region_name, include_self=True)
        (_region_mask, _) = get_region_mask_and_bbox(hemi_atlas_img, _region_ids, margin=CLOSEUP_MARGIN)
        _region_indices = np.flatnonzero(_region_mask.ravel())
        print(f'Region voxels: {len(_region_indices):,}')
        print('Loading gene matrix subset...')
        gene_subset = gene_hemi_view[:, _region_indices].compute()
        print('gene_subset:', gene_subset.shape)
        for _factor_idx in FACTOR_INDICES:
            print(f'  - factor {_factor_idx}')
            _factor_path = FACTOR_DIR / f'factor{_factor_idx}.npy'
            _factor_array = np.load(_factor_path)
            _factor_hemi = average_hemispheres(_factor_array)
            _x = _factor_hemi.ravel()[_region_indices]
            (r_gene, p_gene, p_gene_fdr) = permutation_test_correlations(Y=gene_subset, x=_x, n_perm=N_PERMUTATIONS, seed=RNG_SEED)
            gene_results = feature_table_from_results(meta_df=gene_meta[['gene_symbol']].copy(), label_col='gene_symbol', r=r_gene, p=p_gene, p_fdr=p_gene_fdr)
            gene_results.insert(0, 'region', _region_name)  # Dask partial loading: only load target region voxels
            gene_results.insert(1, 'factor', _factor_idx)
            gene_top = top_significant(gene_results, top_n=TOP_N)
            save_tables(gene_results, gene_top, dirs['gene_results'] / f'{_region_name}_factor{_factor_idx}_genes')
            all_gene_results.append(gene_results)
    combined_gene_results = pd.concat(all_gene_results, ignore_index=True)
    combined_gene_results.to_csv(dirs['summary'] / 'all_gene_results_combined.csv', index=False)
    combined_gene_top = combined_gene_results[combined_gene_results['significant']].sort_values(['region', 'factor', 'abs_r'], ascending=[True, True, False]).groupby(['region', 'factor'], group_keys=False).head(TOP_N).reset_index(drop=True)
    combined_gene_top.to_csv(dirs['summary'] / f'all_gene_top{TOP_N}_combined.csv', index=False)
    gene_manifest = {'analysis_type': 'gene_correlation', 'target_regions': TARGET_REGIONS, 'factor_indices': FACTOR_INDICES, 'top_n': TOP_N, 'n_permutations': N_PERMUTATIONS, 'alpha': ALPHA}
    with open(dirs['summary'] / 'gene_analysis_manifest.json', 'w') as _f:
        json.dump(gene_manifest, _f, indent=2)
    print('\nGene preprocessing complete.')
    # Combined gene summary tables
    print('Saved gene result tables to:', OUTPUT_DIR.resolve())  # Gene correlations
    return


@app.cell
def _(
    ALPHA,
    CLOSEUP_MARGIN,
    FACTOR_DIR,
    FACTOR_INDICES,
    MIN_CLUSTER_NONZERO_VOXELS_IN_REGION,
    MIN_CLUSTER_TOTAL_SIGNAL_IN_REGION,
    N_PERMUTATIONS,
    OUTPUT_DIR,
    RNG_SEED,
    TARGET_REGIONS,
    TOP_N,
    atlas_df,
    average_hemispheres,
    cluster_hemi_view,
    cluster_meta,
    dirs,
    feature_table_from_results,
    get_descendant_ids,
    get_region_mask_and_bbox,
    hemi_atlas_img,
    json,
    np,
    pd,
    permutation_test_correlations,
    save_tables,
    select_clusters_present_in_region,
    top_significant,
):
    # ============================================================
    # Data preprocessing only: CLUSTER correlations
    all_cluster_results = []
    # This cell computes and saves all cluster correlation tables.
    # It does not generate figures.
    for _region_name in TARGET_REGIONS:
        print(f'\n=== Region: {_region_name} ===')
        _region_ids = get_descendant_ids(atlas_df, _region_name, include_self=True)
        (_region_mask, _) = get_region_mask_and_bbox(hemi_atlas_img, _region_ids, margin=CLOSEUP_MARGIN)
        _region_indices = np.flatnonzero(_region_mask.ravel())
        print(f'Region voxels: {len(_region_indices):,}')
        print('Loading cluster matrix subset...')
        cluster_subset_all = cluster_hemi_view[:, _region_indices].compute()
        print('cluster_subset_all:', cluster_subset_all.shape)
        (keep_clusters, cluster_nonzero_voxels, cluster_total_signal) = select_clusters_present_in_region(cluster_subset_all)
        cluster_subset = cluster_subset_all[keep_clusters]
        cluster_meta_region = cluster_meta.loc[keep_clusters].copy().reset_index(drop=True)
        cluster_meta_region['nonzero_voxels_in_region'] = cluster_nonzero_voxels[keep_clusters]
        cluster_meta_region['total_signal_in_region'] = cluster_total_signal[keep_clusters]
        print(f'Keeping {keep_clusters.sum():,} / {len(keep_clusters):,} clusters present in {_region_name}')
        print('cluster_subset:', cluster_subset.shape)
        for _factor_idx in FACTOR_INDICES:
            print(f'  - factor {_factor_idx}')  # Dask partial loading: only load target region voxels
            _factor_path = FACTOR_DIR / f'factor{_factor_idx}.npy'
            _factor_array = np.load(_factor_path)
            _factor_hemi = average_hemispheres(_factor_array)
            _x = _factor_hemi.ravel()[_region_indices]
            (r_cluster, p_cluster, p_cluster_fdr) = permutation_test_correlations(Y=cluster_subset, x=_x, n_perm=N_PERMUTATIONS, seed=RNG_SEED)
            cluster_results = feature_table_from_results(meta_df=cluster_meta_region[['cluster_alias', 'cluster', 'nonzero_voxels_in_region', 'total_signal_in_region']].copy(), label_col='cluster_alias', r=r_cluster, p=p_cluster, p_fdr=p_cluster_fdr)
            cluster_results.insert(0, 'region', _region_name)
            cluster_results.insert(1, 'factor', _factor_idx)
            cluster_top = top_significant(cluster_results, top_n=TOP_N)
            save_tables(cluster_results, cluster_top, dirs['cluster_results'] / f'{_region_name}_factor{_factor_idx}_clusters')
            all_cluster_results.append(cluster_results)
    combined_cluster_results = pd.concat(all_cluster_results, ignore_index=True)
    combined_cluster_results.to_csv(dirs['summary'] / 'all_cluster_results_combined.csv', index=False)
    combined_cluster_top = combined_cluster_results[combined_cluster_results['significant']].sort_values(['region', 'factor', 'abs_r'], ascending=[True, True, False]).groupby(['region', 'factor'], group_keys=False).head(TOP_N).reset_index(drop=True)
    combined_cluster_top.to_csv(dirs['summary'] / f'all_cluster_top{TOP_N}_combined.csv', index=False)
    cluster_manifest = {'analysis_type': 'cluster_correlation', 'target_regions': TARGET_REGIONS, 'factor_indices': FACTOR_INDICES, 'top_n': TOP_N, 'n_permutations': N_PERMUTATIONS, 'alpha': ALPHA, 'min_cluster_nonzero_voxels_in_region': MIN_CLUSTER_NONZERO_VOXELS_IN_REGION, 'min_cluster_total_signal_in_region': MIN_CLUSTER_TOTAL_SIGNAL_IN_REGION}
    with open(dirs['summary'] / 'cluster_analysis_manifest.json', 'w') as _f:
        json.dump(cluster_manifest, _f, indent=2)
    print('\nCluster preprocessing complete.')
    # Combined cluster summary tables
    print('Saved cluster result tables to:', OUTPUT_DIR.resolve())  # Cluster correlations
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plotting section

    Each cell below generates one saved plot. These cells rely on the saved CSV result tables from the preprocessing cell.
    """)
    return


@app.cell
def _(HIGHLIGHT_GENES, OPIOID_GENES, adjust_text, dirs, np, pd, plt, sns):
    def _plot_ecdf_from_results(results_csv, label_col, title, output_dir, output_name, highlight_labels=None, xmin=-0.25, xmax=0.25, top_pos_n=5, top_neg_n=5, shorten_cluster_label=False):
        """
        ECDF plot consistent with the original notebook.

        Annotates:
          - top_pos_n significant features with largest positive r
          - top_neg_n significant features with most negative r
          - highlight_labels regardless of significance/rank
        """
        highlight_labels = set([] if highlight_labels is None else list(map(str, highlight_labels)))
        df = pd.read_csv(results_csv)
        d = df.sort_values('r').reset_index(drop=True).copy()
        _x = d['r'].to_numpy()
        y = np.arange(1, len(d) + 1) / len(d)
        if 'significant' in d.columns:
            d['color'] = np.where(d['significant'].to_numpy(), 'magenta', 'gray')
        else:
            d['color'] = 'gray'
        colors = d['color'].to_numpy()
        (_fig, ax) = plt.subplots(1, 1, figsize=(1.6, 2.5))
        ax.scatter(_x, y, c=colors, s=10, marker='o', edgecolor='none', alpha=0.9)
        if 'significant' in df.columns:
            sig_df = df[df['significant']].copy()
        else:
            sig_df = df.copy()
        top_pos_df = sig_df[sig_df['r'] > 0].sort_values('r', ascending=False).head(top_pos_n)
        top_neg_df = sig_df[sig_df['r'] < 0].sort_values('r', ascending=True).head(top_neg_n)
        labels_to_annotate = set(top_pos_df[label_col].astype(str).tolist()) | set(top_neg_df[label_col].astype(str).tolist()) | highlight_labels
        d['annotate'] = d[label_col].astype(str).isin(labels_to_annotate)
        texts = []
        for (i, row) in d[d['annotate']].iterrows():
            full_label = str(row[label_col])
            if shorten_cluster_label:
                label_text = full_label.split(' ')[0]
            else:
                label_text = full_label
            texts.append(ax.text(row['r'], y[i], label_text, fontsize=8, ha='left', va='bottom', color='black'))
        if len(texts) > 0:
            adjust_text(texts, ax=ax, arrowprops=dict(color='gray', alpha=1, lw=0.5))
        ax.axvline(0, color='gray', lw=0.7, ls=':', alpha=0.7)
        ax.set_xlabel('Pearson r')
        ax.set_ylabel('ECDF')
        ax.set_title(title, fontsize=9)
        ax.set_xlim(xmin, xmax)
        sns.despine()
        _fig.tight_layout()
        _save_fig_original_style(_fig, output_dir, output_name, dpi=216)
        return _fig

    def plot_gene_ecdf(region_name, factor_idx, xmin=-0.25, xmax=0.25, top_pos_n=5, top_neg_n=5):  # --------------------------------------------------------
        csv_path = dirs['gene_results'] / f'{_region_name}_factor{_factor_idx}_genes_full_results.csv'  # Annotate top positive and top negative separately
        gene_highlights = set(HIGHLIGHT_GENES) | set(OPIOID_GENES)  # --------------------------------------------------------
        return _plot_ecdf_from_results(results_csv=csv_path, label_col='gene_symbol', title=f'{_region_name} factor {_factor_idx} genes', output_dir=dirs['gene_ecdf'], output_name=f'{_region_name}_factor{_factor_idx}_genes_ecdf', highlight_labels=gene_highlights, xmin=xmin, xmax=xmax, top_pos_n=top_pos_n, top_neg_n=top_neg_n, shorten_cluster_label=False)

    def plot_cluster_ecdf(region_name, factor_idx, xmin=-0.25, xmax=0.25, top_pos_n=5, top_neg_n=5):
        csv_path = dirs['cluster_results'] / f'{_region_name}_factor{_factor_idx}_clusters_full_results.csv'
        return _plot_ecdf_from_results(results_csv=csv_path, label_col='cluster', title=f'{_region_name} factor {_factor_idx} clusters', output_dir=dirs['cluster_ecdf'], output_name=f'{_region_name}_factor{_factor_idx}_clusters_ecdf', highlight_labels=[], xmin=xmin, xmax=xmax, top_pos_n=top_pos_n, top_neg_n=top_neg_n, shorten_cluster_label=True)

    return


@app.cell
def _(
    CLOSEUP_MARGIN,
    FACTOR_DIR,
    HIGHLIGHT_GENES,
    N_CLOSEUP_SLICES,
    TOP_N,
    atlas_df,
    atlas_img,
    average_hemispheres,
    choose_region_slices,
    contour_img,
    dirs,
    ensure_dir,
    gene_meta,
    gene_zarr,
    get_descendant_ids,
    get_region_mask_and_bbox,
    hemi_atlas_img,
    np,
    overlap_contour,
    pd,
    plt,
):
    import seaborn as sns
    from adjustText import adjust_text
    from brain_vis import set_transparency, overlay_images
    plt.rcParams.update({'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'legend.facecolor': 'none', 'legend.edgecolor': 'none', 'text.color': 'black', 'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})

    def _save_fig_original_style(fig, output_dir, output_name, dpi=216, also_svg=False):
        output_dir = ensure_dir(output_dir)
        out_png = output_dir / f'{output_name}.png'
        out_pdf = output_dir / f'{output_name}.pdf'
        _fig.savefig(out_png, bbox_inches='tight', dpi=dpi)
        _fig.savefig(out_pdf, bbox_inches='tight', dpi=dpi)
        if also_svg:
            out_svg = output_dir / f'{output_name}.svg'
            _fig.savefig(out_svg, bbox_inches='tight', dpi=dpi)
        print('Saved:', out_png)
        return out_png

    def plot_factor_closeup(region_name, factor_idx):
        """
        Spatial closeup plot in the same style as the original notebook:
          - viridis overlay
          - contour_img overlay through overlap_contour if available
          - transparent outside atlas using set_transparency
          - no ticks
          - invert x-axis
        """
        _region_ids = get_descendant_ids(atlas_df, _region_name, include_self=True)
        (_region_mask, bbox) = get_region_mask_and_bbox(hemi_atlas_img, _region_ids, margin=CLOSEUP_MARGIN)
        (yslice, xslice) = bbox
        _factor_array = np.load(FACTOR_DIR / f'factor{_factor_idx}.npy')
        _factor_hemi = average_hemispheres(_factor_array)
        z_show = choose_region_slices(_region_mask, n_slices=N_CLOSEUP_SLICES)
        if len(z_show) == 0:
            raise ValueError(f'No z slices found for {_region_name}')
        if contour_img is not None:
            contour_hemi = contour_img[:, :, :contour_img.shape[2] // 2]
        else:
            contour_hemi = None
        cmax = np.nanpercentile(_factor_hemi[_region_mask], 99)
        if not np.isfinite(cmax) or cmax <= 0:
            cmax = np.nanmax(_factor_hemi)
        (_, overlayed_image) = overlap_contour(_factor_hemi, contour_hemi if contour_hemi is not None else np.zeros_like(_factor_hemi), cmin=0, cmax=cmax, outputpath=None, colormap=plt.cm.viridis)
        (_fig, axs) = plt.subplots(1, len(z_show), figsize=(len(z_show) * 1.25, 1.25), sharex=True, sharey=True, squeeze=False)
        axs = axs[0]
        for (idx, z) in enumerate(z_show):
            ax = axs[idx]
            if 'atlas_img' in globals():
                atlas_hemi_mask = atlas_img[:, :, :atlas_img.shape[2] // 2] == 0
                trans_img = set_transparency(overlayed_image[z, :, :], atlas_hemi_mask[z, :, :])
                ax.imshow(trans_img[yslice, xslice])
            else:
                ax.imshow(overlayed_image[z, yslice, xslice])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_xaxis()
            ax.set_title(f'z={z}', fontsize=8)
        _fig.suptitle(f'{_region_name} factor {_factor_idx}', fontsize=10)
        _fig.tight_layout()
        _save_fig_original_style(_fig, dirs['closeup'], f'{_region_name}_factor{_factor_idx}_closeup', dpi=216)
        return _fig

    def _plot_ecdf_from_results(results_csv, label_col, title, output_dir, output_name, highlight_labels=None):
        """
        ECDF plot consistent with the original notebook:
          - sorted r-values
          - significant = magenta, nonsignificant = gray
          - small scatter points
          - selected labels annotated with adjust_text
          - no grid  # Original notebook-like contour overlay
          - sns.despine()
        """
        highlight_labels = set([] if highlight_labels is None else list(map(str, highlight_labels)))
        df = pd.read_csv(results_csv)
        d = df.sort_values('r').reset_index(drop=True).copy()
        _x = d['r'].to_numpy()
        y = np.arange(1, len(d) + 1) / len(d)
        if 'significant' in d.columns:
            d['color'] = np.where(d['significant'].to_numpy(), 'magenta', 'gray')
        else:
            d['color'] = 'gray'
        colors = d['color'].to_numpy()
        (_fig, ax) = plt.subplots(1, 1, figsize=(1.6, 2.5))
        ax.scatter(_x, y, c=colors, s=10, marker='o', edgecolor='none', alpha=0.9)
        if 'significant' in df.columns and 'abs_r' in df.columns:
            top_df = df[df['significant']].sort_values('abs_r', ascending=False).head(TOP_N)
            labels_to_annotate = set(top_df[label_col].astype(str).tolist()) | highlight_labels
        else:
            labels_to_annotate = highlight_labels
        d['annotate'] = d[label_col].astype(str).isin(labels_to_annotate)
        texts = []
        for (i, row) in d[d['annotate']].iterrows():
            texts.append(ax.text(row['r'], y[i], str(row[label_col]), fontsize=8, ha='left', va='bottom', color='black'))
        if len(texts) > 0:
            adjust_text(texts, ax=ax, arrowprops=dict(color='gray', alpha=1, lw=0.5))
        ax.axvline(0, color='gray', lw=0.7, ls=':', alpha=0.7)
        ax.set_xlabel('Pearson r')
        ax.set_ylabel('ECDF')
        ax.set_title(title, fontsize=9)
        sns.despine()
        _fig.tight_layout()
        _save_fig_original_style(_fig, output_dir, output_name, dpi=216)
        return _fig

    def plot_gene_ecdf_1(region_name, factor_idx):
        csv_path = dirs['gene_results'] / f'{_region_name}_factor{_factor_idx}_genes_full_results.csv'
        return _plot_ecdf_from_results(results_csv=csv_path, label_col='gene_symbol', title=f'{_region_name} factor {_factor_idx} genes', output_dir=dirs['gene_ecdf'], output_name=f'{_region_name}_factor{_factor_idx}_genes_ecdf', highlight_labels=HIGHLIGHT_GENES)

    def plot_cluster_ecdf_1(region_name, factor_idx):
        csv_path = dirs['cluster_results'] / f'{_region_name}_factor{_factor_idx}_clusters_full_results.csv'
        return _plot_ecdf_from_results(results_csv=csv_path, label_col='cluster', title=f'{_region_name} factor {_factor_idx} clusters', output_dir=dirs['cluster_ecdf'], output_name=f'{_region_name}_factor{_factor_idx}_clusters_ecdf', highlight_labels=[])

    def plot_gene_closeup(region_name, gene_symbol, z_show=None, yslice=None, xslice=None, CLOSEUP_MARGIN=10, crop_size=120):
        """
        Spatial closeup plot for a single gene, in the same style as plot_factor_closeup:
          - viridis overlay
          - contour_img overlay through overlap_contour if available
          - transparent outside atlas using set_transparency
          - no ticks
          - invert x-axis
        z_show / yslice / xslice default to None -> auto:
          - z_show: chosen from the region mask (choose_region_slices)
          - yslice/xslice: square crop_size x crop_size centered on the region
            center-of-mass per z-plane (get_slice).
        Pass explicit values to override either.
        Spatial map comes from the gene matrix (gene_zarr / gene_meta) instead of a factor .npy.
        """
        _region_ids = get_descendant_ids(atlas_df, _region_name, include_self=True)
        (_region_mask, _) = get_region_mask_and_bbox(atlas_img, _region_ids, margin=CLOSEUP_MARGIN)
        if z_show is None:
            z_show = choose_region_slices(_region_mask, n_slices=N_CLOSEUP_SLICES)
        else:
            z_show = list(z_show)
        if len(z_show) == 0:
            raise ValueError(f'No z slices found for {_region_name}')
        auto_crop = yslice is None or xslice is None
        half = crop_size // 2
        matches = np.flatnonzero(gene_meta['gene_symbol'].astype(str).values == str(gene_symbol))
        if len(matches) == 0:
            raise ValueError(f'Gene symbol not found: {gene_symbol}')
        gene_idx = int(matches[0])
        gene_array = np.asarray(gene_zarr[gene_idx].compute()).reshape(atlas_img.shape)
        if contour_img is not None:
            contour_hemi = contour_img[:, :, :contour_img.shape[2] // 2]
        else:
            contour_hemi = None
        cmax = 1
        if not np.isfinite(cmax) or cmax <= 0:
            cmax = np.nanmax(gene_array)
        (_, overlayed_image) = overlap_contour(gene_array, contour_img, cmin=0, cmax=cmax, outputpath=None, colormap=plt.cm.viridis)
        (_fig, axs) = plt.subplots(1, len(z_show), figsize=(len(z_show) * 1.25, 1.25), sharex=True, sharey=True, squeeze=False)
        axs = axs[0]
        for (idx, z) in enumerate(z_show):
            ax = axs[idx]
            if auto_crop:
                print(_region_ids)
                (zyslice, zxslice) = get_slice(z, _region_ids, hemi='left', window=half)
                print(zyslice)
            else:
                (zyslice, zxslice) = (yslice, xslice)
            if 'atlas_img' in globals():
                atlas_mask = atlas_img == 0
                trans_img = set_transparency(overlayed_image[z, :, :], atlas_mask[z, :, :])
                ax.imshow(trans_img[zyslice, zxslice])
            else:
                ax.imshow(overlayed_image[z, zyslice, zxslice])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_xaxis()
            ax.set_title(f'z={z}', fontsize=8)
        _fig.suptitle(f'{_region_name} {gene_symbol}', fontsize=10)
        _fig.tight_layout()
        _save_fig_original_style(_fig, dirs['closeup'], f'{_region_name}_{gene_symbol}_closeup', dpi=1026)
        return _fig

    def get_slice(z, target_site_subids, hemi='left', window=60):
        ys = np.array([])
        xs = np.array([])
        if hemi == 'left':
            hemi_slice = slice(0, atlas_img.shape[2] // 2)
        else:
            hemi_slice = slice(atlas_img.shape[2] // 2, atlas_img.shape[2])
        for ID in target_site_subids:
            print(atlas_img[z, :, hemi_slice].shape)
            (y_, x_) = np.where(atlas_img[z, :, hemi_slice] == ID)
            xs = np.concatenate([xs, x_])
            ys = np.concatenate([ys, y_])
        print(xs, ys)
        y_center = int(np.mean(ys))
        x_center = int(np.mean(xs))
        if hemi == 'left':
            yslice = slice(y_center - window, y_center + window)
            xslice = slice(x_center - window, x_center + window)
        elif hemi == 'right':
            yslice = slice(y_center - window, y_center + window)
            xslice = slice(x_center - window + atlas_img.shape[2] // 2, x_center + window + atlas_img.shape[2] // 2)
        elif hemi == 'center':
            yslice = slice(y_center - window, y_center + window)
            xslice = slice(atlas_img.shape[2] // 2 - window, window + atlas_img.shape[2] // 2)
        return (yslice, xslice)

    return (
        adjust_text,
        plot_cluster_ecdf_1,
        plot_factor_closeup,
        plot_gene_closeup,
        plot_gene_ecdf_1,
        sns,
    )


@app.cell
def _():
    OPIOID_GENES = ['Oprm1','Oprk1','Oprd1']
    return (OPIOID_GENES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb genes
    """)
    return


@app.cell
def _(OPIOID_GENES, plot_gene_closeup, plt):
    for _gene in OPIOID_GENES:
        _fig = plot_gene_closeup('Acb', _gene, z_show=(81, 86, 91, 96, 101, 106), yslice=slice(240, 240 + 120), xslice=slice(325 - 120, 325))
        plt.show()
    return


@app.cell
def _(OPIOID_GENES, plot_gene_closeup, plt):
    for _gene in OPIOID_GENES:
        _fig = plot_gene_closeup('Ce', _gene, z_show=(131, 136, 141, 146, 151, 156))
        plt.show()
    return


@app.cell
def _(gene_meta):
    gene_meta[gene_meta.gene_symbol == 'Gpx3']
    return


@app.cell
def _(plot_gene_closeup):
    _fig = plot_gene_closeup('Ce', 'Gpx3', z_show=(131, 136, 141, 146, 151, 156))
    return


@app.cell
def _(OPIOID_GENES, plot_gene_closeup, plt):
    for _gene in OPIOID_GENES:
        _fig = plot_gene_closeup('VTA', _gene, z_show=(175, 180, 185, 190))
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 0 close-up
    """)
    return


@app.cell
def _():
    from brain_vis import overlap_contour

    return (overlap_contour,)


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('Acb', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 0 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('Acb', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 0 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('Acb', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 1 close-up
    """)
    return


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('Acb', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 1 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('Acb', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 1 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('Acb', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 2 close-up
    """)
    return


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('Acb', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 2 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('Acb', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Acb factor 2 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('Acb', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 0 close-up
    """)
    return


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('Ce', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 0 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('Ce', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 0 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('Ce', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 1 close-up
    """)
    return


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('Ce', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 1 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('Ce', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 1 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('Ce', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 2 close-up
    """)
    return


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('Ce', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 2 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('Ce', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: Ce factor 2 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('Ce', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 0 close-up
    """)
    return


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('VTA', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 0 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('VTA', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 0 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('VTA', 0)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 1 close-up
    """)
    return


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('VTA', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 1 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('VTA', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 1 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('VTA', 1)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 2 close-up
    """)
    return


@app.cell
def _(plot_factor_closeup, plt):
    _fig = plot_factor_closeup('VTA', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 2 gene ECDF
    """)
    return


@app.cell
def _(plot_gene_ecdf_1, plt):
    _fig = plot_gene_ecdf_1('VTA', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot: VTA factor 2 cluster ECDF
    """)
    return


@app.cell
def _(plot_cluster_ecdf_1, plt):
    _fig = plot_cluster_ecdf_1('VTA', 2)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bar plots
    """)
    return


@app.cell
def _(OUTPUT_DIR, dirs, ensure_dir, np, pd, plt, sns):
    # ============================================================
    # Top significant correlation bar plot helper functions
    FACTOR_BAR_COLORS = {0: 'gray', 1: 'magenta', 2: 'cyan', 3: 'orange', 16: 'purple'}
    # Bar graph:
    #   - Uses saved CSV result tables only
    #   - Keeps only significant rows
    #   - Optional xlim filters r before top_n selection
    #   - Sorts by Pearson r
    #   - Plots top_n rows
    #   - Allows fixed xlim, e.g. xlim=(0, 0.4)
    #   - Bar color is based on factor_idx
    #   - Style is kept consistent with the original Figure 7 notebook

    # ------------------------------------------------------------
    # Factor color map
    def _get_factor_color(factor_idx):
    # Edit this dictionary to match the exact colors from your original notebook.
    # These are only fallbacks. If FACTOR_COLORS already exists globally, that will be used first.
        """
        Resolve plotting color from factor_idx.

        Priority:
          1. Existing global FACTOR_COLORS dict/list, if present
          2. FACTOR_BAR_COLORS fallback dict
          3. seaborn tab20 fallback
        """
        _factor_idx = int(_factor_idx)
        if 'FACTOR_COLORS' in globals():
            fc = globals()['FACTOR_COLORS']
            if isinstance(fc, dict) and _factor_idx in fc:
                return fc[_factor_idx]
            if isinstance(fc, (list, tuple)) and _factor_idx < len(fc):
                return fc[_factor_idx]
        if _factor_idx in FACTOR_BAR_COLORS:
            return FACTOR_BAR_COLORS[_factor_idx]
        return sns.color_palette('tab20', 20)[_factor_idx % 20]

    def _get_top_significant_sorted_by_r_from_results(df, label_col, top_n=20, ascending=True, xlim=None, filter_by_xlim=True):
        """
        Select top_n significant features sorted by Pearson r.

        ascending=True:
            most negative r values are selected first.

        ascending=False:
            most positive r values are selected first.

        xlim:
            Optional tuple, e.g. (0, 0.4) or (-0.4, 0).

        filter_by_xlim:
            If True, rows outside xlim are removed before selecting top_n.
            This is useful when plotting only positive or only negative bars.
        """
        d = df.copy()
        if 'significant' in d.columns:
            d = d[d['significant']].copy()
        if xlim is not None and filter_by_xlim:
            (xmin, xmax) = xlim
            d = d[(d['r'] >= xmin) & (d['r'] <= xmax)].copy()
        if len(d) == 0:
            return d
        d = d.sort_values('r', ascending=ascending).head(top_n).sort_values('r', ascending=True).reset_index(drop=True)
        return d

    def plot_top_correlation_bar_from_results(results_csv, label_col, title, output_dir, output_name, factor_idx, top_n=20, ascending=True, shorten_cluster_label=False, xlim=None, filter_by_xlim=True, fig_w=2.4, bar_height_per_item=0.18, ytick_fontsize=8, title_fontsize=10, verbose=True):
        """
        Bar plot of top significant correlations.

        Parameters
        ----------
        factor_idx:
            Used to determine bar color.

        xlim:
            Optional tuple for x-axis limits.
            Example:
                xlim=(0, 0.4)
                xlim=(-0.4, 0)
                xlim=(-0.4, 0.4)

        filter_by_xlim:
            If True, xlim is used both for selecting rows and for display.
            If False, xlim is used only for display.

        fig_w:
            Figure width in inches.
        """
        df = pd.read_csv(results_csv)
        top_df = _get_top_significant_sorted_by_r_from_results(df=df, label_col=label_col, top_n=top_n, ascending=ascending, xlim=xlim, filter_by_xlim=filter_by_xlim)
        if len(top_df) == 0:
            print(f'No significant rows to plot within xlim={xlim}: {results_csv}')
            return None
        if shorten_cluster_label:
            top_df['plot_label'] = top_df[label_col].astype(str).str.split(' ').str[0]
        else:
            top_df['plot_label'] = top_df[label_col].astype(str)
        bar_color = _get_factor_color(_factor_idx)
        if verbose:
            print('Plot:', title)
            print('CSV:', results_csv)
            print('factor_idx:', _factor_idx)
            print('bar_color:', bar_color)
            print('selected n:', len(top_df))
            print('selected r range:', float(top_df['r'].min()), float(top_df['r'].max()))
            if xlim is not None:
                print('xlim:', xlim)
            display_cols = [label_col, 'r']
            for c in ['p', 'p_fdr', 'significant']:
                if c in top_df.columns:
                    display_cols.append(c)
            print(top_df[display_cols].to_string(index=False))
        fig_h = max(2.5, bar_height_per_item * len(top_df))
        (_fig, ax) = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        y = np.arange(len(top_df))
        ax.barh(y, top_df['r'].to_numpy(), color=bar_color, edgecolor='none', height=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels(top_df['plot_label'], fontsize=ytick_fontsize)
        ax.axvline(0, color='gray', lw=0.7, ls=':', alpha=0.7)
        ax.set_xlabel('Pearson r')
        ax.set_title(title, fontsize=title_fontsize)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))
        sns.despine()
        _fig.tight_layout()
        _save_fig_original_style(_fig, output_dir, output_name, dpi=216)
        return _fig

    def _xlim_to_name(xlim):
        if xlim is None:
            return None
        return f'xlim_{xlim[0]}_{xlim[1]}'.replace('.', 'p').replace('-', 'neg').replace(' ', '')

    def plot_gene_top_bar(region_name, factor_idx, top_n=20, ascending=True, xlim=None, filter_by_xlim=True, fig_w=2.4, ytick_fontsize=8, verbose=True):
        csv_path = dirs['gene_results'] / f'{_region_name}_factor{_factor_idx}_genes_full_results.csv'
        output_dir = ensure_dir(dirs.get('gene_bar', OUTPUT_DIR / 'gene_bar'))
        if xlim is not None:
            direction = _xlim_to_name(xlim)
        else:
            direction = 'neg' if ascending else 'pos'
        return plot_top_correlation_bar_from_results(results_csv=csv_path, label_col='gene_symbol', title=f'{_region_name} factor {_factor_idx} genes', output_dir=output_dir, output_name=f'{_region_name}_factor{_factor_idx}_genes_top{top_n}_{direction}_bar', factor_idx=_factor_idx, top_n=top_n, ascending=ascending, shorten_cluster_label=False, xlim=xlim, filter_by_xlim=filter_by_xlim, fig_w=fig_w, ytick_fontsize=ytick_fontsize, verbose=verbose)

    def plot_cluster_top_bar(region_name, factor_idx, top_n=20, ascending=True, xlim=None, filter_by_xlim=True, fig_w=2.4, ytick_fontsize=8, verbose=True):
        csv_path = dirs['cluster_results'] / f'{_region_name}_factor{_factor_idx}_clusters_full_results.csv'
        output_dir = ensure_dir(dirs.get('cluster_bar', OUTPUT_DIR / 'cluster_bar'))
        if xlim is not None:
            direction = _xlim_to_name(xlim)
        else:
            direction = 'neg' if ascending else 'pos'
        return plot_top_correlation_bar_from_results(results_csv=csv_path, label_col='cluster', title=f'{_region_name} factor {_factor_idx} clusters', output_dir=output_dir, output_name=f'{_region_name}_factor{_factor_idx}_clusters_top{top_n}_{direction}_bar', factor_idx=_factor_idx, top_n=top_n, ascending=ascending, shorten_cluster_label=True, xlim=xlim, filter_by_xlim=filter_by_xlim, fig_w=fig_w, ytick_fontsize=ytick_fontsize, verbose=verbose)

    return plot_cluster_top_bar, plot_gene_top_bar


@app.cell
def _(FACTOR_INDICES, TARGET_REGIONS, plot_cluster_top_bar, plot_gene_top_bar):
    for _region_name in TARGET_REGIONS:
        for _factor_idx in FACTOR_INDICES:
            plot_gene_top_bar(_region_name, _factor_idx, top_n=20, ascending=False, xlim=(0, 0.4))
            plot_cluster_top_bar(_region_name, _factor_idx, top_n=20, ascending=False, xlim=(0, 0.4))
    return


@app.cell
def _(cluster_details):
    cluster_details.shape
    return


if __name__ == "__main__":
    app.run()
