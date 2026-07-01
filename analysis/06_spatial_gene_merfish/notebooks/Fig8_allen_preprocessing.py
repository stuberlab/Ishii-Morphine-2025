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
    # Generate Allen MERFISH gene and cluster expression Zarrs in Unified/Kim atlas space

    This notebook starts from only local atlas files plus access to the Allen Brain Cell Atlas API/cache.

    It downloads Allen MERFISH data, generates the ANTs transform if it does not already exist, transforms Allen CCF coordinates into Unified/Kim atlas voxel space, and writes:

    - `gene_expression.zarr`
    - `cluster_expression.zarr`
    - `gene_df.csv`
    - `cluster_details.csv`
    - `merfish_cells_unified_voxels.parquet`

    ## Coordinate-space answer

    For Allen API data, use:

    ```python
    COORDINATE_SPACE = "allen_ccf_mm"
    ```

    The Allen `ccf_coordinates` table gives cell positions in Allen CCF physical coordinates in millimeters. Because the registration here uses coronal TIFF atlas images, those millimeter coordinates are converted to Allen 25 µm image-index coordinates before applying the ANTs transform. The output is Unified/Kim **voxel/index space**, matching the space of the c-Fos atlas maps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optional dependency install

    Run this once in a new environment, then restart the kernel if needed.
    """)
    return


@app.cell
def _():
    # %pip install numpy pandas scipy tifffile zarr numcodecs tqdm anndata antspyx pyarrow
    # %pip install abc_atlas_access
    return


@app.cell
def _():
    from __future__ import annotations

    from pathlib import Path
    import os
    import shutil
    import warnings

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import tifffile
    import zarr
    from tqdm.auto import tqdm

    try:
        from numcodecs import Blosc
    except Exception:
        Blosc = None

    import anndata
    import ants

    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    return (
        AbcProjectCache,
        Blosc,
        Path,
        anndata,
        ants,
        np,
        os,
        pd,
        shutil,
        sp,
        tifffile,
        tqdm,
        warnings,
        zarr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration

    Edit this cell. The key setting is `COORDINATE_SPACE = "allen_ccf_mm"`.

    For the original coronal TIFF atlas files, Allen anatomical coordinates `(x, y, z)` need to be reordered to image axes `(z, y, x)` before ANTs point transformation. This is why `ALLEN_CCF_MM_TO_ANTS_AXIS_ORDER = ("z", "y", "x")` by default.
    """)
    return


@app.cell
def _(Path, np, os):
    # -----------------------------
    # Project paths
    # -----------------------------
    PROJECT_DIR = Path(os.environ.get("OPIOID_ALLEN_PROJECT", "./allen_to_unified_expression_project")).resolve()
    DATA_ROOT = Path(os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")).expanduser()
    ATLAS_DIR = DATA_ROOT / "shared" / "atlas"
    ABC_CACHE_DIR = PROJECT_DIR / "abc_atlas_cache"
    OUTPUT_DIR = DATA_ROOT / "06_spatial_gene_merfish"   # writes gene/cluster zarr + CSVs consumed by Figure 8
    TRANSFORM_DIR = PROJECT_DIR / "transforms"

    for p in [ABC_CACHE_DIR, OUTPUT_DIR, TRANSFORM_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Required atlas files
    # -----------------------------
    # Allen CCFv3 25 µm average/template image, coronal TIFF preferred here.
    ALLEN_CCF_TEMPLATE_PATH = ATLAS_DIR / "template_25_coronal.tif"

    # Unified/Kim anatomical reference image. This is the fixed image for registration.
    UNIFIED_TEMPLATE_PATH = ATLAS_DIR / "Kim_ref_adult_v1_brain.tif"

    # Unified/Kim annotation/label image. Defines output grid and brain mask.
    UNIFIED_ANNOTATION_PATH = ATLAS_DIR / "Kim_ref_adult_FP-label_v4.0.tif"

    # If your TIFF arrays need transposition before ANTs / voxelization, set these.
    # For the atlas files used in the original notebook, None should usually be correct.
    ALLEN_TEMPLATE_TRANSPOSE = None
    UNIFIED_TEMPLATE_TRANSPOSE = None
    UNIFIED_ANNOTATION_TRANSPOSE = None

    # -----------------------------
    # Coordinate settings
    # -----------------------------
    COORDINATE_SPACE = "allen_ccf_mm"  # correct for Allen API ccf_coordinates
    ALLEN_CCF_VOXEL_SIZE_MM = 0.025     # Allen CCFv3 25 µm
    UNIFIED_VOXEL_SIZE_MM = np.array([0.020, 0.020, 0.050], dtype=float)  # for metadata only

    # For coronal TIFF registration: API anatomical x,y,z -> image/ANTs axes.
    # Original notebook used: coords.columns = ['z', 'y', 'x']; coords = coords * (1000/25)
    ALLEN_CCF_MM_TO_ANTS_AXIS_ORDER = ("z", "y", "x")

    # The transformed output is Unified/Kim image voxel/index space.
    OUTPUT_COORDINATE_SPACE = "unified_voxel"

    # -----------------------------
    # Allen API data settings
    # -----------------------------
    ABC_DATASET = "Zhuang-ABCA-1"
    ABC_CCF_DATASET = f"{ABC_DATASET}-CCF"

    # -----------------------------
    # ANTs registration settings
    # -----------------------------
    # Register fixed=Unified/Kim, moving=Allen CCF. Forward transforms map Allen CCF -> Unified/Kim.
    ANTSPY_TRANSFORM_TYPE = "antsRegistrationSyNs"  # accurate, slow; "SyN" or "ElasticSyN" are alternatives
    FORCE_REGENERATE_TRANSFORMS = False

    # -----------------------------
    # Voxelization settings
    # -----------------------------
    # Original notebook used size=(2,2,4), which is approximately 40 x 40 x 100 µm in Unified voxels.
    KERNEL_SIZE_VOX = (2, 2, 4)
    KERNEL_METHOD = "spherical"

    # Allen expression is log2 transformed. The manuscript threshold was expression > 1.
    # If the log transform is log2(x), this corresponds to x > 2 on the original scale.
    LOG2_EXPRESSION_THRESHOLD = 1.0
    USE_LOG2_THRESHOLD = True

    ROW_CHUNKS = 1
    VOXEL_CHUNKS = 1_000_000
    GENE_DTYPE = "float32"
    CLUSTER_DTYPE = "uint16"  # use float32 if local cluster counts can exceed 65535
    OVERWRITE_OUTPUTS = True

    # -----------------------------
    # Output paths
    # -----------------------------
    GENE_ZARR_PATH = OUTPUT_DIR / "gene_expression.zarr"
    CLUSTER_ZARR_PATH = OUTPUT_DIR / "cluster_expression.zarr"
    GENE_DF_CSV = OUTPUT_DIR / "gene_df.csv"
    CLUSTER_DETAILS_CSV = OUTPUT_DIR / "cluster_details.csv"
    CELL_COORDINATE_OUTPUT = OUTPUT_DIR / "merfish_cells_unified_voxels.parquet"
    return (
        ABC_CACHE_DIR,
        ABC_CCF_DATASET,
        ABC_DATASET,
        ALLEN_CCF_MM_TO_ANTS_AXIS_ORDER,
        ALLEN_CCF_TEMPLATE_PATH,
        ALLEN_CCF_VOXEL_SIZE_MM,
        ALLEN_TEMPLATE_TRANSPOSE,
        ANTSPY_TRANSFORM_TYPE,
        CELL_COORDINATE_OUTPUT,
        CLUSTER_DETAILS_CSV,
        CLUSTER_DTYPE,
        CLUSTER_ZARR_PATH,
        COORDINATE_SPACE,
        FORCE_REGENERATE_TRANSFORMS,
        GENE_DF_CSV,
        GENE_DTYPE,
        GENE_ZARR_PATH,
        KERNEL_METHOD,
        KERNEL_SIZE_VOX,
        LOG2_EXPRESSION_THRESHOLD,
        OUTPUT_COORDINATE_SPACE,
        OUTPUT_DIR,
        OVERWRITE_OUTPUTS,
        ROW_CHUNKS,
        TRANSFORM_DIR,
        UNIFIED_ANNOTATION_PATH,
        UNIFIED_ANNOTATION_TRANSPOSE,
        UNIFIED_TEMPLATE_PATH,
        UNIFIED_TEMPLATE_TRANSPOSE,
        UNIFIED_VOXEL_SIZE_MM,
        USE_LOG2_THRESHOLD,
        VOXEL_CHUNKS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.cell
def _(
    Blosc,
    COORDINATE_SPACE,
    KERNEL_SIZE_VOX,
    OUTPUT_COORDINATE_SPACE,
    OVERWRITE_OUTPUTS,
    Path,
    UNIFIED_VOXEL_SIZE_MM,
    ants,
    np,
    shutil,
    sp,
    tifffile,
    zarr,
):
    def read_tiff_xyz(path: Path, transpose=None):
        arr = tifffile.imread(str(path))
        if transpose is not None:
            arr = np.transpose(arr, transpose)
        return np.asarray(arr)

    def read_ants_image(path: Path, transpose=None):
        if transpose is None:
            return ants.image_read(str(path))
        arr = read_tiff_xyz(path, transpose=transpose).astype('float32')
        return ants.from_numpy(arr)

    def create_zarr(path: Path, shape, chunks, dtype):
        path = Path(path)
        if path.exists() and OVERWRITE_OUTPUTS:
            shutil.rmtree(path)
        elif path.exists():
            raise FileExistsError(f'{path} exists. Set OVERWRITE_OUTPUTS=True to overwrite.')
        kwargs = dict(store=str(path), mode='w', shape=shape, chunks=chunks, dtype=dtype)
        if Blosc is not None:
            kwargs['compressor'] = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        try:
            return zarr.open(**kwargs)
        except TypeError:
            kwargs.pop('compressor', None)
            return zarr.open(**kwargs)

    def attach_common_zarr_attrs(arr, row_kind, atlas_shape):  # zarr v3 fallback when compressor API differs.
        arr.attrs['row_kind'] = row_kind
        arr.attrs['axis_0'] = row_kind
        arr.attrs['axis_1'] = 'flattened_unified_voxel'
        arr.attrs['output_coordinate_space'] = OUTPUT_COORDINATE_SPACE
        arr.attrs['input_coordinate_space'] = COORDINATE_SPACE
        arr.attrs['atlas_shape'] = tuple((int(x) for x in atlas_shape))
        arr.attrs['flattening'] = 'np.ravel_multi_index((i, j, k), atlas_shape), C-order'
        arr.attrs['kernel_size_vox'] = tuple((int(x) for x in KERNEL_SIZE_VOX))
        arr.attrs['unified_voxel_size_mm_xyz_metadata'] = tuple((float(x) for x in UNIFIED_VOXEL_SIZE_MM))

    def make_kernel_offsets(size_vox=(2, 2, 4), method='spherical'):
        (rx, ry, rz) = [s / 2 for s in size_vox]
        xs = np.arange(-rx, rx + 1)
        ys = np.arange(-ry, ry + 1)
        zs = np.arange(-rz, rz + 1)
        offsets = []
        for x in xs:
            for y in ys:  # Treat size as full diameter/extent in voxels, matching the original notebook.
                for z in zs:
                    if method.lower() == 'pixel':
                        keep = x == 0 and y == 0 and (z == 0)
                    elif method.lower() == 'rectangular':
                        keep = True
                    else:
                        keep = x * x / (rx * rx) + y * y / (ry * ry) + z * z / (rz * rz) < 1
                    if keep:
                        offsets.append((int(round(x)), int(round(y)), int(round(z))))
        offsets = np.unique(np.asarray(offsets, dtype=np.int16), axis=0)
        return offsets

    def voxelize_points_flat(voxel_ijk, values, atlas_shape, brain_mask_flat, offsets, dtype='float32'):
        voxel_ijk = np.asarray(voxel_ijk, dtype=np.int64)
        n_vox = int(np.prod(atlas_shape))
        out = np.zeros(n_vox, dtype=dtype)
        if voxel_ijk.shape[0] == 0:
            return out
        if values is None:
            values = np.ones(voxel_ijk.shape[0], dtype=np.float32)
        else:
            values = np.asarray(values, dtype=np.float32).reshape(-1)
        for off in offsets:
            pts = voxel_ijk + off.astype(np.int64)
            valid = (pts[:, 0] >= 0) & (pts[:, 0] < atlas_shape[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < atlas_shape[1]) & (pts[:, 2] >= 0) & (pts[:, 2] < atlas_shape[2])
            if not np.any(valid):
                continue
            flat = np.ravel_multi_index((pts[valid, 0], pts[valid, 1], pts[valid, 2]), dims=atlas_shape)
            keep = brain_mask_flat[flat]
            if not np.any(keep):
                continue
            src_idx = np.where(valid)[0][keep]
            np.add.at(out, flat[keep], values[src_idx])
        return out

    def get_expression_vector(adata, obs_indices, gene_idx, gene_label=None, verbose=False):
        """
        Read one gene expression vector from AnnData / backed h5py safely.

        h5py-backed AnnData requires fancy-index arrays to be sorted in increasing
        order. We sort obs_indices for reading, then restore the original order so
        expression values still align to valid_vox.
        """
        import time
        t0 = time.perf_counter()
        label = f'gene_idx={_gene_idx}' if gene_label is None else f'{gene_label} / gene_idx={_gene_idx}'
        obs_indices = np.asarray(obs_indices, dtype=np.int64)
        if obs_indices.ndim != 1:
            raise ValueError('obs_indices must be a 1D array.')
        if len(obs_indices) == 0:
            if verbose:
                print(f'[get_expression_vector] {label}: no obs indices', flush=True)
            return np.array([], dtype=np.float32)
        if verbose:
            print(f'[get_expression_vector] {label}: n_obs={len(obs_indices):,}; sorting obs indices...', flush=True)
        order = np.argsort(obs_indices, kind='mergesort')
        sorted_obs_indices = obs_indices[order]
        t_sort = time.perf_counter()
        if verbose:
            print(f'[get_expression_vector] {label}: sorted in {t_sort - t0:.2f}s; reading adata.X...', flush=True)
        x_sorted = adata.X[sorted_obs_indices, _gene_idx]
        t_read = time.perf_counter()
        if verbose:
            print(f'[get_expression_vector] {label}: adata.X read finished in {t_read - t_sort:.2f}s; converting...', flush=True)
        if sp.issparse(x_sorted):
            x_sorted = x_sorted.toarray()
        x_sorted = np.asarray(x_sorted).reshape(-1).astype(np.float32)
        t_convert = time.perf_counter()
        if verbose:
            print(f'[get_expression_vector] {label}: converted in {t_convert - t_read:.2f}s; restoring order...', flush=True)
        x = np.empty_like(x_sorted)
        x[order] = x_sorted
        t_done = time.perf_counter()
        if verbose:
            print(f'[get_expression_vector] {label}: done in {t_done - t0:.2f}s', flush=True)
        return x

    return (
        attach_common_zarr_attrs,
        create_zarr,
        make_kernel_offsets,
        read_ants_image,
        read_tiff_xyz,
        voxelize_points_flat,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Unified atlas annotation

    The annotation volume defines the output grid. All Zarr rows are flattened versions of this exact array shape.
    """)
    return


@app.cell
def _(
    UNIFIED_ANNOTATION_PATH,
    UNIFIED_ANNOTATION_TRANSPOSE,
    np,
    read_tiff_xyz,
):
    unified_annotation = read_tiff_xyz(UNIFIED_ANNOTATION_PATH, transpose=UNIFIED_ANNOTATION_TRANSPOSE)
    unified_annotation = np.asarray(unified_annotation)
    atlas_shape = tuple(int(x) for x in unified_annotation.shape)
    n_voxels = int(np.prod(atlas_shape))
    brain_mask_flat = unified_annotation.reshape(-1) > 0

    print("Unified annotation shape:", atlas_shape)
    print("Total voxels:", n_voxels)
    print("Brain voxels:", int(brain_mask_flat.sum()))
    return atlas_shape, brain_mask_flat, n_voxels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download/collect Allen MERFISH metadata through the Allen API

    This reproduces the original notebook logic: `AbcProjectCache.from_s3_cache`, then `get_metadata_dataframe` for cell metadata, gene metadata, cluster annotations, and CCF coordinates.
    """)
    return


@app.cell
def _(
    ABC_CACHE_DIR,
    ABC_CCF_DATASET,
    ABC_DATASET,
    AbcProjectCache,
    CLUSTER_DETAILS_CSV,
    GENE_DF_CSV,
):
    abc_cache = AbcProjectCache.from_s3_cache(ABC_CACHE_DIR)
    print("ABC manifest:", abc_cache.current_manifest)

    # Cell metadata.
    cell = abc_cache.get_metadata_dataframe(
        directory=ABC_DATASET,
        file_name="cell_metadata",
        dtype={"cell_label": str},
    )
    cell = cell.set_index("cell_label", drop=False)
    print("Cells in metadata:", len(cell))

    # Gene panel.
    gene = abc_cache.get_metadata_dataframe(directory=ABC_DATASET, file_name="gene")
    gene = gene.set_index("gene_identifier", drop=False)
    gene.to_csv(GENE_DF_CSV, index=False)
    print("Genes:", len(gene), "saved to", GENE_DF_CSV)

    # Cluster annotation.
    cluster_details = abc_cache.get_metadata_dataframe(
        directory="WMB-taxonomy",
        file_name="cluster_to_cluster_annotation_membership_pivoted",
        keep_default_na=False,
    )
    cluster_details = cluster_details.set_index("cluster_alias", drop=False)
    cluster_details.to_csv(CLUSTER_DETAILS_CSV, index=False)
    print("Clusters:", len(cluster_details), "saved to", CLUSTER_DETAILS_CSV)

    # Add cluster annotation to cell table.
    cell_extended = cell.join(cluster_details, on="cluster_alias", rsuffix="_cluster")

    # CCF coordinates from Allen CCF dataset.
    ccf_coordinates = abc_cache.get_metadata_dataframe(
        directory=ABC_CCF_DATASET,
        file_name="ccf_coordinates",
        dtype={"cell_label": str},
    )
    ccf_coordinates = ccf_coordinates.set_index("cell_label", drop=False)
    ccf_coordinates = ccf_coordinates.rename(columns={"x": "x_ccf_mm", "y": "y_ccf_mm", "z": "z_ccf_mm"})

    cell_extended = cell_extended.join(ccf_coordinates[["x_ccf_mm", "y_ccf_mm", "z_ccf_mm", "parcellation_index"]], how="inner")
    print("Cells with CCF coordinates:", len(cell_extended))
    return abc_cache, cell_extended, cluster_details, gene


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate or load ANTs transform: Allen CCF → Unified/Kim

    This notebook registers:

    ```python
    fixed  = Unified/Kim template
    moving = Allen CCF 25 µm template
    ```

    Therefore `fwdtransforms` map **Allen CCF image coordinates → Unified/Kim image coordinates**. This avoids the direction ambiguity in the old notebook, where the registration was named `Kim_to_allen25` and Allen→Kim required inverse transforms.
    """)
    return


@app.cell
def _(
    ALLEN_CCF_TEMPLATE_PATH,
    ALLEN_TEMPLATE_TRANSPOSE,
    ANTSPY_TRANSFORM_TYPE,
    FORCE_REGENERATE_TRANSFORMS,
    OUTPUT_DIR,
    Path,
    TRANSFORM_DIR,
    UNIFIED_TEMPLATE_PATH,
    UNIFIED_TEMPLATE_TRANSPOSE,
    ants,
    read_ants_image,
):
    def existing_forward_transforms(transform_dir: Path):
        transform_dir = Path(transform_dir)
        warp = sorted(transform_dir.glob("allen_ccf_to_unified_*1Warp.nii.gz"))
        affine = sorted(transform_dir.glob("allen_ccf_to_unified_*0GenericAffine.mat"))
        if warp and affine:
            return [str(warp[0]), str(affine[0])]
        return []


    def generate_or_load_transforms():
        if not FORCE_REGENERATE_TRANSFORMS:
            existing = existing_forward_transforms(TRANSFORM_DIR)
            if existing:
                print("Using existing Allen CCF -> Unified transforms:")
                for t in existing:
                    print(" ", t)
                return existing

        print("Generating Allen CCF -> Unified transforms with ANTs...")
        fixed = read_ants_image(UNIFIED_TEMPLATE_PATH, transpose=UNIFIED_TEMPLATE_TRANSPOSE)
        moving = read_ants_image(ALLEN_CCF_TEMPLATE_PATH, transpose=ALLEN_TEMPLATE_TRANSPOSE)

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=ANTSPY_TRANSFORM_TYPE,
            outprefix=str(TRANSFORM_DIR / "allen_ccf_to_unified_"),
        )

        # Save visual check: Allen template warped into Unified template space.
        warped_allen = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg["fwdtransforms"])
        warped_allen.to_file(str(OUTPUT_DIR / "allen_template_warped_to_unified.tif"))

        print("Generated forward transforms:")
        for t in reg["fwdtransforms"]:
            print(" ", t)
        return reg["fwdtransforms"]


    TRANSFORM_LIST_ALLEN_TO_UNIFIED = generate_or_load_transforms()
    return (TRANSFORM_LIST_ALLEN_TO_UNIFIED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convert Allen API CCF coordinates into Unified/Kim voxel coordinates

    For coronal TIFFs, the Allen API coordinates are first converted:

    ```text
    Allen ccf_coordinates x,y,z in mm
    → divide by 0.025 mm
    → reorder to image axes z,y,x
    → ANTs transform Allen image coordinates to Unified/Kim image coordinates
    → round to Unified/Kim voxel indices
    ```
    """)
    return


@app.cell
def _(
    ALLEN_CCF_MM_TO_ANTS_AXIS_ORDER,
    ALLEN_CCF_VOXEL_SIZE_MM,
    CELL_COORDINATE_OUTPUT,
    COORDINATE_SPACE,
    TRANSFORM_LIST_ALLEN_TO_UNIFIED,
    ants,
    atlas_shape,
    brain_mask_flat,
    cell_extended,
    np,
    pd,
):
    def allen_ccf_mm_to_ants_points(df):
        if COORDINATE_SPACE != 'allen_ccf_mm':
            raise ValueError("This notebook is configured for Allen API ccf_coordinates: COORDINATE_SPACE='allen_ccf_mm'.")
        mm = df[['x_ccf_mm', 'y_ccf_mm', 'z_ccf_mm']].copy()
        mm.columns = ['x', 'y', 'z']
        vox = mm / ALLEN_CCF_VOXEL_SIZE_MM
        ordered = pd.DataFrame({'x': vox[ALLEN_CCF_MM_TO_ANTS_AXIS_ORDER[0]].to_numpy(), 'y': vox[ALLEN_CCF_MM_TO_ANTS_AXIS_ORDER[1]].to_numpy(), 'z': vox[ALLEN_CCF_MM_TO_ANTS_AXIS_ORDER[2]].to_numpy()}, index=df.index)
        return ordered  # Convert anatomical mm to Allen 25 µm voxel/image-index units.
    allen_points = allen_ccf_mm_to_ants_points(cell_extended)
    transformed = ants.apply_transforms_to_points(dim=3, points=allen_points.reset_index(drop=True), transformlist=TRANSFORM_LIST_ALLEN_TO_UNIFIED, whichtoinvert=[False] * len(TRANSFORM_LIST_ALLEN_TO_UNIFIED))
    unified_vox = np.rint(transformed[['x', 'y', 'z']].to_numpy(dtype=float)).astype(np.int64)  # Reorder anatomical axes to image axes used by the coronal TIFF registration.
    inside = (unified_vox[:, 0] >= 0) & (unified_vox[:, 0] < atlas_shape[0]) & (unified_vox[:, 1] >= 0) & (unified_vox[:, 1] < atlas_shape[1]) & (unified_vox[:, 2] >= 0) & (unified_vox[:, 2] < atlas_shape[2])
    flat = np.full(len(unified_vox), -1, dtype=np.int64)
    flat[inside] = np.ravel_multi_index((unified_vox[inside, 0], unified_vox[inside, 1], unified_vox[inside, 2]), dims=atlas_shape)
    valid = inside.copy()
    valid[inside] = valid[inside] & brain_mask_flat[flat[inside]]
    valid_cell_df = cell_extended.loc[valid].copy()
    valid_cell_df['unified_i'] = unified_vox[valid, 0]
    valid_cell_df['unified_j'] = unified_vox[valid, 1]
    valid_cell_df['unified_k'] = unified_vox[valid, 2]
    valid_cell_df['unified_flat_index'] = flat[valid]
    valid_cell_df.to_parquet(CELL_COORDINATE_OUTPUT, index=False)
    valid_voxels = unified_vox[valid]
    valid_original_positions = np.where(valid)[0]
    print('Input Allen cells:', len(cell_extended))
    print('Cells inside Unified atlas mask:', len(valid_cell_df))
    print('Saved transformed cell table:', CELL_COORDINATE_OUTPUT)
    return valid_original_positions, valid_voxels


@app.cell
def _(CELL_COORDINATE_OUTPUT, pd):
    valid_cell_df_1 = pd.read_parquet(CELL_COORDINATE_OUTPUT)
    print('Loaded valid_cell_df:', valid_cell_df_1.shape)
    print(valid_cell_df_1.columns)
    return (valid_cell_df_1,)


@app.cell
def _(adata, np, pd, valid_cell_df_1):
    print('Expression matrix shape:', adata.shape)
    _valid_labels = pd.Index(valid_cell_df_1['cell_label'].astype(str))
    _obs_names = pd.Index(adata.obs_names.astype(str))
    if _obs_names.is_unique and _valid_labels.isin(_obs_names).all():
        print('Using cell_label to align valid cells to adata.X rows.')
        _obs_pos = pd.Series(np.arange(len(_obs_names)), index=_obs_names)
        valid_expr_positions = _obs_pos.loc[_valid_labels].to_numpy(np.int64)
    else:
        raise ValueError("Could not align valid_cell_df['cell_label'] to adata.obs_names. Rerun the transform cell once and save original_position.")
    valid_vox = valid_cell_df_1['unified_flat_index'].to_numpy(np.int64)
    print('valid_expr_positions:', valid_expr_positions.shape)
    print('valid_vox:', valid_vox.shape)
    print('First positions:', valid_expr_positions[:5])
    print('First voxels:', valid_vox[:5])
    return valid_expr_positions, valid_vox


@app.cell
def _(adata, n_voxels, valid_expr_positions, valid_vox):
    assert len(valid_expr_positions) == len(valid_vox)
    assert valid_expr_positions.min() >= 0
    assert valid_expr_positions.max() < adata.shape[0]
    assert valid_vox.min() >= 0
    assert valid_vox.max() < n_voxels

    print("Alignment looks OK.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate `cluster_expression.zarr`

    Rows are Allen cluster aliases from the taxonomy table. Columns are flattened Unified/Kim atlas voxels.
    """)
    return


@app.cell
def _(CLUSTER_ZARR_PATH):
    CLUSTER_ZARR_PATH
    return


@app.cell
def _(
    ABC_DATASET,
    CLUSTER_DTYPE,
    CLUSTER_ZARR_PATH,
    KERNEL_METHOD,
    KERNEL_SIZE_VOX,
    ROW_CHUNKS,
    TRANSFORM_LIST_ALLEN_TO_UNIFIED,
    VOXEL_CHUNKS,
    atlas_shape,
    attach_common_zarr_attrs,
    brain_mask_flat,
    cluster_details,
    create_zarr,
    make_kernel_offsets,
    n_voxels,
    tqdm,
    valid_cell_df_1,
    valid_voxels,
    voxelize_points_flat,
):
    kernel_offsets = make_kernel_offsets(KERNEL_SIZE_VOX, method=KERNEL_METHOD)
    print('Kernel offsets:', kernel_offsets.shape)
    cluster_arr = create_zarr(CLUSTER_ZARR_PATH, shape=(len(cluster_details), n_voxels), chunks=(ROW_CHUNKS, min(VOXEL_CHUNKS, n_voxels)), dtype=CLUSTER_DTYPE)
    attach_common_zarr_attrs(cluster_arr, row_kind='cluster', atlas_shape=atlas_shape)
    cluster_arr.attrs['source_dataset'] = ABC_DATASET
    cluster_arr.attrs['transform_list_allen_to_unified'] = [str(t) for t in TRANSFORM_LIST_ALLEN_TO_UNIFIED]
    cell_cluster_alias = valid_cell_df_1['cluster_alias'].astype(str).to_numpy()
    cluster_aliases = cluster_details.index.astype(str).to_numpy()
    for (row_idx, cluster_alias) in tqdm(list(enumerate(cluster_aliases)), desc='Voxelizing clusters'):
        _use = cell_cluster_alias == cluster_alias
        vol = voxelize_points_flat(voxel_ijk=valid_voxels[_use], values=None, atlas_shape=atlas_shape, brain_mask_flat=brain_mask_flat, offsets=kernel_offsets, dtype=CLUSTER_DTYPE)
        cluster_arr[row_idx, :] = vol
    print('Saved:', CLUSTER_ZARR_PATH)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Open Allen log2 expression matrix and align cells

    The original notebook used:

    ```python
    abc_cache.get_data_path(directory="Zhuang-ABCA-1", file_name="Zhuang-ABCA-1/log2")
    ```

    This block keeps the expression matrix on disk (`backed='r'`) and processes one gene at a time.
    """)
    return


@app.cell
def _(adata, n_voxels, np, pd, valid_cell_df_1):
    print('Expression matrix shape:', adata.shape)
    candidate_label_cols = ['cell_label', 'abc_sample_id', 'feature_matrix_label']
    _obs_names = pd.Index(adata.obs_names.astype(str))
    print('adata.obs_names example:')
    print(_obs_names[:5].tolist())
    print('\nvalid_cell_df candidate examples:')
    for _col in candidate_label_cols:
        if _col in valid_cell_df_1.columns:
            vals = pd.Index(valid_cell_df_1[_col].astype(str))
            _n_match = vals.isin(_obs_names).sum()
            print(f'{_col}: {_n_match:,}/{len(vals):,} match')
            print('  example:', vals[:5].tolist())
    valid_expr_positions_1 = None
    matched_col = None
    for _col in candidate_label_cols:
        if _col not in valid_cell_df_1.columns:
            continue
        labels = pd.Index(valid_cell_df_1[_col].astype(str))
        if _obs_names.is_unique and labels.isin(_obs_names).all():
            _obs_pos = pd.Series(np.arange(len(_obs_names)), index=_obs_names)
            valid_expr_positions_1 = _obs_pos.loc[labels].to_numpy(np.int64)
            matched_col = _col
            break
    if valid_expr_positions_1 is None:
        raise RuntimeError('Could not align valid transformed cells to adata.X rows using cell_label, abc_sample_id, or feature_matrix_label. You need to rerun the transform cell once and save original_position.')
    valid_vox_1 = valid_cell_df_1['unified_flat_index'].to_numpy(np.int64)
    print(f"\nAligned using valid_cell_df['{matched_col}']")
    print('valid_expr_positions:', valid_expr_positions_1.shape)
    print('valid_vox:', valid_vox_1.shape)
    print('First expression row positions:', valid_expr_positions_1[:5])
    print('First voxel indices:', valid_vox_1[:5])
    assert len(valid_expr_positions_1) == len(valid_vox_1)
    assert valid_expr_positions_1.min() >= 0
    assert valid_expr_positions_1.max() < adata.shape[0]
    assert valid_vox_1.min() >= 0
    assert valid_vox_1.max() < n_voxels
    print('Alignment looks OK.')
    return (valid_vox_1,)


@app.cell
def _(
    ABC_DATASET,
    GENE_DF_CSV,
    abc_cache,
    anndata,
    cell_extended,
    gene,
    pd,
    valid_cell_df_1,
    valid_original_positions,
    warnings,
):
    expr_path = abc_cache.get_data_path(directory=ABC_DATASET, file_name=f'{ABC_DATASET}/log2')
    print('Expression h5ad:', expr_path)
    adata = anndata.read_h5ad(expr_path, backed='r')
    print('Expression matrix shape:', adata.shape)
    _valid_labels = valid_cell_df_1.index.astype(str)
    _obs_names = pd.Index(adata.obs_names.astype(str))
    if _obs_names.is_unique and _valid_labels.isin(_obs_names).all():
        print('Aligning expression rows by adata.obs_names / cell_label.')
        valid_expr_positions_2 = _obs_names.get_indexer(_valid_labels)
    elif adata.n_obs == len(cell_extended):
        warnings.warn('Could not align by obs_names. Falling back to original notebook assumption: expression rows match cell metadata order.')
        valid_expr_positions_2 = valid_original_positions
    else:
        raise RuntimeError('Could not align expression rows to valid transformed cells. Inspect adata.obs_names and cell labels.')
    adata_genes = pd.Index(adata.var_names.astype(str), name='gene_identifier')
    if adata_genes.isin(gene.index.astype(str)).all():
        gene_df = gene.loc[adata_genes].copy()
    else:
        gene_df = pd.DataFrame({'gene_identifier': adata_genes})
    gene_df.to_csv(GENE_DF_CSV, index=False)
    print('Genes in expression matrix:', len(gene_df))
    print('Saved:', GENE_DF_CSV)
    return adata, valid_expr_positions_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate `gene_expression.zarr`

    Rows are genes; columns are flattened Unified/Kim atlas voxels.

    By default, cells contribute to a gene map only when:

    ```python
    log2_expression > 1
    ```

    This follows the manuscript/current interpretation of Allen log2 expression thresholding.
    """)
    return


@app.cell
def _(adata):
    # Recreate gene_df from adata.var after kernel restart
    gene_df_1 = adata.var.copy()
    gene_df_1.index = gene_df_1.index.astype(str)
    if 'gene_symbol' not in gene_df_1.columns:
        if 'gene_name' in gene_df_1.columns:
    # Make sure there is a gene_symbol column
            gene_df_1['gene_symbol'] = gene_df_1['gene_name'].astype(str)
        elif 'symbol' in gene_df_1.columns:
            gene_df_1['gene_symbol'] = gene_df_1['symbol'].astype(str)
        else:
            gene_df_1['gene_symbol'] = gene_df_1.index.astype(str)
    print('gene_df:', gene_df_1.shape)
    print(gene_df_1.head())
    return (gene_df_1,)


@app.cell
def _(
    ABC_DATASET,
    GENE_DTYPE,
    GENE_ZARR_PATH,
    LOG2_EXPRESSION_THRESHOLD,
    OVERWRITE_OUTPUTS,
    Path,
    ROW_CHUNKS,
    TRANSFORM_LIST_ALLEN_TO_UNIFIED,
    USE_LOG2_THRESHOLD,
    VOXEL_CHUNKS,
    adata,
    atlas_shape,
    attach_common_zarr_attrs,
    create_zarr,
    gene_df_1,
    n_voxels,
    np,
    shutil,
    sp,
    tqdm,
    valid_expr_positions_2,
    valid_vox_1,
):
    import time
    PROGRESS_EVERY_N_GENES = 1
    DEBUG_FIRST_N_GENES = 3
    _needed = ['adata', 'gene_df', 'valid_expr_positions', 'valid_vox', 'n_voxels', 'atlas_shape', 'create_zarr', 'attach_common_zarr_attrs']
    _missing = [x for x in _needed if x not in globals()]
    if _missing:
        raise NameError(f'Missing required variables before gene voxelization: {_missing}')
    assert len(gene_df_1) == adata.shape[1], f'gene_df length ({len(gene_df_1)}) does not match adata columns ({adata.shape[1]})'
    assert len(valid_expr_positions_2) == len(valid_vox_1), 'valid_expr_positions and valid_vox must have the same length'
    print('Starting gene voxelization setup')
    print('adata shape:', adata.shape)
    print('n genes:', len(gene_df_1))
    print('n valid cells:', len(valid_expr_positions_2))
    print('n voxels:', n_voxels)
    print('output:', GENE_ZARR_PATH)
    print('\nPreloading valid-cell expression matrix into memory...')
    print('Target shape:', (len(valid_expr_positions_2), adata.shape[1]))
    t0 = time.perf_counter()
    expr_order = np.argsort(valid_expr_positions_2, kind='mergesort')
    valid_expr_positions_sorted = valid_expr_positions_2[expr_order]
    valid_vox_for_expr = valid_vox_1[expr_order]
    print('Reading adata.X[valid_expr_positions_sorted, :] ...', flush=True)
    X_valid = adata.X[valid_expr_positions_sorted, :]
    if sp.issparse(X_valid):
        X_valid = X_valid.toarray()
    X_valid = np.asarray(X_valid, dtype=np.float32)
    print('Loaded X_valid:', X_valid.shape, X_valid.dtype)
    print(f'Preload finished in {time.perf_counter() - t0:.1f}s')
    if Path(GENE_ZARR_PATH).exists():
        if OVERWRITE_OUTPUTS:
            print(f'\nRemoving existing zarr: {GENE_ZARR_PATH}')
            shutil.rmtree(GENE_ZARR_PATH)
        else:
            raise FileExistsError(f'{GENE_ZARR_PATH} exists. Set OVERWRITE_OUTPUTS=True or delete it manually.')
    gene_arr = create_zarr(GENE_ZARR_PATH, shape=(len(gene_df_1), n_voxels), chunks=(ROW_CHUNKS, min(VOXEL_CHUNKS, n_voxels)), dtype=GENE_DTYPE)
    attach_common_zarr_attrs(gene_arr, row_kind='gene', atlas_shape=atlas_shape)
    gene_arr.attrs['source_dataset'] = ABC_DATASET
    gene_arr.attrs['expression_threshold_log2'] = LOG2_EXPRESSION_THRESHOLD
    gene_arr.attrs['use_log2_threshold'] = USE_LOG2_THRESHOLD
    gene_arr.attrs['transform_list_allen_to_unified'] = [str(t) for t in TRANSFORM_LIST_ALLEN_TO_UNIFIED]
    print(f'\nStarting gene voxelization: {len(gene_df_1):,} genes x {n_voxels:,} voxels')
    print(f'Using preloaded X_valid: {X_valid.shape}')
    total_t0 = time.perf_counter()
    pbar = tqdm(range(len(gene_df_1)), desc='Voxelizing genes', mininterval=1)
    for _gene_idx in pbar:
        gene_id = str(gene_df_1.index[_gene_idx])
        if 'gene_symbol' in gene_df_1.columns:
            gene_symbol = str(gene_df_1.iloc[_gene_idx]['gene_symbol'])
        else:
            gene_symbol = gene_id
        verbose_gene = _gene_idx < DEBUG_FIRST_N_GENES or _gene_idx % PROGRESS_EVERY_N_GENES == 0
        if verbose_gene:
            tqdm.write(f'\n[gene {_gene_idx + 1}/{len(gene_df_1)}] {gene_symbol} ({gene_id})')
            gene_t0 = time.perf_counter()
            tqdm.write('  1/4 Getting expression vector from X_valid...')
        expr = X_valid[:, _gene_idx]
        if USE_LOG2_THRESHOLD:
            _use = expr > LOG2_EXPRESSION_THRESHOLD
        else:
            _use = np.isfinite(expr)
        n_used = int(np.sum(_use))
        if verbose_gene:
            t_filter = time.perf_counter()
            tqdm.write(f'      kept {n_used:,}/{len(expr):,} cells/positions')
            tqdm.write('  2/4 Accumulating expression into voxels...')
        row = np.zeros(n_voxels, dtype=GENE_DTYPE)
        if n_used > 0:
            np.add.at(row, valid_vox_for_expr[_use], expr[_use].astype(GENE_DTYPE, copy=False))
        if verbose_gene:
            t_accum = time.perf_counter()
            tqdm.write(f'      accumulation finished in {t_accum - t_filter:.2f}s; nonzero voxels={np.count_nonzero(row):,}')
            tqdm.write('  3/4 Writing row to zarr...')
        gene_arr[_gene_idx, :] = row
        if verbose_gene:
            t_write = time.perf_counter()
            tqdm.write(f'      zarr write finished in {t_write - t_accum:.2f}s')
            tqdm.write(f'  4/4 Done {gene_symbol} in {t_write - gene_t0:.2f}s')
        pbar.set_postfix(gene=gene_symbol[:12], used=n_used, elapsed=f'{time.perf_counter() - total_t0:.1f}s')
    print(f'\nFinished gene voxelization in {time.perf_counter() - total_t0:.1f}s')
    print('Saved:', GENE_ZARR_PATH)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sanity checks
    """)
    return


@app.cell
def _(CLUSTER_ZARR_PATH, Path, os, zarr):
    path = Path(CLUSTER_ZARR_PATH)
    print('zarr version:', zarr.__version__)
    print('path:', path)
    print('exists:', path.exists())
    print('is_dir:', path.is_dir())
    print('\nTop-level contents:')
    if path.exists():
        print(os.listdir(path)[:50])
    else:
        print('PATH DOES NOT EXIST')
    print('\nRoot metadata:')
    for _f in ['.zarray', '.zgroup', '.zattrs', 'zarr.json']:
        print(_f, (path / _f).exists())
    print('\nSearching for arrays/groups below this folder:')
    if path.exists():
        zarrays = list(path.rglob('.zarray'))
        zgroups = list(path.rglob('.zgroup'))
        print('number of .zarray files:', len(zarrays))
        for _meta in zarrays[:20]:
            print('array folder:', _meta.parent)
        print('number of .zgroup files:', len(zgroups))
        for _meta in zgroups[:20]:
            print('group folder:', _meta.parent)
    return


@app.cell
def _(
    CLUSTER_DTYPE,
    CLUSTER_ZARR_PATH,
    Path,
    ROW_CHUNKS,
    VOXEL_CHUNKS,
    atlas_shape,
    cluster_details,
    create_zarr,
    n_voxels,
    np,
    shutil,
    zarr,
):
    import json
    cluster_path = Path(CLUSTER_ZARR_PATH)
    print('cluster_path:', cluster_path)
    print('exists:', cluster_path.exists())
    print('root .zarray exists:', (cluster_path / '.zarray').exists())
    if (cluster_path / '.zarray').exists():
        print('This zarr already has .zarray metadata. No recovery needed.')
    else:
        if 'cluster_details' not in globals():
            raise NameError('cluster_details is missing. Load cluster metadata first.')
        expected_shape = (len(cluster_details), int(n_voxels))
        expected_chunks = (ROW_CHUNKS, min(VOXEL_CHUNKS, int(n_voxels)))
        expected_dtype = np.dtype(CLUSTER_DTYPE)
        print('Expected shape:', expected_shape)
        print('Expected chunks:', expected_chunks)
        print('Expected dtype:', expected_dtype)
        dummy_path = cluster_path.parent / '_dummy_cluster_metadata.zarr'
        if dummy_path.exists():
            shutil.rmtree(dummy_path)
        dummy = create_zarr(dummy_path, shape=expected_shape, chunks=expected_chunks, dtype=expected_dtype)
        dummy_meta_path = dummy_path / '.zarray'
        if not dummy_meta_path.exists():
            raise RuntimeError('Dummy zarr was created, but .zarray was not found.')
        with open(dummy_meta_path, 'r') as _f:
            _meta = json.load(_f)
        print('\nDummy metadata:')
        print(json.dumps(_meta, indent=2)[:1000])
        recovered_meta_path = cluster_path / '.zarray'
        with open(recovered_meta_path, 'w') as _f:
            json.dump(_meta, _f, indent=4)
        attrs = {'recovered_zarray_metadata': True, 'recovery_note': 'Recovered .zarray metadata from dummy zarr with matching shape/chunks/dtype.', 'row_kind': 'cluster', 'atlas_shape': list(atlas_shape)}
        with open(cluster_path / '.zattrs', 'w') as _f:
            json.dump(attrs, _f, indent=4)
        shutil.rmtree(dummy_path)
        print('\nRecovered metadata written:')
        print(recovered_meta_path)
        print(cluster_path / '.zattrs')
    _cluster_check = zarr.open_array(str(cluster_path), mode='r')
    print('\nOpened recovered cluster zarr:')
    print('shape:', _cluster_check.shape)
    print('chunks:', _cluster_check.chunks)
    print('dtype:', _cluster_check.dtype)
    test = _cluster_check[:1, :1000]
    print('test read shape:', test.shape)
    print('test min/max:', np.nanmin(test), np.nanmax(test))
    print('Recovery successful.')
    return


@app.cell
def _(
    CELL_COORDINATE_OUTPUT,
    CLUSTER_DETAILS_CSV,
    CLUSTER_ZARR_PATH,
    GENE_DF_CSV,
    GENE_ZARR_PATH,
):
    import dask.array as da
    print(CLUSTER_ZARR_PATH)
    _cluster_check = da.from_zarr(str(CLUSTER_ZARR_PATH))
    gene_check = da.from_zarr(str(GENE_ZARR_PATH))
    print('cluster_expression.zarr:', _cluster_check.shape, _cluster_check.dtype)
    print('gene_expression.zarr:', gene_check.shape, gene_check.dtype)
    print('cluster chunks:', _cluster_check.chunksize)
    print('gene chunks:', gene_check.chunksize)
    print('cluster details:', CLUSTER_DETAILS_CSV)
    print('gene metadata:', GENE_DF_CSV)
    print('transformed cells:', CELL_COORDINATE_OUTPUT)
    first_cluster_nonzero = da.count_nonzero(_cluster_check[0, :]).compute()
    first_gene_nonzero = da.count_nonzero(gene_check[0, :]).compute()
    print('first cluster nonzero voxels:', int(first_cluster_nonzero))
    print('first gene nonzero voxels:', int(first_gene_nonzero))
    return


@app.cell
def _(adata, atlas_shape, gene_df_1, np, pd, sp, valid_cell_df_1):
    import matplotlib.pyplot as plt
    GENE_SYMBOL_DEBUG = 'Slc17a6'
    Z_PLANE_DEBUG = 100
    PLANE_HALF_WIDTH = 1.0
    POINT_SIZE = 1
    EXPRESSION_THRESHOLD = 1.0
    ALLEN_CCF_VOXEL_SIZE_MM_1 = 0.025
    _needed = ['adata', 'valid_cell_df']
    _missing = [x for x in _needed if x not in globals()]
    if _missing:
        raise NameError(f'Missing required variables: {_missing}')
    required_cols = ['cell_label', 'x_ccf_mm', 'y_ccf_mm', 'z_ccf_mm']
    missing_cols = [c for c in required_cols if c not in valid_cell_df_1.columns]
    if missing_cols:
        raise ValueError(f'valid_cell_df is missing columns: {missing_cols}')
    print('adata shape:', adata.shape)
    print('valid_cell_df:', valid_cell_df_1.shape)
    if 'gene_df' in globals() and 'gene_symbol' in gene_df_1.columns:
        gene_hits = np.where(gene_df_1['gene_symbol'].astype(str).to_numpy() == GENE_SYMBOL_DEBUG)[0]
    else:
        gene_hits = []
        for _col in ['gene_symbol', 'gene_name', 'symbol']:
            if _col in adata.var.columns:
                gene_hits = np.where(adata.var[_col].astype(str).to_numpy() == GENE_SYMBOL_DEBUG)[0]
                if len(gene_hits) > 0:
                    break
        if len(gene_hits) == 0:
            gene_hits = np.where(adata.var_names.astype(str) == GENE_SYMBOL_DEBUG)[0]
    if len(gene_hits) == 0:
        raise ValueError(f'Could not find gene: {GENE_SYMBOL_DEBUG}')
    _gene_idx = int(gene_hits[0])
    print(f'Gene: {GENE_SYMBOL_DEBUG}, gene_idx={_gene_idx}')
    _obs_names = pd.Index(adata.obs_names.astype(str))
    cell_labels = pd.Index(valid_cell_df_1['cell_label'].astype(str))
    if not _obs_names.is_unique:
        raise ValueError('adata.obs_names is not unique.')
    if not cell_labels.isin(_obs_names).all():
        _n_match = cell_labels.isin(_obs_names).sum()
        raise ValueError(f'cell_label does not fully match adata.obs_names: {_n_match:,}/{len(cell_labels):,} matched')
    _obs_pos = pd.Series(np.arange(len(_obs_names)), index=_obs_names)
    expr_positions = _obs_pos.loc[cell_labels].to_numpy(np.int64)
    print('Aligned expression rows:', expr_positions.shape)
    order = np.argsort(expr_positions, kind='mergesort')
    expr_positions_sorted = expr_positions[order]
    print('Reading expression from adata.X...')
    x_sorted = adata.X[expr_positions_sorted, _gene_idx]
    if sp.issparse(x_sorted):
        x_sorted = x_sorted.toarray()
    x_sorted = np.asarray(x_sorted).reshape(-1).astype(np.float32)
    expr_1 = np.empty_like(x_sorted)
    expr_1[order] = x_sorted
    print('Expression vector:', expr_1.shape)
    print('expr min/max:', float(np.nanmin(expr_1)), float(np.nanmax(expr_1)))
    print('expr > threshold:', int(np.sum(expr_1 > EXPRESSION_THRESHOLD)))
    orig_z = valid_cell_df_1['x_ccf_mm'].to_numpy(float) / ALLEN_CCF_VOXEL_SIZE_MM_1
    orig_y = valid_cell_df_1['y_ccf_mm'].to_numpy(float) / ALLEN_CCF_VOXEL_SIZE_MM_1
    orig_x = valid_cell_df_1['z_ccf_mm'].to_numpy(float) / ALLEN_CCF_VOXEL_SIZE_MM_1
    print('\nOriginal Allen coordinate ranges in voxel units:')
    print('orig_z from x_ccf_mm:', np.nanmin(orig_z), np.nanmax(orig_z))
    print('orig_y from y_ccf_mm:', np.nanmin(orig_y), np.nanmax(orig_y))
    print('orig_x from z_ccf_mm:', np.nanmin(orig_x), np.nanmax(orig_x))
    plane_mask = np.abs(orig_z - Z_PLANE_DEBUG) <= PLANE_HALF_WIDTH
    expr_mask = expr_1 > EXPRESSION_THRESHOLD
    show_mask = plane_mask & expr_mask
    print(f'\nPlane z={Z_PLANE_DEBUG} +/- {PLANE_HALF_WIDTH}')
    print('cells in plane:', int(np.sum(plane_mask)))
    print('expressing cells in plane:', int(np.sum(show_mask)))
    if 'atlas_shape' in globals():
        y_size = int(atlas_shape[1])
        x_size = int(atlas_shape[2])
    else:
        y_size = int(np.nanmax(orig_y)) + 1
        x_size = int(np.nanmax(orig_x)) + 1
    img_sum = np.zeros((y_size, x_size), dtype=np.float32)
    img_count = np.zeros((y_size, x_size), dtype=np.float32)
    yy = np.rint(orig_y[show_mask]).astype(np.int64)
    xx = np.rint(orig_x[show_mask]).astype(np.int64)
    vv = expr_1[show_mask].astype(np.float32)
    inside_2d = (yy >= 0) & (yy < y_size) & (xx >= 0) & (xx < x_size)
    yy = yy[inside_2d]
    xx = xx[inside_2d]
    vv = vv[inside_2d]
    np.add.at(img_sum, (yy, xx), vv)
    np.add.at(img_count, (yy, xx), 1)
    print('2D image shape y,x:', img_sum.shape)
    print('nonzero pixels:', int(np.count_nonzero(img_sum)))
    return (
        EXPRESSION_THRESHOLD,
        GENE_SYMBOL_DEBUG,
        POINT_SIZE,
        Z_PLANE_DEBUG,
        expr_1,
        img_sum,
        orig_x,
        orig_y,
        plane_mask,
        plt,
        show_mask,
    )


@app.cell
def _(
    EXPRESSION_THRESHOLD,
    GENE_SYMBOL_DEBUG,
    POINT_SIZE,
    Z_PLANE_DEBUG,
    expr_1,
    img_sum,
    np,
    orig_x,
    orig_y,
    plane_mask,
    plt,
    show_mask,
):
    # ----------------------------
    # Plot original Allen coronal section
    (fig, axes) = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    im = ax.imshow(img_sum, origin='upper')
    # Panel 1: binned expression image
    ax.set_title(f'Original Allen coordinates\n{GENE_SYMBOL_DEBUG}, z={Z_PLANE_DEBUG}')
    ax.set_xlabel('original x index = z_ccf_mm / 0.025')
    ax.set_ylabel('original y index = y_ccf_mm / 0.025')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[1]
    ax.scatter(orig_x[plane_mask], orig_y[plane_mask], s=POINT_SIZE, alpha=0.15)
    ax.set_title(f'All valid Allen cells in z-plane\nn={int(np.sum(plane_mask)):,}')
    ax.set_xlabel('original x index')
    ax.set_ylabel('original y index')
    # Panel 2: scatter of all cells in plane
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax = axes[2]
    sc = ax.scatter(orig_x[show_mask], orig_y[show_mask], c=expr_1[show_mask], s=POINT_SIZE, alpha=0.8)
    ax.set_title(f'{GENE_SYMBOL_DEBUG}+ cells in original z-plane\nn={int(np.sum(show_mask)):,}, threshold>{EXPRESSION_THRESHOLD}')
    ax.set_xlabel('original x index')
    ax.set_ylabel('original y index')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    # Panel 3: scatter of expressing cells in plane
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load outputs later

    ```python
    import zarr
    import pandas as pd

    cluster_expression = zarr.open("./allen_to_unified_expression_project/output/cluster_expression.zarr", mode="r")
    gene_expression = zarr.open("./allen_to_unified_expression_project/output/gene_expression.zarr", mode="r")
    cluster_details = pd.read_csv("./allen_to_unified_expression_project/output/cluster_details.csv")
    gene_df = pd.read_csv("./allen_to_unified_expression_project/output/gene_df.csv")

    atlas_shape = tuple(gene_expression.attrs["atlas_shape"])
    one_gene_volume = gene_expression[0, :].reshape(atlas_shape)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
