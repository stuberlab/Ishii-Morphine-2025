import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import time
    from pathlib import Path
    import scipy as sp
    import dask.array as da
    import pandas as pd
    import numpy as np
    import os
    import neurolight as nl
    from neurolight.conversion import recommend_chunk_size
    import matplotlib.pyplot as plt
    import tifffile

    return da, nl, np, os, pd, plt, recommend_chunk_size, tifffile, time


@app.cell
def _(plt):
    import seaborn as sns
    from adjustText import adjust_text
    # Set matplotlib parameters for white text on transparent background
    #important for text to be detected when importing saved figures into illustrator
    plt.rcParams.update({'figure.facecolor': 'none', 'axes.facecolor': 'none', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'legend.facecolor': 'none', 'legend.edgecolor': 'none', 'text.color': 'black', 'font.family': 'Arial', 'pdf.fonttype': 42, 'ps.fonttype': 42})  # Transparent figure background  # Transparent axes background  # White axes edge color  # White axis labels  # White tick labels  # Transparent legend background  # Transparent legend edgecolor  # White text color
    return


@app.cell
def _(os):
    # Paths -> Figshare deposit (set OPIOID_DATA_ROOT or edit the default)
    DATA_ROOT = os.environ.get("OPIOID_DATA_ROOT", "/path/to/Figshare_deposit")
    GROUP = os.path.join(DATA_ROOT, "06_spatial_gene_merfish")
    ATLAS = os.path.join(DATA_ROOT, "shared", "atlas")
    return ATLAS, GROUP


@app.cell
def _(ATLAS, os, tifffile):
    # load atlas data
    # read an annotated atlas file
    atlas_img = tifffile.imread(os.path.join(ATLAS, "Kim_ref_adult_FP-label_v4.0.tif"))
    brain_mask = atlas_img > 0
    return atlas_img, brain_mask


@app.cell
def _(GROUP, np, os):
    # load example data
    # here we will be using a c-Fos based neural activity data: "neuro_raw"
    # and a data based on allen spatial transctriptomic data for Penk: "genes_raw"
    # eventually this should be loading a data that is stored on a server/cloud
    # (neuro_raw could alternatively be a condition beta map, e.g. Acute_Morphine_betas.npy)
    factor_path = os.path.join(GROUP, "factors")
    fidx = 1
    neuro_raw = np.load(os.path.join(factor_path, f'factor{fidx}.npy')) # loading the morphine factor
    return (neuro_raw,)


@app.cell
def _(GROUP, da, os, pd):
    # gene matrix
    # load the results

    gene_matrixpath = os.path.join(GROUP, "gene_expression.zarr")

    # write as zarr array to os.path.join(analysis_resultpah,'heatmap_array.zarr')
    gene_matrix = da.from_zarr(gene_matrixpath, mode="r")

    # this also contains the gene meta data
    gene_list = pd.read_csv(os.path.join(GROUP, "gene_df.csv"))
    return gene_list, gene_matrix


@app.cell
def _():
    # target genes
    target_genes = ["Oprk1","Oprm1","Oprd1",]
    return (target_genes,)


@app.cell
def _(gene_list, target_genes):
    gene_mask = gene_list.gene_symbol.isin(target_genes)
    # update the target_genes 
    target_genes_1 = gene_list[gene_list.gene_symbol.isin(target_genes)].gene_symbol.values
    return gene_mask, target_genes_1


@app.cell
def _(atlas_img, gene_mask, gene_matrix):
    G = int(gene_mask.sum())

    # (G, X*Y*Z) → (G, X, Y, Z) → (G, Z, Y, X)  — same as one-gene reshape(…[::-1]).T per row
    genes_raw = (
        gene_matrix[gene_mask]
        .compute()
        .reshape((G,) + tuple(atlas_img.shape[::-1]))
        .transpose(0, 3, 2, 1)
    )
    return (genes_raw,)


@app.cell
def _(atlas_img, genes_raw):
    # preprocessing specific to this data. The spatial data is hemispheric, so fold it
    genes_raw_1 = genes_raw[:, :, :, :atlas_img.shape[2] // 2]
    return (genes_raw_1,)


@app.cell
def _(atlas_img, brain_mask):
    # fold the brain mask
    brain_mask_1 = brain_mask[:, :, :atlas_img.shape[2] // 2]
    return (brain_mask_1,)


@app.cell
def _(genes_raw_1, plt):
    # print the shape of the data and visualize a coronal section from z = 100
    print(f'Shape of the data: {genes_raw_1.shape}')
    plt.imshow(genes_raw_1[0, 100, :, :], cmap='gray')
    return


@app.cell
def _(atlas_img, neuro_raw):
    # for the neural activity data, average the left and right hemisphere
    neuro_raw_1 = (neuro_raw[:, :, :atlas_img.shape[2] // 2] + neuro_raw[:, :, atlas_img.shape[2] // 2:][:, :, ::-1]) / 2
    return (neuro_raw_1,)


@app.cell
def _(neuro_raw_1, plt):
    # print the shape of the data and visualize a coronal section from z = 100
    print(f'Shape of the data: {neuro_raw_1.shape}')
    plt.imshow(neuro_raw_1[95, :, :], cmap='gray')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # set up basic information to write the data to zarr
    """)
    return


@app.cell
def _(GROUP, neuro_raw_1, os, target_genes_1):
    VOLUME_SHAPE = neuro_raw_1.shape
    N_GENES = len(target_genes_1)
    RADIUS = 5
    KERNEL = 'sphere'
    SEED = 42
    _alpha = 0.001
    OUT_DIR = os.environ.get("OPIOID_SEARCHLIGHT_OUT", os.path.join(GROUP, "searchlight_output"))
    os.makedirs(OUT_DIR, exist_ok=True)
    return KERNEL, OUT_DIR, RADIUS, VOLUME_SHAPE


@app.cell
def _(RADIUS, VOLUME_SHAPE, recommend_chunk_size):
    # ---------------------------------------------------------------------------
    # 2. Chunk size recommendation
    # ---------------------------------------------------------------------------

    print("\n[2/5] Chunk size recommendations for real mouse brain:")
    recommend_chunk_size(
        volume_shape=VOLUME_SHAPE,
        radius=RADIUS,
        available_ram_gb=128.0,
    )
    return


@app.cell
def _():
    # set a chunk size
    chunk_dim =  132
    return


@app.cell
def _(nl):
    import importlib
    import neurolight.conversion
    importlib.reload(neurolight.conversion)
    importlib.reload(nl)
    return (importlib,)


@app.cell
def _(OUT_DIR, RADIUS, genes_raw_1, neuro_raw_1, nl, os, time):
    print('\n[3/5] Converting to Zarr with kernel-aligned chunking...')
    _t0 = time.perf_counter()
    neuro_zarr = nl.to_zarr(neuro_raw_1, os.path.join(OUT_DIR, 'neuro.zarr'), radius=RADIUS, chunk_multiplier=6, overwrite=True)
    genes_zarr = nl.to_zarr(genes_raw_1, os.path.join(OUT_DIR, 'genes.zarr'), radius=RADIUS, chunk_multiplier=6, overwrite=True)
    info = nl.zarr_info(os.path.join(OUT_DIR, 'neuro.zarr'))
    print(f"  Compression ratio: {info['compression_ratio']:.1f}x")
    print(f'  Time: {time.perf_counter() - _t0:.2f}s')
    return


@app.cell
def _(KERNEL, OUT_DIR, RADIUS, brain_mask_1, nl, os, target_genes_1, time):
    print('\n[4/5] Running searchlight correlation (multi-gene)...')
    _t0 = time.perf_counter()
    sl = nl.SearchlightCorrelation(neuro_path=os.path.join(OUT_DIR, 'neuro.zarr'), gene_path=os.path.join(OUT_DIR, 'genes.zarr'), radius=RADIUS, kernel=KERNEL, mask=brain_mask_1, scheduler='threads', gene_names=target_genes_1)
    print(sl)
    (r_map, p_map) = sl.run(output_path=os.path.join(OUT_DIR, 'results.zarr'), compute_pvalues=True, overwrite=True)
    elapsed = time.perf_counter() - _t0
    print(f'  r_map shape : {r_map.shape}')
    print(f'  Time        : {elapsed:.2f}s')
    return


@app.cell
def _(OUT_DIR, os):
    # load the results zarr
    import zarr
    from neurolight.stats import fdr_correction
    root = zarr.open(os.path.join(OUT_DIR, 'results.zarr'), mode='r')
    return fdr_correction, root


@app.cell
def _(root):
    # Arrays
    r_map_1 = root['r_map'][:]  # (n_genes, Z, Y, X)
    p_map_1 = root['p_map'][:]
    meta = dict(root.attrs)
    # Metadata
    gene_names = meta['gene_names']
    print(meta)
    return gene_names, p_map_1, r_map_1


@app.cell
def _(brain_mask_1, fdr_correction, gene_names, np, p_map_1, r_map_1):
    _alpha = 0.05
    r_thresh_map = np.zeros_like(r_map_1, dtype=np.float32)
    for (g, name) in enumerate(gene_names):
        (reject, q_map) = fdr_correction(p_map_1[g], alpha=_alpha)
        r_thresh = np.where(reject, r_map_1[g], 0)
        print(f'{name}: {reject[brain_mask_1].sum()} sig voxels')
        r_thresh_map[g] = r_thresh
    return (r_thresh_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualization
    """)
    return


@app.cell
def _(os):
    analysis_figurepath = os.path.join("..", "figures", "Figure_S16")
    if not os.path.exists(analysis_figurepath):
        os.makedirs(analysis_figurepath)
    return (analysis_figurepath,)


@app.cell
def _():
    # pre selected zplanes
    curated_zplanes = [84,104,117,153,186,220]
    return (curated_zplanes,)


@app.cell
def _(importlib):
    import brain_vis
    importlib.reload(brain_vis)
    from brain_vis import overlap_contour, set_transparency

    return overlap_contour, set_transparency


@app.cell
def _(ATLAS, os, pd):
    # load brain atlas to register
    atlas_df = pd.read_csv(os.path.join(ATLAS, "atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv"), index_col=False)
    return (atlas_df,)


@app.cell
def _(ATLAS, atlas_img, os, tifffile):
    contour_img = tifffile.imread(os.path.join(ATLAS, "Kim_ref_adult_FP-label_v2.9_contour_map.tif"))
    hemi_contour_img = contour_img[:,:,:atlas_img.shape[2]//2]
    return (hemi_contour_img,)


@app.cell
def _(atlas_img):
    hemi_atlas_img = atlas_img[:,:,:atlas_img.shape[2]//2]
    return (hemi_atlas_img,)


@app.cell
def _(overlap_contour):
    help(overlap_contour)
    return


@app.cell
def _(
    analysis_figurepath,
    curated_zplanes,
    hemi_atlas_img,
    hemi_contour_img,
    os,
    overlap_contour,
    plt,
    r_thresh_map,
    set_transparency,
    target_genes_1,
):
    import matplotlib.cm as mcolors
    import matplotlib.colors as mcolors_mod
    imy_slice = slice(25, 425)
    imx_slice = slice(50, 600)
    cmaps = [plt.cm.Greens, plt.cm.Reds, plt.cm.Blues]
    for (gidx, _gene) in enumerate(target_genes_1):
        theatmap = r_thresh_map[gidx]
        (_cmin, _cmax) = (-0.5, 0.5)
        _cmap = plt.cm.coolwarm
        (__, _overlayed_image) = overlap_contour(theatmap, hemi_contour_img, cmin=_cmin, cmax=_cmax, outputpath=None, colormap=_cmap, overlap_black=True)
        (_fig, _axs) = plt.subplots(1, len(curated_zplanes), figsize=(3 * len(curated_zplanes), 3), sharey=True)
        _fig.subplots_adjust(wspace=0.25, hspace=0.3)
        for (_idx, _ax) in enumerate(_axs):
            _trans_img = set_transparency(_overlayed_image[curated_zplanes[_idx], :, :], (hemi_atlas_img == 0)[curated_zplanes[_idx], :, :])
            _ax.imshow(_trans_img[imy_slice, imx_slice])
            _ax.axis('off')
            _ax.set_title('')
            _ax.set_ylabel(_gene, color='black')
        sm = plt.cm.ScalarMappable(cmap=_cmap, norm=mcolors_mod.Normalize(vmin=_cmin, vmax=_cmax))
        sm.set_array([])
        cbar = _fig.colorbar(sm, ax=_axs[-1], fraction=0.046, pad=0.04)
        cbar.set_label('r', color='black')
        cbar.ax.yaxis.set_tick_params(color='black')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')
        _fig.savefig(os.path.join(analysis_figurepath, f'spatial_correlation_map_{_gene}.png'), bbox_inches='tight', dpi=1024)
        _fig.savefig(os.path.join(analysis_figurepath, f'spatial_correlation_map_{_gene}.pdf'), bbox_inches='tight', dpi=1024)
    return (theatmap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualization for target acronyms
    """)
    return


@app.cell
def _(atlas_img, np):
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
def _(
    analysis_figurepath,
    atlas_df,
    atlas_img,
    get_slice,
    hemi_atlas_img,
    hemi_contour_img,
    np,
    os,
    overlap_contour,
    plt,
    r_thresh_map,
    set_transparency,
    target_genes_1,
    theatmap,
):
    from brain_vis import get_subregions
    target_acronyms = ['A24a (IL)', 'A25 (DP)', 'A32 (PrL)', 'AO', 'AcbC', 'AcbSh', 'CM', 'CPre', 'Ce', 'Cl', 'DS', 'IAD', 'IMD', 'La', 'MD', 'O', 'PT', 'PVT', 'LS', 'VTA', 'DR']
    for (gidx_1, _gene) in enumerate(target_genes_1):
        target_map = r_thresh_map[gidx_1]
        for target_acronym in target_acronyms:
            target_id = atlas_df[atlas_df['acronym'] == target_acronym]['id'].values[0]
            sub_ids = get_subregions(atlas_df, target_id, return_original=True)['id'].values
            brain_voxels = np.isin(hemi_atlas_img, sub_ids)
            zs = np.array([])
            for ID in sub_ids:
                (z_, y_, x_) = np.where(atlas_img == ID)
                zs = np.concatenate([zs, z_])
            z_unique = np.unique(zs).astype('uint16')
            if target_acronym == 'LS':
                z_unique = np.array(range(98, 120 + 1))
            z_center = int(np.mean(zs))
            (_cmin, _cmax) = (-0.5, 0.5)
            _cmap = plt.cm.coolwarm
            (__, _overlayed_image) = overlap_contour(theatmap, hemi_contour_img, cmin=_cmin, cmax=_cmax, outputpath=None, colormap=_cmap, overlap_black=True)
            (_fig, _axs) = plt.subplots(1, len(z_unique[::5]), figsize=(len(z_unique[::5]) * 1.25, 2), sharex=True, sharey=True)
            (__, _overlayed_image) = overlap_contour(target_map, hemi_contour_img, cmin=_cmin, cmax=_cmax, outputpath=None, colormap=_cmap, overlap_black=True)
            for (_idx, curated_zplane) in enumerate(z_unique[::5]):
                _ax = _axs[_idx]
                _trans_img = set_transparency(_overlayed_image[curated_zplane, :, :], (hemi_atlas_img == 0)[curated_zplane, :, :])
                (yslice, xslice) = get_slice(curated_zplane, sub_ids, hemi='left', window=60)
                _ax.imshow(_trans_img[yslice, xslice])
                _ax.set_xticks([])
                _ax.set_yticks([])
                if _idx == 0:
                    _ax.set_ylabel(f'{_gene}', color='black')
                else:
                    _ax.set_ylabel('', color='black')
                _ax.invert_xaxis()
            _fig.savefig(os.path.join(analysis_figurepath, f'{_gene}_{target_acronym}.png'), bbox_inches='tight', dpi=1024)
            _fig.savefig(os.path.join(analysis_figurepath, f'{_gene}_{target_acronym}.svg'), bbox_inches='tight', dpi=1024)
    return (gidx_1,)


@app.cell
def _(gidx_1, np, r_thresh_map):
    np.nanmin(r_thresh_map[gidx_1])
    return


if __name__ == "__main__":
    app.run()
