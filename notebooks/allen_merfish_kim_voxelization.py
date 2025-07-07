#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Allen Brain MERFISH to Kim Atlas Voxelization Pipeline

This script processes gene expression data from the Allen Brain Institute MERFISH dataset
(Zhuang-ABCA-1) and transforms it into Kim atlas space for downstream analysis.

Pipeline Steps:
1. Load Allen MERFISH cell metadata and gene expression data
2. Load CCF coordinates for each cell
3. Perform ANTs-based registration between Allen CCF and Kim atlas
4. Transform cell coordinates from CCF space to Kim atlas space
5. Apply gene expression thresholds to identify positive cells
6. Voxelize gene expression data in Kim atlas space
7. Save results in Zarr format for efficient access

Author: [Your name]
Date: [Date]
"""

import os
import pathlib
import numpy as np
import pandas as pd
import time
import shutil
import scipy.sparse as sp
import zarr
import ants
import tifffile as tiff
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

# Define data paths
download_base = pathlib.Path(r'\\10.159.50.7\LabCommon\Ken\data\Opioid_cFos\Allen_Alignment')
result_path = r"\\10.159.50.7\LabCommon\Ken\data\Opioid_cFos\Allen_Alignment\summarized_data"

# Atlas registration paths
allen_template_path = r"\\10.159.50.7\Analysis2\Ken\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations\Allen_templates\template_25_coronal.tif"
kim_template_path = r"\\10.159.50.7\Analysis2\Ken\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations\Kim_ref_adult_v1_brain.tif"
kim_annotation_path = r"\\10.159.50.7\Analysis2\Ken\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations\Kim_ref_adult_FP-label_v2.9.tif"
output_path = r"\\10.159.50.7\Analysis2\Ken\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations"

# Registration parameters
registration_key = 'Kim_to_allen25'
transformation_path = os.path.join(output_path, f'{registration_key}_transformation')
resolution_factor = 1000/25  # Convert from microns to voxels (25 micron resolution)

# Dataset configuration
dataset = 'Zhuang-ABCA-1'  # Focus on the richest dataset
gene_expression_threshold = 0  # Threshold for identifying positive cells

print("Starting Allen Brain MERFISH to Kim Atlas Pipeline...")
print(f"Dataset: {dataset}")
print(f"Gene expression threshold: {gene_expression_threshold}")

# =============================================================================
# 1. LOAD ALLEN BRAIN DATA
# =============================================================================

print("\n1. Loading Allen Brain data...")

# Initialize Allen Brain Cache
abc_cache = AbcProjectCache.from_s3_cache(download_base)
print(f"Cache manifest: {abc_cache.current_manifest}")

# Load cell metadata
print("Loading cell metadata...")
cell_metadata = abc_cache.get_metadata_dataframe(
    directory=dataset,
    file_name='cell_metadata',
    dtype={"cell_label": str}
)
cell_metadata.set_index('cell_label', inplace=True)
print(f"Loaded {len(cell_metadata)} cells")

# Load gene information
print("Loading gene metadata...")
gene_metadata = abc_cache.get_metadata_dataframe(
    directory=dataset,
    file_name='gene'
)
gene_metadata.set_index('gene_identifier', inplace=True)
print(f"Loaded {len(gene_metadata)} genes")

# Load cluster information for cell type annotation
print("Loading cluster annotations...")
cluster_details = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_pivoted',
    keep_default_na=False
)
cluster_details.set_index('cluster_alias', inplace=True)

cluster_colors = abc_cache.get_metadata_dataframe(
    directory='WMB-taxonomy',
    file_name='cluster_to_cluster_annotation_membership_color',
)
cluster_colors.set_index('cluster_alias', inplace=True)

# Join cell metadata with cluster information
cell_extended = cell_metadata.join(cluster_details, on='cluster_alias')
cell_extended = cell_extended.join(cluster_colors, on='cluster_alias')

# Load CCF coordinates
print("Loading CCF coordinates...")
ccf_coordinates = abc_cache.get_metadata_dataframe(
    directory=f"{dataset}-CCF", 
    file_name='ccf_coordinates'
)
ccf_coordinates.set_index('cell_label', inplace=True)
ccf_coordinates.rename(columns={'x': 'x_ccf', 'y': 'y_ccf', 'z': 'z_ccf'}, inplace=True)

# Join cell metadata with CCF coordinates
cell_extended = cell_extended.join(ccf_coordinates, how='inner')
print(f"Final cell dataset: {len(cell_extended)} cells with coordinates")

# =============================================================================
# 2. ATLAS REGISTRATION USING ANTS
# =============================================================================

print("\n2. Performing atlas registration...")

# Check if transformation already exists
if os.path.exists(transformation_path):
    print("Using existing transformation files...")
else:
    print("Computing new registration between Allen CCF and Kim atlas...")
    
    # Load atlas images
    fixed_image = ants.image_read(allen_template_path)  # Allen CCF (target)
    moving_image = ants.image_read(kim_template_path)   # Kim atlas (source)
    
    print("Running ANTs registration (this may take several minutes)...")
    # Perform registration using SyN (symmetric normalization)
    registration_result = ants.registration(
        fixed=fixed_image, 
        moving=moving_image, 
        type_of_transform='antsRegistrationSyNs'  # Most accurate but slower
    )
    
    # Save transformation files
    os.makedirs(transformation_path, exist_ok=True)
    
    # Copy transformation files to organized directory
    for idx in range(2):
        for transform_type in ['fwdtransforms', 'invtransforms']:
            original_path = registration_result[transform_type][idx]
            file_extension = os.path.splitext(original_path)[1]
            new_filename = f"{transform_type}_{idx}{file_extension}"
            new_path = os.path.join(transformation_path, new_filename)
            shutil.copy(original_path, new_path)
    
    print("Registration completed and transformation files saved.")

# Load transformation files for coordinate transformation
transform_files = [f for f in os.listdir(transformation_path) 
                  if (f.endswith('.mat') and not f.startswith('invtransforms')) or 
                     (f.endswith('.nii.gz') and not f.startswith('invtransforms'))]
transform_files.sort()
transform_list = [os.path.join(transformation_path, f) for f in transform_files]

print(f"Using transformation files: {transform_files}")

# =============================================================================
# 3. COORDINATE TRANSFORMATION
# =============================================================================

print("\n3. Transforming coordinates from CCF to Kim atlas space...")

# Prepare coordinates for transformation
# Note: Coordinate order is swapped (z,y,x) for coronal-to-coronal registration
coords = ccf_coordinates.loc[:, ['x_ccf', 'y_ccf', 'z_ccf']].copy()
coords.columns = ['z', 'y', 'x']
coords = coords * resolution_factor  # Convert to voxel coordinates

print(f"Transforming {len(coords)} coordinate points...")

# Apply ANTs transformation to convert CCF coordinates to Kim space
kim_coordinates = ants.apply_transforms_to_points(
    dim=3,
    points=coords,
    transformlist=transform_list,
    whichtoinvert=[False, False]  # Use forward transformation
)

# Rename columns for clarity
kim_coordinates.rename(columns={'z': 'z_kim', 'y': 'y_kim', 'x': 'x_kim'}, inplace=True)

# Load Kim atlas for region labeling
print("Loading Kim atlas for region annotation...")
kim_atlas_img = tiff.imread(r"\\10.159.50.7\Analysis2\Ken\ClearMap\clearmap_ressources_mouse_brain\ClearMap_ressources\Regions_annotations\Kim_ref_adult_FP-label_v4.0.tif")

# Get Kim region labels for each transformed coordinate
coords_int = kim_coordinates[['x_kim', 'y_kim', 'z_kim']].values.astype(int)
kim_labels = kim_atlas_img[coords_int[:, 2], coords_int[:, 1], coords_int[:, 0]]  # z,y,x indexing
kim_coordinates['kim_region_labels'] = kim_labels

print("Coordinate transformation completed.")

# =============================================================================
# 4. LOAD GENE EXPRESSION DATA
# =============================================================================

print("\n4. Loading gene expression data...")

# Load gene expression matrix
gene_expression_file = abc_cache.get_data_path(directory=dataset, file_name=f"{dataset}/log2")

print("Reading gene expression matrix (this may take several minutes)...")
start_time = time.time()

# Load as AnnData object (backed for memory efficiency)
import anndata
adata = anndata.read_h5ad(gene_expression_file, backed='r')

# Extract gene expression data
gene_expression_data = adata.to_df()
print(f"Gene expression loading time: {time.time() - start_time:.2f} seconds")

# Close the file to free memory
adata.file.close()
del adata

# Combine cell metadata with gene expression data
cell_expression = cell_extended.join(gene_expression_data)

# Add Kim coordinates
cell_expression = cell_expression.join(kim_coordinates)

print(f"Combined dataset shape: {cell_expression.shape}")

# =============================================================================
# 5. VOXELIZATION FUNCTION
# =============================================================================

def voxelize_points(brain_img, coords, intensity=None, method='Spherical', size=(5,5,5)):
    """
    Voxelize point coordinates into a 3D volume.
    
    Parameters:
    -----------
    brain_img : numpy.ndarray
        3D template image defining output dimensions
    coords : numpy.ndarray or pandas.DataFrame
        Coordinates to voxelize (n_points, 3)
    intensity : numpy.ndarray, optional
        Intensity values for each point. If None, uses count of points.
    method : str
        Voxelization method: 'Pixel', 'Spherical', or 'Rectangular'
    size : tuple
        Size of structure to place at each point
        
    Returns:
    --------
    numpy.ndarray
        Voxelized 3D array
    """
    data_size = brain_img.shape
    coords = np.array(coords, dtype=float)
    output = np.zeros(data_size, dtype=float)
    
    if method.lower() == 'pixel':
        # Simple point voxelization
        for i in range(coords.shape[0]):
            x, y, z = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
            if 0 <= x < data_size[0] and 0 <= y < data_size[1] and 0 <= z < data_size[2]:
                if intensity is None:
                    output[x, y, z] += 1
                else:
                    output[x, y, z] += intensity[i]
                    
    elif method.lower() == 'spherical':
        # Spherical kernel voxelization
        radius_x, radius_y, radius_z = size[0]/2, size[1]/2, size[2]/2
        
        # Generate sphere offsets
        x_range = np.arange(-radius_x, radius_x+1)
        y_range = np.arange(-radius_y, radius_y+1)
        z_range = np.arange(-radius_z, radius_z+1)
        
        offsets = []
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    if (x*x)/(radius_x*radius_x) + (y*y)/(radius_y*radius_y) + (z*z)/(radius_z*radius_z) < 1:
                        offsets.append([x, y, z])
        offsets = np.array(offsets)
        
        # Apply spherical kernel at each point
        for i in range(coords.shape[0]):
            for offset in offsets:
                x = int(coords[i, 0] + offset[0])
                y = int(coords[i, 1] + offset[1])
                z = int(coords[i, 2] + offset[2])
                
                if 0 <= x < data_size[0] and 0 <= y < data_size[1] and 0 <= z < data_size[2]:
                    if intensity is None:
                        output[x, y, z] += 1
                    else:
                        output[x, y, z] += intensity[i]
    
    return output

# =============================================================================
# 6. GENE EXPRESSION VOXELIZATION
# =============================================================================

print("\n5. Voxelizing gene expression data...")

# Load Kim template for voxelization reference
kim_template = ants.image_read(kim_template_path)

# Prepare coordinates for voxelization
voxel_coords = cell_expression[['x_kim', 'y_kim', 'z_kim']].values

# Set up Zarr output
zarr_output_path = os.path.join(result_path, "gene_expression.zarr")
if os.path.exists(zarr_output_path):
    shutil.rmtree(zarr_output_path)

# Create Zarr store
gene_count = len(gene_metadata)
voxel_count = np.prod(kim_template.shape)

zarr_store = zarr.open(
    zarr_output_path, 
    mode='w', 
    shape=(gene_count, voxel_count),
    chunks=(1, voxel_count), 
    dtype='float32'
)

print(f"Processing {gene_count} genes...")
gene_list = []

# Process each gene individually
for gene_idx, gene_id in enumerate(gene_metadata.index):
    if gene_idx % 100 == 0:
        print(f"Processing gene {gene_idx+1}/{gene_count}: {gene_id}")
    
    # Get gene expression values
    gene_expression_values = cell_expression[gene_id].values
    
    # Apply threshold to identify expressing cells
    expressing_cells_mask = gene_expression_values > gene_expression_threshold
    
    if expressing_cells_mask.sum() == 0:
        # No expressing cells - create empty voxel map
        voxelized_data = np.zeros(kim_template.shape)
    else:
        # Voxelize expressing cells
        expressing_coords = voxel_coords[expressing_cells_mask]
        expressing_values = gene_expression_values[expressing_cells_mask]
        
        voxelized_data = voxelize_points(
            kim_template,
            expressing_coords,
            intensity=expressing_values,
            method='Spherical',
            size=(2, 2, 4)  # Adjusted for voxel resolution
        )
    
    # Store flattened voxel data in Zarr
    zarr_store[gene_idx, :] = voxelized_data.flatten()
    gene_list.append(gene_id)

print(f"Voxelization completed. Data saved to: {zarr_output_path}")

# =============================================================================
# 7. SAVE METADATA AND RESULTS
# =============================================================================

print("\n6. Saving metadata and results...")

# Save gene list
gene_list_df = pd.DataFrame({
    'gene_id': gene_list,
    'gene_symbol': [gene_metadata.loc[g, 'gene_symbol'] if g in gene_metadata.index else 'Unknown' 
                   for g in gene_list]
})
gene_list_df.to_csv(os.path.join(result_path, 'gene_list.csv'), index=False)

# Save gene metadata
gene_metadata.to_csv(os.path.join(result_path, 'gene_metadata.csv'), index=True)

# Save processing parameters
processing_info = {
    'dataset': dataset,
    'gene_expression_threshold': gene_expression_threshold,
    'n_cells': len(cell_expression),
    'n_genes': len(gene_metadata),
    'voxelization_method': 'Spherical',
    'voxelization_size': (2, 2, 4),
    'kim_atlas_shape': kim_template.shape,
    'zarr_output_shape': (gene_count, voxel_count)
}

import json
with open(os.path.join(result_path, 'processing_info.json'), 'w') as f:
    json.dump(processing_info, f, indent=2)

print("Pipeline completed successfully!")
print(f"Results saved to: {result_path}")
print(f"Gene expression Zarr file: {zarr_output_path}")
print(f"Gene list: {os.path.join(result_path, 'gene_list.csv')}")
print(f"Processing info: {os.path.join(result_path, 'processing_info.json')}")
