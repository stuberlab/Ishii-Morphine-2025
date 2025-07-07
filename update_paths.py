#!/usr/bin/env python3
"""
Simple script to update hardcoded paths in notebooks to use the new data directory structure.
"""

import os
import re
from pathlib import Path

# Path mappings from old to new
PATH_MAPPINGS = {
    r'\\10\.159\.50\.7\\Analysis2\\Ken\\ClearMap\\clearmap_ressources_mouse_brain\\ClearMap_ressources\\Regions_annotations\\atlas_info_KimRef_FPbasedLabel_v4\.0_brain_with_size_with_curated_with_cleaned_acronyms\.csv': 'data/atlas/atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv',
    r'\\10\.159\.50\.7\\Analysis2\\Ken\\ClearMap\\clearmap_ressources_mouse_brain\\ClearMap_ressources\\Regions_annotations\\Kim_ref_adult_FP-label_v4\.0\.tif': 'data/atlas/Kim_ref_adult_FP-label_v4.0.tif',
    r'\\10\.159\.50\.7\\Analysis2\\Ken\\ClearMap\\clearmap_ressources_mouse_brain\\ClearMap_ressources\\Regions_annotations\\Kim_ref_adult_FP-label_v2\.9_contour_map\.tif': 'data/atlas/Kim_ref_adult_FP-label_v2.9_contour_map.tif',
    r'\\10\.159\.50\.7\\analysis2\\Ken\\LSMS\\Opioid': 'data/lsms/Opioid',
    r'\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\result': 'data/opioid_cfos/result',
    r'\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\heatmap_array': 'data/opioid_cfos/heatmap_array',
    r'\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\gene_expression\.zarr': 'data/opioid_cfos/gene_expression.zarr',
    r'\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\Allen_Alignment\\summarized_data\\gene_df\.csv': 'data/allen_alignment/summarized_data/gene_df.csv',
    r'\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\spatial_clustering_results\\2025_04_02-12_34': 'data/spatial_clustering_results/2025_04_02-12_34',
    r'Y:\\SmartSPIM2\\Ken\\Fentanyl': 'data/fentanyl',
    r'G:/My Drive/Opioid_whole_brain_manuscript/result': 'data/results'
}

def update_file_paths(file_path):
    """Update paths in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all path mappings
        for old_pattern, new_path in PATH_MAPPINGS.items():
            content = re.sub(old_pattern, new_path, content)
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Updated paths in {file_path.name}")
            return True
        else:
            print(f"- No changes needed for {file_path.name}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path.name}: {e}")
        return False

def main():
    """Update paths in all notebooks and Python files."""
    notebooks_dir = Path("notebooks")
    
    if not notebooks_dir.exists():
        print("Notebooks directory not found!")
        return
    
    # Find all notebook and Python files
    files = list(notebooks_dir.glob("*.ipynb")) + list(notebooks_dir.glob("*.py"))
    
    if not files:
        print("No files found!")
        return
    
    print(f"Found {len(files)} files")
    print("Updating paths...")
    
    success_count = 0
    for file_path in files:
        if update_file_paths(file_path):
            success_count += 1
    
    print(f"\nCompleted! Updated {success_count}/{len(files)} files.")
    print("\nPath mappings applied:")
    for old, new in PATH_MAPPINGS.items():
        print(f"  {old} -> {new}")

if __name__ == "__main__":
    main() 