#!/usr/bin/env python3
"""
Simple script to update all hardcoded paths in notebooks.
Run this with: python simple_path_update.py
"""

import os
import re
from pathlib import Path

def update_paths_in_file(file_path):
    """Update all hardcoded paths in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Debug: show a sample of the content
        if "\\\\10.159.50.7" in content:
            print(f"  Found escaped paths in {file_path.name}")
            # Find first occurrence
            import re
            match = re.search(r'\\\\10\.159\.50\.7[^"]*', content)
            if match:
                print(f"  Sample path: {match.group()[:100]}...")
        
        # Path replacements - handle both escaped and unescaped versions
        replacements = [
            # Atlas paths
            (r'\\\\10\.159\.50\.7\\Analysis2\\Ken\\ClearMap\\clearmap_ressources_mouse_brain\\ClearMap_ressources\\Regions_annotations\\atlas_info_KimRef_FPbasedLabel_v4\.0_brain_with_size_with_curated_with_cleaned_acronyms\.csv', 'data/atlas/atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv'),
            (r'\\\\10\.159\.50\.7\\Analysis2\\Ken\\ClearMap\\clearmap_ressources_mouse_brain\\ClearMap_ressources\\Regions_annotations\\Kim_ref_adult_FP-label_v4\.0\.tif', 'data/atlas/Kim_ref_adult_FP-label_v4.0.tif'),
            (r'\\\\10\.159\.50\.7\\Analysis2\\Ken\\ClearMap\\clearmap_ressources_mouse_brain\\ClearMap_ressources\\Regions_annotations\\Kim_ref_adult_FP-label_v2\.9_contour_map\.tif', 'data/atlas/Kim_ref_adult_FP-label_v2.9_contour_map.tif'),
            (r'\\\\10\.159\.50\.7\\Analysis2\\Ken\\LSMS\\ClearMap\\clearmap_ressources_mouse_brain\\ClearMap_ressources\\Regions_annotations\\Kim_ref_adult_FP-label_v4\.0\.tif', 'data/atlas/Kim_ref_adult_FP-label_v4.0.tif'),
            
            # LSMS paths
            (r'\\\\10\.159\.50\.7\\analysis2\\Ken\\LSMS\\Opioid', 'data/lsms/Opioid'),
            
            # Opioid cFos paths
            (r'\\\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\result', 'data/opioid_cfos/result'),
            (r'\\\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\heatmap_array', 'data/opioid_cfos/heatmap_array'),
            (r'\\\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\gene_expression\.zarr', 'data/opioid_cfos/gene_expression.zarr'),
            (r'\\\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\Allen_Alignment\\summarized_data\\gene_df\.csv', 'data/allen_alignment/summarized_data/gene_df.csv'),
            (r'\\\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Opioid_cFos\\spatial_clustering_results\\2025_04_02-12_34', 'data/spatial_clustering_results/2025_04_02-12_34'),
            
            # Fentanyl paths
            (r'Y:\\SmartSPIM2\\Ken\\Fentanyl', 'data/fentanyl'),
            (r'\\\\10\.159\.50\.7\\LabCommon\\Ken\\data\\Fentanyl', 'data/fentanyl'),
            
            # Google Drive paths
            (r'G:/My Drive/Opioid_whole_brain_manuscript/result', 'data/results'),
        ]
        
        # Apply all replacements
        changes_made = False
        for old_pattern, new_path in replacements:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_path, content)
                changes_made = True
                print(f"  Replaced: {old_pattern[:50]}... -> {new_path}")
        
        if not changes_made:
            print(f"  No patterns found in {file_path.name}")
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Updated {file_path.name}")
            return True
        else:
            print(f"- No changes needed for {file_path.name}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path.name}: {e}")
        return False

def main():
    """Update paths in all notebooks."""
    notebooks_dir = Path("notebooks")
    
    if not notebooks_dir.exists():
        print("Notebooks directory not found!")
        return
    
    # Find all notebook files
    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    
    if not notebook_files:
        print("No notebook files found!")
        return
    
    print(f"Found {len(notebook_files)} notebook files")
    print("Updating paths...")
    
    success_count = 0
    for notebook_path in notebook_files:
        if update_paths_in_file(notebook_path):
            success_count += 1
    
    print(f"\nCompleted! Updated {success_count}/{len(notebook_files)} notebooks.")

if __name__ == "__main__":
    main() 