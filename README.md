# Ishii-Morphine-2025

This repository contains analysis notebooks and scripts for opioid research data analysis.

## Data Organization

The project uses a simple data structure. All data should be placed in the `data/` directory with the following structure:

```
data/
├── atlas/                          # Brain atlas files
│   ├── atlas_info_KimRef_FPbasedLabel_v4.0_brain_with_size_with_curated_with_cleaned_acronyms.csv
│   ├── Kim_ref_adult_FP-label_v4.0.tif
│   ├── Kim_ref_adult_FP-label_v2.9_contour_map.tif
│   └── templates/                  # Atlas templates for registration
│       ├── template_25_coronal.tif
│       ├── Kim_ref_adult_v1_brain.tif
│       └── Kim_ref_adult_FP-label_v2.9.tif
├── lsms/                          # LSMS experimental data
│   └── Opioid/
├── opioid_cfos/                   # Opioid cFos data
│   ├── result/                    # Analysis results
│   ├── heatmap_array/             # Heatmap data
│   └── gene_expression.zarr       # Gene expression data
├── allen_alignment/               # Allen Brain alignment data
│   ├── download/                  # Downloaded Allen data
│   └── summarized_data/           # Summarized alignment results
│       └── gene_df.csv
├── spatial_clustering_results/    # Spatial clustering analysis
│   └── 2025_04_02-12_34/
├── fentanyl/                      # Fentanyl experimental data
└── results/                       # General output results
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Ishii-Morphine-2025
   ```

2. **Create the data directory:**
   ```bash
   mkdir -p data
   ```

3. **Place your data files:**
   - Copy your atlas files to `data/atlas/`
   - Copy your experimental data to the appropriate subdirectories
   - Ensure file names match those used in the notebooks

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Notebooks

The notebooks in the `notebooks/` directory have been updated to use relative paths:

- `Opioid_Figure 1.ipynb` - Main analysis notebook
- `Opioid_Figure 2.ipynb` - Additional analysis
- `Opioid_Figure 3.ipynb` - Further analysis
- `Opioid_Figure 4.ipynb` - Analysis continuation
- `Opioid_Figure 5.ipynb` - Extended analysis
- `Opioid_Figure 6.ipynb` - Additional figures
- `Opioid_Figure 7.ipynb` - Gene expression analysis
- `Opioid_Figure 8.ipynb` - Final analysis
- `update_labels.ipynb` - Label processing
- `active_sunburst.py` - Visualization utilities
- `allen_merfish_kim_voxelization.py` - Allen Brain data processing
- `TreeBH.py` - Statistical analysis
- `create_mask_for_region.py` - Region masking utilities

## Usage

1. Ensure your data is properly organized in the `data/` directory
2. Open and run the notebooks in order
3. Results will be saved to the appropriate output directories

## Dependencies

Key dependencies include:
- pandas
- numpy
- matplotlib
- seaborn
- tifffile
- ants
- zarr
- plotly
- networkx
- statsmodels
- scipy

See `requirements.txt` for the complete list.

## License

[Add your license information here]