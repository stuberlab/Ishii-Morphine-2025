# Ishii-Morphine-2025

This repository contains analysis notebooks and scripts for Ishii et al-2025.

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

Helper functions
- `active_sunburst.py` - Visualization utilities
- `create_mask_for_region.py` - Region masking utilities

For BRANCH testing check the following repository
- TreeBH:
- BRANCH test: a wrapper for TreeBH and associated functions to better suit whole-brain datasets


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
