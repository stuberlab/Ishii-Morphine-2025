#!/usr/bin/env python3
"""
Simple script to clear all notebook outputs to reduce file sizes.
"""

import json
import os
from pathlib import Path

def clear_notebook_outputs(notebook_path):
    """Clear all outputs from a notebook."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Clear outputs from all cells
        for cell in notebook['cells']:
            if 'outputs' in cell:
                cell['outputs'] = []
            if 'execution_count' in cell:
                cell['execution_count'] = None
        
        # Write back the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✓ Cleared outputs from {notebook_path.name}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {notebook_path.name}: {e}")
        return False

def main():
    """Clear outputs from all notebooks."""
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
    print("Clearing all outputs...")
    
    success_count = 0
    for notebook_path in notebook_files:
        if clear_notebook_outputs(notebook_path):
            success_count += 1
    
    print(f"\nCompleted! Cleared outputs from {success_count}/{len(notebook_files)} notebooks.")

if __name__ == "__main__":
    main() 