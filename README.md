# Limbal and Basal Stem Cell Counting
By: Samir Akre and Henry Zheng

## Purpose
The goal of this repository is to aide in the cell counting of images taken on cultered limbal stem cells (LSCs) and Basal Epithelial Cells (BECs) as part of a project for the BE224B Course at UCLA.

## Directory Structure

### JSON files
- __config.json__: File to guide processing of images and configure for different machines, example version in repository. Modify to fit local machine.
- __optimal_tile_params.json__: File used to set best parameters for searching for tiles used in cell counting. Already optimized. Modify if desired.

### Python Files
- __segment_tiles.py__: File containing class used to return best tiles for cell counting from an image
- __load_data.py__: File containing helper functions to load data as given reproducibly
- __format_axis.py__: File containing functions used to make figures pretty

### Python Notebooks
- __find_tiles_basal_cell_example.ipynb__: Notebook to serve as an example of finging optimal tiles on basal epithelial cell images using the TileSegmenter from `segment_tiles.py`
- __find_tiles_basal_cell_example.ipynb__: Notebook to serve as an example of finging optimal tiles on limbal stem cell slide images using the TileSegmenter from `segment_tiles.py`
- __get_LSC_tile_params.ipynb__: Performs a grid search over paramaters for tile segmentation and logs GLCM properties of resultant tiles for comparison to hand selected tiles


### Folders
- __lsc_gridsearch_tile_glcm/__: Directory containing tables which document the GLCM properties of tiles selected with different parameters fed to TileSegmenter from `segment_tiles.py`. Generated in notebook `get_LSC_tile_params.ipynb`
