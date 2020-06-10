# Limbal and Basal Stem Cell Counting
By: Samir Akre and Henry Zheng   
_Python Version 3.7.6_

## Purpose
The goal of this repository is to aide in the cell counting of images taken on cultered limbal stem cells (LSCs) and Basal Epithelial Cells (BECs) as part of a project for the BE224B Course at UCLA.

## Usage:
### Calculating Cell Density (full pipeline)
1. Configure python 3.7.X environment with required packages from `requirements.txt`
2. Run the `calculate_cell_density.py` file
```bash
python calculate_cell_density.py [path_to_input_image] \
    -t [cell_type (can be either BEC or LSC)]
```

- Set the `-s` flag to show tile segmentation results
- If using custom configuration, modify `config/config.json` to change file paths and `config/optimal_tile_params.json` to change how tiles are segmented


### Just Segmenting Tiles
1. Configure python 3.7.X environment with required packages from `requirements.txt`
2. Run the `segment_tiles.py` file
```bash
python segment_tiles.py [path_to_input_image] \
    -t [cell_type (can be either BEC or LSC)] \
    -o [path_to_output_directory]
```
- Set the `-s` flag to show tile segmentation results
- If using custom configuration, modify `config/config.json` to change file paths and `config/optimal_tile_params.json` to change how tiles are segmented


## Directory Structure

### Configuration files
- __config/__
    - __config.json__: File to guide processing of images and configure for different machines, example version in repository. Modify to fit local machine.
    - __optimal_tile_params.json__: File used to set best parameters for searching for tiles used in cell counting. Already optimized. Modify if desired.
    - __regressiondata.json__: Parameters for a pre-trained linear regression model. Used in conjunction with predictDensity. Retraining on a new training dataset can be performed using trainModel. 
    - __pcaparams.csv__: Parameters for PCA transformation. Used in conjunction with predictDensity. Retraining on a new training dataset can be performed using trainModel. 

### Python Files
- __calculate_cell_density.py__: File used to calculate cell density from LSC or BEC images
- __segment_tiles.py__: File containing class used to return best tiles for cell counting from an image. Can be run from command line.
- __src/__
    - __load_data.py__: File containing helper functions to load data as given reproducibly
    - __format_axis.py__: File containing functions used to make figures pretty
    - __basal_pipe.py__: File containing end-to-end operation of generating basal stem cell density predictions from provided whole images

### Python Notebooks
- __notebooks/__
    - __find_tiles_basal_cell_example.ipynb__: Notebook to serve as an example of finding optimal tiles on basal epithelial cell images using the TileSegmenter from `segment_tiles.py`
    - __find_tiles_basal_cell_example.ipynb__: Notebook to serve as an example of finding optimal tiles on limbal stem cell slide images using the TileSegmenter from `segment_tiles.py`
    - __get_LSC_tile_params.ipynb__: Performs a grid search over paramaters for tile segmentation and logs GLCM properties of resultant tiles for comparison to hand selected tiles
    - __count_cells_bcd.ipynb__: Attempts to count BEC cells using Otsu thresholding



### Folders
- __lsc_gridsearch_tile_glcm/__: Directory containing tables which document the GLCM properties of tiles selected with different parameters fed to TileSegmenter from `segment_tiles.py`. Generated in notebook `get_LSC_tile_params.ipynb`
