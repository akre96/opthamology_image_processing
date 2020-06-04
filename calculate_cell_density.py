""" Calculate Cell density on Basal Epithelial Cell or Limbal Stem Cell images

## Command Line Usage:
python calculate_cell_density.py \
    [path to image] \
    -t [BSC | LSC]

## Options
To use custom parameters for tile segmentation, create a 
parameter JSON file in the style of 'optimal_tile_params.json'
and set `-p` option with path to that new params file.

Set `-s` flag to show plots of tile segmentation results

Author:
Samir Akre
"""

import argparse
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# Henry's parts
from radfuncs import radfeatures
import basal_pipeline as BEC
import centroidfuncs as LSC

# Samir's parts
import segment_tiles
import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Segment Tiles and Count Cells"
    )
    parser.add_argument('input_image', type=str)
    parser.add_argument('-t', '--image-type', dest='image_type', type=str)
    parser.add_argument(
        '-p',
        '--params',
        dest='param_file',
        type=str,
        default='./optimal_tile_params.json',
        help='Tile selection parameters JSON file'
    )
    parser.add_argument(
        '-s',
        '--show-plots',
        dest='show_plots',
        action='store_true',
        help='Set flag to show plots and attention map'
    )
    args = parser.parse_args()

    # Catch invalid image type
    if args.image_type not in ['LSC', 'BEC']:
        raise ValueError('Invalid Image Type, must be BEC or LSC')

    # Catch invalid parameter file
    p_f = Path(args.param_file)
    if not p_f.is_file():
        raise ValueError('Param File not found ' + str(p_f))

    # Catch invalid image path
    img_path = Path(args.input_image)
    if not img_path.is_file():
        raise ValueError('Image File not found ' + str(img_path))

    img = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)

    # Segment tiles for counting
    print(' -- Segmenting Tiles -- ')
    tile_seg_params = load_data.get_tile_select_params(
        args.param_file
    )[args.image_type]
    tile_seg_params['plot_patches'] = args.show_plots
    if args.show_plots:
        print('Plotting enabled... Close plots to continue to cell counting')
    segmenter = segment_tiles.TileSegmenter(
        **tile_seg_params,
    )
    tiles = segmenter.segment_tiles(img)

    print('\n', ' -- Counting Cells -- ')
    # Basal Epithelial Cell Pipeline
    if args.image_type == 'BEC':
        features = pd.DataFrame()
        for i, tile in enumerate(tiles):
            features[i] = radfeatures(img, args.image_type)
        features = features.transpose()
        cols = list(features.columns)
        cols[-1] = 'label'
        features.columns = cols
        density = BEC.predictDensity(features)
        print('\n', ' -- Density Report -- ')
        print('Density:', density.values.tolist()[0])

    # Basal Epithelial Cell Pipeline
    elif args.image_type == 'LSC':
        scale_factor = 147.05893166097917 ** 2
        tile_area = tile_seg_params['tile_size'] ** 2

        densities = []
        for tile in tiles:
            centroids = LSC.generateCentroid(tile, 20, 35)
            n_cells = centroids.shape[0]
            density = (1000 ** 2 * n_cells)/tile_area
            densities.append(density)
        
        print('\n', ' -- Density Report -- ')
        print('Densities (cells/mm^2):', densities)
        densities_arr = np.array(densities)
        print('Average Density (cells/mm^2):', densities_arr.mean())
        print('Std Density (cells/mm^2):', densities_arr.std())

    else:
        print('Image type, -t flag must be set to BEC or LSC')
else:
    print('Importing not supported, run with __name__ == "__main__"')
