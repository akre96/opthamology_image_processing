import argparse
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
from typing import Dict

from radfuncs import radfeatures
import basal_pipeline as BEC
import centroidfuncs as LSC
import segment_tiles
import load_data

def get_image_scale(img_path: Path) -> Dict:
    """ Get width, height in microns of image

    Args:
        img_path (Path): Path to tiff image

    Returns:
        Dict - image resolution properties
    """
    with Image.open(img_path) as img:
        meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.keys()}
    xres = meta_dict['XResolution']
    yres = meta_dict['YResolution']
    scalex = xres[0][0]/xres[0][1]
    scaley = yres[0][0]/yres[0][1]

    img_props = {
        'scaled_x': scalex,
        'scaled_y': scaley,
        'length': meta_dict['ImageLength'][0],
        'width': meta_dict['ImageWidth'][0],
    }
    return img_props


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    p_f = Path(args.param_file)
    if not p_f.is_file():
        raise ValueError('Param File not found ' + str(p_f))

    img_path = Path(args.input_image)
    if not img_path.is_file():
        raise ValueError('Image File not found ' + str(img_path))
    img = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)

    tile_seg_params = load_data.get_tile_select_params(
        args.param_file
    )[args.image_type]
    tile_seg_params['plot_patches'] = args.show_plots
    segmenter = segment_tiles.TileSegmenter(
        **tile_seg_params,
    )
    tiles = segmenter.segment_tiles(img)
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
        print('Image type, -t flag must be set to BEC')
