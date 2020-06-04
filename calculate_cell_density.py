import argparse
import pandas as pd
import cv2

from radfuncs import radfeatures
import basal_pipeline as BEC
import segment_tiles
import load_data


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
        print('Density:', density.values.tolist()[0])
    else:
        print('Image type, -t flag must be set to BEC')
    

