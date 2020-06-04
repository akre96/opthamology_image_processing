""" Get best tiles to use for cell density counting
Usage:
    Import TileSegmenter class. Use dataloader from load_data.py to
    load optimal tile selection parameters for the type of image being
    analyzed.

    Alternatively this file can be called via command line. Input image
    is segmented with parameters according to input cell type. Output
    tiles are saved to specified directory.

# Example script call:
python segment_tiles.py Mild/12-od-2/-33.jpg \
    -t BEC \
    -o test_output \
    -p optimal_tile_params.json \



# Example loading Limbal Stem cell parameters:
import load_data as ld
import segment_tiles as st

tile_select_params = ld.get_tile_select_params()['LSC']
segmenter = st.TileSegmenter(
    **tile_select_params,
    glcm_n_samples=500,
    plot_patches=True,
    show_tqdm=True,
)

Author: Samir Akre
"""

from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Any
from tqdm import tqdm
from skimage.feature import greycomatrix, greycoprops, canny
from skimage.filters import rank
from skimage.morphology import dilation, disk
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib import patches as Patch
import cv2
import argparse
import load_data as ld


class TileSegmenter():
    def __init__(
        self,
        tile_size: int,
        n_tiles: int = 5,
        min_tile_dist: float = 200,
        glcm_n_samples: int = 5000,
        edge_weight: float = -5.0,
        contrast_weight: float = 1.5,
        energy_weight: float = 1.0,
        canny_sigma: float = 2.0,
        plot_patches: bool = False,
        rand_seed: int = 0,
        show_tqdm: bool = True,
    ):
        """ Initialize Tile Segmenter

        Arguments:
            tile_size {int} -- Size of tile, tiles are
                size (tile_size, tile_size)

        Keyword Arguments:
            n_tiles {int} -- Number of top tiles to return (default: {5})
            min_tile_dist {float} -- Minimum euclidean distance between
                centroid of two tiles (default: {200})
            glcm_n_samples {int} -- Number of points to sample and calculate
                properties as. Kind of like a resolution. (default: {5000})
            edge_weight {float} -- Amount to weigh edges detected in selecting
                optimal tiles (default: {-5.0})
            contrast_weight {float} -- Amount to weigh contrast GLCM property
                in selecting optimal tiles (default: {1.5})
            energy_weight {float} -- Amount to weigh energy GLCM property in
                selecting optimal tiles (default: {1.0})
            canny_sigma {float} -- Sigma value for canny filter used in edge
                detection (default: {2.0})
            plot_patches {bool} -- Set True to visualize results in matplotlib
                (default: {False})
            rand_seed {int} -- Random seed for point selection (default: {0})
            show_tqdm {bool} -- Set False to hide progressbars for internal
                processes (default: {True})
        """
        self.tile_size = tile_size
        self.half_ts = int(round(tile_size/2))
        self.n_tiles = n_tiles
        self.min_tile_dist = min_tile_dist
        self.glcm_n_samples = glcm_n_samples
        self.edge_weight = edge_weight
        self.contrast_weight = contrast_weight
        self.energy_weight = energy_weight
        self.canny_sigma = canny_sigma
        self.plot_patches = plot_patches
        self.show_tqdm = show_tqdm
        self.rand_seed = rand_seed
        self.xys: Any = None

    def segment_tiles(
        self,
        img,
    ) -> List:
        """ Find best tiles for cell density counting

        Arguments:
            img -- input image
            n_tiles {int} -- number of top tiles to return
            tile_size {int} -- height/width of tiles to return

        Keyword Arguments:
            min_tile_dist {float} -- minimum distance between two tiles center
                (default: {200})
            glcm_n_samples {int} -- number of points to sample when calculating
                glcm features (default: {5000})
            edge_weight {float} -- amount to weight blobs found
                (default: {-5.0})
            contrast_weight {float} -- amount to weight contrast
                (default: {1.5})
            energy_weight {float} -- amount to weight energy (default: {1.0})
            canny_sigma {float} -- Sigma parameter of canny filter for edge
                detection penalization (default: {2.0})
            plot_patches {bool} -- set true to plot results (default: {False})

        Returns:
            List -- A list of `n_tiles` images of size (tile_size, tile_size)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Calculate gray level features for a subset of tiles
        [contrast_mat, energy_mat], xys = self.get_greycoprop_matrices(
            gray,
            props=['contrast', 'energy'],
        )

        # Find edges using canny filter
        edge_mat = self.calc_edge_penalty(
            gray,
            dilate_disk_size=5,
            mean_disk_size=25
        )

        # Normalize values betweeo 0 and 1
        norm_contrast = self.min_max_normalize(contrast_mat)
        norm_energy = self.min_max_normalize(energy_mat)
        norm_edge = self.min_max_normalize(edge_mat)

        # Sum features based on weights
        sum_mat = self.edge_weight * norm_edge \
            + self.contrast_weight * norm_contrast \
            + self.energy_weight * norm_energy

        # Convolution over summed features to get attention map
        conv = self.quick_conv_sum(
            sum_mat,
            xys,
            func=np.mean,
        )

        # Remove edge areas from attention map
        conv = self.border_to_min(conv)

        # Get n_tiles top tiles
        centers = self.get_top_xy(conv)
        patches = self.get_patches(img, centers)

        # Plot to view results
        if self.plot_patches:
            print('Plotting output')
            fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
            cycle = ['red', 'green', 'blue', 'yellow', 'purple']
            ec = [cycle[i % len(cycle)] for i in range(self.n_tiles)]

            ax = axes[0]
            ax.set_title('Attention Map', fontsize=20)
            ax.imshow(conv)
            for i, [x, y] in enumerate(centers):
                rect = Patch.Rectangle(
                    (x-self.half_ts, y-self.half_ts),
                    self.tile_size,
                    self.tile_size,
                    linewidth=2,
                    edgecolor=ec[i],
                    facecolor='none'
                )
                ax.add_patch(rect)

            ax = axes[1]
            ax.set_title('Original Image', fontsize=20)
            ax.imshow(img, cmap='gray')
            for i, [x, y] in enumerate(centers):
                rect = Patch.Rectangle(
                    (x-self.half_ts, y-self.half_ts),
                    self.tile_size,
                    self.tile_size,
                    linewidth=2,
                    edgecolor=ec[i],
                    facecolor='none'
                )
                ax.add_patch(rect)

            if self.n_tiles >= 5:
                cols = 5
            else:
                cols = self.n_tiles
            rows = int(np.ceil(self.n_tiles/5))
            fig, axes = plt.subplots(
                ncols=cols,
                nrows=rows,
                figsize=(cols*3, 4*rows)
            )
            for i, p in enumerate(patches):
                axes.flatten()[i].imshow(p, cmap='gray')
                axes.flatten()[i].set_title(
                    'Rank: ' + str(i) + ' ' + ec[i],
                    fontsize=15
                )

            plt.show()

        return patches

    def calc_edge_penalty(
        self,
        img,
        dilate_disk_size: int = 5,
        mean_disk_size: int = 25,
    ) -> np.array:
        """ Uses Canny filter to make penalty for edges
        Used to down weight areas with splotches

        Arguments:
            img {[type]} -- input image

        Keyword Arguments:
            sigma {int} --  sigma value for canny filter (default: {2})
            dilate_disk_size {int} -- dilation size (default: {5})
            mean_disk_size {int} -- mean filter size (default: {25})

        Returns:
            np.array -- array used to penalize edge areas
        """
        dilated = dilation(
            canny(img, sigma=self.canny_sigma),
            disk(dilate_disk_size)
        )
        smooth_edge = self.min_max_normalize(
            rank.mean(
                img_as_ubyte(dilated), selem=disk(mean_disk_size)
            )
        )
        return smooth_edge

    def get_patches(self, img, centers: List) -> List:
        """ Get tiles from calcualted top centers
        patches are of size (tile_size, tile_size)

        Arguments:
            img  -- input image
            centers {List} -- list of [x,y] coordinates

        Returns:
            List -- list of patches
        """
        patches = []
        for x, y in centers:
            patches.append(
                img[
                    y-self.half_ts:y+self.half_ts,
                    x-self.half_ts:x+self.half_ts
                ]
            )
        return patches

    def quick_conv_sum(
        self,
        attention_matrix: np.array,
        xys: List = None,
        func=np.mean,
    ) -> np.array:
        """ Applies a function to tiles in an image simulating a convolution on
        Uses a subsample of points `xys` to act as centers. if `xys` not input,
        creates sample based on n_samples and rand_seed.

        Arguments:
            attention_matrix {np.array} -- [description]

        Keyword Arguments:
            xys {List} -- [description] (default: {None})
            func {function} -- Function to use for convolution

        Returns:
            np.array -- [description]
        """
        total_pixels = attention_matrix.shape[0] * attention_matrix.shape[1]
        radius = int(round(total_pixels / self.glcm_n_samples / 2**5))
        if self.xys is None:
            self.xys = self.get_xy_subsample_noedge(
                attention_matrix.shape,
            )
        conv = np.zeros(attention_matrix.shape)
        conv[:, :] = attention_matrix.min()
        iterator = tqdm(
            xys,
            total=self.glcm_n_samples,
            desc='Quick Convolution',
            disable=not self.show_tqdm
        )
        for x, y in iterator:
            val = func(
                attention_matrix[
                    x-self.half_ts:x+self.half_ts,
                    y-self.half_ts:y+self.half_ts
                ]
            )
            conv[x-radius: x+radius, y-radius:y+radius] = val
        return conv

    def border_to_min(
        self,
        attention_matrix: np.array,
    ) -> np.array:
        """ Sets tile overflow area to minimum of attention matrix

        Arguments:
            attention_matrix {np.array} -- image weighing attention to area

        Keyword Arguments:
            tile_size {int} -- size of tile (default: {200})

        Returns:
            np.array -- input with 0 set to edges
        """
        min_val = attention_matrix.min()
        attention_matrix[0:self.half_ts, :] = min_val
        attention_matrix[:, 0:self.half_ts] = min_val
        attention_matrix[attention_matrix.shape[0]-self.half_ts:, :] = min_val
        attention_matrix[:, attention_matrix.shape[1]-self.half_ts:] = min_val
        return attention_matrix

    def get_greycoprop_matrices(
        self,
        img: np.array,
        props: List[str] = ['contrast', 'energy'],
    ) -> Tuple[List, List]:
        """ Create matrices with values from GLCM greycooccurence matrix over
        tiles sub samples `n_samples` points and calculates GLCM values
        from `props` list for tiles of shape (tile_size, tile_size).
        Returns numpy matrices of same shape as image one per propert,
        in order of `props` list.
        Arguments:
            img {np.array} -- Input image, must be grayscale
            n_samples {int} -- Number of points to subsample from/tiles to use

        Keyword Arguments:
            props {List[str]} -- List of GLCM properties
                (default: {['contrast', 'energy']})
            tile_size {int} -- size of tile to calculate property over
                (default: {200})
            rand_seed {int} -- seeds random for reproducibility (default: {0})

        Returns:
            List[np.array] -- List of GLCM images. shape: len(props)
        """
        glcm_prop_matrices = [np.zeros(img.shape) for _ in props]
        xys = self.get_xy_subsample_noedge(img.shape)
        half_ts = self.half_ts
        iterator = tqdm(
            xys,
            total=self.glcm_n_samples,
            desc='Calculating grey level features',
            disable=not self.show_tqdm
        )
        for x, y in iterator:
            chunk = img[x-half_ts:x+half_ts, y-half_ts:y+half_ts]
            glcm = greycomatrix(chunk, [10], [0], symmetric=True, normed=True)
            for i, prop in enumerate(props):
                prop_val = greycoprops(glcm, prop)
                glcm_prop_matrices[i][x, y] = prop_val

        return glcm_prop_matrices, xys

    def get_xy_subsample_noedge(
        self,
        img_shape: Tuple,
    ) -> List:
        """ Get a list of random x,y coordinates in the image excluding edges
        A point is considered an edge if a tile centered at it would fall
        off the image.

        Arguments:
            img_shape {Tuple} -- (height, width), value from np.array.shape

        Returns:
            List -- list of (x,y) coordinates
        """
        l, w = img_shape

        xs, ys = np.meshgrid(np.arange(l), np.arange(w))
        xs = xs.flatten()
        ys = ys.flatten()

        half_ts = self.half_ts
        ind_0 = (xs > half_ts) * (xs < l-half_ts)
        ind_1 = (ys > half_ts) * (ys < w-half_ts)
        zero_ind = ind_0 * ind_1

        np.random.seed(self.rand_seed)
        xs_nz = xs[zero_ind]
        ys_nz = ys[zero_ind]

        subsamp = np.random.choice(
            xs_nz.shape[0],
            self.glcm_n_samples,
            replace=False
        )
        return list(zip(xs_nz[subsamp], ys_nz[subsamp]))

    @staticmethod
    def min_max_normalize(
        data: np.ndarray,
        min_val: float = 0.0,
        scale: float = 1.0
    ) -> np.ndarray:
        """ shift and scale data based on data range (min-max normalization)

        Arguments:
            data {np.ndarray} -- numpy float array

        Keyword Arguments:
            min_val {float} -- minimum value after shift (default: {0.0})
            scale {float} -- output range of data (default: {1.0})

        Returns:
            np.ndarray -- shifted and scaled data
        """
        normed = (data - data.min())/(data.max() - data.min())
        normed = (normed - min_val) * scale
        normed[np.isnan(normed)] = min_val
        return normed

    def get_top_xy(
        self,
        attention_matrix,
    ):
        """ Find the top N xy coordinates from attention matrix
        Returned coordinates used as center of rectangles to isolate tiles

        Arguments:
            attention_matrix {np.array} -- matrix of size image

        Returns:
            List -- list of x,y pairs
        """
        sorted_pixels = np.argsort(attention_matrix, axis=None)
        xys = []
        for p in sorted_pixels[::-1]:
            y, x = np.unravel_index(p, attention_matrix.shape)
            if len(xys) == 0:
                xys.append([x, y])
            else:
                dists = cdist(xys, [[x, y]])
                if not (dists < self.min_tile_dist).any():
                    xys.append([x, y])
            if len(xys) == self.n_tiles:
                break
        return xys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', type=str)
    parser.add_argument('-t', '--image-type', dest='image_type', type=str)
    parser.add_argument(
        '-o',
        '--output-dir',
        dest='output_dir',
        type=str,
        default='./'
    )
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

    params = ld.get_tile_select_params(args.param_file)

    if args.image_type not in params.keys():
        raise ValueError(
            '''
            Invalid image type --
            Image type must be in the tile parameter file keys
            '''
        )
    tile_params = params[args.image_type]
    tile_params['plot_patches'] = args.show_plots
    segmenter = TileSegmenter(**params[args.image_type])

    img = cv2.imread(args.input_image)
    ext = img_path.name.split('.')[-1]

    tiles = segmenter.segment_tiles(img)
    output_dir = Path(args.output_dir)
    for i, tile in enumerate(tiles):
        tile_name = '.'.join(img_path.name.split('.')[:-1])\
            + '_tile-' + str(i) + '.'\
            + ext

        tile_file = Path(
            output_dir,
            tile_name
        )
        cv2.imwrite(str(tile_file), tile)
