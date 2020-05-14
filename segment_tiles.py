""" Get best tiles to use for cell density counting
Usage: Import function `segment_tiles` and use in pipeline

Author: Samir Akre
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple
from tqdm import tqdm
from skimage.feature import greycomatrix, greycoprops, canny
from skimage.filters import rank
from skimage.morphology import dilation, disk
import matplotlib.pyplot as plt
from matplotlib import patches as Patch


def segment_tiles(
    img,
    n_tiles: int,
    tile_size: int,
    min_tile_dist: float = 200,
    glcm_n_samples: int = 5000,
    edge_weight: float = -5.0,
    contrast_weight: float = 1.5,
    energy_weight: float = 1.0,
    canny_sigma: float = 2.0,
    plot_patches: bool = False,
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
        edge_weight {float} -- amount to weight blobs found (default: {-5.0})
        contrast_weight {float} -- amount to weight contrast (default: {1.5})
        energy_weight {float} -- amount to weight energy (default: {1.0})
        canny_sigma {float} -- Sigma parameter of canny filter for edge
            detection penalization (default: {2.0})
        plot_patches {bool} -- set true to plot results (default: {False})

    Returns:
        List -- A list of `n_tiles` images of size (tile_size, tile_size)
    """
    # Calculate gray level features for a subset of tiles
    [contrast_mat, energy_mat], xys = get_greycoprop_matrices(
        img,
        glcm_n_samples,
        props=['contrast', 'energy'],
        tile_size=tile_size,
        rand_seed=0
    )

    # Find edges using canny filter
    edge_mat = calc_edge_penalty(
        img,
        sigma=canny_sigma,
        dilate_disk_size=5,
        mean_disk_size=25
    )

    # Normalize values betweeo 0 and 1
    norm_contrast = min_max_normalize(contrast_mat)
    norm_energy = min_max_normalize(energy_mat)
    norm_edge = min_max_normalize(edge_mat)

    # Sum features based on weights
    sum_mat = edge_weight * norm_edge \
        + contrast_weight * norm_contrast \
        + energy_weight * norm_energy

    # Convolution over summed features to get attention map        
    conv = quick_conv_sum(
        sum_mat,
        glcm_n_samples,
        xys,
        func=np.mean,
        tile_size=tile_size,
        rand_seed=0
    )

    # Remove edge areas from attention map
    conv = border_to_min(conv, tile_size)

    # Get n_tiles top tiles
    centers = get_top_xy(conv, n=n_tiles, min_dist=min_tile_dist)
    patches = get_patches(img, centers, tile_size)

    # Plot to view results
    if plot_patches:
        half_ts = int(round(tile_size/2))
        fig, axes = plt.subplots(ncols=2, figsize=(16, 8))
        cycle = ['red', 'green', 'blue', 'yellow', 'purple']
        ec = [cycle[i % len(cycle)] for i in range(n_tiles)]

        ax = axes[0]
        ax.set_title('Attention Map', fontsize=20)
        ax.imshow(conv)
        for i, [x, y] in enumerate(centers):
            rect = Patch.Rectangle(
                (x-half_ts, y-half_ts),
                tile_size,
                tile_size,
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
                (x-half_ts, y-half_ts),
                tile_size,
                tile_size,
                linewidth=2,
                edgecolor=ec[i],
                facecolor='none'
            )
            ax.add_patch(rect)

        fig, axes = plt.subplots(
            ncols=5,
            nrows=int(np.ceil(n_tiles/5)),
            figsize=(5*3, 4*int(np.ceil(n_tiles/5)))
        )
        for i, p in enumerate(patches):
            axes.flatten()[i].imshow(p, cmap='gray')
            axes.flatten()[i].set_title('Rank: ' + str(i) + ' ' + ec[i], fontsize=15)
        
        plt.show()

    return patches

def calc_edge_penalty(
    img,
    sigma=2,
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
    dilated = dilation(canny(img, sigma=sigma), disk(dilate_disk_size))
    smooth_edge = min_max_normalize(
        rank.mean(
            dilated, selem=disk(mean_disk_size)
        )
    )
    return smooth_edge

def get_patches(img, centers: List, tile_size: int = 200) -> List:
    """ Get tiles from calcualted top centers
    patches are of size (tile_size, tile_size)

    Arguments:
        img  -- input image
        centers {List} -- list of [x,y] coordinates

    Keyword Arguments:
        tile_size {int} -- size of tiles to return (default: {200})

    Returns:
        List -- list of patches
    """
    patches = []
    half_ts = int(round(tile_size/2))
    for x, y in centers:
        patches.append(img[y-half_ts:y+half_ts, x-half_ts:x+half_ts])
    return patches

def quick_conv_sum(
    attention_matrix: np.array,
    n_samples: int,
    xys: List = None,
    func=np.mean,
    tile_size: int = 200,
    rand_seed: int = 0,
) -> np.array:
    """ Applies a function to tiles in an image simulating a convolution on
    Uses a subsample of points `xys` to act as centers. if `xys` not input,
    creates sample based on n_samples and rand_seed.

    Arguments:
        attention_matrix {np.array} -- [description]
        n_samples {int} -- [description]

    Keyword Arguments:
        xys {List} -- [description] (default: {None})
        func {function} -- Function to use for convolution
        tile_size {int} -- [description] (default: {200})
        rand_seed {int} -- [description] (default: {0})

    Returns:
        np.array -- [description]
    """
    total_pixels = attention_matrix.shape[0] * attention_matrix.shape[1]
    radius = int(round(total_pixels / n_samples / 2**5))
    if xys is None:
        xys = get_xy_subsample_noedge(attention_matrix.shape, n_samples, tile_size, rand_seed)
    half_ts = int(round(tile_size/2))
    conv = np.zeros(attention_matrix.shape)
    conv[:, :] = attention_matrix.min()
    for x, y in tqdm(xys, total=n_samples, desc='Quick Convolution'):
        val = func(attention_matrix[x-half_ts:x+half_ts, y-half_ts:y+half_ts])
        conv[x-radius: x+radius, y-radius:y+radius] = val
    return conv

def border_to_min(
    attention_matrix: np.array,
    tile_size: int = 200,
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
    half_ts = int(round(tile_size/2))
    attention_matrix[0:half_ts, :] = min_val
    attention_matrix[:, 0:half_ts] = min_val
    attention_matrix[attention_matrix.shape[0]-half_ts:, :] = min_val
    attention_matrix[:, attention_matrix.shape[1]-half_ts:] = min_val
    return attention_matrix


def get_greycoprop_matrices(
    img: np.array,
    n_samples: int,
    props: List[str] = ['contrast', 'energy'],
    tile_size: int = 200,
    rand_seed: int = 0,
) -> List[np.array]:
    """ Create matrices with values from GLCM greycooccurence matrix over tiles
    sub samples `n_samples` points and calculates GLCM values from `props` list
    for tiles of shape (tile_size, tile_size). Returns numpy matrices of same
    shape as image one per propert, in order of `props` list.
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
    xys = get_xy_subsample_noedge(img.shape, n_samples, tile_size, rand_seed)
    half_ts = int(round(tile_size/2))
    for x, y in tqdm(xys, total=n_samples, desc='Calculating grey level features'):
        chunk = img[x-half_ts:x+half_ts, y-half_ts:y+half_ts]
        glcm = greycomatrix(chunk, [10], [0], symmetric=True, normed=True)
        for i, prop in enumerate(props):
            prop_val = greycoprops(glcm, prop)
            glcm_prop_matrices[i][x, y] = prop_val

    return glcm_prop_matrices, xys

def get_xy_subsample_noedge(
    img_shape: Tuple,
    n_samples: int,
    tile_size: int,
    rand_seed: int = 0,
) -> List:
    """ Get a list of random x,y coordinates in the image excluding edges
    A point is considered an edge if a tile centered at it would fall off the
    image.

    Arguments:
        img_shape {Tuple} -- (height, width), value from np.array.shape
        n_samples {int} -- number of points to sample
        tile_size {int} -- size of a tile to determine edges

    Keyword Arguments:
        rand_seed {int} --  seed for reproducibility (default: {0})

    Returns:
        List -- list of (x,y) coordinates
    """
    l, w = img_shape

    xs, ys = np.meshgrid(np.arange(l), np.arange(w))
    xs = xs.flatten()
    ys = ys.flatten()

    half_ts = int(round(tile_size/2))
    ind_0 = (xs > half_ts) * (xs < l-half_ts)
    ind_1 = (ys > half_ts) * (ys < w-half_ts)
    zero_ind = ind_0 * ind_1

    np.random.seed(rand_seed)
    xs_nz = xs[zero_ind]
    ys_nz = ys[zero_ind]

    subsamp = np.random.choice(xs_nz.shape[0], n_samples, replace=False)
    return list(zip(xs_nz[subsamp], ys_nz[subsamp]))


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
    return normed

def get_top_xy(attention_matrix, n=10, min_dist=100):
    """ Find the top N xy coordinates from attention matrix
    Returned coordinates used as center of rectangles to isolate tiles

    Arguments:
        attention_matrix {np.array} -- matrix of size image

    Keyword Arguments:
        n {int} --  number of cooridnates to return (default: {10})
        min_dist {int} --  minimum distance between coordinates
            (default: {100})

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
            if not (dists < min_dist).any():
                xys.append([x, y])
        if len(xys) == n:
            break
    return xys
