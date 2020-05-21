""" Group of functions that make plots prettier

"""
from matplotlib.axis import Axis
import seaborn as sns


def despine_thicken_axes(
    ax: Axis,
    lw: float = 4,
    fontsize: float = 30,
    rotate_x: float = 0,
    rotate_y: float = 0,
    x_tick_fontsize: float = None,
    y_tick_fontsize: float = None,
):
    """ Despine axes, rotate x or y, thicken axes

    Arguments:
        ax -- matplotlib axis to modify

    Keyword Arguments:
        lw {float} -- line width for axes (default: {4})
        fontsize {float} --  fontsize for axes labels/ticks (default: {30})
        rotate_x {float} -- rotation in degrees for x-axis ticks (default: {0})
        rotate_y {float} -- rotation in degrees for y-axis ticks (default: {0})

    Returns:
        ax -- modified input axis
    """
    ax.xaxis.set_tick_params(width=lw, length=lw*2)
    ax.yaxis.set_tick_params(width=lw, length=lw*2)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)
    if x_tick_fontsize is None:
        x_tick_fontsize = fontsize
    if y_tick_fontsize is None:
        y_tick_fontsize = fontsize

    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    sns.despine()
    for var in ['x', 'y']:
        fs = y_tick_fontsize
        rot = rotate_y
        if var == 'x':
            fs = x_tick_fontsize
        rot = rotate_x
        ax.tick_params(axis=var, which='major', labelsize=fs)
        ax.tick_params(axis=var, which='minor', labelsize=fs*.8)
        ax.tick_params(axis=var, rotation=rot)

    return ax
