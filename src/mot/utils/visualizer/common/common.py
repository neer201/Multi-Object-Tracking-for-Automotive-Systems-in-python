import logging

import colorcet
import matplotlib
import matplotlib.pyplot as plt


logging.getLogger("matplotlib").setLevel(logging.WARNING)


def set_mpl_params():

    assert "matplotlib" in globals() or "matplotlib" in locals(), "matplotlib not imported"

    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.ERROR)
    plt.set_loglevel("error")

    titlesize = 20
    labelsize = 16
    legendsize = labelsize
    xticksize = 14
    yticksize = xticksize
    # matplotlib.use("svg")
    matplotlib.rcParams["legend.markerscale"] = 1.5  # the relative size of legend markers vs. original
    matplotlib.rcParams["legend.handletextpad"] = 0.5
    matplotlib.rcParams[
        "legend.labelspacing"
    ] = 0.4  # the vertical space between the legend entries in fraction of fontsize
    matplotlib.rcParams["legend.borderpad"] = 0.5  # border whitespace in fontsize units
    matplotlib.rcParams["font.size"] = 14
    # matplotlib.rcParams["font.family"] = "serif"
    # matplotlib.rcParams["font.serif"] = "Times"
    matplotlib.rcParams["axes.labelsize"] = labelsize
    matplotlib.rcParams["axes.titlesize"] = titlesize

    matplotlib.rc("xtick", labelsize=xticksize)
    matplotlib.rc("ytick", labelsize=yticksize)
    matplotlib.rc("legend", fontsize=legendsize)

    matplotlib.rc("font", **{"family": "serif"})
    # matplotlib.rc("text", usetex=True)


def create_figure(figsize=(5, 5), title=None, load_mpl_params=True, dpi=100, *args, **kwargs):
    if load_mpl_params:
        set_mpl_params()
    fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
    ax = plt.subplot(111, aspect="equal")
    ax.grid(which="both", linestyle="-", alpha=0.5)
    ax.set_title(label=title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    return fig, ax


def get_list_of_mcolors():
    # colors = mcolors.BASE_COLORS
    colors = colorcet.glasbey_dark
    return colors


def get_list_of_markers():
    markers = matplotlib.linesLine2D.filled_markers
    return markers
