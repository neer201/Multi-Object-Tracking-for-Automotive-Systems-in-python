import numpy as np


def autoscale(ax, axis="y", margin=10.0):
    """Autoscales the x or y axis of a given matplotlib ax object
    to fit the margins set by manually limits of the other axis,
    with margins in fraction of the width of the plot

    Defaults to current axes object if not specified.
    """

    newlow, newhigh = np.inf, -np.inf

    for artist in ax.collections + ax.lines:
        x, y = get_xy(artist)
        if axis == "y":
            setlim = ax.set_ylim
            lim = ax.get_xlim()
            fixed, dependent = x, y
        else:
            setlim = ax.set_xlim
            lim = ax.get_ylim()
            fixed, dependent = y, x

        low, high = calculate_new_limit(fixed, dependent, lim)
        newlow = int(low if low < newlow else newlow)
        newhigh = int(high if high > newhigh else newhigh)

        margin = margin * (newhigh - newlow)
        setlim(newlow - margin, newhigh + margin)
    return ax


def calculate_new_limit(fixed, dependent, limit):
    """Calculates the min/max of the dependent axis given
    a fixed axis with limits
    """
    if len(fixed) > 2:
        mask = (fixed > limit[0]) & (fixed < limit[1])
        window = dependent[mask]
        low, high = window.min(), window.max()
    else:
        low = dependent[0]
        high = dependent[-1]
        if low == 0.0 and high == 1.0:
            # This is a axhline in the autoscale direction
            low = np.inf
            high = -np.inf
    return low, high


def get_xy(artist):
    """Gets the xy coordinates of a given artist"""
    if "Collection" in str(artist):
        x, y = artist.get_offsets().T
    elif "Line" in str(artist):
        x, y = artist.get_xdata(), artist.get_ydata()
    else:
        raise ValueError("This type of object isn't implemented yet")
    return x, y
