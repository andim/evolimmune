import string, itertools
import numpy as np
import matplotlib.pyplot as plt

def heatmap(df, imshow=True, zlabel='z', ax=None, cax=None, cbarkwargs=dict(), cbarlabelkwargs=dict(), **kwargs):
    """Plot a heat-map of a pivoted data frame with automatic labeling of axes.

    imshow: if True use imshow, otherwise use pcolormesh (needed if afterwards nonlinear scaling applied to axes)
    
    """
    if ax is None:
        ax = plt.gcf().add_subplot(111)
    if imshow:
        im = ax.imshow(df, extent=(min(df.columns), max(df.columns), min(df.index), max(df.index)), aspect='auto', **kwargs)
    else:
        X, Y = np.meshgrid(df.columns, df.index)
        # automatic axis scaling does not work if there are nans
        defaultkwargs = dict(vmin=np.nanmin(df), vmax=np.nanmax(df))
        defaultkwargs.update(kwargs)
        im = ax.pcolormesh(X, Y, df, **defaultkwargs)
        #FIXME: Once color matplotlib colormesh is fixed (PR submitted) the following line should suffice
        #im = ax.pcolormesh(X, Y, df, **kwargs)
    if cax is None:
        cbar = plt.gcf().colorbar(im, ax=ax, **cbarkwargs)
    else:
        cbar = plt.gcf().colorbar(im, cax=cax, **cbarkwargs)
    if zlabel is not None:
        cbar.set_label(zlabel, **cbarlabelkwargs)
    # workaround for pdf/svg export for more smoothness
    # see matplotlib colorbar documentation
    cbar.solids.set_edgecolor("face")
    # lower limit
    ax.set_xlim(min(df.columns), max(df.columns))
    ax.set_ylim(min(df.index), max(df.index))
    ax.set_xlabel(df.columns.name)
    ax.set_ylabel(df.index.name)
    return im, cbar

def contour(df, levels=None, ax=None, clabel=True,
            label=None, clabelkwargs=dict(), **kwargs):
    """Plot a heat-map of a pivoted data frame with automatic labeling of axes."""
    if ax is None:
        ax = plt.gcf().add_subplot(111)

    X, Y = np.meshgrid(df.columns, df.index)
    cs = ax.contour(X, Y, df, levels=levels,
                    #extent=(min(df.columns), max(df.columns),
                    #        min(df.index), max(df.index)),
                    aspect='auto',
                    # color lines from default color cycle, if colors not defined and levels explicitely specified
                    colors=kwargs.pop('colors', [plt.gca()._get_lines.prop_cycler.next()['color'] for level in levels] if levels else None),
                    **kwargs)
    if clabel:
        thisclabelkwargs = dict(fmt='%1.1f')
        thisclabelkwargs.update(clabelkwargs)
        ax.clabel(cs, **thisclabelkwargs)
    # lower limit
    ax.set_xlim(min(df.columns), max(df.columns))
    ax.set_ylim(min(df.index), max(df.index))
    ax.set_xlabel(df.columns.name)
    ax.set_ylabel(df.index.name)
    if label:
        cs.collections[0].set_label(label)
    return cs 

def label_axes(fig_or_axes, labels=string.uppercase,
               labelstyle=r'{\sf \textbf{%s}}',
               xy=(-0.05, 0.95), xycoords='axes fraction', **kwargs):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure or Axes to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : Where to put the label units (len=2 tuple of floats)
    xycoords : loc relative to axes, figure, etc.
    kwargs : to be passed to annotate
    """
    # re-use labels rather than stop labeling
    labels = itertools.cycle(labels)
    axes = fig_or_axes.axes if isinstance(fig_or_axes, plt.Figure) else fig_or_axes
    for ax, label in zip(axes, labels):
        ax.annotate(labelstyle % label, xy=xy, xycoords=xycoords,
                    **kwargs)

def despine(ax, spines=['top', 'right']):
    if spines == 'all':
        spines = ['top', 'bottom', 'left', 'right']
    for spine in spines:
        ax.spines[spine].set_visible(False)

def jumpify(x, y, threshold=1.0):
    """Add extrapolated intermediate point at positions of jumps"""
    oldx, oldy = np.asarray(x), np.asarray(y)
    for ind in np.where(np.abs(np.diff(oldy)) > threshold)[0]:
        newx = list(oldx[:ind+1])
        midx = 0.5*(oldx[ind]+oldx[ind+1])
        newx.extend([midx, midx])
        newx.extend(oldx[ind+1:])
        newy = list(oldy[:ind+1])
        newy.extend([1.5*oldy[ind]-0.5*oldy[ind-1], 1.5*oldy[ind+1]-0.5*oldy[ind+2]])
        newy.extend(oldy[ind+1:])
        oldx, oldy = newx, newy
    return oldx, oldy

def box_from_text(text, pad=0.01):
    ax = text.axes
    renderer = ax.figure.canvas.get_renderer()
    bbox = text.get_window_extent(renderer)
    bbox_points = bbox.transformed(ax.transAxes.inverted()).get_points()
    bbox_points[0] -= pad
    bbox_points[1] += pad
    bbox_points = ax.transData.inverted().transform(ax.transAxes.transform(bbox_points))
    return bbox_points

def latexboldmultiline(text):
    """make a LaTeX multiline text bold in matplotlib"""
    return '\n'.join([r'{\bf %s}' % line for line in text.split()])

# presentation colors
cdarkprime = '#0E1E97'
cmediumprime = '#3246e0'
clightprime = '#e2e5fe'
cdarkcomp = '#9b7c0f'
cmediumcomp = '#f6c51c'
clightcomp = '#fef8e2'
cdarkgrey = '#303030'
cmediumgrey = '#888888'
clightgrey = '#e5e5e5'
cdarkthree = '#11990f'
cmediumthree = '#2dea2a'
clightthree = '#e4fce3'
cdarkfour = '#950e0e'
cmediumfour = '#eb2828'
clightfour = '#fce3e3'
