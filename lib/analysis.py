import numpy as np
import pandas as pd
import shapely.geometry
import shapely.geometry.polygon
import shapely.geometry.multilinestring
import shapely.ops

import matplotlib.figure
import matplotlib.patches
from matplotlib.path import Path
import matplotlib.transforms

def collapsemax(group, collapsecolumn='Lambda'):
    """Collapse to maximum along column""" 
    index = group[collapsecolumn].idxmax()
    return pd.Series({column : group[column].loc[index] for column in group.columns})

def printunique(df, nunique=1, returnnonunique=False):
    if returnnonunique:
        nonuniquecolumns = []
    for key in sorted(df.columns):
        uniques = df[key].unique()
        if len(uniques) == 1:
            print('{0}: {1}'.format(key, uniques[0]))
        elif len(uniques) <= nunique:
            print('{0}: {1}'.format(key, '; '.join(str(s) for s in uniques)))
        elif returnnonunique:
            nonuniquecolumns.append(key)
    if returnnonunique:
        return nonuniquecolumns

def intelligent_describe(df, **kwargs): 
    ukwargs = dict(returnnonunique=kwargs.get('returnnonunique', True),
                   nunique=kwargs.get('nunique', 1))
    print('-----------------------------------------------------')
    print('values of columns with no more than {0} unique entries'.format(ukwargs['nunique']))
    print('')
    nonuniquecolumns = printunique(df, **ukwargs)
    print('-----------------------------------------------------')
    print('summary statistics of other columns')
    print('')
    dkwargs = dict(include=kwargs.get('include', None),
                   exclude=kwargs.get('exclude', None),
                   percentiles=kwargs.get('percentiles', None))
    print(df[nonuniquecolumns].describe(**dkwargs))
    print('-----------------------------------------------------')

def flatten(l):
    return [item for sublist in l for item in sublist]

def loadnpz(filename):
    f = np.load(filename)
    df = pd.DataFrame(f['data'], columns=f['columns'])
    df = df.apply(pd.to_numeric, errors='ignore')
    df.sort_values(list(df.columns), inplace=True)
    return df

def shapely_contour(df, level=0.0, extrapolate=0.05):
    """Return contour as a shapely object."""
    ax = matplotlib.figure.Figure().add_subplot(111)
    Xi, Yi = np.meshgrid(df.columns.values, df.index.values)
    cs = ax.contourf(Xi, Yi, df.values, levels=[level, 10.0])
#    cs = ax.contourf(df, levels=[0.0, 10.0],
#                  extent=(min(df.columns), max(df.columns), min(df.index), max(df.index)))
    ps = cs.collections[0].get_paths()
    # extrapolate along x axis down to 0 and up to 1
    polygons = []
    for p in ps:
        v = p.vertices
        v[v[:, 0] <= extrapolate, 0] = 0.0
        v[v[:, 0] >= 1-extrapolate, 0] = 1.0
        polygons.append(shapely.geometry.Polygon(v))
    return shapely.ops.cascaded_union(polygons)
#    return shapely.ops.cascaded_union([shapely.geometry.Polygon(p.vertices) for p in ps])

def polygon_from_boundary(xs, ys, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, xtol=0.0):
    """Polygon within box left of boundary given by (xs, ys)
    
    xs, ys: coordinates of boundary (ys ordered increasingly)
    """

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    xs[xs > xmax-xtol] = xmax
    xs[xs < xmin+xtol] = xmin

    index = -1
    while xs[index] == xmin:
        index -= 1
    if index < -2:
        xs, ys = xs[:index+2], ys[:index+2]
    vertices = zip(xs, ys)
    if len(xs) == 1:
        vertices.append((xs[0], ymax))
        vertices.append((xmin, ymax))
    elif xs[-1] >= xmax-xtol:
        if xs[-1] < xmax:
            vertices.append((xmax, ys[-1]))
        if ys[-1] < ymax:
            vertices.append((xmax, ymax))
        vertices.append((xmin, ymax))
    elif xs[-1] > xmin:
        vertices.append((xmin, ys[-1]))
    if (xs[0] > xmin) or (ys[0] > ymin):
        vertices.append((xmin, ymin)) 
    if ys[0] > ymin:
        vertices.append((xs[0], ymin)) 

    vertices = np.asarray(vertices)
    return vertices

def plot_interior_boundary(ax, shape, ylimmin=0.0, ylimmax=1.0, eps=1e-9, **kwargs):
    """plot only part of boundary that is in the interior"""
    verts = np.asarray(shape.exterior.coords)
    lines = []
    for i in range(len(verts)-1):
        if not ((verts[i, 0]==0.0 and verts[i+1, 0]==0.0) or
            (verts[i, 1]<=ylimmin+eps and verts[i+1, 1]<=ylimmin+eps) or
            (verts[i, 0]==1.0 and verts[i+1, 0]==1.0) or
            (verts[i, 1]>=ylimmax-eps and verts[i+1, 1]>=ylimmax-eps)):
            lines.append(shapely.geometry.LineString(verts[i:i+2]))
    lines = shapely_to_mpl(shapely.ops.linemerge(lines), **kwargs)
    try:
        ax.add_artist(lines)
    except:
        ax.add_collection(lines)

def cascaded_intersection(objs):
    out = objs[0]
    for obj in objs:
        out = out.intersection(obj)
    return out

def mpl_to_shapely(mpl, **kwargs):
    if isinstance(mpl, matplotlib.transforms.Bbox):
        return shapely.geometry.box(mpl.x0, mpl.y0, mpl.x0+mpl.width, mpl.y0+mpl.height)

def shapely_to_mpl(shape, **kwargs):
    """Convert shapely to mpl Polygon.
        
        If PolygonCollection try instead to convert largest polygon.
    """
    if isinstance(shape, shapely.geometry.polygon.Polygon):
        # return mpl Polygon if possible
        if isinstance(shape.boundary, shapely.geometry.linestring.LineString):
            return matplotlib.patches.Polygon(zip(*shape.exterior.xy),
                                      **kwargs)
        elif isinstance(shape.boundary, shapely.geometry.multilinestring.MultiLineString):
            verts = []
            codes = []
            for geom in shape.boundary.geoms:
                nverts = zip(*geom.xy)
                ncodes = [Path.MOVETO]
                ncodes.extend([Path.LINETO for i in range(len(nverts)-2)])
                ncodes.append(Path.CLOSEPOLY)
                verts.extend(nverts) 
                codes.extend(ncodes)
            path = Path(verts, codes)
            return matplotlib.patches.PathPatch(path, **kwargs)
    elif isinstance(shape, shapely.geometry.multipolygon.MultiPolygon):
        # MultiPolygon instead of Polygon --> convert largest polygon in collection
        polygon = sorted(shape.geoms, key=lambda x: x.area)[-1]
        return matplotlib.patches.Polygon(zip(*polygon.exterior.xy),
                                      **kwargs)
    elif isinstance(shape, shapely.geometry.multilinestring.MultiLineString):
        lines = [zip(*line.xy) for line in shape.geoms]
        return matplotlib.collections.LineCollection(lines,
                                                     **kwargs)
    elif isinstance(shape, shapely.geometry.linestring.LineString):
        return matplotlib.lines.Line2D(*shape.xy, **kwargs)

