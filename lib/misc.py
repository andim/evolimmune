import os.path
import numbers
import itertools
import numpy as np

def filterarray(arr, low, high):
    return arr[(arr >= low) & (arr <= high)]

class DefaultIdentityDict(dict):
    def __missing__(self, key):
        return r'$%s$' % key

def minus(func):
    def minusFunc(*args, **kwargs):
        return -func(*args, **kwargs)
    return minusFunc

def progressbar(iterator):
    # if available add progress indicator
    try:
        import pyprind
        iterator = pyprind.prog_bar(iterator)
    except:
        pass
    return iterator

def parametercheck(datadir, argv, paramscomb, nbatch):
    if not os.path.exists(datadir):
        print 'datadir missing!'
        return False
    if not len(argv) > 1:
        if len(paramscomb) % nbatch != 0.0:
            print 'incompatible nbatch', len(paramscomb), nbatch
        print len(paramscomb) / nbatch
        return False
    return True

def params_combination(params):
    """Make a list of all combinations of the parameters."""
    # for itertools.product to work float entries have to be converted to 1-element lists
    params = [[p] if isinstance(p, numbers.Number) or isinstance(p, str) or hasattr(p, '__call__') else p for p in params]
    return list(itertools.product(*params))

def expand_params(inlist):
    """Expand a nested list to parameter combinations."""
    outlist = []
    for item in inlist:
        outlist.extend(params_combination(item))
    return outlist
