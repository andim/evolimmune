import sys, os
sys.path.append('../lib')
import numpy as np
import evolimmune
import noisyopt
import strategies
from misc import *

# model definitions
from params import *
# parameters of numerical algorithms
niter = 1e6
nburnin = 1e4
deltainit = 0.02
deltatol = 0.0005
feps = 1e-9
boundtol = 0.005
qboundtol = deltatol
xtol = 0.025
xtolbound = 0.01
alpha = 0.005
disp = True
# kwargs for the optimization algorithms
commonoptkwargs = dict(deltatol=deltatol, deltainit=deltainit, feps=feps,
                       errorcontrol=True, paired=True, alpha=alpha, disp=disp)
coptkwargs = dict(bounds=np.array([[0.0+qboundtol, 1.0], [0.0+boundtol, 1.0]]), maxf=400, **commonoptkwargs)
moptkwargs = dict(bounds=np.array([[0.0+boundtol, 1.0], [0.0+qboundtol, 1.0], [0.0+boundtol, 1.0]]), maxf=5000, **commonoptkwargs)
ioptkwargs = dict(bounds=np.array([[0.0+boundtol, 1.0], [0.0+qboundtol, 1.0]]), maxf=400, **commonoptkwargs)
poptkwargs = dict(bounds=np.array([[0.0+boundtol, 1.0-boundtol]]), maxf=20, **commonoptkwargs)
# script parameters
aenvs = evolimmune.from_tau(np.logspace(np.log10(0.09), np.log10(20.0), num=20, endpoint=True))
nbatch = 1
datadir = 'data/'

# define different boundaries to test and where to look for them
# use some aenv cutoffs to save unnecessary computations
paramscomb = expand_params([('ap', aenvs[[0, -1]]), # no change with aenv so it's enough to evaluate at extremities
                               ('ac', aenvs),
                               ('cm', filterarray(aenvs, 0.2, 1.0)),
                               ('mi', filterarray(aenvs, 0.01, 1.0)),
                               ('io', filterarray(aenvs, 0.01, 1.0)),
                               ('pm', aenvs),
                               ('pi', filterarray(aenvs, 0.0, 0.9)),
                               ('po', aenvs[[0, -1]]),
                               ('pc', aenvs),
                               ])

# define cut functions
cuts = strategies.cuts(coptkwargs=coptkwargs, moptkwargs=moptkwargs,
                       ioptkwargs=ioptkwargs, poptkwargs=poptkwargs)
# run simulations
if parametercheck(datadir, sys.argv, paramscomb, nbatch):
    njob = int(sys.argv[1])
    data = []
    for i in progressbar(range(nbatch)):
        n = (njob-1) * nbatch + i
        boundary, aenv = paramscomb[n]
        if disp:
            print 'boundary %s, aenv %s' % (boundary, aenv)
        bisect_kwargs = dict(xtol=xtol, errorcontrol=True, testkwargs=dict(alpha=alpha),
                             ascending=False, disp=disp)
        fargs = lambda_, mus, cup, aenv, niter, nburnin
        pienvbnd = noisyopt.bisect(noisyopt.DifferenceFunction(cuts[boundary[0]], cuts[boundary[1]],
                                                               fargs1=fargs, fargs2=fargs, paired=True),
                                   0, 1, **bisect_kwargs)
        data.append([boundary, lambda_, mus, cup, aenv, niter, nburnin, deltainit, deltatol,
                     np.log10(feps), boundtol, qboundtol, xtol, xtolbound, pienvbnd])
    columns = ['boundary', 'lambda_', 'mus', 'cup', 'aenv', 'niter', 'nburnin', 'deltainit',
               'deltatol', 'logfeps', 'boundtol', 'qboundtol', 'xtol', 'xtolbound', 'pienvbnd']
    np.savez_compressed(datadir + 'scan_phases_%g' % (njob), data=data, columns=columns)
