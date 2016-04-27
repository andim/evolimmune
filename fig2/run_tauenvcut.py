import os.path, sys
sys.path.append('../lib/')
import numpy as np
import evolimmune
from evolimmune import to_api, from_api, from_tau
from misc import *
import noisyopt
import scipydirect

# model definitions
from params import *
# parameters of numerical algorithms
niter = 1e6
nburnin = 1e4
# first stage numerical parameters
maxfs = 10000
# second stage numerical parameters
deltainit = 0.02
deltatols = 0.0005
alpha = 0.005
feps = 1e-9
# script parameters
aenvs = from_tau([12, 0.8])
pienvs = np.linspace(0.0, 1.0, 101)[1:-1]

nbatch = 1
disp = True
datadir = 'data/'

paramscomb = params_combination((aenvs, pienvs, maxfs, deltatols))
niter = int(niter)
nburnin = int(nburnin)
if parametercheck(datadir, sys.argv, paramscomb, nbatch):
    njob = int(sys.argv[1])
    data = []
    for i in progressbar(range(nbatch)):
        n = (njob-1) * nbatch + i
        aenv, pienv, maxf, deltatol = paramscomb[n]
        if disp:
            print paramscomb[n]
        args = lambda_, mus, cup, aenv, pienv, niter, nburnin
        bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        res1 = scipydirect.minimize(minus(evolimmune.Lambda_pq),
                   maxf=maxf, args=args, bounds=bounds, disp=disp)
        if disp:
            print 'results of first phase optimization', res1
        res2 = noisyopt.minimize(minus(evolimmune.Lambda_pq),
		   res1.x,
                   scaling=(1.0, 1.0, 5.0, 1.0),
                   args=args, bounds=bounds,
                   deltainit=deltainit,
                   deltatol=deltatol,
                   alpha=alpha,
                   feps=feps,
                   errorcontrol=True,
                   paired=True,
                   disp=disp)
        res2.x[res2.free] = np.nan
        p, q, epsilon, pup = res2.x
        if disp:
            print 'result', res2.x
        Lambdaopt = res2.fun
        Lambdaoptse = res2.funse
        data.append([lambda_, mus, cup, aenv, pienv, p, q, pup, epsilon, niter,
                     nburnin, maxf, deltainit, deltatol, alpha, np.log10(feps), Lambdaopt, Lambdaoptse])
    columns = ['lambda_', 'mus', 'cup', 'aenv', 'pienv', 'p', 'q', 'pup', 'epsilon',
               'niter', 'nburnin', 'maxf', 'deltainit', 'deltatol', 'alpha', 'logfeps',
               'Lambdaopt', 'Lambdaoptse']
    np.savez_compressed(datadir + 'scan_tauenvcut_%g' % (njob), data=data, columns=columns)
