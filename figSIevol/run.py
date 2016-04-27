# script for running parameter scans of agent based simulation

import os.path, sys
sys.path.append('../lib/')
import numpy as np
from evolimmune import (from_tau, mus_from_str, cup_from_str,
                        agentbasedsim_evol, zstogrowthrate)
import cevolimmune
from misc import *

# general model parameters
lambdas = 3.0
muss = '1.0-2.0*epsilon/(1.0+epsilon), 1.0+0.8*epsilon'
cups = '0.1*pup+pup**2'

# finite population model parameters
Ls = 1
ninds = [50, 100, 1000]

aenvs = from_tau(np.logspace(np.log10(0.09), np.log10(11.0),  40, True))
pienvs = [0.3, 0.5, 0.7]

# numerical parameters
ngens = [100000]

# parameter evolution parameters
mutrates = lambda gen: 1e-2 * np.exp(-gen/1e4)
mutsizes = lambda gen: 0.25 * np.exp(-gen/1e4)

# script parameters
nbatch = 1
nruns = 50
datadir = 'data'

paramscomb = params_combination((Ls, lambdas, muss, cups, aenvs, pienvs, ninds, mutrates, mutsizes))

if parametercheck(datadir, sys.argv, paramscomb, nbatch):
    ngens = np.asarray([ngens]) if isinstance(ngens, int) else np.asarray(ngens)
    index = int(sys.argv[1])
    njob = (index-1) % len(paramscomb)
    data = []
    for run in progressbar(range(nbatch * nruns)):
        n = njob * nbatch + run // nruns
        L, lambda_, mus, cup, aenv, pienv, nind, mutrate, mutsize = paramscomb[n]
        print paramscomb[n]
        fmus = mus_from_str(mus)
        fcup = cup_from_str(cup)
        def xi(ind, env, epsilon, pup):
            mu1, mu2 = fmus(epsilon)
            return cevolimmune.xi_pa(ind, env, lambda_, mu1, mu2) * np.exp(-fcup(pup).sum(axis=1))
        # ngens is handled by running one long simulation and storing intermediate results
        zs, pis, as_, pups, epsilons = agentbasedsim_evol(aenv, pienv, xi,
                                                          L=L,
                                                          nind=nind,
                                                          ngens=ngens,
                                                          mutrate=mutrate,
                                                          mutsize=mutsize,
                                                          evolvep=True,
                                                          evolveq=True,
                                                          evolvepup=True,
                                                          evolveepsilon=True
                                                         )
        Lambdas = [zstogrowthrate(zs[(ngens[i-1] if i > 0 else 0):ngens[i]]) for i in range(len(ngens))]
        for j, ngen in enumerate(ngens):
            data.append([L, lambda_, mus, cup, aenv, pienv, nind, ngen, as_[j], pis[j], pups[j], epsilons[j], Lambdas[j]])
    columns = ['L', 'lambda_', 'mus', 'cup', 'aenv', 'pienv', 'nind', 'ngen', 'a', 'pi', 'pup', 'epsilon', 'Lambda']
    np.savez_compressed(datadir + 'scan_ind%g' % (index), data=data, columns=columns)
