import functools
import numpy as np
import evolimmune
import scipydirect
import noisyopt
from noisyopt import OptimizeResult
from evolimmune import from_api, to_api
from misc import minus


def wrap_cutpienv(func):
    def wrapped(x, *args, **kwargs):
        pienv = x
        lambda_, mus, cup, aenv, niter, nburnin = args
        return func(lambda_, mus, cup, aenv, pienv, niter, nburnin, **kwargs)
    return wrapped


def cuts(coptkwargs=dict(), moptkwargs=dict(), ioptkwargs=dict(), poptkwargs=dict()):
    ccutpienv = wrap_cutpienv(functools.partial(cLambdaopt, optkwargs=coptkwargs))
    mcutpienv = wrap_cutpienv(functools.partial(mLambdaopt, optkwargs=moptkwargs))
    icutpienv = wrap_cutpienv(functools.partial(iLambdaopt, optkwargs=ioptkwargs))
    pcutpienv = wrap_cutpienv(functools.partial(pLambdaopt, optkwargs=poptkwargs))
    acutpienv = wrap_cutpienv(aLambdaopt)
    ocutpienv = wrap_cutpienv(oLambdaopt)
    cuts = dict(c=ccutpienv, m=mcutpienv, p=pcutpienv, a=acutpienv, o=ocutpienv, i=icutpienv)
    return cuts

def twostage_optimization(func, args, kwargs):
    res1 = scipydirect.minimize(func, args=args, **kwargs)
    return noisyopt.minimize(func, res1.x, args=args, **kwargs)


## CRISPR

def cLambda(x, *args, **kwargs):
    q, pup = x
    epsilon = 0
    p = 0
    return evolimmune.Lambda_pq((p, q, epsilon, pup), *args, **kwargs)

@noisyopt.memoized
def copt(args, optkwargs=dict()):
    return twostage_optimization(minus(cLambda), args, optkwargs)


def cLambdaopt(*args, **kwargs):
    optkwargs = kwargs.pop('optkwargs', dict())
    res = copt(args, optkwargs=optkwargs)
    return cLambda(res.x, *args, **kwargs)

def cagent(xinit, *args, **kwargs):
    aenv, pienv, xi = args
    ainit, pupinit = xinit
    epsiloninit = 0
    piinit = 0
    return evolimmune.agentbasedsim_evol(aenv, pienv, xi,
                                        ainit=ainit, piinit=piinit, pupinit=pupinit, epsiloninit=epsiloninit,
                                        evolvep=False,
                                        evolveq=True,
                                        evolvepup=True,
                                        evolveepsilon=False,
                                        **kwargs)

def cLambda_agent(xinit, *args, **kwargs):
    nburnin = kwargs.pop('nburnin', 1)
    return evolimmune.zstogrowthrate(cagent(xinit, *args, **kwargs)[0], nburnin=nburnin)

@noisyopt.memoized
def copt_agent(args, agentargs, optkwargs=dict(), agentoptkwargs=dict()):
    res = copt(args, optkwargs=optkwargs) 
    nburnin = agentoptkwargs.pop('nburnin', 1)
    zs, pis, as_, pups, epsilons = cagent(res.x, *agentargs, **agentoptkwargs)
    return OptimizeResult(x=(as_[-1], pups[-1]), fun=evolimmune.zstogrowthrate(zs, nburnin=nburnin))

def cLambdaopt_agent(*args, **kwargs):
    optkwargs = kwargs.pop('optkwargs', dict())
    res = copt_agent(args, optkwargs=optkwargs)
    return cLambda_agent(res.x, *args, **kwargs)

## mixed

def mLambda(x, *args, **kwargs):
    p, q, pup = x
    epsilon = 0
    return evolimmune.Lambda_pq((p, q, epsilon, pup), *args, **kwargs)

@noisyopt.memoized
def mopt(args, optkwargs=dict()):
    return twostage_optimization(minus(mLambda), args, optkwargs)

def mLambdaopt(*args, **kwargs):
    optkwargs = kwargs.pop('optkwargs', dict())
    res = mopt(args, optkwargs=optkwargs)
    return mLambda(res.x, *args, **kwargs)

def magent(xinit, *args, **kwargs):
    aenv, pienv, xi = args
    ainit, piinit, pupinit = xinit
    epsiloninit = 0.0
    return evolimmune.agentbasedsim_evol(aenv, pienv, xi,
                                        ainit=ainit, piinit=piinit, pupinit=pupinit, epsiloninit=epsiloninit,
                                        evolvep=True,
                                        evolveq=True,
                                        evolvepup=True,
                                        evolveepsilon=False,
                                        **kwargs)

## innate

def iLambda(x, *args, **kwargs):
    p, q = x
    pup = 0.0
    epsilon = 0.0
    return evolimmune.Lambda_pq((p, q, epsilon, pup), *args, **kwargs)

@noisyopt.memoized
def iopt(args, optkwargs=dict()):
    return twostage_optimization(minus(iLambda), args, optkwargs)

def iLambdaopt(*args, **kwargs):
    optkwargs = kwargs.pop('optkwargs', dict())
    res = iopt(args, optkwargs=optkwargs)
    return iLambda(res.x, *args, **kwargs)

## proto-adaptive

def pLambda(x, *args, **kwargs):
    epsilon = x
    p = 1.0
    q = 0.0
    pup = 0.0
    return evolimmune.Lambda_pq((p, q, epsilon, pup), *args, **kwargs)

def pagent(xinit, *args, **kwargs):
    aenv, pienv, xi = args
    epsiloninit = xinit
    piinit = 1.0
    pupinit = 0.0
    return evolimmune.agentbasedsim_evol(aenv, pienv, xi,
                                        ainit=ainit, piinit=piinit, pupinit=pupinit, epsiloninit=epsiloninit,
                                        evolvep=False,
                                        evolveq=False,
                                        evolvepup=False,
                                        evolveepsilon=True,
                                        **kwargs)

@noisyopt.memoized
def popt(args, optkwargs=dict()):
    return twostage_optimization(minus(pLambda), args, optkwargs)

def pLambdaopt(*args, **kwargs):
    optkwargs = kwargs.pop('optkwargs', dict())
    res = popt(args, optkwargs=optkwargs)
    return pLambda(res.x, *args, **kwargs)

## one

def oLambdaopt(*args, **kwargs):
    lambda_, mus, cup, aenv, pienv, niter, nburnin, = args
    return -evolimmune.mus_from_str(mus)(0)[0]
    
## adaptive
def aLambda(*args, **kwargs):
    epsilon = 1.0
    p = 1.0
    q = 0.0
    pup = 0.0
    return evolimmune.Lambda_pq((p, q, epsilon, pup), *args, **kwargs)

def aLambdaopt(*args, **kwargs):
    return aLambda(*args, **kwargs)
