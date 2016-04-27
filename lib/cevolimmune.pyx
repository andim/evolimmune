# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, srand
cdef extern from "stdlib.h":
        int RAND_MAX
        int INT_MAX
from libc.math cimport exp, log


def sumlog(np.ndarray[np.double_t, ndim=1] data):
    # speed up calculation of sum of log by calculating log of product
    cdef double res = 0.0
    cdef double tmp = 1.0
    for i in range(data.shape[0]):
        tmp *= data[i]
        if not (1e-14 < tmp < 1e14):
            res += log(tmp)
            tmp = 1.0
    res += log(tmp)
    return res

def zrecursion(double lambda_, double mu,
           double p, double q, double alpha, double beta,
           double pup=0.0, double kup=0.0,
           mu2=None,
           int niter=1000, seed=None):
    """
    If mu2 is defined then mu will be the cost in the absence of pathogen (mu1).
    
    seed should be in [2, uint32max]
    """
    cdef double elambda = np.exp(-lambda_)
    cdef double emu1 = np.exp(-mu)
    cdef double emu2 = np.exp(-mu2)
    cdef double ekup = np.exp(-kup)
    cdef double dRAND_MAX = <double> RAND_MAX
    if seed is None:
        seed = np.random.randint(0, INT_MAX)
    srand(seed)
    # set x to 1 with probability pienv
    cdef int x = (alpha/(alpha+beta)) < (rand() / dRAND_MAX)
    cdef double r = 0.5
    cdef np.ndarray[np.double_t, ndim=1] zs = np.zeros(niter, dtype = np.double)
    cdef double z, prob
    cdef unsigned int i
    with nogil:
        for i in range(niter):
            prob = rand() / dRAND_MAX
            if x:
                x = prob > beta
            else:
                x = prob < alpha
            if x:
                if pup:
                    z = (1.0-r)*(pup*emu2*ekup+(1.0-pup)*elambda) + r*emu2
                    r = ((1.0-r)*(pup*emu2*ekup*(1.0-q)+p*(1.0-pup)*elambda) + r*(1.0-q)*emu2)/z
                else:
                    z = (1.0-r)*elambda + r*emu2
                    r = ((1.0-r)*p*elambda + r*(1.0-q)*emu2)/z
            else:
                z = (1.0-r) + r*emu1
                r = ((1.0-r)*p + r*(1.0-q)*emu1)/z
            zs[i] = z
    return zs

def Lambda_grad(double lambda_, double mu, double p, double q, double alpha, double beta,
        double dp = 0.01, double dq = 0.01, int niter = 1000000, int nburnin = 10000, seed = None):
    cdef int x = 1
    cdef double r = 0.0
    cdef double r_p = r
    cdef double r_q = r
    cdef double elambda = np.exp(-lambda_)
    cdef double emu = np.exp(-mu)
    cdef double dRAND_MAX = <double> RAND_MAX
    cdef np.ndarray[np.double_t, ndim=1] zs = np.zeros(niter, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1] zs_p = np.zeros(niter, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1] zs_q = np.zeros(niter, dtype = np.double)
    cdef double z, z_p, z_q, prob
    cdef unsigned int i
    srand(seed if seed else np.random.randint(1000))
    with nogil:
        for i in range(niter):
            prob = rand() / dRAND_MAX
            if x:
                x = prob > beta
            else:
                x = prob < alpha
            if x:
                z = (1.0-r)*elambda + r*emu
                r = ((1.0-r)*p*elambda + r*(1.0-q)*emu)/z
                z_p = (1.0-r_p)*elambda + r_p*emu
                r_p = ((1.0-r_p)*(p+dp)*elambda + r_p*(1.0-q)*emu)/z_p
                z_q = (1.0-r_q)*elambda + r_q*emu
                r_q = ((1.0-r_q)*p*elambda + r_q*(1.0-(q+dq))*emu)/z_q
            else:
                z = (1.0-r) + r*emu
                r = ((1.0-r)*p + r*(1.0-q)*emu)/z
                z_p = (1.0-r_p) + r_p*emu
                r_p = ((1.0-r_p)*(p+dp) + r_p*(1.0-q)*emu)/z_p
                z_q = (1.0-r_q) + r_q*emu
                r_q = ((1.0-r_q)*p + r_q*(1.0-(q+dq))*emu)/z_q
            zs[i] = z
            zs_p[i] = z_p
            zs_q[i] = z_q
    f = np.mean(np.log(zs)[nburnin:])
    fp = np.mean(np.log(zs_p)[nburnin:])
    fq = np.mean(np.log(zs_q)[nburnin:])
    return f, np.asarray(((fp - f)/dp, (fq - f)/dq))

def xi(np.ndarray[np.uint8_t, cast=True, ndim=2] phen,
       np.ndarray[np.uint8_t, cast=True, ndim=1] env,
       double lambda_,
       double mu,
       double nu):
    cdef Py_ssize_t nind = phen.shape[0]
    cdef Py_ssize_t L = phen.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] res = np.empty(nind, dtype = np.float64)
    cdef Py_ssize_t i, j
    cdef unsigned int nprotected, ninfected
    cdef double tmp_res
    for i in range(nind):
        nprotected = 0
        ninfected = 0
        for j in range(L):
            if phen[i, j]:
                nprotected += 1
            elif env[j]:
                ninfected += 1
        tmp_res = -lambda_ * ninfected - mu * nprotected
        if nu:
            tmp_res += - nu * nprotected**2
        res[i] = exp(tmp_res)
    return res

def xi_pa(np.ndarray[np.uint8_t, cast=True, ndim=2] phen,
       np.ndarray[np.uint8_t, cast=True, ndim=1] env,
       double lambda_,
       np.ndarray[np.float64_t, cast=True, ndim=1] mu1,
       np.ndarray[np.float64_t, cast=True, ndim=1] mu2):
    cdef Py_ssize_t nind = phen.shape[0]
    cdef Py_ssize_t L = phen.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] res = np.empty(nind, dtype = np.float64)
    cdef Py_ssize_t i, j
    cdef unsigned int nused, nunused, ninfected
    for i in range(nind):
        nused = 0
        nunused = 0
        ninfected = 0
        for j in range(L):
            if phen[i, j]:
                if env[j]:
                    nused += 1
                else:
                    nunused += 1
            elif env[j]:
                ninfected += 1
        res[i] = exp(-lambda_ * ninfected - mu1[i] * nunused - mu2[i] * nused)
    return res

def stepcrispr(np.ndarray[np.uint8_t, cast=True, ndim=1] env,
               np.ndarray[np.uint8_t, cast=True, ndim=2] gen,
               np.ndarray[np.double_t, ndim=2] pup,
               np.ndarray[np.double_t, ndim=2] rand):
    cdef Py_ssize_t nind = gen.shape[0]
    cdef Py_ssize_t L = gen.shape[1]
    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] out = np.empty((nind, L), dtype = np.bool)
    cdef np.ndarray[np.uint32_t, ndim=1] nacquired = np.zeros(nind, dtype = np.uint32)
    cdef Py_ssize_t i, j
    for i in range(nind):
        for j in range(L):
            if gen[i, j]:
                out[i, j] = True
            else:
                if env[j] and (rand[i, j] < pup[i, j]):
                    out[i, j] = True
                    nacquired[i] += 1
                else:
                    out[i, j] = False
    return out, nacquired

# numpy boolean array can not be used in cython, therefore need to cast to uint8
def stepmarkov(np.ndarray[np.uint8_t, cast=True] x,
               np.ndarray[np.double_t] a,
               np.ndarray[np.double_t] b,
               np.ndarray[np.double_t] rand):
    cdef np.ndarray[np.uint8_t, cast=True] out = np.empty(x.shape[0], dtype=np.bool)
    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        if x[i]:
            out[i] = rand[i] < 1.0-b[i]
        else:
            out[i] = rand[i] < a[i]
    return out

# numpy boolean array can not be used in cython, therefore need to cast to uint8
def stepmarkov2d(np.ndarray[np.uint8_t, cast=True, ndim=2] x,
                 np.ndarray[np.double_t, ndim=2] a,
                 np.ndarray[np.double_t, ndim=2] b,
                 np.ndarray[np.double_t, ndim=2] rand):
    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] out = np.empty((x.shape[0], x.shape[1]), dtype=np.bool)
    cdef Py_ssize_t i, j
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j]:
                out[i, j] = rand[i, j] < 1.0-b[i, j]
            else:
                out[i, j] = rand[i, j] < a[i, j]
    return out
