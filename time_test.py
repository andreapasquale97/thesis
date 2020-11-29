"""

Testing importance and stratified sampling
based on execution time

"""

import vegas
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.optimize import rosen



dim = 4
n_eval =  1e5
n_iter = 10
n_eval_warmup = 1e3
n_iter_warmup = 5
domain_per_dimension = [[0.,1.]]

def f(x,dim=None):
    if dim is None:
        dim = x.shape[-1]
    dx2 = 0
    a = 0.1
    coef = (1.0/a/np.sqrt(np.pi))**dim
    for d in range(dim):
        dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * coef

def int_photo(s, t, u):
    alpha0 = 1.0 / 137.03599911
    return alpha0 * alpha0 / 2.0 / s * (t / u + u / t)


def hadronic_pspgen(xarr, mmin, mmax):
    smin = mmin * mmin
    smax = mmax * mmax

    r1 = xarr[0]
    r2 = xarr[1]
    r3 = xarr[2]

    tau0 = smin / smax
    tau = pow(tau0, r1)
    y = pow(tau, 1.0 - r2)
    x1 = y
    x2 = tau / y
    s = tau * smax

    jacobian = tau * np.log(tau0) * np.log(tau0) * r1

    # theta integration (in the CMS)
    cos_theta = 2.0 * r3 - 1.0
    jacobian *= 2.0

    t = -0.5 * s * (1.0 - cos_theta)
    u = -0.5 * s * (1.0 + cos_theta)

    # phi integration
    jacobian *= 2.0 * np.math.acos(-1.0)

    return s, t, u, x1, x2, jacobian

def integrand_pineappl(xarr, dim=None, **kwargs):
    if dim is None:
        dim = xarr.shape[-1]
    # in GeV^2 pbarn
    hbarc2 = 389379372.1
    s, t, u, x1, x2, jacobian = hadronic_pspgen(xarr, 10.0, 7000.0)

    ptl = np.sqrt((t * u / s))
    mll = np.sqrt(s)
    yll = 0.5 * np.log(x1 / x2)
    ylp = np.abs(yll + np.math.acosh(0.5 * mll / ptl))
    ylm = np.abs(yll - np.math.acosh(0.5 * mll / ptl))

    jacobian *= hbarc2

    # cuts for LO for the invariant-mass slice containing the
    # Z-peak from CMSDY2D11
    if ptl < 14.0 or np.abs(yll) > 2.4 or ylp > 2.4 \
        or ylm > 2.4 or mll < 60.0 or mll > 120.0:
        weight = 0.0
    else:
        weight = jacobian * int_photo(s, u, t)
    #print(weight)
    #q2 = 90.0 * 90.0    
    #grid.fill(x1, x2, q2, 0, np.abs(yll), 0, weight)
    return weight

def integration(integrand,isStratified,isImportance,n_eval):

    #assign integration volume to integrator
    region = dim * domain_per_dimension

    if isStratified and isImportance:
        integ = vegas.Integrator(region)
    elif not isStratified:
        integ = vegas.Integrator(region,max_nhcube=1) # stratified off
    else:
        integ = vegas.Integrator(region,adapt=False) # no adaptation

    # adapt to the integrand; discard results
    integ(integrand,nitn=n_iter_warmup,neval=n_eval)

    # proper integration
    result = integ(integrand,nitn=n_iter,neval=n_eval)
    #print(result.summary())

    return result.mean, result.sdev

def timing_per_type(isStratified=None,isImportance=None,integrand=None):
    if isStratified and isImportance:
        integ_type = "vegas+" 
    elif isStratified: 
        integ_type = "stratified"
    else:
        integ_type = "importance"
    #print(f"{integ_type.upper()} INTEGRATION")
    start = time.time()
    result, error = integration(integrand,isStratified,isImportance,n_eval)
    end = time.time()
    print(f"{integ_type} result: {result} +/- {error} took time (s) {end-start}")
    #print(f"{integ_type} took: time (s): {end-start}")

def timing(integrand=None,label=None):
    print(f"{label.upper()} INTEGRATION:")
    #
    timing_per_type(isStratified=True,isImportance=False,integrand=integrand)
    timing_per_type(isStratified=False,isImportance=True,integrand=integrand)
    timing_per_type(isStratified=True,isImportance=True,integrand=integrand)
    print('-'*100)




if __name__ == '__main__':
    timing(f,"Gaussian function dim = 4")
    dim = 8
    timing(f,"Gaussian function dim = 8")
    dim = 3    
    timing(integrand_pineappl,"Physical integrand")
    dim = 4
    domain_per_dimension = [[-1.,1.]]
    timing(f,"Rosenbrock function dim = 4")
    dim = 8
    timing(f,"Rosenbrock function dim = 8")

