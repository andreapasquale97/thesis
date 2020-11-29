"""

Testing importance and stratified sampling in vegas+
varying the number of samples of the integrand

"""


import vegas
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.optimize import rosen


SHOW_PLOT = False

dim = 3
# data set
n_eval_min = 1e3
n_eval_max = 1e6
n_evals = np.logspace(3,6,6,dtype=np.int32)
#print(n_evals)
n_iter = 10
domain_per_dimension = [[0.,1.]]
plot_title = f"Comparison for physics integrand dim = {dim}"
file_name = "plots/dy_aa_samples.png"


n_eval_warmup = n_eval_min
n_iter_warmup = 5

# possible integrands

# 1) gaussian
def f(x,dim=None):
    if dim is None:
        dim = x.shape[-1]
    dx2 = 0
    a = 0.1
    coef = (1.0/a/np.sqrt(np.pi))**dim
    for d in range(dim):
        dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * coef

# 2) Drell-Yan
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
    #print(f"Stratifications {np.array(integ.nstrat)}")
    #print("\n Grid \n")
    #print(integ.map.settings())
    print(result.summary())

    return result.mean, result.sdev

def generate_simulation_per_type(integrand,isStratified,isImportance):
    errors = []
    integrals = []

    for n_eval in n_evals:
        #print(f"Gaussian Integration with {n_eval} samples")
        result = integration(integrand,isStratified,isImportance,n_eval)
        integrals.append(result[0])
        errors.append(result[1])

    return integrals, errors

def generate_simulation(integrand=f):
    print("--------IMPORTANCE SAMPLING INTEGRATION--------")
    importance = generate_simulation_per_type(integrand,isStratified=False,isImportance=True)
    print("--------ADAPTIVE STRATIFIED INTEGRATION--------")
    stratified = generate_simulation_per_type(integrand,isStratified=True,isImportance=False)
    print("--------VEGAS+ INTEGRATION---------")
    vegas = generate_simulation_per_type(integrand,isStratified=True,isImportance=True)

    return importance,stratified,vegas

def make_plot(importance,stratified,vegas):
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,6))

    perc_err_vegas = [i/j for i,j in zip(vegas[1],vegas[0])]
    perc_err_stratified = [i/j for i,j in zip(stratified[1],stratified[0])]
    perc_err_importance = [i/j for i,j in zip(importance[1],importance[0])]

    ax[0].plot(n_evals, perc_err_vegas,label="vegas+")
    ax[0].plot(n_evals, perc_err_importance,label="importance sampling")
    ax[0].plot(n_evals, perc_err_stratified,label="stratified")

    plt.xlabel('samples')
    ax[0].set_ylabel(' Percent uncertainty')
    fig.suptitle(plot_title)
    ax[0].legend()

    ax[1].plot(n_evals, perc_err_vegas,label="vegas+")
    ax[1].plot(n_evals, perc_err_importance,label="importance sampling")
    #ax[1].plot(n_evals, perc_err_stratified,label="stratified")
    ax[1].set_ylabel(' Percent uncertainty')
    ax[1].legend()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig(file_name,bbox_inches='tight')

if __name__ == '__main__':

    importance, stratified, vegas = generate_simulation(integrand_pineappl)
    make_plot(importance,stratified,vegas)
