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

dim = 8
# data set
n_eval_min = 1e3
n_eval_max = 1e6
n_evals = np.logspace(3,6,6,dtype=np.int32)
#print(n_evals)
n_iter = 10


n_eval_warmup = n_eval_min
n_iter_warmup = 5

def f(x,dim=None):
    if dim is None:
        dim = x.shape[-1]
    dx2 = 0
    a = 0.1
    coef = (1.0/a/np.sqrt(np.pi))**dim
    for d in range(dim):
        dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * coef


def integration(integrand,isStratified,isImportance,n_eval):

    #assign integration volume to integrator
    region = dim * [[-1.,1.]]


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
    fig.suptitle(f'Comparison for Rosen function [-1,1] dim = {dim}')
    ax[0].legend()

    ax[1].plot(n_evals, perc_err_vegas,label="vegas+")
    ax[1].plot(n_evals, perc_err_importance,label="importance sampling")
    #ax[1].plot(n_evals, perc_err_stratified,label="stratified")
    ax[1].set_ylabel(' Percent uncertainty')
    ax[1].legend()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig(f'rosen_samples_symm_dim{dim}.png',bbox_inches='tight')

if __name__ == '__main__':

    importance, stratified, vegas = generate_simulation(rosen)
    make_plot(importance,stratified,vegas)
