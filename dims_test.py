"""

Testing importance and stratified sampling
varying the integral dimension

"""

import vegas
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.optimize import rosen

SHOW_PLOT = False


dim_min = 2
dim_max = 10
# data set
n_eval = 12000
n_eval_testing = 11000
n_iter = 50
n_iter_warmup = 5
n_eval_warmup = 1e3

def f(x,dim=None):
    if dim is None:
        dim = x.shape[-1]
    dx2 = 0
    a = 0.1
    coef = (1.0/a/np.sqrt(np.pi))**dim
    for d in range(dim):
        dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * coef


def integration(dim,integrand,isStratified,isImportance):

    #assign integration volume to integrator
    region = dim * [[-1.,1.]]


    if isStratified and isImportance:
        integ = vegas.Integrator(region)
    elif not isStratified:
        integ = vegas.Integrator(region,max_nhcube=1) # stratified off
    else:
        integ = vegas.Integrator(region,adapt=False) # no adaptation


    # adapt to the integrand; discard results
    integ(integrand,nitn=n_iter_warmup,neval=n_eval_warmup)



    # proper integration
    result = integ(integrand,nitn=n_iter,neval=n_eval_testing)
    #print(f"Stratifications {np.array(integ.nstrat)}")
    #print("\n Grid \n")
    print(result.summary())

    return result.mean, result.sdev

def generate_simulation_per_type(integrand,isStratified,isImportance):
    errors = []
    integrals = []
    dim = dim_min
    while dim < dim_max+1:
        result = integration(dim,integrand,isStratified,isImportance)
        integrals.append(result[0])
        errors.append(result[1])
        dim += 1

    return integrals, errors

def generate_simulation(integrand):
    print("--------IMPORTANCE SAMPLING INTEGRATION--------")
    importance = generate_simulation_per_type(integrand,isStratified=False,isImportance=True)
    print("--------ADAPTIVE STRATIFIED INTEGRATION--------")
    stratified = generate_simulation_per_type(integrand,isStratified=True,isImportance=False)
    print("--------VEGAS+ INTEGRATION---------")
    vegas = generate_simulation_per_type(integrand,isStratified=True,isImportance=True)

    return importance,stratified,vegas

def make_plot(importance,stratified,vegas):
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,6))
    plt.xlabel('dimension (D)')
    fig.suptitle('Comparison for Rosen Function [-1,1]')

    dims = [i for i in range(dim_min,dim_max+1)]

    ax[0].errorbar(dims, vegas[0],
                yerr=vegas[1],
                fmt='-o',label="vegas+")

    ax[0].errorbar(dims, importance[0],
                yerr=importance[1],
                fmt='-o',label="importance sampling")

    ax[0].errorbar(dims, stratified[0],
                yerr=stratified[1],
                fmt='-o',label="stratified")

    #ax.set_xlabel('dimensions (D)')
    ax[0].set_ylabel('integral estimate (I)')
    #ax.set_title('Comparison between Importance and Stratified Sampling')
    ax[0].legend()

    #fig,ax1 = plt.subplots()
    #ax1.set_xlabel('dimensions (D)')
    ax[1].set_ylabel('integral estimate (I)')
    #ax1.set_title('Comparison between Importance and Stratified Sampling')
    ax[1].legend()

    ax[1].errorbar(dims, vegas[0],
                yerr=vegas[1],
                fmt='-o',label="vegas+")

    ax[1].errorbar(dims, importance[0],
                yerr=importance[1],
                fmt='-o',label="importance sampling")
    ax[1].legend()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig('rosen_symm_dimensions.png',bbox_inches='tight')

def make_second_plot(importance,stratified,vegas):
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,6))
    dims = [i for i in range(dim_min,dim_max+1)]

    perc_err_vegas = [i/j for i,j in zip(vegas[1],vegas[0])]
    perc_err_stratified = [i/j for i,j in zip(stratified[1],stratified[0])]
    perc_err_importance = [i/j for i,j in zip(importance[1],importance[0])]

    ax[0].plot(dims, perc_err_vegas,label="vegas+")
    ax[0].plot(dims, perc_err_importance,label="importance sampling")
    ax[0].plot(dims, perc_err_stratified,label="stratified")

    plt.xlabel('dimensions D')
    ax[0].set_ylabel(' Percent uncertainty')
    fig.suptitle('Comparison for Rosen Function [-1,1]')
    ax[0].legend()

    ax[1].plot(dims, perc_err_vegas,label="vegas+")
    ax[1].plot(dims, perc_err_importance,label="importance sampling")
    ax[1].set_ylabel(' Percent uncertainty')
    ax[1].legend()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig('rosen_symm_dims_2.png',bbox_inches='tight')

def make_single_plot(importance,stratified,vegas):
    fig, ax = plt.subplots(figsize=(8,6))
    dims = [i for i in range(dim_min,dim_max+1)]
    perc_err_vegas = [i/j for i,j in zip(vegas[1],vegas[0])]
    perc_err_stratified = [i/j for i,j in zip(stratified[1],stratified[0])]
    perc_err_importance = [i/j for i,j in zip(importance[1],importance[0])]

    ax.plot(dims, perc_err_vegas,label="vegas+")
    ax.plot(dims, perc_err_importance,label="importance sampling")
    ax.plot(dims, perc_err_stratified,label="stratified")

    ax.set_title('Comparison for Rosenbrock Function [-1,1] after 50 iterations')
    ax.set_xlabel('dimensions D')
    ax.set_ylabel('Percent uncertainty')
    ax.legend()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig('plots/rosen_dims_final.png',bbox_inches='tight')
    
if __name__ == '__main__':

    importance, stratified, vegas = generate_simulation(f)
    make_single_plot(importance,stratified,vegas)
    #make_second_plot(importance,stratified,vegas)
