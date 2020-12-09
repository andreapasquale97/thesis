"""

    Perfomance comparsion between 
    vegas+, importance and stratified

"""

import vegas
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.optimize import rosen
import pandas as pd
import json


# data set
EVAL_MIN= 1e3
EVAL_MAX = 1e6
MAX_ITER = 100 #data 200
ITER_WARMUP = 5
EVAL_WARMUP = 1e3

SHOW_PLOT = True





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
    return weight


def integration(dim=None, integrand=None,isStratified=True,
                isImportance=True,n_eval=None,rtol=None, domain=[[0.,1.]]):

    #assign integration volume to integrator
    region = dim * domain
    if integrand == f:
        label_integrand = f"SymGauss {dim}-d"
    elif integrand == rosen:
        label_integrand= f"Rosenbrock {dim}-d"
    elif integrand == integrand_pineappl:
        label_integrand = f"Physical Integrand {dim}-d"
    else:
        raise RuntimeError("Unknown integrand")

    if isStratified and isImportance:
        integ = vegas.Integrator(region,rtol=rtol)
        label_method = "vegas+"
    elif not isStratified:
        integ = vegas.Integrator(region,max_nhcube=1,rtol=rtol) # stratified off
        label_method = "importance"
    else:
        integ = vegas.Integrator(region,adapt=False,rtol=rtol) # no adaptation
        label_method = "stratified"
    # adapt to the integrand; discard results
    integ(integrand,nitn=ITER_WARMUP,neval=EVAL_WARMUP)

    # proper integration
    start = time.time()
    result = integ(integrand,nitn=MAX_ITER,neval=n_eval)
    end = time.time()
    print(result.summary())

    #create dictionary for output
    
    output = {
        "integrand" : label_integrand,
        #"domain" : domain,
        "integrator" : label_method,
        "perc_uncertainty" : rtol,
        "time" : end-start,
        "iter" : len(result.itn_results),
        "samples/iter" : int(n_eval),
        "result" : f"{result.mean} +/- {result.sdev}" 
    }

    return output

def simulation(integrand=None,dim=None,domain=None,rtol=None):
    samples = list(map(int,[1e3,1e4,1e5,1e6]))
    result = []

    
    for sample in samples:
        vegas = integration(integrand=integrand,dim=dim,n_eval=sample,rtol=rtol)
        importance = integration(integrand=integrand,dim=dim,n_eval=sample,rtol=rtol,isStratified=False)
        stratified = integration(integrand=integrand,dim=dim,n_eval=sample,rtol=rtol,isImportance=False)
        result += [vegas,importance,stratified]
    
    return result
def run_simulation(integrand=None,dim=None,domain=[[0.,1.]]):

    first = simulation(integrand=integrand,dim=dim,rtol=1e-2)
    second = simulation(integrand=integrand,dim=dim,rtol=1e-3)
    third = simulation(integrand=integrand,dim=dim,rtol=1e-4) 

    return first+second+third

def save_data(filename=None,data=None):
    with open(filename, "w") as f:
        json.dump(data,f,indent=True)

def prepare_data(data=None,rtol=1e-2,index=None,showStratified=True):
    vegas_iter = [i["iter"] for i in data if i["integrator"] == "vegas+" and i["perc_uncertainty"] == rtol ]
    importance_iter = [i["iter"] for i in data if i["integrator"] == "importance" and i["perc_uncertainty"] == rtol]
    stratified_iter = [i["iter"] for i in data if i["integrator"] == "stratified" and i["perc_uncertainty"] == rtol ]
    vegas_time = [i["time"] for i in data if i["integrator"] == "vegas+" and i["perc_uncertainty"] == rtol ]
    importance_time = [i["time"] for i in data if i["integrator"] == "importance" and i["perc_uncertainty"] == rtol]
    stratified_time = [i["time"] for i in data if i["integrator"] == "stratified" and i["perc_uncertainty"] == rtol]

    if showStratified:
        df_iter = pd.DataFrame({'vegas+' : vegas_iter, 'importance' : importance_iter,'stratified' : stratified_iter },index=index)
        df_time = pd.DataFrame({'vegas+' : vegas_time, 'importance' : importance_time,'stratified' : stratified_time },index=index)
    else:
        df_iter = pd.DataFrame({'vegas+' : vegas_iter, 'importance' : importance_iter},index=index)
        df_time = pd.DataFrame({'vegas+' : vegas_time, 'importance' : importance_time},index=index)

    return df_iter, df_time

def make_histo(infile=None, outfile=None, save=True, showStratified=True):
 


    with open(infile, "r") as f:
        data = json.load(f)

    fig, axs  = plt.subplots(3, 2, sharey='row',figsize=(10, 10))#12
    #fig.suptitle('Time/iterations comparison')
    fig.text(0.02, 0.5, 'samples/iteration', ha='center', va='center', rotation='vertical')
    index = ['1000', '10000', '100000', '1000000']

    df_iter1, df_time1 = prepare_data(data=data,rtol=1e-2,index=index,showStratified=showStratified)
    df_iter2, df_time2 = prepare_data(data=data,rtol=1e-3,index=index,showStratified=showStratified)
    df_iter3, df_time3 = prepare_data(data=data,rtol=1e-4,index=index,showStratified=showStratified)
    

    axs[0,0] = df_time1.plot.barh(ax=axs[0,0],legend=False)
    axs[0,0].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,1] = df_iter1.plot.barh(ax=axs[0,1],legend=False)
    axs[0,1].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,0] = df_time2.plot.barh(ax=axs[1,0],legend=False)
    axs[1,0].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,1] = df_iter2.plot.barh(ax=axs[1,1],legend=False)
    axs[1,1].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,0] = df_time3.plot.barh(ax=axs[2,0],legend=False)
    axs[2,0].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,1] = df_iter3.plot.barh(ax=axs[2,1],legend=False)
    axs[2,1].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})


    #labelling
    #axs[0,0].set_xlabel('time (s)')
    #axs[0,1].set_xlabel('iterations')
    #axs[1,0].set_xlabel('time (s)')
    #axs[1,1].set_xlabel('iterations')
    axs[2,0].set_xlabel('time (s)')
    axs[2,1].set_xlabel('iterations')
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels,loc='center right')
    #for ax in axs.flat:
    #    ax.label_outer()
    if save:
        plt.savefig(outfile,bbox_inches='tight')
    else:
        plt.show()
        



if __name__ == '__main__':
    
    #data = run_simulation(integrand=rosen,dim=8,domain=[[-1.,1.]])
    #save_data('data/rosen8-d.json',data)
    make_histo(infile='data/rosen8-d.json',outfile='performance_plots/rosen8-d_no_stratified.png',save=True,showStratified=False)
    
