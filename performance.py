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
MAX_ITER = 200
ITER_WARMUP = 5
EVAL_WARMUP = 1e3

SHOW_PLOT = False





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
 
def add_integrand(filename=None,integrand=None,dim=None,domain=[[0.,1.]],rtol=1e-2):
    results = simulation(integrand=integrand,dim=dim,rtol=rtol)
    with open(filename, "w") as f:
        json.dump(results,f,indent=True)
            
def make_histo(first=None,second=None,third=None,filename=None):
 
    fig, axs  = plt.subplots(3, 2, sharey='row',figsize=(10, 10))#12
    #fig.suptitle('Time/iterations comparison')
    index = ['1000', '10000', '100000', '1000000']
    plt.ylabel("samples")

    vegas_iter1 = [i["iter"] for i in first if i["integrator"] == "vegas+"]
    importance_iter1 = [i["iter"] for i in first if i["integrator"] == "importance"]
    stratified_iter1 = [i["iter"] for i in first if i["integrator"] == "stratified"]
    vegas_time1 = [i["time"] for i in first if i["integrator"] == "vegas+"]
    importance_time1 = [i["time"] for i in first if i["integrator"] == "importance"]
    stratified_time1 = [i["time"] for i in first if i["integrator"] == "stratified"]
    df_iter1 = pd.DataFrame({'vegas+' : vegas_iter1, 'importance' : importance_iter1,
                        'stratified' : stratified_iter1 },index=index)
    df_time1 = pd.DataFrame({'vegas+' : vegas_time1, 'importance' : importance_time1,
                        'stratified' : stratified_time1 },index=index)
                    
    vegas_iter2 = [i["iter"] for i in second if i["integrator"] == "vegas+"]
    importance_iter2 = [i["iter"] for i in second if i["integrator"] == "importance"]
    stratified_iter2 = [i["iter"] for i in second if i["integrator"] == "stratified"]
    vegas_time2 = [i["time"] for i in second if i["integrator"] == "vegas+"]
    importance_time2 = [i["time"] for i in second if i["integrator"] == "importance"]
    stratified_time2 = [i["time"] for i in second if i["integrator"] == "stratified"]
    df_iter2 = pd.DataFrame({'vegas+' : vegas_iter2, 'importance' : importance_iter2,
                        'stratified' : stratified_iter2 },index=index)
    df_time2 = pd.DataFrame({'vegas+' : vegas_time2, 'importance' : importance_time2,
                        'stratified' : stratified_time2 },index=index)

    vegas_iter3 = [i["iter"] for i in third if i["integrator"] == "vegas+"]
    importance_iter3 = [i["iter"] for i in third if i["integrator"] == "importance"]
    stratified_iter3 = [i["iter"] for i in third if i["integrator"] == "stratified"]
    vegas_time3 = [i["time"] for i in third if i["integrator"] == "vegas+"]
    importance_time3 = [i["time"] for i in third if i["integrator"] == "importance"]
    stratified_time3 = [i["time"] for i in third if i["integrator"] == "stratified"]
    df_iter3 = pd.DataFrame({'vegas+' : vegas_iter3, 'importance' : importance_iter3,
                        'stratified' : stratified_iter3 },index=index)
    df_time3 = pd.DataFrame({'vegas+' : vegas_time3, 'importance' : importance_time3,
                        'stratified' : stratified_time3 },index=index)

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
    plt.ylabel("samples")
    #for ax in axs.flat:
    #    ax.label_outer()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig(filename,bbox_inches='tight')


if __name__ == '__main__':
    #make_histo('data.json')
    #result = simulation(integrand=f,dim=4,rtol=1e-2)
    #filename = 'data.json'
    #with open(filename, "r") as f:
    #    json_dict = json.load(f)
    #add_integrand(filename=filename,integrand=f,dim=1,rtol=1e-2)
    #print(simulation(integrand=f,dim=1))
    first = simulation(integrand=f,dim=2,rtol=1e-2)
    second = simulation(integrand=f,dim=2,rtol=1e-3)
    third = simulation(integrand=f,dim=2,rtol=1e-4) 
    make_histo(first=first,second=second,third=third,filename='test.png')
    
