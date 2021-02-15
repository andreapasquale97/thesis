from benchmark import Integrator, MAX_ITERATIONS
from functions import gauss_v
import matplotlib.pyplot as plt
import numpy as np

import time
import vegas

#auxiliary functions


class Vegas(Integrator):
    """
    Class for benchmarks with vegas integrator
    """
    def __init__(self,n_dim,n_calls,rtol,integrator,domain=[[0.,1.]],**kwargs):

        super().__init__(n_dim,n_calls,rtol,**kwargs) 
        self.domain = domain
        self.integrator = integrator
        if integrator == 'vegas+':
            self.train=True
            self.stratified = True
        elif integrator == 'vegas-importance':
            self.train = True
            self.stratified = False
        elif integrator == 'vegas-stratified':
            self.train = False
            self.stratified = True
        else:
            raise RuntimeError("Unknowm vegas integrator")

        #warmup setting
        self.n_iter_warmup = 5
        self.n_calls_warmup = 1000
        
    def recognize_integrand(self,integrand_name=None):
        if integrand_name == 'gauss':
           self.integrand = gauss_v
           self.integrand_name = integrand_name 
        else: 
            raise RuntimeError("Integrand not recognized")
  
    def run_integration(self):
        region = self.n_dim * self.domain
        if not self.train:
            integ = vegas.Integrator(region,adapt=False,rtol=self.rtol)
        elif not self.stratified:
            integ = vegas.Integrator(region,max_nhcube=1,rtol=self.rtol)
        else:
            integ = vegas.Integrator(region,rtol=self.rtol)

        integ(self.integrand,nitn=self.n_iter_warmup,neval=self.n_calls_warmup)

        start=time.time()
        result = integ(self.integrand,nitn=self.n_iter_limit)
        end=time.time()
        self.time = end - start
        self.n_iter = len(result.itn_results)
        if self.n_iter == MAX_ITERATIONS:
            print("INFO: Max iterations reached \n")
            self.rtol = result.sdev/result.mean
            self.fail=1

        self.integral = result.mean
        self.error = result.sdev

        return result.mean, result.sdev

    def generate_dims_data(self,dim_min,dim_max):
        
        vegas_data = []
        importance_data = []
        stratified_data = []

        dim = dim_min
        while dim <= dim_max:
            region = dim * self.domain
            vegas_integrator = vegas.Integrator(region)
            importance_integrator = vegas.Integrator(region,max_nhcube=1)
            stratified_integrator = vegas.Integrator(region,adapt=False)

            #warmup
            vegas_integrator(self.integrand,nitn=self.n_iter_warmup,neval=self.n_calls_warmup)
            importance_integrator(self.integrand,nitn=self.n_iter_warmup,neval=self.n_calls_warmup)
            stratified_integrator(self.integrand,nitn=self.n_iter_warmup,neval=self.n_calls_warmup)

            #integration
            vegas_result = vegas_integrator(self.integrand,nitn=50,neval=1100)
            importance_result = importance_integrator(self.integrand,nitn=50,neval=1100)
            stratified_result = stratified_integrator(self.integrand,nitn=50,neval=1100)


            vegas_data.append([vegas_result.mean,vegas_result.sdev])
            importance_data.append([importance_result.mean, importance_result.sdev])
            stratified_data.append([stratified_result.mean, stratified_result.sdev])
            
            dim += 1

        return vegas_data, importance_data, stratified_data

    def make_dims_plot(self,dim_min=1,dim_max=10):

        vegas, importance, stratified = self.generate_dims_data(dim_min,dim_max)

        fig, ax = plt.subplots(figsize=(8,6))
        dims = [i for i in range(dim_min,dim_max+1)]

        perc_err_vegas = [ i[1]/i[0] for i in vegas]
        perc_err_importance = [i[1]/i[0] for i in importance]
        perc_err_stratified = [i[1]/i[0] for i in stratified]


        ax.plot(dims, perc_err_vegas,label="vegas+")
        ax.plot(dims, perc_err_importance,label="importance sampling")
        ax.plot(dims, perc_err_stratified,label="stratified")

        ax.set_xlabel('dimensions D')
        ax.set_ylabel('Percent uncertainty')
        ax.legend()
        
        plt.show()

    def generate_eval_data(self,n_evals):

        vegas_data = []
        importance_data = []
        stratified_data = []

        region = self.n_dim * self.domain

        vegas_integrator = vegas.Integrator(region)
        importance_integrator = vegas.Integrator(region,max_nhcube=1)
        stratified_integrator = vegas.Integrator(region,adapt=False)

        #warmup
        vegas_integrator(self.integrand,nitn=self.n_iter_warmup,neval=self.n_calls_warmup)
        importance_integrator(self.integrand,nitn=self.n_iter_warmup,neval=self.n_calls_warmup)
        stratified_integrator(self.integrand,nitn=self.n_iter_warmup,neval=self.n_calls_warmup)

        for n_eval in n_evals:

            vegas_result = vegas_integrator(self.integrand,nitn=10,neval=n_eval)
            importance_result = importance_integrator(self.integrand,nitn=10,neval=n_eval)
            stratified_result = stratified_integrator(self.integrand,nitn=10,neval=n_eval)

            vegas_data.append([vegas_result.mean,vegas_result.sdev])
            importance_data.append([importance_result.mean, importance_result.sdev])
            stratified_data.append([stratified_result.mean, stratified_result.sdev])

        return vegas_data, importance_data, stratified_data

    def make_eval_plot(self,min_log_eval=3,max_log_eval=6,number=4):


        log_evals =np.linspace(min_log_eval,max_log_eval,number)
        evals = 10**log_evals.astype(int)

        vegas, importance, stratified = self.generate_eval_data(evals)

        perc_err_vegas = [ i[1]/i[0] for i in vegas]
        perc_err_importance = [i[1]/i[0] for i in importance]
        perc_err_stratified = [i[1]/i[0] for i in stratified]

        fig, ax = plt.subplots(figsize=(8,6))
    
        ax.plot(log_evals, perc_err_vegas,label="vegas+")
        ax.plot(log_evals, perc_err_importance,label="importance sampling")
        ax.plot(log_evals, perc_err_stratified,label="stratified")

        ax.set_xlabel('Log[samples]')
        ax.set_ylabel('Percent uncertainty')
        ax.legend()

        plt.show()













    
                
        
    