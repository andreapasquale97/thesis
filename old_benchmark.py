from abc import ABC, abstractmethod
from vegasflow.configflow import DTYPE
import time
import argparse
import numpy as np 
import math
import tensorflow as tf

MAX_ITERATIONS = 100

# possible integrands

@tf.function
def gauss_vf(xarr, n_dim=None, **kwargs):
    """symgauss test function"""
    if n_dim is None:
        n_dim = xarr.shape[-1]
    a = tf.constant(0.1, dtype=DTYPE)
    n100 = tf.cast(100 * n_dim, dtype=DTYPE)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


def gauss_v(x,dim=None):
    if dim is None:
        dim = x.shape[-1]
    dx2 = 0
    a = 0.1
    coef = (1.0/a/np.sqrt(np.pi))**dim
    for d in range(dim):
        dx2 += (x[d] - 0.5) ** 2
    return np.exp(-dx2 * 100.) * coef



class Integrator(ABC):
    """
    Parent class of all integrators for the benchmark

    Parameters
    ----------
        `n_dim` : dimensions of the integrand
        `n_calls`: number of evalutations of the integrand per iteration
        `rtol`:  percent uncertainty required   
    """
    def __init__(self,n_dim,n_calls,rtol):
        self.n_calls = n_calls
        self.n_dim = n_dim
        self.rtol = rtol
        self.integrand = None
        self.integrand_name=None

        self.n_iter_limit = MAX_ITERATIONS
        self.integrator=None

        self.fail=0
        self.time=None
        self.n_iter=0

        self.integral=None
        self.error=None

        

    @abstractmethod
    def run_integration(self):
        """Run integration at fixed calls and accuracy"""
    
    @abstractmethod
    def recognize_integrand(self,integrand_name=None):
        """Assign integrand to the integrator"""

    def show_content(self):
        if self.fail == 0:
            print(f"Integrator {self.integrator} converged in {self.n_iter} iterations and took time {self.time} s")
            print(f"RESULT: {self.integral} +/- {self.error}")
        else:
            print(f"Integrator {self.integrator} cannot converge to the required percent uncertainty")

    def set_integrand(self,integrand):
        self.integrand = integrand

    def compare(self,*integrators):
   
        fig, (ax1,ax2)  = plt.subplots(1, 2,sharey=True,figsize=(8,4))

        data = [wrapper(i,self.integrand,self.n_dim,self.n_calls,self.rtol) for i in integrators]
        times = [self.time] + [i[0] for i in data]
        iterations = [self.n_iter] + [i[1] for i in data]

    
        integrators = [i.replace("-", "\n" ) for i in [self.integrator]+list(integrators) ]
    
        ax1.barh(integrators,times)
        ax1.set_yticks(integrators)
        ax1.set_xlabel("Time (s)")

        ax2.barh(integrators,iterations)
        ax2.set_yticks(integrators)
        ax2.set_xlabel("Iterations")

        plt.show()     
    

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
    
                
        
    

class VegasFlow(Integrator):
    """
    Class for benchmark with vegasflow integrator
    """
    def __init__(self,n_dim,n_calls,rtol,integrator,**kwargs):

        super().__init__(n_dim,n_calls,rtol,**kwargs) 
        self.integrator = integrator
        if integrator == 'vegasflow':
            self.train=True
        else:
            raise RuntimeError("Unknowm vegas integrator")

    def recognize_integrand(self,integrand_name=None):
        if integrand_name == 'gauss':
           self.integrand = gauss_vf
           self.integrand_name = integrand_name
        else: 
            raise RuntimeError("Integrand not recognized")

    def run_integration(self):
        instance = vegasflow.VegasFlow(self.n_dim,self.n_calls,simplify_signature=True)
        instance.compile(self.integrand)
        start = time.time()
        self.n_iter = 0
        all_results = []
        for i in range(MAX_ITERATIONS):
            self.n_iter += 1
            res, error = instance._run_iteration()
            all_results.append((res, error))
            aux_res = 0.0
            weight_sum = 0.0
            for i, result in enumerate(all_results):
                res = result[0]
                sigma = result[1]
                wgt_tmp = 1.0 / pow(sigma, 2)
                aux_res += res * wgt_tmp
                weight_sum += wgt_tmp
            
            final_result = aux_res / weight_sum
            sigma = np.sqrt(1.0 / weight_sum)
            if (sigma/final_result <= self.rtol):
                end = time.time()
                self.integral = final_result
                self.error = sigma
                self.time = end - start
                return self.integral, self.error
            if self.n_iter == MAX_ITERATIONS:
                print("INFO: Max iterations reached \n")
                self.integral = final_result
                self.error = sigma
                self.rtol = self.error/self.integral
                self.fail=1
    

        

def wrapper(integrator,integrand,n_dim,n_calls,rtol,**kwargs):

    if integrator == "vegasflow":
        instance = VegasFlow(n_dim,n_calls,rtol,integrator)
    else:
        instance = Vegas(n_dim,n_calls,rtol,integrator)

    instance.set_integrand(integrand)
    instance.run_integration()
    instance.show_content()

    return instance.time, instance.n_iter



if __name__ == '__main__':

    #SCRIPT EXAMPLE

    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrator', default='vegas+', type=str, help='Integrator type.')
    parser.add_argument('--integrand',  default='gauss_4', type=str, help=' Integrand wrote as "integrand_<dim>" ')
    parser.add_argument('--ncalls', default=1000000,type=int, help='Number of calls.')
    parser.add_argument('--accuracy', default=1e-4,type=float, help='Percent uncertainty required')

    args = vars(parser.parse_args())

    ncalls = args['ncalls']
    integrator = args['integrator']
    integrand_name = args['integrand'].split("_")[0]
    n_dim = int(args['integrand'].split("_")[1])
    accuracy = args['accuracy']

    vegas_options = ['vegas+', 'vegas-importance', 'vegas-stratified']
    
    if any([integrator == i for i in vegas_options]):
        try:
            import vegas
            instance = Vegas(n_dim,ncalls,accuracy,integrator)
            instance.recognize_integrand(integrand_name)
            instance.run_integration()
            instance.show_content()
            

        except ImportError:
            raise ImportError("Install vegas module")  
        
    elif integrator == 'vegasflow':
        try: 
            import vegasflow
            instance = VegasFlow(n_dim,ncalls,accuracy,integrator)
            instance.recognize_integrand(integrand_name)
            instance.run_integration()
            instance.show_content()
        except ImportError:
            raise ImportError("Install vegasflow module")  

    
    """

    #COMPARE EXAMPLE

    import vegas
    import matplotlib.pyplot as plt
    instance = Vegas(4,1000,1e-2,"vegas+")
    instance.set_integrand(gauss_v)
    instance.run_integration()
    instance.compare("vegas-stratified","vegas-importance")
    #time, iteration = wrapper('vegas+',gauss_v,4,10000,1e-2)
    #instance.show_content()
    

