from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

MAX_ITERATIONS = 100

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
        """Show results of the integration"""
        if self.fail == 0:
            print(f"Integrator {self.integrator} converged in {self.n_iter} iterations and took time {self.time} s")
            print(f"RESULT: {self.integral} +/- {self.error}")
        else:
            print(f"Integrator {self.integrator} cannot converge to the required percent uncertainty")

    def set_integrand(self,integrand):
        self.integrand = integrand
    
    def compare(self,*integrators):
        """
        Compare integrator with other integrators
        
        Parameter:
            integrators: sequence of tuples (Integrator Class, integrator type)
        """

        fig, (ax1,ax2)  = plt.subplots(1, 2,sharey=True,figsize=(8,4))



        data = [wrapper(i[0],i[1],self.integrand,self.n_dim,self.n_calls,self.rtol) for i in integrators]
        # wrapper function dove la devo mettere?
        times = [self.time] + [i[0] for i in data]
        iterations = [self.n_iter] + [i[1] for i in data]

    
        integrators = [i.replace("-", "\n" ) for i in [self.integrator]+[j[1] for j in integrators] ]
    
        ax1.barh(integrators,times)
        ax1.set_yticks(integrators)
        ax1.set_xlabel("Time (s)")

        ax2.barh(integrators,iterations)
        ax2.set_yticks(integrators)
        ax2.set_xlabel("Iterations")

        plt.show()

def wrapper(integrator_class,integrator_type,integrand,n_dim,n_calls,rtol,**kwargs):

    instance = integrator_class(n_dim,n_calls,rtol,integrator_type)
    instance.set_integrand(integrand)
    instance.run_integration()
    instance.show_content()

    return instance.time, instance.n_iter


  
