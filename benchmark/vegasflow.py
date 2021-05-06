#from vegasflow.stratified import StratifiedFlow
from benchmark.benchmark import Integrator, MAX_ITERATIONS, WARMUP_ITERATIONS,WARMUP_CALLS,generate_data
from benchmark.functions.gauss import gauss_vf
import numpy as np

import time
import vegasflow


def simulation(integrand=None,dim=None,rtol=None):
    samples = list(map(int,[1e3,1e4,1e5,1e6]))
    result = []

    
    for sample in samples:
        vegasplus = generate_data(VegasFlow,"VegasFlowPlus",integrand,dim,sample,rtol)
        vegas =  generate_data(VegasFlow, "VegasFlowPlus",integrand,dim,sample,rtol,adaptive=False)
        importance = generate_data(VegasFlow,"VegasFlow", integrand,dim,sample,rtol)
        adaptive_stratified = generate_data(VegasFlow,"StratifiedFlow",integrand,dim,sample,rtol)
        stratified = generate_data(VegasFlow,"StratifiedFlow",integrand,dim,sample,rtol,adaptive=False)
        plain = generate_data(VegasFlow,"PlainFlow",integrand,dim,sample,rtol)

        #result += [vegas]
        #result += [vegasplus,vegas,importance,adaptive_stratified,stratified,plain]
    
    return result


class VegasFlow(Integrator):
    """
    Class for benchmark with vegasflow integrator
    """
    def __init__(self,n_dim,n_calls,rtol,integrator,train=True,adaptive=True,warmup=0):

        super().__init__(n_dim,n_calls,rtol) 
        self.integrator = integrator
        if all((not(self.integrator == i) for i in ["VegasFlow","StratifiedFlow",
                                                    "PlainFlow","VegasFlowPlus"])):
            raise RuntimeError("Unknown vegas integrator")

        self.train = train
        self.adaptive = adaptive
        self.warmup = warmup

    def recognize_integrand(self,integrand_name=None):
        if integrand_name == 'gauss':
           self.integrand = gauss_vf
           self.integrand_name = integrand_name
        else: 
            raise RuntimeError("Integrand not recognized")

    def run_integration(self):

        if self.integrator == "VegasFlowPlus":
            instance = getattr(vegasflow, self.integrator)(self.n_dim,self.n_calls,
                                                       #simplify_signature=True,
                                                       train=self.train,
                                                       adaptive=self.adaptive)
        elif self.integrator == "VegasFlow":
            instance = getattr(vegasflow, self.integrator)(self.n_dim,self.n_calls,
                                                       #simplify_signature=True,
                                                       train=self.train)
        elif self.integrator == "StratifiedFlow":
            instance = getattr(vegasflow, self.integrator)(self.n_dim,self.n_calls,
                                                       #simplify_signature=True,
                                                       adaptive=self.adaptive)
        else:
            instance = getattr(vegasflow, self.integrator)(self.n_dim,self.n_calls)
                                                       #simplify_signature=True)
        
        instance.compile(self.integrand)
        start = time.time()
        if self.warmup == 1:
            instance.run_integration(WARMUP_ITERATIONS)
            instance.adaptive=False
            instance.freeze_grid()
            print(" > Freezing the grid")
        elif self.warmup == 2:
            instance.run_integration(WARMUP_ITERATIONS)
            instance.freeze_grid()
            print(" > Freezing the grid")

        self.n_iter = 0
        all_results = []
        local_times = []
        for i in range(MAX_ITERATIONS):
            self.n_iter += 1
            local_time = time.time()
            res, error = instance._run_iteration()
            local_time = time.time() - local_time
            time_str = f"(took {local_time:.5f} s)"
            vegasflow.monte_carlo.print_iteration(self.n_iter,res,error,extra=time_str)
            local_times.append(local_time)
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
                result = {
                          "integrator" : self.integrator,
                          "adaptive" : self.adaptive,
                          "train" : self.train,
                          "perc_uncertainty" : self.rtol,
                          "time" : end-start,
                          "iter" : self.n_iter,
                          "samples/iter" : self.n_calls,
                          "result" : f"{self.integral} +/- {self.error}",
                          "rtol_reached": (self.error/self.integral).numpy(),
                          "warmup" : self.warmup,
                          "local_times" : local_times,
                          "avg_time_per_iteration" : np.mean(local_times),
                          "total_iteration_time" : np.sum(local_times),
                          "warmup_time" : end-start-np.sum(local_times)
                        }
                return result
                #return self.integral, self.error
            if self.n_iter == MAX_ITERATIONS:
                #print("INFO: Max iterations reached \n")
                end = time.time()
                self.time = end - start
                self.integral = final_result
                self.error = sigma
                #self.rtol = self.error/self.integral
                self.fail=1
                result = {
                          "integrator" : self.integrator,
                          "adaptive" : self.adaptive,
                          "train" : self.train,
                          "perc_uncertainty" : self.rtol,
                          "time" : end-start,
                          "iter" : self.n_iter,
                          "samples/iter" : self.n_calls,
                          "result" : f"{self.integral} +/- {self.error}",
                          "rtol_reached": (self.error/self.integral).numpy(),
                          "warmup": self.warmup,
                          "local_times" : local_times,
                          "avg_time_per_iteration" : np.mean(local_times),
                          "total_iteration_time" : np.sum(local_times),
                          "warmup_time" : end-start-np.sum(local_times)
                        }
                return result

        #return result
    