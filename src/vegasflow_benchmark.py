from benchmark import Integrator, MAX_ITERATIONS

import time
import vegasflow


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