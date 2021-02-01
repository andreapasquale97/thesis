from functions import gauss_v
from vegas_benchmark import Vegas





if __name__ == '__main__':

    
    instance = Vegas(8,1000,1e-3,"vegas+")
    instance.set_integrand(gauss_v)
    instance.make_eval_plot(min_log_eval=3,max_log_eval=4,number=6)
    #instance.make_dims_plot(dim_min=4,dim_max=5)
    