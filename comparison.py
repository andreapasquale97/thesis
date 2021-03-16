from functions import gauss_vf
from vegas_benchmark import Vegas
from vegasflow_benchmark import VegasFlow
import sys

if __name__ == '__main__':
    
    samples = [1e5,1e5,1e6,1e6]
    rtols = [1e-2,1e-3,1e-2,1e-3]

    instance = VegasFlow(4,int(samples[0]),rtols[0],"VegasFlow")
    instance.compare("VegasFlowPlus")
    
    """
    with open('benchmark.txt','w') as f:
        sys.stdout  = f
        for i in range(4):
            print("Gauss 4-d ", samples[i], "samples per iterations rtol", rtols[i])
            instance = VegasFlow(4,int(samples[i]),rtols[i],"VegasFlow")
            instance.set_integrand(gauss_vf)
            instance.run_integration()
            instance.show_content()
            instance1 = VegasFlow(4,int(samples[i]),rtols[i],"VegasFlowPlus",adaptive=False)
            instance1.set_integrand(gauss_vf)
            instance1.run_integration() 
            instance1.show_content()

    f.close()
    """
        
