from benchmark import gauss_vf, pineappl, singletop, drellyan
from benchmark.functions.gauss import gauss_vf1
from benchmark.functions.singletop_lo_tf import singletop1
from benchmark.functions.drellyan_lo_tf import drellyan1
from vegasflow.configflow import float_me, int_me,DTYPE,DTYPEINT
from vegasflow import VegasFlowPlus
from benchmark import VegasFlow
from benchmark.benchmark import generate_data
import argparse
import json


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrator', default='VegasFlow', type=str, help='Integrator type.')
    parser.add_argument('--adaptive', default=True, help='If True apply adaptive stratified sampling')
    parser.add_argument('--train', default=True,help='If True apply importance sampling')
    parser.add_argument('--integrand',  default='gauss_vf', type=str, help='Integrand')
    parser.add_argument('--dim',  default=4, type=int, help='Dimension of the integrand')
    parser.add_argument('--ncalls', default=1000000,type=int, help='Number of calls')
    parser.add_argument('--accuracy', default=1e-2,type=float, help='Percent uncertainty required')
    args = vars(parser.parse_args())

    integrator = args['integrator']
    adaptive = False if args['adaptive']=='False' else True
    train = False if args['train']=='False' else True
    integrand = eval(args['integrand'])
    dim = args['dim']
    ncalls = args['ncalls']
    accuracy = args['accuracy']

    #print(args)
    
    data = generate_data(VegasFlow,
                         integrator,
                         integrand,
                         dim,
                         ncalls,
                         accuracy,
                         train,
                         adaptive
                         )


    print(data)
    
    #print(data)
    with open("simulation_improved/drellyan/drellyan.json", "a") as f:
        json.dump(data,f,indent=True)
    
    

    #instance=VegasFlow(3,int(1e5),1e-2,"VegasFlowPlus",train=True,adaptive=True)
    #instance.set_integrand(pineappl)
    #result = instance.run_integration()
    #instance.show_content()

    #vegas_instance = VegasFlowPlus(3, int(1e5),adaptive=True)
    #vegas_instance.compile(pineappl)
    #vegas_instance.run_integration(5)

