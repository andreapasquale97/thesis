from benchmark import gauss_vf, pineappl, singletop, drellyan
from benchmark import vfh_production_real, vfh_production_nlo, vfh_production_leading_order
from benchmark.functions.gauss import gauss_vf1
from benchmark.functions.singletop_lo_tf import singletop1
from benchmark.functions.drellyan_lo_tf import drellyan1
from benchmark.functions.pineappl import pineappl1
from benchmark.functions.higgs.higgs import vfh_production_real1, vfh_production_nlo1, vfh_production_leading_order1
from vegasflow.configflow import float_me, int_me,DTYPE,DTYPEINT
import vegasflow
import argparse
import time



if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrator', default='VegasFlow', type=str, help='Integrator type.')
    parser.add_argument('--adaptive', default=True, help='If True apply adaptive stratified sampling')
    parser.add_argument('--train', default=True,help='If True apply importance sampling')
    parser.add_argument('--integrand',  default='gauss_vf', type=str, help='Integrand')
    parser.add_argument('--dim',  default=4, type=int, help='Dimension of the integrand')
    parser.add_argument('--ncalls', default=1000000,type=int, help='Number of calls')
    parser.add_argument('--niter', default=10,type=int, help='Number of iterations')

    #parser.add_argument('--accuracy', default=1e-2,type=float, help='Percent uncertainty required')
    #parser.add_argument('--outfile',default=None,help='Output file')
    #parser.add_argument('--warmup',default=True)
    args = vars(parser.parse_args())

    integrator = args['integrator']
    adaptive = False if args['adaptive']=='False' else True
    train = False if args['train']=='False' else True
    integrand = eval(args['integrand'])
    dim = args['dim']
    ncalls = args['ncalls']
    #accuracy = args['accuracy']
    #outfile = args['outfile']
    #warmup = False if args['warmup']=='False' else True
    niter = args['niter']


    if adaptive:
        # set integrand with input signature if adaptive is ON
        integrand = eval(args['integrand']+"1")

    if integrator == 'VegasFlowPlus':
        instance = getattr(vegasflow,integrator)(dim,ncalls,train=train,adaptive=adaptive)
    else:
        instance = getattr(vegasflow,integrator)(dim,ncalls,train=train)   

    # training with half iterations
    niter_warmup = int(niter/2)
    instance.compile(integrand)
    start = time.time()
    instance.run_integration(niter_warmup)
    print("> Freeze grid and stop sampling redistribution in hypercubes")
    instance.freeze_grid()
    instance.adaptive=False
    # integration at fixed grid and fixed samples per hypercube
    instance.run_integration(niter-niter_warmup)
    end = time.time()
    print(f"{integrator} took time {end-start}.")


    
    