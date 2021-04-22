from benchmark import gauss_vf, pineappl, singletop, drellyan
from benchmark import vfh_production_real, vfh_production_nlo, vfh_production_leading_order
from benchmark.functions.gauss import gauss_vf1
from benchmark.functions.singletop_lo_tf import singletop1
from benchmark.functions.drellyan_lo_tf import drellyan1
from benchmark.functions.pineappl import pineappl1
from benchmark.functions.higgs.higgs import vfh_production_real1, vfh_production_nlo1, vfh_production_leading_order1
from vegasflow.configflow import float_me, int_me,DTYPE,DTYPEINT, run_eager
from vegasflow import VegasFlowPlus
from benchmark import VegasFlow
from benchmark.benchmark import generate_data
from benchmark.utils import updateJsonFile
import argparse



if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrator', default='VegasFlow', type=str, help='Integrator type.')
    parser.add_argument('--adaptive', default=True, help='If True apply adaptive stratified sampling')
    parser.add_argument('--train', default=True,help='If True apply importance sampling')
    parser.add_argument('--integrand',  default='gauss_vf', type=str, help='Integrand')
    parser.add_argument('--dim',  default=4, type=int, help='Dimension of the integrand')
    parser.add_argument('--ncalls', default=1000000,type=int, help='Number of calls')
    parser.add_argument('--accuracy', default=1e-2,type=float, help='Percent uncertainty required')
    parser.add_argument('--outfile',default=None,help='Output file')
    parser.add_argument('--warmup',default=0,type=int)
    args = vars(parser.parse_args())

    integrator = args['integrator']
    adaptive = False if args['adaptive']=='False' else True
    train = False if args['train']=='False' else True
    integrand = eval(args['integrand'])
    dim = args['dim']
    ncalls = args['ncalls']
    accuracy = args['accuracy']
    outfile = args['outfile']
    #warmup = False if args['warmup']=='False' else True
    warmup = args['warmup']
    
    
    elem = generate_data(VegasFlow,
                         integrator,
                         integrand,
                         dim,
                         ncalls,
                         accuracy,
                         train,
                         adaptive,
                         warmup
                         )

    
    #elem['test'] = 'max_nhcube_1e4'
    
    if args['outfile'] is not None:
        updateJsonFile(outfile,elem,args['integrand'])
    
    
    #instance=VegasFlow(3,int(1e5),1e-2,"VegasFlowPlus",train=True,adaptive=True)
    #instance.set_integrand(pineappl)
    #result = instance.run_integration()
    #instance.show_content()

    #vegas_instance = VegasFlowPlus(6, int(1e6),adaptive=False)
    #vegas_instance.compile(vfh_production_leading_order)
    #vegas_instance.run_integration(5)

