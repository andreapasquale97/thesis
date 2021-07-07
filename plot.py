from benchmark.utils import make_histo
from benchmark.utils import make_histo3
from benchmark.utils import make_histo4
from benchmark.utils import dim_comparison

if __name__ == '__main__':
    #make_histo("simulation_improved/higgs/lo_final_beta_025.json",outfile="lo_training_beta_025",save=True,showPlus=True)
    #make_histo3("simulation_max_iter_200/drellyan.json",outfile="new_plots/max_iterations_200/drellyan", save=True,title="Perfomance comparison for DrellYan")
    
    #make_histo4(function='gauss12d',save=False,title='Comparison for Gaussian Integral in 12 dimensions')
    dim_comparison(device='GPU')
    '''
    make_histo3("simulation_CPU/higgs_REAL.json",  
                outfile="CPU_plots/higgs_REAL",
                save=True,
                title="Perfomance comparison for higgs REAL")

    
    make_histo3("simulation_GPU/simulation_both_gpus/singletop.json",
                outfile="GPU_plots/singletop/gpu01",
                save=True,
                title="Perfomance comparison for singletop")

    make_histo3("simulation_GPU/simulation_rtx/singletop.json",
                outfile="GPU_plots/singletop/gpu1",
                save=True,
                title="Perfomance comparison for singletop")
    '''
    