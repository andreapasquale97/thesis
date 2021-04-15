from benchmark.utils import make_histo
from benchmark.utils import make_histo2

if __name__ == '__main__':
    #make_histo("simulation_improved/higgs/lo_final_beta_025.json",outfile="lo_training_beta_025",save=True,showPlus=True)
    make_histo2("new_simulation/singletop.json",outfile="new_plots/vflowplus_fixed/singletop", save=True,title="Perfomance comparison for Drellyan")