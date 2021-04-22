from benchmark.utils import make_histo
from benchmark.utils import make_histo3

if __name__ == '__main__':
    #make_histo("simulation_improved/higgs/lo_final_beta_025.json",outfile="lo_training_beta_025",save=True,showPlus=True)
    make_histo3("simulation_max_iter_200/gauss12d.json",outfile="new_plots/max_iterations_200/gauss12d", save=True,title="Perfomance comparison for Gauss 12 d")
    #make_histo3("simulation_max_iter_200/gauss8d.json", save=False,title="Perfomance comparison for gauss 8d")