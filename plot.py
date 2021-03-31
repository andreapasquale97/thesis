from test import make_histo
from benchmark.utils import make_histo

if __name__ == '__main__':
    make_histo("simulation_improved/higgs/real_final.json",outfile="higgs_real_training",save=True,showPlus=False)