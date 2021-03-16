from benchmark import pineappl
from benchmark import singletop, gauss_vf, VegasFlow,drellyan
from vegasflow.configflow import run_eager 
from benchmark.vegasflow import simulation
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import json

def prepare_data(data=None,data1=None,rtol=1e-2,index=None,showEager=False):

    vegas_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False and i["samples/iter"] != 1000000]
    vegasplus_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True  and i["samples/iter"] != 1000000] 
    importance_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol  and i["samples/iter"] != 1000000]
    #vegasplus_eager_iter = [i["iter"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["samples/iter"] != 1000000] 
    #adaptive_stratified_iter = [i["iter"] for i in data if i["integrator"] == "adaptive stratified" and i["perc_uncertainty"] == rtol ]
    #stratified_iter = [i["iter"] for i in data if i["integrator"] == "stratified" and i["perc_uncertainty"] == rtol ]
   


    vegas_time = [i["time"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False  and i["samples/iter"] != 1000000 ]
    vegasplus_time = [i["time"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True  and i["samples/iter"] != 1000000]
    importance_time = [i["time"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol  and i["samples/iter"] != 1000000]
    #vegasplus_eager_time = [i["time"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["samples/iter"] != 1000000]
    #adaptive_stratified_time = [i["time"] for i in data if i["integrator"] == "adaptive stratified" and i["perc_uncertainty"] == rtol]
    #stratified_time = [i["time"] for i in data if i["integrator"] == "stratified" and i["perc_uncertainty"] == rtol]

    #if showEager:
    #    df_iter = pd.DataFrame({'vegas+' : vegasplus_iter, 'vegas' : vegas_iter,
    #                            'importance' : importance_iter,
    #                            'adaptive stratified' : adaptive_stratified_iter,
    #                            'stratified' : stratified_iter 
    #                            },index=index)
    #    df_time = pd.DataFrame({'vegas+' : vegasplus_time, 'vegas' : vegas_time,
    #                            'importance' : importance_time,
    #                            'adaptive stratified' : adaptive_stratified_time,
    #                            'stratified' : stratified_time
    #                            },index=index)
    #else:
    df_iter = pd.DataFrame({'vegas+' : vegasplus_iter, 'importance' : importance_iter,'vegas' : vegas_iter },index=index)
    df_time = pd.DataFrame({'vegas+' : vegasplus_time, 'importance' : importance_time,'vegas' : vegas_time },index=index)

    #df_iter = pd.DataFrame({'importance' : importance_iter,'vegas' : vegas_iter },index=index)
    #df_time = pd.DataFrame({'importance' : importance_time,'vegas' : vegas_time },index=index)

    #df_iter = pd.DataFrame({'vegas+' : vegasplus_iter, 'vegas+ eager' : vegasplus_eager_iter,'importance' : importance_iter,'vegas' : vegas_iter },index=index)
    #df_time = pd.DataFrame({'vegas+' : vegasplus_time, 'vegas+ eager': vegasplus_eager_time,'importance' : importance_time,'vegas' : vegas_time },index=index)

    return df_iter, df_time

def make_histo(infile=None,infile1=None, outfile=None, save=False, showEager=True):
 


    with open(infile, "r") as f:
        data = json.load(f)

    #with open(infile1, "r") as f1:
    #    data1 = json.load(f1)

    fig, axs  = plt.subplots(2, 2, sharey='row',figsize=(10, 10))#12
    fig.suptitle('Single Top')
    fig.text(0.02, 0.5, 'samples/iteration', ha='center', va='center', rotation='vertical')
    #index = ['1000', '10000', '100000', '1000000']
    index = ['1000', '10000', '100000']
    #index = ['1000','10000']

    df_iter1, df_time1 = prepare_data(data=data,rtol=1e-2,index=index)
    df_iter2, df_time2 = prepare_data(data=data,rtol=1e-3,index=index)
    #df_iter3, df_time3 = prepare_data(data=data,rtol=1e-4,index=index)

    #df_iter1, df_time1 = prepare_data(data=data,data1=data1,rtol=1e-2,index=index)
    #df_iter2, df_time2 = prepare_data(data=data,data1=data1,rtol=1e-3,index=index)
    #df_iter3, df_time3 = prepare_data(data=data,data1=data1,rtol=1e-4,index=index)
    

    axs[0,0] = df_time1.plot.barh(ax=axs[0,0],legend=False)
    axs[0,0].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,1] = df_iter1.plot.barh(ax=axs[0,1],legend=False)
    axs[0,1].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,0] = df_time2.plot.barh(ax=axs[1,0],legend=False)
    axs[1,0].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,1] = df_iter2.plot.barh(ax=axs[1,1],legend=False)
    axs[1,1].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    #axs[2,0] = df_time3.plot.barh(ax=axs[2,0],legend=False)
    #axs[2,0].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    #axs[2,1] = df_iter3.plot.barh(ax=axs[2,1],legend=False)
    #axs[2,1].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})


    #labelling
    #axs[0,0].set_xlabel('time (s)')
    #axs[0,1].set_xlabel('iterations')
    axs[1,0].set_xlabel('time (s)')
    axs[1,1].set_xlabel('iterations')
    #axs[2,0].set_xlabel('time (s)')
    #axs[2,1].set_xlabel('iterations')
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels,loc='center right')
    #for ax in axs.flat:
    #    ax.label_outer()
    if save:
        plt.savefig(outfile,bbox_inches='tight')
    else:
        plt.show()
        


if __name__ == '__main__':

    #make_histo("simulation/gauss/gauss_8d.json","simulation/gauss/gauss_8d_eager.json", outfile="gauss8d_vegasflow_zoom_eager",save=True)
    make_histo("simulation_improved/singletop/singletop.json",outfile="new_singletop1",save=True)
