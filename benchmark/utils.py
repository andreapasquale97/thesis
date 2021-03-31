import os
import json
from benchmark import pineappl
from benchmark import singletop, gauss_vf, VegasFlow,drellyan
from vegasflow.configflow import run_eager 
from benchmark.vegasflow import simulation
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(data=None,data1=None,rtol=1e-2,index=None):

    if data1 == None:
        vegas_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False] 
        importance_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol]
        vegas_time = [i["time"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False]
        importance_time = [i["time"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol]

        df_iter = pd.DataFrame({'vegasflow' : importance_iter,'vegasflowplus' : vegas_iter },index=index)
        df_time = pd.DataFrame({'vegasflow' : importance_time,'vegasflowplus' : vegas_time },index=index)
    else:
        vegas_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False]
        vegasplus_iter = [i["iter"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True] 
        importance_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol]
        vegas_time = [i["time"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False]
        vegasplus_time = [i["time"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True]
        importance_time = [i["time"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol] 

        df_iter = pd.DataFrame({'vegasflowplus adaptive' : vegasplus_iter, 'vegasflow' : importance_iter,'vegasflowplus' : vegas_iter },index=index)
        df_time = pd.DataFrame({'vegasflowplus adaptive' : vegasplus_time, 'vegasflow' : importance_time,'vegasflowplus' : vegas_time },index=index)

    return df_iter, df_time

def make_histo(infile=None, outfile=None, save=False, showPlus=True):
 
    with open(infile, "r") as f:
        all = json.load(f)
        data = []
        for key in all.keys():
            data.append(all[key]) 
        
    index = ['1000', '10000', '100000', '1000000']
    index1 = ['1000', '10000', '100000']
    
    if not showPlus:
        fig, axs  = plt.subplots(3, 2, sharey='row',figsize=(10, 12))
        fig.text(0.02, 0.5, 'samples/iteration', ha='center', va='center', rotation='vertical')
        df_iter1, df_time1 = prepare_data(data=data[0],rtol=1e-2,index=index)
        df_iter2, df_time2 = prepare_data(data=data[0],rtol=1e-3,index=index)
        df_iter3, df_time3 = prepare_data(data=data[0],rtol=1e-4,index=index)
    else:
        fig, axs  = plt.subplots(3, 2, sharey='row',figsize=(10, 12))
        fig.text(0.02, 0.5, 'samples/iteration', ha='center', va='center', rotation='vertical')     
        df_iter1, df_time1 = prepare_data(data=data[0],data1=data[1],rtol=1e-2,index=index)
        df_iter2, df_time2 = prepare_data(data=data[0],data1=data[1],rtol=1e-3,index=index)
        df_iter3, df_time3 = prepare_data(data=data[0],data1=data[1],rtol=1e-4,index=index)
    

    axs[0,0] = df_time1.plot.barh(ax=axs[0,0],legend=False)
    axs[0,0].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,1] = df_iter1.plot.barh(ax=axs[0,1],legend=False)
    axs[0,1].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,0] = df_time2.plot.barh(ax=axs[1,0],legend=False)
    axs[1,0].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,1] = df_iter2.plot.barh(ax=axs[1,1],legend=False)
    axs[1,1].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,0] = df_time3.plot.barh(ax=axs[2,0],legend=False)
    axs[2,0].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,1] = df_iter3.plot.barh(ax=axs[2,1],legend=False)
    axs[2,1].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,0].set_xlabel('time (s)')
    axs[2,1].set_xlabel('iterations')

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels,loc='center right')
    if save:
        plt.savefig(outfile,bbox_inches='tight')
    else:
        plt.show()
        
def updateJsonFile(file,elem,key):

    if os.path.isfile(file):
        jsonFile = open(file, "r")
        data = json.load(jsonFile) 
        jsonFile.close()
    else: 
        data = {}
        data[key] = []

    if key in data:
        data[key].append(elem)
    else:
        data[key] = []
        data[key].append(elem)

    jsonFile = open(file, "w+")
    jsonFile.write(json.dumps(data,indent=True))
    jsonFile.close()