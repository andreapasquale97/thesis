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

def compute_rtol_from_result(results):
    rtol = [ float(i.split("+/-")[1])/float(i.split("+/-")[0])  for i in results]
    return rtol

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

def prepare_data1(data=None,data1=None,rtol=1e-2,index=None):

    vegas_iter = [i["iter"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False] 
    importance_iter = [i["iter"] for i in data1 if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol]
    vegas_nhcube1_iter = [i["iter"] for i in data if i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["test"]== 'max_neval_hcube_1e3'] 
    vegas_nhcube2_iter = [i["iter"] for i in data if i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["test"]== 'max_neval_hcube_1e4']
    vegas_numpy_indices_iter = [i["iter"] for i in data if i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["test"]== 'numpy indices'] 
    vegas_nhcube_iter = [i["iter"] for i in data if i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["test"]== 'max_nhcube_1e4'] 



    vegas_rtol = compute_rtol_from_result([i["result"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False])
    importance_rtol = compute_rtol_from_result([i["result"] for i in data1 if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol])
    vegas_nhcube1_rtol = compute_rtol_from_result([i["result"] for i in data if i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["test"]== 'max_neval_hcube_1e3'] )
    vegas_nhcube2_rtol = compute_rtol_from_result([i["result"] for i in data if i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["test"]== 'max_neval_hcube_1e4'])
    vegas_numpy_indices_rtol = compute_rtol_from_result([i["result"] for i in data if i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["test"]== 'numpy indices'] )
    vegas_nhcube_rtol = compute_rtol_from_result([i["result"] for i in data if i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["test"]== 'max_nhcube_1e4'])

    df_iter = pd.DataFrame({'vf' : importance_iter,
                            'vf+ not adaptive' : vegas_iter,
                            'vf+ max samples per hcube 1e3': vegas_nhcube1_iter, 
                            'vf+ max samples per hcube 1e4': vegas_nhcube2_iter,
                            'vf+ with numpy indices': vegas_numpy_indices_iter,
                            'vf+ max hcube 1e4': vegas_nhcube_iter
                            },index=index)

    df_rtol= pd.DataFrame({'vf' : importance_rtol,
                            'vf+ not adaptive' : vegas_rtol,
                            'vf+ max samples per hcube 1e3': vegas_nhcube1_rtol, 
                            'vf+ max samples per hcube 1e4': vegas_nhcube2_rtol,
                            'vf+ with numpy indices': vegas_numpy_indices_rtol,
                            'vf+ max hcube 1e4': vegas_nhcube_rtol
                            },index=index)
    return df_iter, df_rtol

def prepare_data2(data=None,data1=None,rtol=1e-2,index=None):

    vegas_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False]
    vegasplus_iter = [i["iter"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True] 
    importance_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol]

    vegas_rtol = compute_rtol_from_result([i["result"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False])
    importance_rtol = compute_rtol_from_result([i["result"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol])
    vegasplus_rtol = compute_rtol_from_result([i["result"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True])


    vegas_time = [i["time"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False]
    vegasplus_time = [i["time"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True]
    importance_time = [i["time"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol] 

    df_iter = pd.DataFrame({'vegasflowplus adaptive' : vegasplus_iter, 'vegasflow' : importance_iter,'vegasflowplus' : vegas_iter },index=index)
    df_time = pd.DataFrame({'vegasflowplus adaptive' : vegasplus_time, 'vegasflow' : importance_time,'vegasflowplus' : vegas_time },index=index)
    df_rtol= pd.DataFrame({'vegasflowplus adaptive' : vegasplus_rtol, 'vegasflow' : importance_rtol,'vegasflowplus' : vegas_rtol },index=index)

    return df_iter, df_time, df_rtol

def prepare_data3(data=None,data1=None,rtol=1e-2,index=None):

    vegas_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False]
    importance_iter = [i["iter"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol]
    vegasplus_iter = [i["iter"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["warmup"] == 1] 
    vegasplus1_iter = [i["iter"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["warmup"] == 2] 

    vegas_rtol = [i["rtol_reached"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False]
    importance_rtol = [i["rtol_reached"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol]
    vegasplus_rtol = [i["rtol_reached"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["warmup"] == 1]
    vegasplus1_rtol = [i["rtol_reached"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["warmup"] == 2]

    vegas_time = [i["time"] for i in data if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == False]
    importance_time = [i["time"] for i in data if i["integrator"] == "VegasFlow" and i["perc_uncertainty"] == rtol] 
    vegasplus_time = [i["time"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["warmup"] == 1]
    vegasplus1_time = [i["time"] for i in data1 if i["integrator"] == "VegasFlowPlus" and i["perc_uncertainty"] == rtol and i["adaptive"] == True and i["warmup"] == 2]
    
    df_iter = pd.DataFrame({'vegasflow' : importance_iter,
                            'vegasflowplus' : vegas_iter,
                            'vegasflowplus adaptive' : vegasplus_iter,
                            'vegasflowplus adaptive after warmup' : vegasplus1_iter   
                            },index=index)

    df_time = pd.DataFrame({'vegasflow' : importance_time,
                            'vegasflowplus' : vegas_time,
                            'vegasflowplus adaptive' : vegasplus_time,
                            'vegasflowplus adaptive after warmup' : vegasplus1_time
                             },index=index)

    df_rtol= pd.DataFrame({'vegasflow' : importance_rtol,
                           'vegasflowplus' : vegas_rtol,
                           'vegasflowplus adaptive' : vegasplus_rtol,
                           'vegasflowplus adaptive after warmup' : vegasplus1_rtol
                            },index=index)

    return df_iter, df_time, df_rtol



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

def make_histo1(infile=None, outfile=None, save=False):
 
    with open(infile, "r") as f:
        all = json.load(f)
        data = []
        for key in all.keys():
            data.append(all[key]) 
    
    #print(data)
    #print(data[1])
    #print(data[1])
    index = ['100000', '1000000']

    fig, axs  = plt.subplots(3, 2, sharey='row',figsize=(14, 10))
    fig.suptitle("Perfomance comparison for Gauss8d", fontsize=14)
    fig.text(0.02, 0.5, 'samples/iteration', ha='center', va='center', rotation='vertical')
    df_iter1, df_rtol1 = prepare_data1(data=data[0],data1=data[1],rtol=1e-2,index=index)
    df_iter2, df_rtol2 = prepare_data1(data=data[0],data1=data[1],rtol=1e-3,index=index)
    df_iter3, df_rtol3 = prepare_data1(data=data[0],data1=data[1],rtol=1e-4,index=index)



    axs[0,0] = df_rtol1.plot.barh(ax=axs[0,0],legend=False)
    axs[0,0].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))
    axs[0,1] = df_iter1.plot.barh(ax=axs[0,1],legend=False)
    axs[0,1].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,0] = df_rtol2.plot.barh(ax=axs[1,0],legend=False)
    axs[1,0].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))
    axs[1,1] = df_iter2.plot.barh(ax=axs[1,1],legend=False)
    axs[1,1].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,0] = df_rtol3.plot.barh(ax=axs[2,0],legend=False)
    axs[2,0].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))
    axs[2,1] = df_iter3.plot.barh(ax=axs[2,1],legend=False)
    axs[2,1].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,0].set_xlabel('rtol')
    axs[2,1].set_xlabel('iterations')

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels)
    if save:
        plt.savefig(outfile,bbox_inches='tight')
    else:
        plt.show()

    """
    axs[0] = df_iter1.plot.barh(ax=axs[0],legend=False)
    axs[0].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1] = df_iter2.plot.barh(ax=axs[1],legend=False)
    axs[1].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2] = df_iter3.plot.barh(ax=axs[2],legend=False)
    axs[2].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2].set_xlabel('iterations')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    if save:
        plt.savefig(outfile,bbox_inches='tight')
    else:
        plt.show()
    """

def make_histo2(infile=None, outfile=None, save=False,title=str):
    with open(infile, "r") as f:
        all = json.load(f)
        data = []
        for key in all.keys():
            data.append(all[key]) 
    



    index = ['1000', '10000', '100000', '1000000']
    index1 = ['1000', '10000', '100000']
    

    fig, axs  = plt.subplots(3, 3, sharey='row',figsize=(13, 9))
    fig.text(0.02, 0.5, 'samples/iteration', ha='center', va='center', rotation='vertical')
    df_iter1, df_time1, df_rtol1 = prepare_data2(data=data[0],data1=data[1],rtol=1e-2,index=index)
    df_iter2, df_time2, df_rtol2 = prepare_data2(data=data[0],data1=data[1],rtol=1e-3,index=index)
    df_iter3, df_time3, df_rtol3 = prepare_data2(data=data[0],data1=data[1],rtol=1e-4,index=index)
    

    fig.suptitle(title, fontsize=14)

    axs[0,0] = df_time1.plot.barh(ax=axs[0,0],legend=False)
    axs[0,0].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,1] = df_iter1.plot.barh(ax=axs[0,1],legend=False)
    axs[0,1].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,2] = df_rtol1.plot.barh(ax=axs[0,2],legend=False)
    axs[0,2].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,2].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))

    axs[1,0] = df_time2.plot.barh(ax=axs[1,0],legend=False)
    axs[1,0].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,1] = df_iter2.plot.barh(ax=axs[1,1],legend=False)
    axs[1,1].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,2] = df_rtol2.plot.barh(ax=axs[1,2],legend=False)
    axs[1,2].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,2].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))

    axs[2,0] = df_time3.plot.barh(ax=axs[2,0],legend=False)
    axs[2,0].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,1] = df_iter3.plot.barh(ax=axs[2,1],legend=False)
    axs[2,1].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,2] = df_rtol3.plot.barh(ax=axs[2,2],legend=False)
    axs[2,2].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,2].ticklabel_format(style = 'sci', axis='x', scilimits=(0,0))
    axs[2,2].set_xlim([0,1e-3])

    axs[2,0].set_xlabel('time (s)')
    axs[2,1].set_xlabel('iterations')
    axs[2,2].set_xlabel('rtol')


    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels)
    if save:
        plt.savefig(outfile,bbox_inches='tight')
    else:
        plt.show()


def make_histo3(infile=None, outfile=None, save=False,title=str):
    with open(infile, "r") as f:
        all = json.load(f)
        data = []
        for key in all.keys():
            data.append(all[key]) 

    #index = ['100000', '1000000']
    #index1 = ['1000', '10000', '100000']
    index = ['1000000']
    fig, axs  = plt.subplots(3, 3, sharey='row',figsize=(13, 10))
    fig.text(0.02, 0.5, 'samples/iteration', ha='center', va='center', rotation='vertical')
    df_iter1, df_time1, df_rtol1 = prepare_data3(data=data[0],data1=data[1],rtol=1e-2,index=index)
    df_iter2, df_time2, df_rtol2 = prepare_data3(data=data[0],data1=data[1],rtol=1e-3,index=index)
    df_iter3, df_time3, df_rtol3 = prepare_data3(data=data[0],data1=data[1],rtol=1e-4,index=index)
    

    fig.suptitle(title, fontsize=14)

    axs[0,0] = df_time1.plot.barh(ax=axs[0,0],legend=False)
    #axs[0,0].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,1] = df_iter1.plot.barh(ax=axs[0,1],legend=False)
    #axs[0,1].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,2] = df_rtol1.plot.barh(ax=axs[0,2],legend=False)
    #axs[0,2].set_title('Percent Uncertainty 0.01',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[0,2].axvline(0.01,color='black',ls='--')
    axs[0,2].ticklabel_format(style = 'sci', axis='x',  scilimits=(-2,-2))

    axs[1,0] = df_time2.plot.barh(ax=axs[1,0],legend=False)
    #axs[1,0].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,1] = df_iter2.plot.barh(ax=axs[1,1],legend=False)
    #axs[1,1].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,2] = df_rtol2.plot.barh(ax=axs[1,2],legend=False)
    #axs[1,2].set_title('Percent Uncertainty 0.001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[1,2].axvline(0.001,color='black',ls='--')
    axs[1,2].ticklabel_format(style = 'sci', axis='x', scilimits=(-3,-3))

    axs[2,0] = df_time3.plot.barh(ax=axs[2,0],legend=False)
    #axs[2,0].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,1] = df_iter3.plot.barh(ax=axs[2,1],legend=False)
    #axs[2,1].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,2] = df_rtol3.plot.barh(ax=axs[2,2],legend=False)
    #axs[2,2].set_title('Percent Uncertainty 0.0001',fontdict={'fontsize': 8, 'fontweight': 'medium'})
    axs[2,2].axvline(0.0001,color='black',ls='--')
    axs[2,2].ticklabel_format(style = 'sci', axis='x', scilimits=(-4,-4))
    #axs[2,2].set_xlim([0,1e-3])

    axs[2,0].set_xlabel('time (s)')
    axs[2,1].set_xlabel('iterations')
    axs[2,2].set_xlabel('rtol')


    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels)
    if save:
        plt.savefig(outfile,bbox_inches='tight')
    else:
        plt.show()


def make_histo4(function=str, save=False, title=str):
    
    path_CPU= f'simulation_CPU/{function}.json'
    path_GPU0 = f'simulation_GPU/simulation_titan/{function}.json'
    path_GPU1 = f'simulation_GPU/simulation_rtx/{function}.json'
    path_GPU01 = f'simulation_GPU/simulation_both_gpus/{function}.json'

    index = ['CPU', 'GPU0', 'GPU1', 'GPU01']

    files = [path_CPU, path_GPU0, path_GPU1, path_GPU01]

    data =[]
    for file in files:
        with open(file, "r") as f:
            all = json.load(f)
            for key in all.keys():
                data.append(all[key])

    not_adaptive = data[0]+data[2]+data[4]+data[6]
    adaptive = data[1]+data[3]+data[5]+data[7]

    fig, axs  = plt.subplots(3, 3, sharey='row',figsize=(13, 10))
    #fig.text(0.02, 0.5, 'samples/iteration', ha='center', va='center', rotation='vertical')
    df_iter1, df_time1, df_rtol1 = prepare_data3(data=not_adaptive,data1=adaptive,rtol=1e-2,index=index)
    df_iter2, df_time2, df_rtol2 = prepare_data3(data=not_adaptive,data1=adaptive,rtol=1e-3,index=index)
    df_iter3, df_time3, df_rtol3 = prepare_data3(data=not_adaptive,data1=adaptive,rtol=1e-4,index=index)

    fig.suptitle(title, fontsize=14)

    axs[0,0] = df_time1.plot.barh(ax=axs[0,0],legend=False)
    axs[0,1] = df_iter1.plot.barh(ax=axs[0,1],legend=False)
    axs[0,2] = df_rtol1.plot.barh(ax=axs[0,2],legend=False)
    axs[0,2].axvline(0.01,color='black',ls='--')
    axs[0,2].ticklabel_format(style = 'sci', axis='x',  scilimits=(-2,-2))

    axs[1,0] = df_time2.plot.barh(ax=axs[1,0],legend=False)
    axs[1,1] = df_iter2.plot.barh(ax=axs[1,1],legend=False)
    axs[1,2] = df_rtol2.plot.barh(ax=axs[1,2],legend=False)
    axs[1,2].axvline(0.001,color='black',ls='--')
    axs[1,2].ticklabel_format(style = 'sci', axis='x', scilimits=(-3,-3))

    axs[2,0] = df_time3.plot.barh(ax=axs[2,0],legend=False)
    axs[2,1] = df_iter3.plot.barh(ax=axs[2,1],legend=False)
    axs[2,2] = df_rtol3.plot.barh(ax=axs[2,2],legend=False)
    axs[2,2].axvline(0.0001,color='black',ls='--')
    axs[2,2].ticklabel_format(style = 'sci', axis='x', scilimits=(-4,-4))

    axs[2,0].set_xlabel('time (s)')
    axs[2,1].set_xlabel('iterations')
    axs[2,2].set_xlabel('rtol')


    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels)
    outfile = f'plots_CPU_GPU/{function}'
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