import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import pickle
import networkx as nx
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from boltzmann_machine import *
from nilearn.datasets import fetch_abide_pcp
from joblib import Parallel,delayed

abide_data_control=fetch_abide_pcp(
                               pipeline="cpac",
                               derivatives=["rois_aal"],
                               band_pass_filtering=True,
                               data_dir="/AAL_atlas",
                               AGE_AT_SCAN=(15,50),
                               DX_GROUP=2,
                               SEX=1,
                               )

abide_data_autism=fetch_abide_pcp(
                               pipeline="cpac",
                               derivatives=["rois_aal"],
                               band_pass_filtering=True,
                               data_dir="AAL_atlas",
                               AGE_AT_SCAN=(15,50),
                               DX_GROUP=1,
                               SEX=1,
                               )
all_signals_control=[abide_data_control["rois_aal"][i][:,0:90] for i in range(len(abide_data_control["rois_aal"]))]
all_signals_autism=[abide_data_autism["rois_aal"][i][:,0:90] for i in range(len(abide_data_autism["rois_aal"]))]



even_ind=np.arange(0,90,2)
odd_ind=np.arange(1,90,2)[::-1]
rearrange_arr=np.hstack((even_ind,odd_ind))

labels=np.array(pd.read_csv("aal_labels.csv", skiprows=1).iloc[:,-1])[0:90]
labels=labels[rearrange_arr]

all_signals_control=[all_signals_control[i][:,rearrange_arr] for i in range(len(all_signals_control))]
all_signals_autism=[all_signals_autism[i][:,rearrange_arr] for i in range(len(all_signals_autism))]

num_rois=all_signals_control[0].shape[-1]
edge_ind=np.triu_indices(num_rois,k=1)

all_data={"control":[],"autism":[]}

all_data["autism"]=all_signals_autism
all_data["control"]=all_signals_control


#standardization

for signal in all_data["control"]:
    m=signal.mean()
    signal[signal>m]=1
    signal[signal<m]=-1
    

for signal in all_data["autism"]:
    m=signal.mean()
    signal[signal>m]=1
    signal[signal<m]=-1

for signal in all_data["parkinson"]:
    m=signal.mean()
    signal[signal>m]=1
    signal[signal<m]=-1   

mem_models={"control":[],"autism":[]} #data including J,h,beta
mem_objects={"control":[],"autism":[]} #data including all metropolis objects
mem_dispersion={"control":[],"autism":[]} #data to find critical temp
mem_critical_temp={"control":[],"autism":[]} 

for key in mem_models.keys():

    mem_models[key]=Parallel(n_jobs=-1)(delayed(MEM_estimator)(binarized_signal)
                                                 for binarized_signal in all_data[key])
    
    for model in mem_models[key]:
        mem_objects[key].append(MEM_Metropolis(model[1],model[0]))



with open("mem_models.pkl","wb") as file:
    pickle.dump(mem_models,file)

with open("mem_objects.pkl","wb") as file:
    pickle.dump(mem_objects,file)      


temp_list=np.arange(0.001,1,0.01)

def dispersion_index(temporal_matrix):
    cov=np.cov(temporal_matrix)
    return (cov.max()-cov.mean())/(np.var(cov)+0.0000000000000001) 


def dispersion_plot(mem_object,temp_list=temp_list):
    d_list=[]
    if mem_object.minima_matrix is None:
        mem_object.find_minima(10)
    
    for T in temp_list:
        beta=1/T
        temporal_matrix=mem_object.sample(beta)
        d_list.append(dispersion_index(temporal_matrix))

    return [temp_list,d_list]       


for key in mem_dispersion.keys():

    mem_dispersion[key]=Parallel(n_jobs=-1)(delayed(dispersion_plot)(mem_object)
                                                 for mem_object in mem_objects[key])
    
    for disp in mem_dispersion[key]:
        max_ind=np.argmax(np.array(disp[1]))
        mem_critical_temp[key].append(disp[0][max_ind])    


with open("mem_dispersion.pkl","wb") as file:
    pickle.dump(mem_dispersion,file)

with open("mem_critical_temp.pkl","wb") as file:
    pickle.dump(mem_critical_temp,file)      

