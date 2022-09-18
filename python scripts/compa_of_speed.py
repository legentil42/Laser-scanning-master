#%%
from base64 import b16decode
from cmath import exp
from ensurepip import version
from ipaddress import ip_address
import itertools
from tracemalloc import start
import pandas as pd
import pptk
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from tkinter import *
import time


def import_my_3DPC(i,L_names,show = True,r=500):
    
    L_SCAN = []
    if results[i][2] == 1:
            
        with open(L_names[i], 'r') as f:
            for line  in f:
                L_SCAN.append(line.strip().replace(" ","").replace("[","").replace("]","").split(","))
                L_SCAN[-1] = [float(i) for i in L_SCAN[-1]]
                if L_SCAN[-1] == [0,0,0]:
                    print("hi")
                    L_SCAN.pop(-1)
        if show:
            v = pptk.viewer(L_SCAN)
            v.set(point_size=0.1)
            v.set(r=r)
            print(L_names[i])

        return L_SCAN
    else:
        with open(L_names[i], 'r') as f:
            for line  in f:
                L_SCAN.append(line.strip().replace(" ","").replace("[","").replace("]","").split(","))
                L_SCAN[-1] = [float(i) for i in L_SCAN[-1]]
                L_SCAN[-1][0],L_SCAN[-1][1]=L_SCAN[-1][1],L_SCAN[-1][0]
                if L_SCAN[-1] == [0,0,0]:
                    print("hi")
                    L_SCAN.pop(-1)
        if show:
            v = pptk.viewer(L_SCAN)
            v.set(point_size=0.1)

        return L_SCAN
        
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return nodes[np.argmin(dist_2)]

    
def import_my_3DPC_simple(name,show = True):
    
    L_SCAN = []
            
    with open(name, 'r') as f:
        for line  in f:
            L_SCAN.append(line.strip().replace(" ","").replace("[","").replace("]","").split(","))
            L_SCAN[-1] = [float(i) for i in L_SCAN[-1]]
            if L_SCAN[-1] == [0,0,0]:
                print("hi")
                L_SCAN.pop(-1)
    if show:
        v = pptk.viewer(L_SCAN)
        v.set(point_size=0.1)

    return L_SCAN


def keep_top_z(L_SCAN,z,tolerance,show = True):
    L_kept = []
    for i in range(len(L_SCAN)):
        if abs(L_SCAN[i][2]-z) < tolerance:
            L_kept.append(L_SCAN[i])
    if show:
        v = pptk.viewer(L_kept)
        v.set(point_size=0.1)

    return L_kept

def PC_view_of_these_lists(L_Lists,point_size=0.1):
    to_view = []
    colors = []
    for i in range(len(L_Lists)):
        to_view += L_Lists[i]
        colors += [i] * len(L_Lists[i])
    v = pptk.viewer(to_view,colors)
    v.set(point_size=point_size)

def bufcount(filename):
    f = open(filename)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines


def sum_dist_each_point(L1,L2):
    import sys
    sum = 0
    if len(L1) == len(L2):
        for i in range(len(L1)):
            sum+= np.linalg.norm(np.array(L1[i])-np.array(L2[i]))**2
            sys.stdout.write('\r'
            + str(round(100*i/len(L2),2))+'%')
        return sum/len(L1)
    else:
        if len(L1)>len(L2):
    
            for i in range(len(L2)):
                closest_to_i = closest_node(L2[i],L1)
                sum+= np.linalg.norm(np.array(closest_to_i)-np.array(L2[i]))**2
                sys.stdout.write('\r'
                + str(round(100*i/len(L2),2))+'%')
            return sum/len(L2)

        elif len(L2)>len(L1):
    
            for i in range(len(L1)):
                closest_to_i = closest_node(L1[i],L2)
                sum+= np.linalg.norm(np.array(closest_to_i)-np.array(L1[i]))**2
                sys.stdout.write('\r'
                + str(round(100*i/len(L1),2))+'%')
            return sum/len(L1) 


def db_generation():
    L_res = [4.6943197481337,
    4.168721917205156,
    3.4458427580970734,
    3.7586734910604895,
    3.7586734910604895,
    4.823765524444423,
    2.8591633506372793,
    4.81655088797619,
    4.295238012366545,
    4.361035051436667,
    4.277354423736898,
    3.829772817538342,
    0.0,
    3.029190277273864,
    4.085293462445963,
    3.7243279792999973,
    4.26313692339939,
    4.700902549515067,
    3.9645235363815488,
    4.088406426242154,
    3.8884353378498515]

    results_path = r"C:\Users\legen\Desktop\dissertation\Results_cylinder_speed"

    results = [f for f in listdir(results_path) if isfile(join(results_path, f))]
    file_paths = results.copy()
    file_paths = [results_path+"\\" +results[i_file_name] for i_file_name in range(len(file_paths))]

    for i_res in range(len(results)):
        num_lines = sum(1 for _ in open(results_path+"\\" +results[i_res]))
        results[i_res] = list(map(float,(results[i_res][:-4].split(" ")))) + [bufcount(results_path+"\\" +results[i_res])]

    print(results)

  
    #Create the pandas DataFrame
    df = pd.DataFrame(results, columns=['Time (s)', 'Nb_steps', 'direction','speed_factor','Nb_points'])
    df['L_res'] = L_res
    # print dataframe.
    print(df)
    return df, file_paths,results


#%% DB GENERATION
df, file_paths,results=db_generation()
df = df.sort_values(by=['speed_factor'])

# %%

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
df_bis = df.copy()
df_bis = df_bis.rename(columns={'Time (s)': "Duration (s)", "Nb_steps": "N_steps","direction":"Direction","speed_factor":"Speed_factor"})


# %%

L_SCAN = import_my_3DPC(14,file_paths,show = True,r=350)
# %%
L_SCAN = import_my_3DPC(7,file_paths,show = True,r=350)

L_SCAN = import_my_3DPC(1,file_paths,show = True,r=350)

L_SCAN = import_my_3DPC(16,file_paths,show = True,r=350)


# %%
L_factors = [25,50,75,100,125,150,175,200]
L_SCAN = import_my_3DPC(12,file_paths,show = False)
L_kept = keep_top_z(L_SCAN,8.5,1,show = False)
L_small_area = []
# %%
L_res = []
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14, 15, 16, 17, 18, 19, 20]:
    L_SCAN2 = import_my_3DPC(i,file_paths,show = False)
    L_kept2 = keep_top_z(L_SCAN2,8.5,1,show = False)
    L_small_area2 = []

    print(i)
    L_res.append(sum_dist_each_point(L_small_area,L_kept2))

#%%

import numpy as np
L_speeds = list(set(df['speed_factor']))
L_times_by_speed = []
L_res_by_speed = []
L_std_times_by_speed = []
L_std_res_by_speed = []

for speed in L_speeds:
    L_times_this_speed = []
    L_res_this_speed = []
    for index, row in df.iterrows():
        if speed == row['speed_factor'] and row['L_res']<10 and row['L_res'] != 0:
            print(row['L_res'])
            L_times_this_speed.append(row['Time (s)'])
            L_res_this_speed.append(row['L_res'])
        
    L_times_by_speed.append(np.mean(L_times_this_speed))
    L_res_by_speed.append(np.mean(L_res_this_speed))
    L_std_times_by_speed.append(np.std(L_times_this_speed))
    L_std_res_by_speed.append(np.std(L_res_this_speed))


plt.errorbar(L_speeds,L_times_by_speed,L_std_times_by_speed, linestyle='None', marker='^')
plt.xlabel('speed of scanner')
plt.ylabel('Time')
plt.title("crack_")
plt.ylim(0,max(L_times_by_speed)*1.2)
plt.legend()
plt.grid()
plt.show()

plt.errorbar(L_speeds,L_res_by_speed,L_std_res_by_speed, linestyle='None', marker='^')
plt.xlabel('speed of scanner')
plt.ylabel('Error')
plt.title("crack_")
plt.ylim(0,max(L_res_by_speed)*1.2)
plt.legend()
plt.grid()
plt.show()

# %%

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 100

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.errorbar(L_speeds,L_times_by_speed,L_std_times_by_speed, linestyle='None', marker='o',color = 'r')
# set x-axis label
ax.set_xlabel('Speed factor of the scan', fontsize = 14)
# set y-axis label
ax.set_ylabel("Duration (s)",
              color="red",
              fontsize=14)
ax.set_ylim(0,max(L_times_by_speed)*1.2)
ax.set_xticks(L_speeds)
[i.set_color("red") for i in plt.gca().get_yticklabels()]   
	
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.errorbar(L_speeds,L_res_by_speed,L_std_res_by_speed, linestyle='None', marker='o')
ax2.plot([0,6.5],[np.mean(L_res_by_speed),np.mean(L_res_by_speed)],'b--',label='Mean distance = '+str(round(np.mean(L_res_by_speed),2)) + " mm")
ax2.set_xlim(0,6.5)
ax2.set_ylabel("Mean distance (mm)",color="blue",fontsize=14)
ax2.set_ylim(0,max(L_res_by_speed)*1.2)
ax.grid()
ax2.legend(loc="lower right")
[i.set_color("blue") for i in plt.gca().get_yticklabels()]  
plt.title("Mean distance to baseline scan (speed factor = 0.5) and\nduration of the scan for different scanning speed (n=3)")
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')
# %%
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 100

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.errorbar(L_speeds,L_times_by_speed,L_std_times_by_speed, linestyle='None', marker='o',color = 'r')
# set x-axis label
ax.set_xlabel('Speed factor of the scan', fontsize = 14)
# set y-axis label
ax.set_ylabel("Duration (s)",
              color="red",
              fontsize=14)
ax.set_ylim(0,max(L_times_by_speed)*1.2)
ax.set_xticks(L_speeds)
ax.grid()
[i.set_color("red") for i in plt.gca().get_yticklabels()]   
plt.title("Mean distance to baseline scan (speed factor = 0.5) and\nduration of the scan for different scanning speed (n=3)")
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')
# %%
