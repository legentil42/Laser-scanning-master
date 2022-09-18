#%%
import numpy as np
import laspy as lp
import random as rd
input_path=""
dataname="NEO"
point_cloud=lp.read(input_path+dataname+".las")

points = []


#%%
good_points = []
for i in range(len(points)):
    if points[i][2] != 0:
        good_points.append(points[i])
#%%

#import .txt
import ast

with open("3DPC_lego_dir_0.txt", "r") as f:
    L_dir_0 = []
    for item in f:
        L_dir_0.append(ast.literal_eval(item[:-1]))

with open("3DPC_lego_dir_1.txt", "r") as f:
    L_dir_1 = []
    for item in f:
        L_dir_1.append(ast.literal_eval(item[:-1]))
#%%

import pptk
v = pptk.viewer(L_dir_0)
v.set(point_size=0.01)

v = pptk.viewer(L_dir_1)
v.set(point_size=0.01)
#%%
