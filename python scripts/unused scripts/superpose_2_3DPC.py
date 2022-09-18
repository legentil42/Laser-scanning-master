#%%
import numpy as np
import numpy as np
from scipy.optimize import fmin
from scipy import optimize
import math
import numdifftools as nd
import random
import ast
import pptk


def import_my_3DPC(name,factor = 100,show = True):
    
    L_SCAN = [];
    with open(name, 'r') as f:
        for line  in f:
            L_SCAN.append(line.strip().replace(" ","").replace("[","").replace("]","").split(","))
            L_SCAN[-1] = [float(i) for i in L_SCAN[-1]]
            if L_SCAN[-1] == [0,0,0]:
                print("hi")
                L_SCAN.pop(-1)
    return L_SCAN

L_SCAN = import_my_3DPC("3DPC_koi.txt")



v = pptk.viewer(L_SCAN)
v.set(point_size=0.1)

# %%


# with open("scy_CAD.txt", "r") as f:
#     L_CAD = []
#     for item in f:
#         point = item.split(" ")
#         point = [float(point[0]),float(point[1]),float(point[2][:-1])]
#         if point != [0,0,0]:
#             L_CAD.append(point)
#     #L_CAD = np.array(L_CAD)

#%%

# random.shuffle(L_CAD)

# L_CAD_full = L_CAD.copy()
# factor=25
# L_CAD = L_CAD[::factor]


def quickconv(L):
    x,y,z,ones = [],[],[],[]
    for i in range(len(L)):
        x.append(L[i][0])
        y.append(L[i][1])
        z.append(L[i][2])
        ones.append(1)
    return np.array([x,y,z,ones])

def revertconv(xyzones_array):
    return np.asarray(list(zip(xyzones_array[0],xyzones_array[1],xyzones_array[2])))

L_SCAN_work = quickconv(L_SCAN)


v = pptk.viewer(L_SCAN+L_CAD_full)
v.set(point_size=0.1)
# %%
v = pptk.viewer(L_SCAN)
v.set(point_size=0.1)

T = [[1,0,0,0],
        [0,1,0,0],
        [0,0,1,100],
        [0,0,0,1]]



v = pptk.viewer(revertconv(np.dot(T,quickconv(L_SCAN))))
v.set(point_size=0.1)


#%%

  

from math import cos, sin, radians

def trig(angle):
    r = radians(angle)
    return cos(r), sin(r)

def matrix(rotation, translation):

  xC, xS = trig(rotation[0])
  yC, yS = trig(rotation[1])
  zC, zS = trig(rotation[2])
  dX = translation[0]
  dY = translation[1]
  dZ = translation[2]

  Translate_matrix = np.array([[1, 0, 0, dX],
                               [0, 1, 0, dY],
                               [0, 0, 1, dZ],
                               [0, 0, 0, 1]])
  Rotate_X_matrix = np.array([[1, 0, 0, 0],
                              [0, xC, -xS, 0],
                              [0, xS, xC, 0],
                              [0, 0, 0, 1]])
  Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                              [0, 1, 0, 0],
                              [-yS, 0, yC, 0],
                              [0, 0, 0, 1]])
  Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                              [zS, zC, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
  return np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))


def distance_old(point):
    nodes = np.asarray(L_CAD)
    dist_2 = np.sum((nodes - point)**2, axis=1)
    return dist_2[np.argmin(dist_2)]

def distance(node):
    deltas = L_CAD - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return dist_2[np.argmin(dist_2)]


factor =  2

def Fitness_Func(Rota_Trans):
    Cost = 0
    T_mat  = matrix(Rota_Trans[0:3],Rota_Trans[3:])
    L_SCAN_MODIFIED = revertconv(np.dot(T_mat,quickconv(L_SCAN[::factor])))
    for i_point in range(len(L_SCAN[::factor])):
        i_dist = distance(np.array(L_SCAN_MODIFIED[i_point]))
        Cost += i_dist**2
    print(Cost,T_mat)
    return Cost


Rota_Trans = [ 0,  90,   0,-405, -13, -155]


Fitness_Func(Rota_Trans)


#%%


Res =optimize.minimize(Fitness_Func,Rota_Trans,method =  "trust-constr",options={'maxiter': 500})
#%%
Res = fmin(Fitness_Func,Rota_Trans,maxfun=500)

# %%
best_Rota_Trans = Res
best_T_mat  = matrix(best_Rota_Trans[0:3],best_Rota_Trans[3:])

T_SCAN = list(revertconv(np.dot(best_T_mat,quickconv(L_SCAN))))
v = pptk.viewer(T_SCAN+L_CAD_full,[0]*len(T_SCAN)+[1]*len(L_CAD_full))
v.set(point_size=0.1)

# %%
v = pptk.viewer(revertconv(np.dot(best_T_mat,quickconv(L_SCAN))))
v.set(point_size=0.1)
# %%

# %%
v = pptk.viewer(L_SCAN[::factor])
v.set(point_size=0.1)

#%%


# %%


pptk.viewer(L_CAD)
v.set(point_size=1)
# %%

# %%