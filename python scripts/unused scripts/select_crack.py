#%%
import numpy as np
from tkinter import *
import ast
import pptk
import os


def save_selected_points(L,L_SCAN,name = "Selected_points.txt"):
    
    if os.path.exists(name):
        os.remove(name)

    with open(name, 'w') as f:
        for i in range(len(L)):
            f.write("%s\n" % L_SCAN[L[i]])


def quickconv(L):
    x,y,z = [],[],[]
    for i in range(len(L)):
        if L[i][2] != 0:
            x.append(L[i][0])
            y.append(L[i][1])
            z.append(L[i][2])
    return np.array([x,y,z])



def show_selected_and_traj(name = "Selected_points.txt"):
    #import .txt
    L_traj = []

    with open(name, "r") as f:
        good_points = []
        for item in f:
            if ast.literal_eval(item[:-1]) != [0,0,0]:
                good_points.append(ast.literal_eval(item[:-1]))
                traj_point = [good_points[-1][0],good_points[-1][1],good_points[-1][2]]
                L_traj.append(traj_point) #remove points in [0,0,0]

    #v = pptk.viewer(good_points)
    #v.set(point_size=0.1)
    return L_traj

def selection():

    with open("scy_scan.txt", "r") as f:
        L_SCAN = []
        for item in f:
            L_SCAN.append(ast.literal_eval(item[:-1]))


    def save_points():
        Selected = v1.get('selected')
        print(Selected)
        v1.close()
        save_selected_points(Selected,L_SCAN)
        
        

    v1 = pptk.viewer(L_SCAN)
    v1.set(point_size=0.1)
    # root window


    # Creating the tkinter window
    root = Tk()
    root.geometry("200x100")

    # Button for closing
    exit_button = Button(root, text="Select", command=root.destroy)
    exit_button.pack(pady=20)

    root.mainloop()
    save_points()
    L_traj = show_selected_and_traj()
    return L_traj,L_SCAN

L_selected_points,L_SCAN = selection()

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def K_farthest(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts

def K_closest(pts, K,my_point):
    import numpy as np
    from sklearn.neighbors import KDTree
    pts = np.array([list(my_point)]+pts)
    tree = KDTree(pts)
    nearest_dist, nearest_ind = tree.query(pts, k=K+1)  # k=2 nearest neighbors where k1 = identity
    # drop id 
    return nearest_ind[0]


Farthest = K_farthest(L_selected_points,2)

closest = K_closest(L_selected_points,4,Farthest[0])


def getEquidistantPoints(p1, p2, parts):
    return list(zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1),
               np.linspace(p1[2], p2[2], parts+1),))


My_Line = getEquidistantPoints(Farthest[0],Farthest[1],20)
My_Line = [list(My_Line[i]) for i in range(len(My_Line))]

for i in range(len(My_Line)):
    distances = calc_distances(np.array(My_Line[i]), np.array(L_SCAN))
    closest_point = L_SCAN[np.argmin(distances)]
    My_Line[i] = list(np.array(My_Line[i])*0.5+np.array(closest_point)*0.5)




v1 = pptk.viewer(list(L_SCAN)+list(My_Line),len(L_SCAN)*[0]+len(My_Line)*[1])
v1.set(point_size=0.2)



# %%
