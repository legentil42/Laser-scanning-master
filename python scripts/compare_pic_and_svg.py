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
from svg.path import parse_path
from xml.dom import minidom


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return nodes[np.argmin(dist_2)]


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

def get_point_at(path, distance, scale, offset):
    pos = path.point(distance)
    pos += offset
    pos *= scale
    return pos.real, pos.imag


def points_from_path(path, density, scale, offset):
    step = int(path.length() * density)
    last_step = step - 1

    if last_step == 0:
        yield get_point_at(path, 0, scale, offset)
        return

    for distance in range(step):
        yield get_point_at(
            path, distance / last_step, scale, offset)


def points_from_doc(doc, density=5, scale=1, offset=0):
    offset = offset[0] + offset[1] * 1j
    points = []
    for element in doc.getElementsByTagName("path"):
        for path in parse_path(element.getAttribute("d")):
            points.extend(points_from_path(
                path, density, scale, offset))
    
    for i in range(len(points)):
        points[i] = list(points[i]) + [8.5]

    return points

def show_img(path_or_img,title="Image"):
    
    if type(path_or_img) == str:
        image = cv.imread(path_or_img)
        image_conv = image[:,:,::-1]
        plt.figure(figsize = (15,15))
        plt.axis('off')
        plt.title(title)
        plt.imshow(image_conv)
        plt.show()
        return image

    elif type(path_or_img[0][0]) != list:
        plt.figure(figsize = (15,15))
        plt.axis('off')
        plt.title(title)
        plt.imshow(path_or_img,cmap="gray")
        plt.imsave("images/"+title+'.png',path_or_img,cmap="gray")
        
    else:
        plt.figure(figsize = (15,15))
        plt.axis('off')
        plt.title(title)
        plt.imshow(path_or_img[:,:,::-1])
        plt.imsave("images/"+title+'.png',path_or_img[:,:,::-1])



def PC_view_of_these_lists(L_Lists,point_size=0.1):
    to_view = []
    colors = []
    for i in range(len(L_Lists)):
        to_view += L_Lists[i]
        colors += [i] * len(L_Lists[i])
    v = pptk.viewer(to_view,colors)
    v.set(point_size=point_size)


def selection(name = "crack_palette_svg.svg"):

    with open(name, 'r') as file:
        string = file.read().replace('\n', '')


    doc = minidom.parseString(string)
    points = points_from_doc(doc, density=100, scale=1, offset=(0, 0))
    return points

     
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

def generate_PC_of_crack_only_no_L(mask_of_crack,res_factor,new_z):
    L_crack_only =[]


    L_i,L_j = [],[]

    for i in range(len(mask_of_crack)):
        for j in range(len(mask_of_crack[0])):
            if mask_of_crack[i][j] == 255:
                
                L_i.append(i)
                L_j.append(j)


    for i in range(len(L_i)):
        L_crack_only.append([L_i[i],L_j[i],new_z])

    return L_crack_only


def mean_of_zs(L_corner_SCAN):
    L_zs = [L_corner_SCAN[i][2] for i in range(len(L_corner_SCAN))]
    return np.mean(L_zs)



def save_selected_points(L,L_SCAN,name = "Selected_points_svg.txt"):
    
    if os.path.exists(name):
        os.remove(name)

    with open(name, 'w') as f:
        for i in range(len(L)):
            f.write("%s\n" % L_SCAN[L[i]])

def selection_crack(point_cloud_to_select_from,show = True):


    L_SCAN = point_cloud_to_select_from


    def save_points():
        Selected = v1.get('selected')
        #print(Selected)
        v1.close()
        save_selected_points(Selected,L_SCAN,"mycrack.txt")
        
        

    v1 = pptk.viewer(L_SCAN)
    v1.set(point_size=0.05)
    v1.set(phi = np.pi/2,theta = np.pi/2)
    # root window


    # Creating the tkinter window
    root = Tk()
    root.geometry("200x100")

    # Button for closing
    exit_button = Button(root, text="Select", command=root.destroy)
    exit_button.pack(pady=20)

    root.mainloop()
    save_points()
    L_crack = import_my_3DPC_simple("mycrack.txt",show = show)
    return L_crack


def selection_4_corners(L_kept):

    def save_points(full_cloud,name_txt):
        Selected = v1.get('selected')
        #print(Selected)
        v1.close()
        save_selected_points(Selected,full_cloud,name = name_txt)
        
    v1 = pptk.viewer(L_kept)
    v1.set(point_size=0.003)
    # root window

    # Creating the tkinter window
    root = Tk()
    root.geometry("200x100")
    # Button for closing
    exit_button = Button(root, text="Select corners", command=root.destroy)
    exit_button.pack(pady=20)

    root.mainloop()

    save_points(L_kept,"SCAN_corners.txt")

    L_corners_SCAN = import_my_3DPC_simple("SCAN_corners.txt",show=False)


    print(L_corners_SCAN)
    return L_corners_SCAN

def apply_homography_to_svg(L_SVG,H,new_z):
    L_2D = []
    a11 = H[0][0]
    a12 = H[0][1]
    a21 = H[1][0]
    a22 = H[1][1]
    b1=H[0][2]
    b2=H[1][2]
    for i in range(len(L_SVG)): 
          L_2D.append(list(np.dot(np.array([[a11, a12],[a21, a22]]),
                np.array(L_SVG[i][0:2]))+ np.array([b1, b2]))+ [new_z])
    

    return L_2D

def Homography_from_corners_two_lists(Scan_pts,corners_ori, show = True):
    
    Scan_pts_2D=Scan_pts.copy()
    corners_ori_2D = corners_ori.copy()
    for i in range(len(Scan_pts)):
        Scan_pts_2D[i] = Scan_pts[i][0:2]
        corners_ori_2D[i] = corners_ori[i][0:2]



    Homography, _ = cv.estimateAffine2D(np.float32(corners_ori_2D),
                                    np.float32(Scan_pts_2D))

    
    return Homography



def compare_photo_to_svg(i_photo,show=True):
    L_svg = selection("images\\palette_full.svg")

    short_path = "images\\"+str(i_photo)+"_test.png"

    crack_0_mask = show_img(short_path)
        
    crack_0_mask = cv.cvtColor(crack_0_mask, cv.COLOR_BGR2GRAY)
    crack_0_mask = np.uint8(crack_0_mask)

    im2, contours = cv.findContours(255-crack_0_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    L_coords_contours = []
    for i in range(len(im2)):
        for j in range(len(im2[i])):
            L_coords_contours.append(list(im2[i][j][0])+[8.5])
            L_coords_contours[-1][0],L_coords_contours[-1][1]=\
            L_coords_contours[-1][1],-L_coords_contours[-1][0]   

    return L_coords_contours

    PC_view_of_these_lists([L_coords_contours])

    L_svg = selection("images\\palette_full.svg")

    print('done')

    PC_view_of_these_lists([L_svg,L_coords_contours])

    L_corners_1 = selection_4_corners(L_svg)

    L_corners_2 = selection_4_corners(L_coords_contours)

    L_corners_1 = L_corners_1[::2]
    L_corners_1[1],L_corners_1[2] = L_corners_1[2],L_corners_1[1]


    H = Homography_from_corners_two_lists(L_corners_1,L_corners_2)


    new_z = mean_of_zs(L_svg) 

    L_img_procs_wraped = apply_homography_to_svg(L_coords_contours,H,new_z)
    L_only_my_crack = selection_crack(L_svg,show = True)
    res = sum_dist_each_point(L_img_procs_wraped,L_only_my_crack)
    print(res)
    PC_view_of_these_lists([L_svg,L_coords_contours,L_img_procs_wraped])

    return res,L_img_procs_wraped




# %%


def import_original_cracks_and_trajs(show = True):
    
    warped_img_path = r"C:\Users\legen\Desktop\dissertation\python scripts\warped_img_crack_processed"

    warped_img_individual_paths = [f for f in listdir(warped_img_path) if isfile(join(warped_img_path, f))]
    warped_img_individual_paths = warped_img_individual_paths[:-1]
    L_PCs= []
    if show:
        print(warped_img_individual_paths )

    for i_files in range(len(warped_img_individual_paths)):
        L_thing =  import_my_3DPC_simple(warped_img_path+"\\"+warped_img_individual_paths[i_files], show = False)
        L_PCs.append(L_thing)
        if show:
            print(i_files)

    return L_PCs

L_PCs = import_original_cracks_and_trajs(show = True)


#%%

#%%

L_svg = selection("images\\palette_full.svg")
#%%
PC_view_of_these_lists(L_PCs+[L_svg])
#%%
L_res = [0.8038580275678444,
 0.6907749222270351,
 0.6345024930724225,
 0.7121751580955458,
 0.6022989038595221,
 0.39391734000649714,
 1.6529644771857674]

#%%

def sum_dist_each_point_and_std(L1,L2):
    import sys
    L_errors = []

    for i in range(len(L2)):
        closest_to_i = closest_node(L2[i],L1)
        L_errors.append(np.linalg.norm(np.array(closest_to_i)-np.array(L2[i])))
        
        sys.stdout.write('\r'
        + str(round(100*i/len(L2),2))+'%')
    return np.mean(L_errors),np.std(L_errors)


def sum_dist_each_point_and_std_only_crack(L1,L2):
    import sys
    L_errors = []

    for i in range(len(L2)):
        if L2[i][1]<242.7 and L2[i][1]>72.1:
            closest_to_i = closest_node(L2[i],L1)
            L_errors.append(np.linalg.norm(np.array(closest_to_i)-np.array(L2[i])))
        
        sys.stdout.write('\r'
        + str(round(100*i/len(L2),2))+'%')
    return np.mean(L_errors),np.std(L_errors)

def sum_x_disp_each_point_and_std_with_direction(L1,L2):
    import sys
    L_errors = []

    for i in range(len(L2)):
        closest_to_i = closest_node(L2[i],L1)
        L_errors.append(np.linalg.norm(closest_to_i[0]-L2[i][0]))
        
        sys.stdout.write('\r'
        + str(round(100*i/len(L2),2))+'%')
    return np.mean(L_errors),np.std(L_errors)

def get_centroid_dist(PC1,PC2):
    x = [p[0] for p in PC1]
    y = [p[1] for p in PC1]
    z = [p[1] for p in PC1]
    centroid1 = np.array([sum(x) / len(PC1), sum(y) / len(PC1),sum(z) / len(PC1)])
    x = [p[0] for p in PC2]
    y = [p[1] for p in PC2]
    z = [p[1] for p in PC2]
    centroid2 = np.array([sum(x) / len(PC2), sum(y) / len(PC2),sum(z) / len(PC2)])
    return np.linalg.norm(centroid1-centroid2)


# %%
L_all_warped_img = []
L_res = []
L_res_centro = []
L_res_only_crack = []
L_only_craks_i = []
for i in range(len(L_PCs)):
    L_only_craks_i.append(selection_crack(L_svg,show = False))


L_svg = selection("images\\palette_full.svg")
# %%
for i in range(len(L_PCs)):

    mean,std = sum_dist_each_point_and_std_only_crack(L_PCs[i],L_only_craks_i[i])
    print(mean,std )
    print(i)
    L_res_only_crack.append([mean,std])

# %%
L_res_only_crack = [[0.7809745614863284, 0.5512397845034112],
 [0.741584007063508, 0.5608814706716333],
 [0.6976382516180003, 0.5348028742880201],
 [0.7565349859935966, 0.5631904301720555],
 [0.6208861010895104, 0.4467044582942057],
 [0.41826916888845805, 0.27917015026132896],
 [0.4068066436884569, 0.1741394781139351]]
# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 100
import numpy as np 

L_res = [
 [0.6537128620903777, 0.538741761466744],
 [0.6315748243133962, 0.5458892560776845],
 [0.6052850557899743, 0.519924215569525],
 [0.6460649802683769, 0.5456738939681726],
 [0.5583317348510389, 0.47703723786926894],
 [0.42812090589433344, 0.36932246506007155],
 [0.4788220775107537, 0.4129857105843213]]


x_svg = list(range(7))
y_svg,e_svg = zip(*L_res)
mean_error = np.mean(np.array(y_svg))
plt.errorbar(x_svg,y_svg,e_svg, linestyle='None', marker='o')
plt.xlabel('Number of the crack')
plt.ylabel('Mean distance between points (mm)')
plt.title("Mean distance (mm) and standard deviation between\nthe svg and photo point clouds")
plt.ylim(0,1.4)
plt.xlim(-0.5,6.5)
plt.plot([-0.5,6.5],[mean_error,mean_error],'r--',label='Mean distance = '+str(round(mean_error,2)) + " mm")

plt.legend()
plt.grid()
plt.show()

#%%
L_res_disp = [[0.4808937511871854, 0.43878866288812857],
 [0.44946212021917364, 0.4225899074190851],
 [0.4318993262134641, 0.4026533871186507],
 [0.45005798167925887, 0.4369077184015877],
 [0.40789227232189323, 0.3811533652044575],
 [0.26705215712133257, 0.26632105499070474],
 [0.24686155266528859, 0.2685308486539567]]
#%%

def import_original_cracks_and_trajs_ori_ori(show = True):
    
    points_path = r"C:\Users\legen\Desktop\dissertation\python scripts\SVG_cracks_3DPC"

    trajs_and_cracks_paths = [f for f in listdir(points_path) if isfile(join(points_path, f))]

    L_cracks_and_trajs = []
    if show:
        print(trajs_and_cracks_paths )

    for i_files in range(len(trajs_and_cracks_paths)):
        L_thing =  import_my_3DPC_simple(points_path+"\\"+trajs_and_cracks_paths[i_files], show = False)
        L_cracks_and_trajs.append(L_thing)
        if show:
            print(i_files)

    crack_paths = trajs_and_cracks_paths[:len(trajs_and_cracks_paths)//2]
    traj_paths = trajs_and_cracks_paths[len(trajs_and_cracks_paths)//2:]
    

    L_cracks_ori = L_cracks_and_trajs[:len(L_cracks_and_trajs)//2]
    L_trajs_ori = L_cracks_and_trajs[len(L_cracks_and_trajs)//2:]
    
    
    return crack_paths,traj_paths,L_cracks_ori,L_trajs_ori

crack_paths,traj_paths,L_cracks_ori,L_trajs_ori=import_original_cracks_and_trajs_ori_ori(show = True)
# %%
PC_view_of_these_lists(L_PCs+L_cracks_ori+[L_svg])

# %%
LALA = compare_photo_to_svg(1,True)
# %%
PC_view_of_these_lists(L_PCs+[L_svg])

# %%
