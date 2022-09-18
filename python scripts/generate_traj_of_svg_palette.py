#%%
from base64 import b16decode
from cmath import exp
from ensurepip import version
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



from svg.path import parse_path
from xml.dom import minidom


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


def save_selected_points(L,L_SCAN,name = "Selected_points_svg.txt"):
    
    if os.path.exists(name):
        os.remove(name)

    with open(name, 'w') as f:
        for i in range(len(L)):
            f.write("%s\n" % L_SCAN[L[i]])


def import_my_3DPC(i,L_names,show = True):
    
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

def selection():

    with open("crack_palette_svg.svg", 'r') as file:
        string = file.read().replace('\n', '')


    doc = minidom.parseString(string)
    points = points_from_doc(doc, density=100, scale=1, offset=(0, 0))
    doc.unlink()

    def save_points():
        Selected = v1.get('selected')
        print(Selected)
        v1.close()
        save_selected_points(Selected,points)
        
        

    v1 = pptk.viewer(points)
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

    points_output = points.copy()

    def remove_wrong_points(points):

        L_wrong = import_my_3DPC("Selected_points_svg.txt")

        for selected_point in L_wrong:
            points_output.remove(selected_point)

        v = pptk.viewer(points_output)
        v.set(point_size=0.1)

    remove_wrong_points(points_output)

    return points_output

def selection_to_remove(L_3DPC):
    

    def save_points(points):
        Selected = v1.get('selected')
        print(Selected)
        v1.close()
        save_selected_points(Selected,points,name = "temp.txt")
        
        

    v1 = pptk.viewer(L_3DPC)
    v1.set(point_size=0.1)
    # root window


    # Creating the tkinter window
    root = Tk()
    root.geometry("200x100")

    # Button for closing
    exit_button = Button(root, text="Select", command=root.destroy)
    exit_button.pack(pady=20)

    root.mainloop()
    save_points(L_3DPC)

    points_output = L_3DPC.copy()

    def remove_wrong_points(points_output):

        L_wrong = import_my_3DPC_simple("temp.txt")

        for selected_point in L_wrong:
            points_output.remove(selected_point)

        v = pptk.viewer(points_output)
        v.set(point_size=0.1)
        return points_output

    points_output = remove_wrong_points(points_output)

    return points_output


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




def selection_4_corners(L_kept):

    def save_points(full_cloud,name_txt):
        Selected = v1.get('selected')
        print(Selected)
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

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return nodes[np.argmin(dist_2)]


def farthest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return nodes[np.argmax(dist_2)]


def Homography_from_corners(Scan_pts, show = True):
    
    corners_ori = [[-318.510232, -770.433475, 8.5],
  [-318.510232, -20.559587, 8.5],
  [-8.003429, -770.433475, 8.5],
  [-8.003429, -20.559587, 8.5] ]
    Scan_pts_2D=Scan_pts.copy()
    corners_ori_2D = corners_ori.copy()
    for i in range(len(Scan_pts)):
        Scan_pts_2D[i] = Scan_pts[i][0:2]
        corners_ori_2D[i] = corners_ori[i][0:2]

    if False:
        SCAN_pts_ordered = []
        for i in range(len(corners_ori)):
            SCAN_pts_ordered.append(list(closest_node(corners_ori[i],Scan_pts)))
            Scan_pts.remove(list(SCAN_pts_ordered[-1]))

        print(SCAN_pts_ordered)
        Scan_pts=SCAN_pts_ordered
        if show:
            print(Scan_pts)
            print(corners_ori)



    Homography, _ = cv.estimateAffine2D(np.float32(corners_ori_2D), np.float32(Scan_pts_2D))

    print()
    return Homography


def apply_homography_to_svg(L_SVG,H,new_z):
    L_2D = []
    a11 = H[0][0]
    a12 = H[0][1]
    a21 = H[1][0]
    a22 = H[1][1]
    b1=H[0][2]
    b2=H[1][2]
    for i in range(len(L_SVG)): 
          L_2D.append(list(np.dot(np.array([[a11, a12],[a21, a22]]),np.array(L_SVG[i][0:2]))+ np.array([b1, b2]))+ [new_z])
    

    return L_2D

def mean_of_zs(L_corner_SCAN):
    L_zs = [L_corner_SCAN[i][2] for i in range(len(L_corner_SCAN))]
    return np.mean(L_zs)

L_corners_SCAN = [[-452.28494053200046, 31.587132055760957, 8.375963914085247],
 [-455.3167732662149, 228.96464934565765, 8.514024778846704],
 [-370.64812776079515, 31.695657136222696, 7.920277322215725],
 [-373.90917288397424, 230.16036197930592, 7.599699323425739]]

corners_ori = [[-318.510232, -770.433475, 8.5],
  [-318.510232, -20.559587, 8.5],
  [-8.003429, -770.433475, 8.5],
  [-8.003429, -20.559587, 8.5],
]



def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)



def K_farthest(pts):
    pt_start = pts[0]
    pt_1 = farthest_node(pt_start,pts)
    pt_2 = farthest_node(pt_1,pts)
    pt_3 = farthest_node(pt_2,pts)
    return [pt_2,pt_3]


def getEquidistantPoints(p1, p2, parts):
    return list(zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1),
               np.linspace(p1[2], p2[2], parts+1),))


def selection_crack(point_cloud_to_select_from):


    L_SCAN = point_cloud_to_select_from


    def save_points():
        Selected = v1.get('selected')
        print(Selected)
        v1.close()
        save_selected_points(Selected,L_SCAN,"mycrack.txt")
        
        

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
    L_crack = import_my_3DPC_simple("mycrack.txt")
    return L_crack



def calc_L_plane_for_ith_point(My_Line_no_weight,i,show=True):
    # vector normal to plane
    n = np.array(My_Line_no_weight)[i+1]-np.array(My_Line_no_weight)[i]

    n =n/abs(np.linalg.norm(n))

    point = np.array(My_Line_no_weight[i])

    

    Lx = []
    Lz = []

    mini = 1
    maxi = 1
    n_points = 100

    xx = np.linspace(My_Line_no_weight[i][0]-mini,My_Line_no_weight[i][0]+maxi,n_points)
    zz = np.linspace(My_Line_no_weight[i][2]-mini,My_Line_no_weight[i][2]+maxi,n_points)

    Lx, Lz = np.meshgrid(xx, zz)
    XZpairs = np.vstack([ Lx.reshape(-1), Lz.reshape(-1) ])
    Lx,Lz = XZpairs[0],XZpairs[1]

    d = -point.dot(n)


    Ly = (-n[0] * Lx - n[2] * Lz - d) / n[1]

    L_plane = list(zip(Lx,Ly,Lz))

    return point,n,L_plane



def calculate_traj_with_planes(L_crack,n_my_line,nb_best = 30,proj_on_plane = True,show = True):


    def dist_from_plane(p_test,p0,n):
        v = np.array(p_test)-p0
        dist = abs(np.dot(v,n/np.linalg.norm(n)))
        return dist

    Farthest = K_farthest(L_crack)

    My_Line = getEquidistantPoints(Farthest[0],Farthest[1],n_my_line)
    My_Line = [list(My_Line[i]) for i in range(len(My_Line))]

    My_Line_no_weight = My_Line.copy()

    L_all_planes = []

    for i in range(len(My_Line_no_weight)-1):

        point,n,L_plane = calc_L_plane_for_ith_point(My_Line_no_weight,i,show=False)
        L_all_planes+= L_plane
        L_dist_point_of_crack_to_ith_plane = []

        for i_point_crack in range(len(L_crack)):
            L_dist_point_of_crack_to_ith_plane.append([i_point_crack,dist_from_plane(L_crack[i_point_crack],point,n)])
        
        
        
        L_dist_point_of_crack_to_ith_plane = sorted(L_dist_point_of_crack_to_ith_plane, 
        key=lambda x: x[1])


        L_closest_points = []
        

        avgx,avgy,avgz = 0,0,0
        for _ in range(nb_best):
            try:
                L_closest_points.append(L_dist_point_of_crack_to_ith_plane.pop(0)[0])
                avgx+=L_crack[L_closest_points[-1]][0]
                avgy+=L_crack[L_closest_points[-1]][1]
                avgz+=L_crack[L_closest_points[-1]][2]
            except:
                pass

        avgx,avgy,avgz= avgx/len(L_closest_points),avgy/len(L_closest_points),avgz/len(L_closest_points)
        
        
        if proj_on_plane:
            #i have the closest point avg of crack points
            #now i have to firugre out the closet point to this one that is on the plane
                
            #Make a vector from your orig point to the point of interest:

            v = np.array([avgx,avgy,avgz])-np.array(My_Line_no_weight[i]);

            #Take the dot product of that vector with the unit normal vector n:

            dist = abs(np.dot(v,n))
            #; dist = scalar distance from point to plane along the normal

            #Multiply the unit normal vector by the distance, and subtract that vector from your point.

            projected_point =  np.array([avgx,avgy,avgz]) - dist*n;

            My_Line[i] = list(projected_point)

        else:
            My_Line[i] = np.array([avgx,avgy,avgz])

    if show:
        v = pptk.viewer(L_crack+My_Line+My_Line_no_weight,[0]*len(L_crack)+[1]*len(My_Line)+[2]*len(My_Line_no_weight))
        v.set(point_size=0.1)

    return My_Line


def avg_My_Line(My_Line,n_moving=3,show = True):
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


    My_Line_x = [pt[0] for pt in My_Line]
    My_Line_y = [pt[1] for pt in My_Line]
    My_Line_z = [pt[2] for pt in My_Line]

    My_Line_x_avg = moving_average(My_Line_x,n_moving)
    My_Line_y_avg = moving_average(My_Line_y,n_moving)
    My_Line_z_avg = moving_average(My_Line_z,n_moving)

    My_Line_avg = [[x,y,z] for x,y,z in zip(My_Line_x_avg,My_Line_y_avg,My_Line_z_avg)]

    if show:
        v = pptk.viewer(My_Line+My_Line_avg,[0]*len(My_Line)+[1]*len(My_Line_avg))
        v.set(point_size=0.1)
    return My_Line_avg

def sum_dist_each_point(L1,L2):
    sum = 0
    if len(L1) == len(L2):
        for i in range(len(L1)):
            sum+= np.linalg.norm(np.array(L1[i])-np.array(L2[i]))**2
        return sum/len(L1)
    else:
        if len(L1)>len(L2):
    
            for i in range(len(L2)):
                closest_to_i = closest_node(L2[i],L1)
                sum+= np.linalg.norm(np.array(closest_to_i)-np.array(L2[i]))**2
            return sum/len(L2)

        elif len(L2)>len(L1):
    
            for i in range(len(L1)):
                closest_to_i = closest_node(L1[i],L2)
                sum+= np.linalg.norm(np.array(closest_to_i)-np.array(L1[i]))**2
            return sum/len(L1)      

def avg_until_convergence(My_Line,thres = 0.9,iter_lim = 5):
    My_Line_start = My_Line.copy()
    My_Line_iter_1 = avg_My_Line(My_Line_start,n_moving=3,show = True)
    dist_one =  sum_dist_each_point(My_Line_iter_1,My_Line_start)

    My_Line_iter_2 = avg_My_Line(My_Line_iter_1,n_moving=3,show = True)
    dist_two=  sum_dist_each_point(My_Line_iter_2,My_Line_iter_1)
    print(dist_two/dist_one)
    i = 2

    while dist_two/dist_one<thres and i < iter_lim:
        dist_one=dist_two
        My_Line_iter_1 = My_Line_iter_2
        My_Line_iter_2 = avg_My_Line(My_Line_iter_1,n_moving=3,show = True)
        dist_two=  sum_dist_each_point(My_Line_iter_2,My_Line_iter_1)

        if dist_one == 0:
            break
        print(dist_two/dist_one)
        i+=1

    return My_Line_iter_2


def PC_view_of_these_lists(L_Lists,point_size=0.1):
    to_view = []
    colors = []
    for i in range(len(L_Lists)):
        to_view += L_Lists[i]
        colors += [i] * len(L_Lists[i])
    v = pptk.viewer(to_view,colors)
    v.set(point_size=point_size)


def compress_list(values):
    result = []

    for key, group in itertools.groupby(range(len(values)), values.__getitem__):
        indices = list(group)

        if len(indices) > 1:
            result.append([indices[-1]-indices[0]+1, key])
        else:
            result.append([1, key])
    return result

def PC_to_image(L_kept,i_PC_to_do,res_factor = 10,show = True):
    
    L_x = np.array([L_kept[i][0] for i in range(len(L_kept))])*res_factor
    L_y = np.array([L_kept[i][1] for i in range(len(L_kept))])*res_factor


    height = max(L_x)-min(L_x)
    width = max(L_y)-min(L_y)

    L_x = L_x - min(L_x)
    L_y = L_y - min(L_y)

    

    if show:
        print(max(L_x))
        print(max(L_y))

    lim_1 =  round(max(L_x))
    lim_2 = round(max(L_y))

    output_image = np.ones((lim_1+1,lim_2+1))*255

    for i in range(len(L_kept)):
        output_image[round(L_x[i]),round(L_y[i])] = 0

    if show:
        show_img(output_image,"3DPC to img and no processing")
        
    mask = output_image.copy()



    if results[i_PC_to_do][2] == 1:
        compressed_slice = compress_list(output_image[:, 1])[1:-2]
        if show:
            print(len(output_image[:, 1]))
            print(output_image[:, 1])
    if results[i_PC_to_do][2] == 0:
        compressed_slice = compress_list(output_image[1])[1:-2]
        if show:
            print(len(output_image[1]))
            print(output_image[1])

    if show:
        print(compressed_slice)

    compressed_slice = [compressed_slice[i][0] for i in range(len(compressed_slice)) if compressed_slice[i][1] == 255]

    size_kernel = [max(compressed_slice)+1]

    if show:
        print(size_kernel)

    for i in range(len(size_kernel)):
            
        # Creating kernel
        if results[i_PC_to_do][2] == 1:
            kernel = np.ones((size_kernel[i], 1), np.uint8)
        if results[i_PC_to_do][2] == 0:
            kernel = np.ones((1,size_kernel[i]), np.uint8)
        # Using cv.erode() method 
        mask = cv.erode(mask, kernel) 
        if show:
            show_img(mask,"eroded_3DPC")

    return mask

def complete_lacking_traj(lacking_traj):
    missing = 420-len(lacking_traj)
    per_line = round(missing/len(lacking_traj)-1)
    extra_points = []
    for i in range(len(lacking_traj)-1):
        extra_points += list(getEquidistantPoints(lacking_traj[i],lacking_traj[i+1],per_line))
    
    return extra_points

def generate_PC_of_crack_only(mask_of_crack,L_crack,res_factor):
    L_crack_only =[]

    L_x = np.array([L_crack[i][0] for i in range(len(L_crack))])
    L_y = np.array([L_crack[i][1] for i in range(len(L_crack))])

    height = max(L_y)-min(L_y)
    width = max(L_x)-min(L_x)

    L_i,L_j = [],[]

    for i in range(len(mask_of_crack)):
        for j in range(len(mask_of_crack[0])):
            if mask_of_crack[i][j] == 255:
                
                L_i.append(i)
                L_j.append(j)

    L_i = np.array(L_i) 
    L_j = np.array(L_j) 

    L_i =( width*L_i /len(mask_of_crack) )   + min(L_x)
    L_j =( height*L_j /len(mask_of_crack[0])) + min(L_y)

    for i in range(len(L_i)):
        L_crack_only.append([L_i[i],L_j[i],8.5])

    return L_crack_only
#%% DB GENERATION
results_path = r"C:\Users\legen\Desktop\dissertation\Results"

results = [f for f in listdir(results_path) if isfile(join(results_path, f))]
file_paths = results.copy()
file_paths = [results_path+"\\" +results[i_file_name] for i_file_name in range(len(file_paths))]

for i_res in range(len(results)):
    num_lines = sum(1 for _ in open(results_path+"\\" +results[i_res]))
    results[i_res] = list(map(float,(results[i_res][:-4].split(" ")))) + [bufcount(results_path+"\\" +results[i_res])]

print(results)


# Create the pandas DataFrame
df = pd.DataFrame(results, columns=['Time (s)', 'Nb_steps', 'direction','speed_factor','Nb_points'])
  
# print dataframe.
print(df)

original_palette_3DPC = import_my_3DPC_simple("L_palette_3DPC_original.txt",show = False)
#%% IMPORT THE 3DPC

i_PC_to_do = 2
L_SCAN = import_my_3DPC(i_PC_to_do,file_paths,show = False)
L_kept = keep_top_z(L_SCAN,8.5,1,show = True)

#%% WARP THE CRACK PALETTE TO THE SCAN

L_corners_ori = import_my_3DPC_simple("ori_corners.txt",show=False)
L_corners_SCAN = selection_4_corners(L_kept)

H = Homography_from_corners(L_corners_SCAN, show = False)
new_z = mean_of_zs(L_corners_SCAN)
L_warped = apply_homography_to_svg(original_palette_3DPC,H,new_z)

v = pptk.viewer(L_warped+L_kept,[0]*len(L_warped)+[1]*len(L_kept))
v.set(point_size=0.01)



#%% get crack from svg
L_crack_ori = selection_crack(L_warped)
PC_view_of_these_lists([L_warped])




# %% get crack from scan
res_factor = 10

L_crack_SCAN = selection_crack(L_kept)
mask_of_crack = PC_to_image(L_crack_SCAN,i_PC_to_do,res_factor = 5,show = True)

L_crack_SCAN_only = generate_PC_of_crack_only(mask_of_crack,L_crack_SCAN,res_factor)

L_crack_SCAN_only_removed_wrong = selection_to_remove(L_crack_SCAN_only)

PC_view_of_these_lists([L_crack_SCAN_only_removed_wrong,L_kept])

#%%
PC_view_of_these_lists([L_crack_SCAN_only_removed_wrong,L_crack_ori,L_warped])


#%%
My_Line_SCAN = calculate_traj_with_planes(L_crack_SCAN_only_removed_wrong,30,proj_on_plane=False,show = False)
#%%
My_Line_ori = calculate_traj_with_planes(L_crack_ori,100,nb_best = 30,proj_on_plane=False,show = False)

#%%
extra_for_ori = complete_lacking_traj(My_Line_ori)
#%%
sum_dist_each_point(My_Line_ori,My_Line_SCAN)

#%%
My_Line_ori_temp = calculate_traj_with_planes(L_crack_ori,50,nb_best = 2,proj_on_plane=False,show = False)
PC_view_of_these_lists([L_warped,My_Line_ori_temp,L_kept])
#%%
PC_view_of_these_lists([L_warped,My_Line_ori,L_kept])
#%%
PC_view_of_these_lists([L_warped,My_Line_ori],  point_size=0.01)
