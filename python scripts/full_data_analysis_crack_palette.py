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

def easy_traj_calc(L_crack,n_point):
            
    L_x = np.array([L_crack[i][0] for i in range(len(L_crack))])
    L_y = np.array([L_crack[i][1] for i in range(len(L_crack))])
    L_z = np.array([L_crack[i][1] for i in range(len(L_crack))])

    my_x_range = np.linspace(min(L_x),max(L_x),n_point)
    My_Line = []
    step = (max(L_x)-min(L_x))/n_point
    for cur_x in my_x_range:
        L_x_in_range = []
        for i_point in range(len(L_crack)):
            if L_x[i_point]<cur_x+step and  L_x[i_point]>=cur_x:
                L_x_in_range.append(L_crack[i_point])
        try:
            pt1,pt2 = K_farthest(L_x_in_range)
            My_Line.append([(pt1[0]+pt2[0])/2,(pt1[1]+pt2[1])/2,(pt1[2]+pt2[2])/2])
        except:
            pass
    
    return My_Line


def import_original_cracks_and_trajs(show = True):
    
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

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

def generate_true_traj_of_crack_palette_svg(show = True):
    original_palette_3DPC = import_my_3DPC_simple(
                    "L_palette_3DPC_original.txt",show = False)

    n_cracks = 7
    L_trajs_ori = []
    L_the_cracks = []
    for i_crack in range(n_cracks):

        L_crack_ori = selection_crack(original_palette_3DPC)

        L_crack_cropped = []
        for pt in L_crack_ori:
            if pt[0] < -77.7 and pt[0] > -248.5:
                L_crack_cropped.append(pt)


        My_Line_ori = easy_traj_calc(L_crack_cropped,100)

        full_traj = complete_lacking_traj(My_Line_ori)

        L_the_cracks.append(L_crack_cropped)
        L_trajs_ori.append(full_traj)


    return L_trajs_ori,L_the_cracks


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
            v.set(r=r)

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


def keep_top_z(L_SCAN,z,tolerance,show = True,r = 350):
    L_kept = []
    for i in range(len(L_SCAN)):
        if abs(L_SCAN[i][2]-z) < tolerance:
            L_kept.append(L_SCAN[i])
    if show:
        v = pptk.viewer(L_kept)
        v.set(point_size=0.1)
        v.set(r=r)

    return L_kept

def selection(name = "crack_palette_svg.svg"):

    with open(name, 'r') as file:
        string = file.read().replace('\n', '')


    doc = minidom.parseString(string)
    points = points_from_doc(doc, density=100, scale=1, offset=(0, 0))
    doc.unlink()
    return points
    doc.unlink()

    def save_points():
        Selected = v1.get('selected')
        print(Selected)
        v1.close()
        save_selected_points(Selected,points)
        
        

    v1 = pptk.viewer(points)
    v1.set(point_size=0.1)
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

    points_output = points.copy()

    def remove_wrong_points(points_output):

        L_wrong = import_my_3DPC_simple("Selected_points_svg.txt")

        for selected_point in L_wrong:
            points_output.remove(selected_point)

        v = pptk.viewer(points_output)
        v.set(point_size=0.1)

    points_output= remove_wrong_points(points_output)

    return points_output

def selection_to_remove(L_3DPC,show = True):
    

    def save_points(points):
        Selected = v1.get('selected')
        print(Selected)
        v1.close()
        save_selected_points(Selected,points,name = "temp.txt")
        
        

    v1 = pptk.viewer(L_3DPC)
    v1.set(point_size=0.1)
    v1.set(phi = np.pi/2,theta = np.pi/2)
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

    def remove_wrong_points(points_output,show = True):

        L_wrong = import_my_3DPC_simple("temp.txt",show=show)

        for selected_point in L_wrong:
            points_output.remove(selected_point)
        if show:
            v = pptk.viewer(points_output)
            v.set(point_size=0.1)
        return points_output

    points_output = remove_wrong_points(points_output,show = show)

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



    Homography, _ = cv.estimateAffine2D(np.float32(corners_ori_2D),
                                    np.float32(Scan_pts_2D))

    
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
          L_2D.append(list(np.dot(np.array([[a11, a12],[a21, a22]]),
                np.array(L_SVG[i][0:2]))+ np.array([b1, b2]))+ [new_z])
    

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

    xx = np.linspace(My_Line_no_weight[i][0]-mini,
                    My_Line_no_weight[i][0]+maxi,n_points)

    zz = np.linspace(My_Line_no_weight[i][2]-mini,
                    My_Line_no_weight[i][2]+maxi,n_points)

    Lx, Lz = np.meshgrid(xx, zz)
    XZpairs = np.vstack([ Lx.reshape(-1), Lz.reshape(-1) ])
    Lx,Lz = XZpairs[0],XZpairs[1]

    d = -point.dot(n)


    Ly = (-n[0] * Lx - n[2] * Lz - d) / n[1]

    L_plane = list(zip(Lx,Ly,Lz))

    return point,n,L_plane



def calculate_traj_with_planes(L_crack,n_my_line,nb_best = 30,
                            proj_on_plane = True,show = True):

    if len(L_crack) == 0:
        return [[0,0,0],[0,0,0]]

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

        PC_view_of_these_lists([L_crack[::30],getEquidistantPoints(Farthest[0],Farthest[1],n_my_line),My_Line])
    return My_Line


def avg_My_Line(My_Line,n_moving=3,show = True):
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret / n


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
    L_sum = []
    if len(L1) == len(L2):
        for i in range(len(L1)):
            L_sum.append(np.linalg.norm(np.array(L1[i])-np.array(L2[i]))**2)
        return L_sum
    else:
        if len(L1)>len(L2):
    
            for i in range(len(L2)):
                closest_to_i = closest_node(L2[i],L1)
                L_sum.append(np.linalg.norm(np.array(closest_to_i)-np.array(L2[i]))**2)
            return L_sum

        elif len(L2)>len(L1):
    
            for i in range(len(L1)):
                closest_to_i = closest_node(L1[i],L2)
                L_sum.append(np.linalg.norm(np.array(closest_to_i)-np.array(L1[i]))**2)
            return L_sum    

def avg_until_convergence(My_Line,thres = 0.9,iter_lim = 5):
    My_Line_start = My_Line.copy()
    My_Line_iter_1 = avg_My_Line(My_Line_start,n_moving=3,show = True)
    dist_one =  sum_dist_each_point(My_Line_iter_1,My_Line_start)
    dist_one = sum(dist_one)/len(dist_one)

    My_Line_iter_2 = avg_My_Line(My_Line_iter_1,n_moving=3,show = True)
    dist_two=  sum_dist_each_point(My_Line_iter_2,My_Line_iter_1)
    dist_two = sum(dist_two)/len(dist_two)
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
    return v

def compress_list(values):
    result = []

    for key, group in itertools.groupby(range(len(values)), values.__getitem__):
        indices = list(group)

        if len(indices) > 1:
            result.append([indices[-1]-indices[0]+1, key])
        else:
            result.append([1, key])
    return result

def PC_to_image(L_kept,i_PC_to_do,L_flat_area_scan,res_factor = 10,show = True):
    
    L_x = np.array([L_kept[i][0] for i in range(len(L_kept))])*res_factor
    L_y = np.array([L_kept[i][1] for i in range(len(L_kept))])*res_factor

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

    

    L_x_flat = np.array([L_flat_area_scan[i][0] for i in range(len(L_flat_area_scan))])*res_factor
    L_y_flat = np.array([L_flat_area_scan[i][1] for i in range(len(L_flat_area_scan))])*res_factor

    L_x_flat = L_x_flat - min(L_x_flat)
    L_y_flat = L_y_flat - min(L_y_flat)



    lim_1_flat =  round(max(L_x_flat))
    lim_2_flat = round(max(L_y_flat))

    output_image_flat = np.ones((lim_1_flat+1,lim_2_flat+1))*255

    for i in range(len(L_flat_area_scan)):
        output_image_flat[round(L_x_flat[i]),round(L_y_flat[i])] = 0


    mask_flat = output_image_flat.copy()


    if results[i_PC_to_do][2] == 1:
        compressed_slice = compress_list(output_image_flat[:, round(len(output_image_flat[0])/2)])[1:-2]
        if show:
            print(len(output_image_flat[:, 1]))
            print(output_image_flat[:, 1])
    if results[i_PC_to_do][2] == 0:
        compressed_slice = compress_list(output_image_flat[round(len(output_image_flat)/2)])[1:-2]
        if show:
            print(len(output_image_flat[1]))
            print(output_image_flat[1])

    if show:
        print(compressed_slice)

    compressed_slice = [compressed_slice[i][0] for i in range(len(compressed_slice)) if compressed_slice[i][1] == 255]

    size_kernel = [max(compressed_slice)+1]

    if show:
        print("size kernel: ")
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

def save_selected_points_simple(L,name = "Selected_points_svg.txt"):
    
    if os.path.exists(name):
        os.remove(name)

    with open(name, 'w') as f:
        for i in range(len(L)):
            f.write("%s\n" % L[i])

def generate_PC_of_crack_only(mask_of_crack,L_crack,res_factor,new_z):
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
        L_crack_only.append([L_i[i],L_j[i],new_z])

    return L_crack_only

def complete_lacking_traj(lacking_traj):
    full_traj = []
    missing = 420-len(lacking_traj)
    per_line = round(missing/len(lacking_traj)-1)
    for i in range(len(lacking_traj)-1):
        
        full_traj.append(list(lacking_traj[i]))
        extra_now = list(getEquidistantPoints(lacking_traj[i],lacking_traj[i+1],per_line))
        extra_now = [list(pt) for pt in extra_now]
        full_traj += extra_now
    
    full_traj.append(list(lacking_traj[-1]))
    return full_traj



def full_analysis_of_a_scan(i_PC_to_do,mask_res_factor,kept_z=8.5,kept_z_tol=1,
                            nb_points_traj=50,nb_best_kept_for_traj=30,proj_on_plane_bool = False,
                            show = True,start_at=0):

    path_all_results = r"C:\Users\legen\Desktop\dissertation\python scripts\palette_results"
    original_palette_3DPC = import_my_3DPC_simple("L_palette_3DPC_original.txt",show = show)
    crack_paths,traj_paths,L_cracks_ori,L_trajs_ori=import_original_cracks_and_trajs(show = show)


    #import the ith scan and keep the top part
    L_SCAN = import_my_3DPC(i_PC_to_do,file_paths,show = show)
    L_kept = keep_top_z(L_SCAN,kept_z,kept_z_tol,show = show)


    print("select a flat area for kernel size")
    
    L_flat_area_scan = selection_crack(L_kept,show = show)
    



    # WARP THE CRACK PALETTE TO THE SCAN
    print("select the 4 corners")

    L_corners_ori = import_my_3DPC_simple("ori_corners.txt",show=show)
    L_corners_SCAN =[[-452.3122607683059, 31.57889375830999, 8.401394754937314], [-455.358494567466, 225.4550887218903, 8.529425184766467], [-370.63992938919654, 31.650613871715482, 7.981165903320232], [-373.8600878471785, 229.55347905037564, 7.568748786055778]]
    L_corners_SCAN = selection_4_corners(L_kept)

    H = Homography_from_corners(L_corners_SCAN, show = show)
    new_z = mean_of_zs(L_corners_SCAN) 
    L_warped = apply_homography_to_svg(original_palette_3DPC,H,new_z)
    
    L_trajs_ori_warped,L_cracks_ori_warped = [],[]

    for i_crack in range(len(L_trajs_ori)):

        L_trajs_ori_warped.append(apply_homography_to_svg(L_trajs_ori[i_crack],H,new_z))
        L_cracks_ori_warped.append(apply_homography_to_svg(L_cracks_ori[i_crack],H,new_z))
    

    

    for i_crack in range(len(L_cracks_ori_warped)):
        if start_at!=0:
            i_crack = start_at
            if start_at==666:
                return L_warped,L_trajs_ori_warped,L_cracks_ori_warped,L_kept,H
        save_yn = 2
        while save_yn == 2:
            print("crack number "+str(i_crack))

            if show:
                PC_view_of_these_lists([L_cracks_ori_warped[1],L_warped,L_kept])

            print("select location of crack")
            L_crack_SCAN = selection_crack(L_kept,show = show)

            mask_of_crack = PC_to_image(L_crack_SCAN,i_PC_to_do,L_flat_area_scan,
                    res_factor = mask_res_factor, show = show)


            L_crack_SCAN_only = generate_PC_of_crack_only(mask_of_crack,
                                L_crack_SCAN,mask_res_factor,new_z)


            print("select points that are NOT the crack")
            L_crack_SCAN_only_removed_wrong = selection_to_remove(
                                            L_crack_SCAN_only, show = show)

            start_time = time.time()

            My_Line_SCAN = calculate_traj_with_planes(
                L_crack_SCAN_only_removed_wrong,nb_points_traj,
                nb_best = nb_best_kept_for_traj,proj_on_plane=proj_on_plane_bool,
                show = show)

            My_Line_SCAN_full_traj = complete_lacking_traj(My_Line_SCAN)

            taken_time = time.time()-start_time
            L_traj_error = sum_dist_each_point(L_trajs_ori_warped[i_crack],My_Line_SCAN_full_traj)
            
            PC_view_of_these_lists([L_crack_SCAN_only_removed_wrong])

            if show:

                PC_view_of_these_lists([My_Line_SCAN_full_traj,My_Line_SCAN],
                                    point_size=0.1)
                
                PC_view_of_these_lists([My_Line_SCAN_full_traj,L_cracks_ori_warped[i_crack]],
                                    point_size=0.1)

                PC_view_of_these_lists([L_trajs_ori_warped[i_crack],L_cracks_ori_warped[i_crack],My_Line_SCAN_full_traj],
                                    point_size=0.05)

            text_to_append = str(i_PC_to_do) + " " + str(i_crack) + " " + str(round(taken_time,2))+" "+str(sum(L_traj_error)/len(L_traj_error))


            save_yn = input("save and keep going (1)\n dont save and do again (2)\n stop (3)\n ")

            if save_yn == "1":
                append_new_line("scan_traj_results.txt",text_to_append)
                save_selected_points_simple(L_traj_error,name = path_all_results+ "\\traj_error "+ text_to_append +".txt")
                save_selected_points_simple(My_Line_SCAN_full_traj,name = path_all_results+ "\\traj_points "+ text_to_append +".txt")

            elif save_yn == "2":
                print("doing again")

            else:
                return L_warped,L_traj_error,L_crack_SCAN_only_removed_wrong,My_Line_SCAN_full_traj,My_Line_SCAN,L_trajs_ori_warped,L_cracks_ori_warped,L_kept
                break



    return L_warped,L_traj_error,L_crack_SCAN_only_removed_wrong,My_Line_SCAN_full_traj,My_Line_SCAN,L_trajs_ori_warped,L_cracks_ori_warped,L_kept,H
    

def get_results_of_line_generation(scan_to_display,show = True):

    path_all_results= r"C:\Users\legen\Desktop\dissertation\python scripts\palette_results"

    trajs_and_errors_paths = [f for f in listdir(path_all_results) if isfile(join(path_all_results, f))]


    trajs_and_errors_paths_for_loop = [trajs_and_errors_paths[i].split(" ") for i in range(len(trajs_and_errors_paths))]

    L_errors_files_of_this_scan = []
    L_trajs_files_of_this_scan = []
    L_trajs_of_this_scan,L_errors_of_this_scan = [],[]
    for i in range(len(trajs_and_errors_paths)):
        if int(trajs_and_errors_paths_for_loop[i][1])==scan_to_display:
            if trajs_and_errors_paths_for_loop[i][0] == "traj_error":
                L_errors_files_of_this_scan.append(trajs_and_errors_paths[i])
            else:  
                L_trajs_files_of_this_scan.append(trajs_and_errors_paths[i])

    for i_traj in range(len(L_trajs_files_of_this_scan)):
        L_thing =  import_my_3DPC_simple(path_all_results+"\\"+L_trajs_files_of_this_scan[i_traj], show = False)
        L_trajs_of_this_scan.append(L_thing)

    for i_error in range(len(L_errors_files_of_this_scan)):
        L_thing =  import_my_3DPC_simple(path_all_results+"\\"+L_errors_files_of_this_scan[i_error], show = False)
        L_thing = [x[0] for x in L_thing]
        L_errors_of_this_scan.append(L_thing)


    for i in range(len(L_errors_of_this_scan)):
        plt.scatter(list(range(len(L_errors_of_this_scan[i]))),L_errors_of_this_scan[i])
        
    if show:
        plt.show()

        # L_to_show = []
        # for i in range(len(L_trajs_of_this_scan)):
        #     L_to_show.append(L_trajs_of_this_scan[i])
        #     L_to_show.append(L_trajs_ori_warped[i])



        # L_to_show.append(L_warped)

        # PC_view_of_these_lists(L_to_show,
        #                     point_size=0.05)

    return L_trajs_of_this_scan,L_errors_of_this_scan,trajs_and_errors_paths_for_loop


def db_generation():
    results_path = r"C:\Users\legen\Desktop\dissertation\Results"

    results = [f for f in listdir(results_path) if isfile(join(results_path, f))]
    file_paths = results.copy()
    file_paths = [results_path+"\\" +results[i_file_name] for i_file_name in range(len(file_paths))]

    for i_res in range(len(results)):
        num_lines = sum(1 for _ in open(results_path+"\\" +results[i_res]))
        results[i_res] = list(map(float,(results[i_res][:-4].split(" ")))) + [bufcount(results_path+"\\" +results[i_res])]



    # Create the pandas DataFrame
    df = pd.DataFrame(results, columns=['Time (s)', 'N_steps', 'Direction','Speed factor','Nb_points'])
    L_size_steps = []
    for i in range(len(results)):
        if results[i][2]==0:
            L_size_steps.append(round(226/results[i][1],2))
        else:
            L_size_steps.append(round(97/results[i][1],2))
    df['Width steps (mm)'] = L_size_steps

    cols = ['N_steps','Width steps (mm)', 'Direction','Speed factor','Nb_points','Time (s)']
    df = df[cols]
    return df, file_paths,results


def append_avg_errors_to_db(df):
    L_avg_errors = []
    L_avg_errors_except_last=[]
    L_avg_errors_expect_last_std = []
    L_i_scan_j_crack = []
    L_std_i_scan_j_each = [] 
        
    for i_scan in range(len(df)):
        L_trajs_of_this_scan,L_errors_of_this_scan,trajs_and_errors_paths_for_loop =\
        get_results_of_line_generation(i_scan,show = False)
        L_score_each_crack = []
        L_std_each_crack = []
        L_score_each_crack_except_last = []
        
        for i_crack in range(len(L_errors_of_this_scan)):
            L_score_each_crack.append(sum(L_errors_of_this_scan[i_crack])/len(L_errors_of_this_scan[i_crack]))
            L_std_each_crack.append(np.std(L_errors_of_this_scan[i_crack]))
        for i_crack in range(len(L_errors_of_this_scan)-1):
            L_score_each_crack_except_last.append(sum(L_errors_of_this_scan[i_crack])/len(L_errors_of_this_scan[i_crack]))
            
        
        L_i_scan_j_crack.append(L_score_each_crack)
        L_std_i_scan_j_each.append(L_std_each_crack)
        L_avg_errors.append(sum(L_score_each_crack)/len(L_score_each_crack))
        
        L_avg_errors_except_last.append(sum(L_score_each_crack_except_last)/len(L_score_each_crack_except_last))
        L_avg_errors_expect_last_std.append(np.std(np.array(L_score_each_crack_except_last)))

    df['accuracy_mean'] = L_avg_errors
    df['accuracy_mean_except_crack_6'] = L_avg_errors_except_last
    df['std_accuracy_mean_except_crack_6'] = L_avg_errors_expect_last_std

    for j_crack in range(len(L_i_scan_j_crack[0])):
        df['crack_'+str(j_crack)] = [L_i_scan_j_crack[i][j_crack] for i in range(len(L_i_scan_j_crack))]
        df['std_crack_'+str(j_crack)] = [L_std_i_scan_j_each[i][j_crack] for i in range(len(L_std_i_scan_j_each))]
        
    
    #TIME FOR TRAJ GEN
    process_time_trajs = np.zeros((len(df),7))
    for i_line in range(len(trajs_and_errors_paths_for_loop)):
        if trajs_and_errors_paths_for_loop[i_line][0] =='traj_error':

            process_time_trajs[int(trajs_and_errors_paths_for_loop[i_line][1])] \
                [int(trajs_and_errors_paths_for_loop[i_line][2])] =\
                    float(trajs_and_errors_paths_for_loop[i_line][3])

    for j_crack in range(len(L_i_scan_j_crack[0])):
        df['crack_time_'+str(j_crack)] = process_time_trajs[:,j_crack]

    return df,L_i_scan_j_crack,trajs_and_errors_paths_for_loop

#%% DB GENERATION
df, file_paths,results=db_generation()
df.drop(6).sort_values(by=['N_steps']).rename(columns={"Time (s)": "Duration (s)"}).to_html('dataa.html')
original_palette_3DPC = import_my_3DPC_simple(
                    "L_palette_3DPC_original.txt",show = True)


df,L_i_scan_j_crack,trajs_and_errors_paths_for_loop = append_avg_errors_to_db(df)
# %%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
L_SCAN = import_my_3DPC(11,file_paths,show = True)
xs = [i[0] for i in L_SCAN]
ys = [i[1] for i in L_SCAN]
zs = [i[2] for i in L_SCAN]
# %%
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 100
import numpy as np 

nb_min = 25

for j_crack in range(7):
    x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Size_steps']
    y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['crack_'+str(j_crack)]
    e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_crack_3']

    x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Size_steps']
    y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['crack_'+str(j_crack)]
    e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_crack_'+str(j_crack)]



    plt.errorbar(x_dir_1,y_dir_1,e_dir_1,label = "Direction 1", linestyle='None', marker='^')
    plt.errorbar(x_dir_0,y_dir_0,e_dir_0,label = "Direction 0", linestyle='None', marker='^')
    plt.xlabel('Resolution of scanner')
    plt.ylabel('Average error')
    plt.title("crack_"+str(j_crack))
    plt.ylim(0,2)
    plt.legend()
    plt.grid()
    plt.show()



    print(stats.ttest_ind(y_dir_1,y_dir_0))

#%%
L_avg_by_n_steps = []
L_std_by_n_steps = []
for n_step in list(set(df['N_steps'])):
    x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] == n_step)]['N_steps']
    y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] == n_step)]['accuracy_mean_except_crack_6']
    e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] == n_step)]['std_accuracy_mean_except_crack_6']

    L_avg_by_n_steps.append(np.mean(y_dir_1))
    L_std_by_n_steps.append(np.mean(e_dir_1))


plt.errorbar(list(set(df['N_steps'])),L_avg_by_n_steps,L_std_by_n_steps,label = 'Direction 1', linestyle='None', marker='^')

L_avg_by_n_steps = []
L_std_by_n_steps = []

for n_step in list(set(df['N_steps'])):
    x_dir_1 = df[(df['Direction'] == 0) & (df['N_steps'] == n_step)]['N_steps']
    y_dir_1 = df[(df['Direction'] == 0) & (df['N_steps'] == n_step)]['accuracy_mean_except_crack_6']
    e_dir_1 = df[(df['Direction'] == 0) & (df['N_steps'] == n_step)]['std_accuracy_mean_except_crack_6']

    L_avg_by_n_steps.append(np.mean(y_dir_1))
    L_std_by_n_steps.append(np.mean(e_dir_1))

plt.errorbar(list(set(df['N_steps'])),L_avg_by_n_steps,L_std_by_n_steps,label = 'Direction 0', linestyle='None', marker='^')
plt.xlabel('Resolution of scanner')
plt.ylabel('Average error')
plt.title("crack_"+str(j_crack))
plt.ylim(0,2)
plt.legend()
plt.grid()
plt.show()



print(stats.ttest_ind(y_dir_1,y_dir_0))
 #%%

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 100
import numpy as np 

nb_min = 10


x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['N_steps']
y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['accuracy_mean_except_crack_6']
e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_accuracy_mean_except_crack_6']

x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['N_steps']
y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['accuracy_mean_except_crack_6']
e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_accuracy_mean_except_crack_6']



plt.errorbar(x_dir_1,y_dir_1,0,label = "Direction 1", linestyle='None', marker='^')
plt.errorbar(x_dir_0,y_dir_0,0,label = "Direction 0", linestyle='None', marker='^')
plt.xlabel('Acc mean')
plt.ylabel('Time (s)')
plt.title("crack_"+str(j_crack))
plt.ylim(0,2)
plt.legend()
plt.grid()
plt.show()



 #%%
nb_min = 10


x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['accuracy_mean_except_crack_6']
y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Time (s)']
e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_crack_3']

x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['accuracy_mean_except_crack_6']
y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Time (s)']
e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_crack_'+str(j_crack)]



plt.errorbar(x_dir_1,y_dir_1,0,label = "Direction 1", linestyle='None', marker='^')
plt.errorbar(x_dir_0,y_dir_0,0,label = "Direction 0", linestyle='None', marker='^')
plt.xlabel('Acc mean')
plt.ylabel('Time (s)')
plt.title("crack_all")

plt.legend()
plt.grid()
plt.show()



print(stats.ttest_ind(y_dir_1,y_dir_0))
#%%

#%%

Dir_0 = df[df['Direction'] == 0]

Dir_1 = df[df['Direction'] == 1]

print(np.mean(Dir_0['accuracy_mean_except_crack_6']))
print(np.mean(Dir_1['accuracy_mean_except_crack_6']))
# %%
plt.scatter(df['crack_3'],df['Time (s)'],c = df['Direction'])
# %%
plt.scatter(df['Nb_points'],df['accuracy_mean'],c = df['Direction'])
# %%
plt.scatter(df['Direction'],df['crack_3'],c = df['Direction'])
# %%
for j_crack in range(7):
    plt.scatter(df['N_steps'],df['crack_'+str(j_crack)],c = df['Direction'])
    plt.grid()
    plt.xlabel('Resolution of scanner')
    plt.ylabel('error (mm)')
    ymax = max(21.74,max(df['crack_'+str(j_crack)]))
    plt.ylim([0, ymax])
    
    plt.title('crack_'+str(j_crack))
    plt.show()
# %%

L_trajs_of_this_scan,L_errors_of_this_scan = get_results_of_line_generation(9)
# %%
for j_crack in range(7):
    plt.scatter(df['N_steps'],df['crack_time_'+str(j_crack)],c = df['Direction'])
    plt.grid()
    plt.xlabel('Resolution of scanner')
    plt.ylabel('error (mm)')
    ymax = max(21.74,max(df['crack_time_'+str(j_crack)]))
    plt.ylim([0, ymax])
    
    plt.title('crack_time_'+str(j_crack))
    plt.show()

#%%


# %%
#y = accuracy crack, x = crack for one direction
L_avg_by_crack = []
L_std_by_crack = []

for j_crack in range(7):
    x_dir_1 = df[(df['direction'] == 1) & (df['N_steps'] > nb_min)]['N_steps']
    y_dir_1 = df[(df['direction'] == 1) & (df['N_steps'] > nb_min)]['crack_'+str(j_crack)]
    e_dir_1 = df[(df['direction'] == 1) & (df['N_steps'] > nb_min)]['std_crack_3']
    L_avg_by_crack.append(np.mean(y_dir_1))
    L_std_by_crack.append(np.std(y_dir_1))


plt.errorbar(list(range(7)),L_avg_by_crack,L_std_by_crack, linestyle='None', marker='^',label = 'direction 1')
L_avg_by_crack = []
L_std_by_crack = []
for j_crack in range(7):
    x_dir_1 = df[(df['direction'] == 0) & (df['N_steps'] > nb_min)]['N_steps']
    y_dir_1 = df[(df['direction'] == 0) & (df['N_steps'] > nb_min)]['crack_'+str(j_crack)]
    e_dir_1 = df[(df['direction'] == 0) & (df['N_steps'] > nb_min)]['std_crack_3']
    L_avg_by_crack.append(np.mean(y_dir_1))
    L_std_by_crack.append(np.std(y_dir_1))


plt.errorbar(list(range(7)),L_avg_by_crack,L_std_by_crack, linestyle='None', marker='^',label = 'direction 0')


plt.xlabel('Crack')
plt.ylabel('Average error')
plt.title("crack_"+str(j_crack))
plt.ylim(0,10)
plt.legend()
plt.grid()
plt.show()



print(stats.ttest_ind(y_dir_1,y_dir_0))
# %%


#L_trajs_ori,L_the_cracks = generate_true_traj_of_crack_palette_svg(False)

# %%

# %%

# %%
columns=['Time (s)', 'N_steps', 'direction','speed_factor','Nb_points']

plt.scatter(df['N_steps'],df['Time (s)'])
    
# %%
#%%
    
L_warped,L_traj_error,L_crack_SCAN_only_removed_wrong,My_Line_SCAN_full_traj,My_Line_SCAN,L_trajs_ori_warped,\
    L_cracks_ori_warped,L_kept = full_analysis_of_a_scan(5,10,show = True)

# %%

L_trajs_of_this_scan,L_errors_of_this_scan =\
     get_results_of_line_generation(16,L_trajs_ori_warped,L_warped)


# %%
#%%
L_SCAN1 = import_my_3DPC(1,file_paths,show = False)
L_SCAN2 = import_my_3DPC(2,file_paths,show = False)
L_SCAN3 = import_my_3DPC(4,file_paths,show = False)
L_SCAN4 = import_my_3DPC(5,file_paths,show = True)

# %%
L_kept = keep_top_z(L_SCAN1,8.5,1,show = True,r=270)
L_kept = keep_top_z(L_SCAN2,8.5,1,show = True,r=270)
L_kept = keep_top_z(L_SCAN3,8.5,1,show = True,r=270)
L_kept = keep_top_z(L_SCAN4,8.5,1,show = True,r=270)

# %%
path_all_results = r"C:\Users\legen\Desktop\dissertation\python scripts\palette_results\\"
L_SCAN4 = import_my_3DPC_simple(str(path_all_results)+"traj_points 16 4 1.91 4.739961090731926.txt",show = True)
# %%
def import_results_cracks_and_trajs(show = True):
    
    points_path = r"C:\Users\legen\Desktop\dissertation\python scripts\palette_results"

    trajs_and_cracks_paths = [f for f in listdir(points_path) if isfile(join(points_path, f))]

    L_cracks_and_trajs = []
    if show:
        print(trajs_and_cracks_paths )

    for i_files in range(len(trajs_and_cracks_paths)):
        L_thing =  import_my_3DPC_simple(points_path+"\\"+trajs_and_cracks_paths[i_files], show = False)
        L_cracks_and_trajs.append(L_thing)
        if show:
            print(i_files)


    error_paths,traj_paths,L_trajs_res,L_error_res = [],[],[],[]
    for i in range(len(trajs_and_cracks_paths)):
        if trajs_and_cracks_paths[i][:10] == 'traj_error':

            error_paths.append(trajs_and_cracks_paths[i])
            L_error_res.append(L_cracks_and_trajs[i])
        else:
            traj_paths.append(trajs_and_cracks_paths[i])
            L_trajs_res.append(L_cracks_and_trajs[i])
    
    
    return error_paths,traj_paths,L_trajs_res,L_error_res


# %%

PC_view_of_these_lists([L_warped,L_crack_SCAN_only_removed_wrong],point_size=0.05)

# %%===================================================================
nb_min = 10


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 100
import numpy as np 

x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['N_steps']
y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['crack_6']
e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_crack_6']

x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['N_steps']
y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['crack_6']
e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_crack_6']


# create figure and axis objects with subplots()
fig,ax = plt.subplots()

            
ax.errorbar(x_dir_1,y_dir_1,e_dir_1,label = "Parallel scan", linestyle='None', marker='o')
ax.errorbar(x_dir_0,y_dir_0,e_dir_0,label = "Perpendicular scan", linestyle='None', marker='o')
ax.set_xlabel('N_steps')
ax.set_ylabel('Mean distance between\ntrue and generated path (mm)')

ax.set_xticks(list(range(0,1001,100)))
plt.title("Analysis of crack 6 with various\nN_steps and scanning directions")
ax.set_ylim(0,1.2*max(max(y_dir_1),max(y_dir_0)))

ax.legend()
ax.grid()
plt.show()

# %%===================================================================
nb_min = 10
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 100
import numpy as np 

x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['N_steps']
y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Time (s)']
e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_crack_3']

x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['N_steps']
y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Time (s)']
e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_crack_3']

#Parallel = 1
#Perpendicular = 0

plt.errorbar(x_dir_1,y_dir_1,0,label = "Parallel", linestyle='None', marker='^')
plt.errorbar(x_dir_0,y_dir_0,0,label = "Perpendicular", linestyle='None', marker='^')
plt.xlabel('N steps')
plt.ylabel('Duration (s)')
plt.title("Scan duration (s) for various number of steps\nand two scanning directions")

plt.legend()
plt.grid()

from scipy import stats
#linregress() renvoie plusieurs variables de retour. On s'interessera 
# particulierement au slope et intercept

#la variable fitLine sera un tableau de valeurs prédites depuis la tableau de variables X
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x_dir_1, y_dir_1)

def predict(x):
       return slope1 * x + intercept1

fitLine1 = predict(x_dir_1)
plt.plot(x_dir_1, fitLine1, c='b',alpha=0.35)

slope0, intercept0, r_value0, p_value0, std_err0 = stats.linregress(x_dir_0, y_dir_0)

def predict(x):
       return slope0 * x + intercept0

fitLine0 = predict(x_dir_0)
plt.plot(x_dir_0, fitLine0, c='orange',alpha=0.5)

plt.show()
print("dir 0 :")
print("slope, intercept, r_value, p_value, std_err")
L_p = [slope0, intercept0, r_value0, p_value0, std_err0]
stri = ""
for i in L_p:
    stri += str(i) + ","
print(stri)
print("dir 1 :")
L_p = [slope1, intercept1, r_value1, p_value1, std_err1]
stri = ""
for i in L_p:
    stri += str(i) + ","
print(stri)
# %%===================================================================
#Parallel = 1
#Perpendicular = 0
nb_min = 10


x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Width steps (mm)']
y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Time (s)']
e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_crack_3']

x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Width steps (mm)']
y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Time (s)']
e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_crack_3']



plt.errorbar(x_dir_1,y_dir_1,0,label = "Parallel", linestyle='None', marker='^')
plt.errorbar(x_dir_0,y_dir_0,0,label = "Perpendicular", linestyle='None', marker='^')
plt.xlabel('Width steps (mm)')
plt.ylabel('Duration (s)')
plt.title("Scan duration (s) against the width of the\nscanning steps (mm) and two scanning directions")

x01 = sorted(np.array(list(np.array(x_dir_0)) + list(np.array(x_dir_1))))

def predict(x):
       return slope1 * 1/x + intercept1

fitLine1 = predict(x01)
plt.plot(x01, fitLine1, c='tab:brown',alpha=0.75)

plt.legend()
plt.grid()



#%%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX GOOD
nb_min = 10


x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Width steps (mm)']
y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Time (s)']
e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_crack_3']

x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Width steps (mm)']
y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Time (s)']
e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_crack_3']



plt.errorbar(1/x_dir_1,y_dir_1,0,label = "Parallel", linestyle='None', marker='^')
plt.errorbar(1/x_dir_0,y_dir_0,0,label = "Perpendicular", linestyle='None', marker='^')
plt.xlabel('Inverse width steps (mm⁻¹)')
plt.ylabel('Duration (s)')
plt.title("Scan duration (s) against the inverse of the width of the\nscanning steps (mm⁻¹) and two scanning directions")


plt.legend()
plt.grid()

from scipy import stats
#linregress() renvoie plusieurs variables de retour. On s'interessera 
# particulierement au slope et intercept

x01 = np.array(list(1/np.array(x_dir_0)) + list(1/np.array(x_dir_1)))
y01 = np.array(list(y_dir_0)+list(y_dir_1))
#la variable fitLine sera un tableau de valeurs prédites depuis la tableau de variables X
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x01, y01)

def predict(x):
       return slope1 * x + intercept1

fitLine1 = predict(x01)
plt.plot(x01, fitLine1, c='tab:brown',alpha=0.75)

plt.show()
print("slope, intercept, r_value, p_value, std_err")
L_p = [slope1, intercept1, r_value1, p_value1, std_err1]
stri = ""
for i in L_p:
    stri += str(i) + ","
print(stri)

# %%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX GOOD
L_scan1 = import_my_3DPC(5,file_paths,show = True)
L_scan2 = import_my_3DPC(8,file_paths,show = True)

#%%

L_k1 = keep_top_z(L_scan1,8.5,1)
v1 = PC_view_of_these_lists([L_k1])
v1.set(phi = 0,theta = np.pi/2)

L_k2 = keep_top_z(L_scan2,8.5,1)
#%%
v1 = PC_view_of_these_lists(L_cracks_ori_warped+[L_k2])
v1.set(phi = 0,theta = np.pi/2)

# %%============================CRACK 0 EXAMPLE=======================================

def cool_plot(crack=-1):
    
    nb_min = 10

    #Parallel = 1
    #Perpendicular = 0
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['figure.dpi'] = 100
    import numpy as np 

    if crack == -1:
        my_title = "Mean LSED over 6 cracks for various\nwidth steps (mm) and scanning directions"
        x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Width steps (mm)']
        y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['accuracy_mean_except_crack_6']
        e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_accuracy_mean_except_crack_6']

        x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Width steps (mm)'].drop(6)
        y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['accuracy_mean_except_crack_6'].drop(6)
        e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_accuracy_mean_except_crack_6'].drop(6)

    else:
        my_title = "Mean LSED of crack "+ str(crack)+ " for various\nwidth steps (mm) and scanning directions"
        x_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['Width steps (mm)']
        y_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['crack_'+str(crack)]
        e_dir_1 = df[(df['Direction'] == 1) & (df['N_steps'] > nb_min)]['std_crack_'+str(crack)]

        x_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['Width steps (mm)'].drop(6)
        y_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['crack_'+str(crack)].drop(6)
        e_dir_0 = df[(df['Direction'] == 0) & (df['N_steps'] > nb_min)]['std_crack_'+str(crack)].drop(6)

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()

                
    ax.errorbar(x_dir_1,y_dir_1,e_dir_1,label = "Parallel scan", linestyle='None', marker='o')
    ax.errorbar(x_dir_0,y_dir_0,e_dir_0,label = "Perpendicular scan", linestyle='None', marker='o')
    ax.set_xlabel('Width steps (mm)')
    ax.set_ylabel('Mean LSED  between\ntrue and generated paths (mm)')

    plt.title(my_title)
    ax.set_ylim(0,20)#1.2*max(max(y_dir_1),max(y_dir_0)))
    #ax.set_xscale('log')
    ax.legend()
    plt.minorticks_on()
    ax.grid()
    ax.grid(which='minor',alpha=0.5, linestyle='--')
    plt.show()

    fig,ax = plt.subplots()


    mean0 = np.mean(np.array(y_dir_0[2:-1]))
    mean1 = np.mean(np.array(y_dir_1[2:]))
    std0 = np.std(np.array(y_dir_0[2:-1]))
    std1 = np.std(np.array(y_dir_1[2:]))
    ax.plot([0,1.3],[mean0,mean0],c="tab:orange", linestyle='dashed',alpha = 0.75)
    ax.plot([0,1.3],[mean1,mean1],c="tab:blue", linestyle='dashed',alpha=0.75)
    ax.errorbar(x_dir_1,y_dir_1,e_dir_1,label = "Parallel scan (mean = " + str(round(mean1,2))+")", linestyle='None', marker='o')
    ax.errorbar(x_dir_0,y_dir_0,e_dir_0,label = "Perpendicular scan (mean = " + str(round(mean0,2))+")", linestyle='None', marker='o')
    ax.set_xlabel('Width steps (mm)')
    ax.set_ylabel('Mean LSED between\ntrue and generated paths (mm)')
    plt.minorticks_on()
    plt.title(my_title)
    ax.set_ylim(0,0.6)#1.2*max(max(y_dir_1),max(y_dir_0)))
    #ax.set_xscale('log')
    ax.set_xlim(0,1.25)
    ax.legend(loc="upper right")
    ax.grid()
    ax.grid(which='minor',alpha=0.5, linestyle='--')
    plt.show()

    return mean0,mean1,std0,std1
# %%
L_mean_per_crack_dir_0 = []
L_mean_per_crack_dir_1 = []
L_std_per_crack_dir_0 = []
L_std_per_crack_dir_1 = []
for i in range(6):
    mean0,mean1,std0,std1 = cool_plot(i)
    L_mean_per_crack_dir_0.append(mean0)
    L_mean_per_crack_dir_1.append(mean1)
    L_std_per_crack_dir_0.append(std0)
    L_std_per_crack_dir_1.append(std1)

plt.scatter(range(len(L_mean_per_crack_dir_0)),L_mean_per_crack_dir_0,c="tab:orange")
plt.scatter(range(len(L_mean_per_crack_dir_1)),L_mean_per_crack_dir_1,c="tab:blue")
plt.show()
# %%
fig,ax = plt.subplots()
x = list(range(6))

ax.errorbar(x,L_mean_per_crack_dir_1,L_std_per_crack_dir_1,label = "Parallel scans", linestyle='None', marker='o')
ax.errorbar(x,L_mean_per_crack_dir_0,L_std_per_crack_dir_0,label = "Perpendicular scans", linestyle='None', marker='o')
ax.set_xlabel('Crack number')
ax.set_ylabel('Mean LSED between\ntrue and generated paths (mm)')
plt.minorticks_on()
plt.title("Average of the mean LSED (mm) of\nthe finest scans (Width < 1.2 mm) by crack")
ax.set_ylim(0,0.7)#1.2*max(max(y_dir_1),max(y_dir_0)))
#ax.set_xscale('log')
#ax.set_xlim(0,1.25)
ax.legend(loc="upper right")
ax.grid()
ax.grid(which='minor',alpha=0.5, linestyle='--')
plt.show()
# %%
L_warped,L_trajs_ori_warped,L_cracks_ori_warped,L_kept,H = full_analysis_of_a_scan(0,10,show = False,start_at=666)


crack_paths,traj_paths_ori,L_cracks_ori,L_trajs_ori=import_original_cracks_and_trajs(show = False)
# %%
error_paths,traj_paths,L_trajs_res,L_error_res=import_results_cracks_and_trajs()

# %%
i_my=0

trajs_ori_same_col = []
for i in L_trajs_ori_warped:
    trajs_ori_same_col += i

crack_ori_same_col = []
for i in  L_cracks_ori_warped:
    crack_ori_same_col += i

traj_res_same_col = []
for i in  L_trajs_res[0+i_my*7:7*i_my+7]:
    traj_res_same_col += i


v1 = PC_view_of_these_lists([L_warped]+[trajs_ori_same_col]+[traj_res_same_col])

v1.set(phi = np.pi/2,theta = np.pi/2)


# %%
