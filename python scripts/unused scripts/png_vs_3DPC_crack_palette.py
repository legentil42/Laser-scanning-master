#%%
from unittest import expectedFailure
import numpy as np
import numpy as np
from scipy.optimize import fmin
from scipy import optimize
import math
import numdifftools as nd
import random
import ast
import pptk
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.pyplot as plt

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



def PC_to_image(L_kept,size_kernel,show = True,turn = False):

    L_x = np.array([L_kept[i][0] for i in range(len(L_kept))])*5
    L_y = np.array([L_kept[i][1] for i in range(len(L_kept))])*5



    width = round(max(L_x)-min(L_x))
    height = round(max(L_y)-min(L_y))


    L_x = L_x - min(L_x)
    L_y = L_y - min(L_y)


        

    output_image = np.ones((width+1,height+1))*255

    for i in range(len(L_kept)):
        output_image[round(L_x[i]),round(L_y[i])] = 0
        if True == False:
            output_image[round(L_x[i]),round(L_y[i])] = 0
            output_image[round(L_x[i])+1,round(L_y[i])] = 0
            output_image[round(L_x[i])-1,round(L_y[i])] = 0
            output_image[round(L_x[i]),round(L_y[i])+1] = 0
            output_image[round(L_x[i]),round(L_y[i])-1] = 0

        
    if turn:
        output_image = np.transpose(output_image)
        output_image = cv.flip(output_image, 1)
    
    if show:
        show_img(output_image,"3DPC_untouched")

    mask = output_image.copy()

    for i in range(len(size_kernel)):
            
        # Creating kernel
        kernel = np.ones((size_kernel[i], 1), np.uint8)
        if turn:
            kernel = np.ones((1,size_kernel[i]), np.uint8)
        # Using cv.erode() method 
        mask = cv.erode(mask, kernel) 
        if show:
            show_img(mask,"eroded_3DPC")




    return mask


class MousePts:
    def __init__(self,windowname,img):
        self.windowname = windowname
        self.img1 = img.copy()
        self.img = self.img1.copy()
        cv.namedWindow(windowname,cv.WINDOW_NORMAL)
        cv.imshow(windowname,img)
        self.curr_pt = []
        self.point   = []

    def select_point(self,event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.point.append([x,y])
            #print(self.point)
            cv.circle(self.img,(x,y),5,(0,255,0),-1)
        elif event == cv.EVENT_MOUSEMOVE:
            self.curr_pt = [x,y]
            #print(self.point)

    def getpt(self,count=1,img=None):
        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()
        cv.namedWindow(self.windowname,cv.WINDOW_NORMAL)
        cv.imshow(self.windowname,self.img)
        cv.setMouseCallback(self.windowname,self.select_point)
        self.point = []
        while(1):
            cv.imshow(self.windowname,self.img)
            k = cv.waitKey(20) & 0xFF
            if k == 27 or len(self.point)>=count:
                break
            #print(self.point)
        cv.setMouseCallback(self.windowname, lambda *args : None)
        #cv.destroyAllWindows()
        return self.point, self.img

def points_on_image(img):
    windowname = 'image'
    coordinateStore = MousePts(windowname,img)


    pts,img = coordinateStore.getpt(3,img)
    print(pts)

    cv.imshow(windowname,img)
    cv.waitKey(0)

    return np.float32(pts)

def manualdHomography(img1, img2):
    
    rows,cols = img1.shape
    theta_mean_rad = 0
    kernel = np.ones((10,10), np.uint8)
    # Using cv.erode() method 
    img1 = cv.erode(img1, kernel) 
    img1 = cv.dilate(img1, kernel) 
    img1 = np.uint8(img1)
    edges = cv.Canny(img1,50,150,apertureSize = 3)
    lines = cv.HoughLines(edges,1,np.pi/180,200)

    for line in lines:
        rho,theta = line[0]
        theta_mean_rad+= theta

    theta_mean_rad = theta_mean_rad/len(lines)

    theta_deg = (theta_mean_rad*360/(2*np.pi)-90)
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),theta_deg,1)

    print(theta_deg)
    return M


def findHomography(img1, img2, afficher_les_images):
    """
    Trouver la matrice d'homography entre deux arrays (images)
    à l'aide des points SIFT

    retourne H = la matrice OU False si rien trouvé
    
    img1 : array de la première image
    img2 : array de la deuxieme image
    """
    # define constants
    MIN_MATCH_COUNT = 10
    MIN_DIST_THRESHOLD = 0.9

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()



    kernel = np.ones((10,10), np.uint8)
    # Using cv.erode() method 
    img1 = cv.erode(img1, kernel) 
    img1 = cv.dilate(img1, kernel) 

    #convertir en 8bits

    image1_8bit = cv.normalize(img1, None, 0, 255,
    cv.NORM_MINMAX).astype('uint8')
    image2_8bit = cv.normalize(img2, None, 0, 255,
    cv.NORM_MINMAX).astype('uint8')

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1_8bit, None)
    kp2, des2 = sift.detectAndCompute(image2_8bit, None)

    # find matches
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    if not ( des1 is None ) and not ( des2 is None ):

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < MIN_DIST_THRESHOLD * n.distance:
                good.append(m)



        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])\
            .reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])\
            .reshape(-1, 1, 2)
            
        #si on a assez de bons matchs on calcule et renvoie l'homography
        if len(good) > MIN_MATCH_COUNT:
            
            Homography, _ = cv.estimateAffine2D(src_pts, dst_pts)
            if afficher_les_images:
                print(Homography)
            
            img3 = cv.drawMatches(image1_8bit, kp1, image2_8bit, kp2, good[:10], image2_8bit, flags=2)
            show_img(img3)

            #if abs(Homography[0][0]-1) > 0.1:
                #return np.float32([[1,0,0],[0,1,0]])

            return Homography
        #sinon on renvoie False

        else:
            if afficher_les_images:
                print("Not enough matches are found - {}/{}".format(len(good),
                MIN_MATCH_COUNT))

            return False
    else:
        if afficher_les_images:
            print("Pas de point sift trouvé sur une des images")
        return False


def import_my_3DPC(name,factor = 100,show = True):

    L_SCAN = [];
    with open(name, 'r') as f:
        for line  in f:
            L_SCAN.append(line.strip().replace(" ","").replace("[","").replace("]","").split(","))
            L_SCAN[-1] = [float(i) for i in L_SCAN[-1]]
            if L_SCAN[-1] == [0,0,0]:
                print("hi")
                L_SCAN.pop(-1)

    

    L_SCAN_deprecated = L_SCAN[::factor]

    if show:
        v = pptk.viewer(L_SCAN)
        v.set(point_size=0.1)

        v = pptk.viewer(L_SCAN_deprecated)
        v.set(point_size=0.1)

    return L_SCAN,L_SCAN_deprecated


def show_above_z(L_points,nb_z_slice):
    L_z = [L_points[i][2] for i in range(len(L_points))]
    L_z_slices = []
    for z_i in np.linspace(min(L_z),max(L_z),nb_z_slice):

        L_kept = []
        for i in range(len(L_points)):
            
            if L_points[i][2] > z_i :
                L_kept.append(L_points[i])
    
        #v = pptk.viewer(L_kept)
        #v.set(point_size=0.1)
        L_z_slices.append(L_kept)
    return L_z_slices,np.linspace(min(L_z),max(L_z),nb_z_slice)



def keep_above_z(L_SCAN,this_z,show = True):

    L_kept = []
    for i in range(len(L_SCAN)):
        
        if L_SCAN[i][2] > this_z :
            L_kept.append(L_SCAN[i])
            L_kept[-1][0] = L_kept[-1][0]

    if show:
        v = pptk.viewer(L_kept)
        v.set(point_size=0.1)

    return L_kept





def dif_between_palette_and_scan(L_SCAN,L_SCAN_deprecated,kernel_size,name_png="crack palette.png",show_anything = True,turn=False):
   
   #import the 3DPC and make a deprecated version for faster computing

    crack_palette_png = show_img(name_png,"original crack palette")

    NB_OF_SLICES = 6
    L_z_slices,L_z_intervals = show_above_z(L_SCAN_deprecated,NB_OF_SLICES)

    #CHOOSE WHICH SLICE
    good_slice = 4
    L_kept = keep_above_z(L_SCAN,L_z_intervals[good_slice],show = False)

    

    #transform 3DPC to an image with erosion and kernel size
    png_of_3DPC = PC_to_image(L_kept,kernel_size,show = show_anything,turn = turn)


    Homography = manualdHomography(png_of_3DPC,crack_palette_png)


    png_of_3DPC_translate_H = cv.warpAffine(png_of_3DPC, Homography, (crack_palette_png.shape[1],
        crack_palette_png.shape[0]), borderValue = 255)

    _,png_of_3DPC_translate_H = cv.threshold(png_of_3DPC_translate_H, 127,255 , cv.THRESH_TOZERO)



    dif_red = crack_palette_png.copy()

    for i in range(89,313):
        for j in range(len(dif_red[0])):
            if png_of_3DPC_translate_H[i,j] != crack_palette_png[i,j,0]:
                dif_red[i,j] = [255,0,0]

    if show_anything:
        
        show_img(png_of_3DPC_translate_H,"eroded 3DPC homography")
        show_img(crack_palette_png,"original crack palette")
        show_img(dif_red,"difference between the 2")


    return dif_red,png_of_3DPC_translate_H

def results(dif_red,show = True):
    crack_palette_png = show_img("crack palette.png","original crack palette")
    show_img(dif_red[89:313],"dif")
    plt.show()
    L_results = []
    for n_crack in range(7):

        count_red = 0
        count_white = 0
        for i in range(89,313):
            for j in range(n_crack*138,n_crack*138+138):
                if list(dif_red[i,j]) == [255,0,0]:
                    count_red += 1
                if list(crack_palette_png[i,j]) == [255,255,255]:
                    count_white += 1
        
        L_results.append(100*count_red/count_white)

    if show:
        plt.grid()
        plt.title("ratio of red pixels over white pixels for each crack")

        for i, j in zip(list(range(1,8)), L_results):
            plt.text(i, j+5, '{}%'.format(round(j,2)))

        plt.scatter(list(range(1,8)),L_results)
    return L_results


# %%
L_SCAN_0_200,L_SCAN_deprecated_0_200 = import_my_3DPC("238.2 200 0 4.0.txt",factor = 100,show = False)
dif_red,png_of_3DPC_translate_H = dif_between_palette_and_scan(L_SCAN_0_200,L_SCAN_deprecated_0_200,[2,2,2,2,2],turn = True)

L_res = results(dif_red)

# %%
L_res_by_ker = []
for kernel_size_i in [[2],[2,2],[2,2,2],[2,2,2,2]]:
    dif_red,png_of_3DPC_translate_H = dif_between_palette_and_scan(L_SCAN_0_200,L_SCAN_deprecated_0_200,kernel_size_i,show_anything=False)
    L_res = results(dif_red,True)
    L_res_by_ker.append(L_res)

print(L_res_by_ker)
#%%
L_sums = [sum(i) for i in L_res_by_ker]
# %%
# %%
L_reverted = L_SCAN_0_200.copy()

for i in range(len(L_reverted)):
    L_reverted[i][0] = -L_reverted[i][0]

v = pptk.viewer(L_reverted)
v.set(point_size=0.1)
