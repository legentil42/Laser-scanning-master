#%%
from ensurepip import version
import itertools
import pandas as pd
import pptk
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#%%
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
# %%
df.plot(x ='Nb_points', y='Time (s)',c='direction', colormap='Paired'  ,kind = 'scatter')
plt.grid()
plt.show()
# %%
df.plot(x ='Nb_points', y='Time (s)',c='direction', colormap='Paired'  ,kind = 'scatter')
plt.grid()
plt.show()

#%%

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


def keep_top_z(L_SCAN,z,tolerance,show = True):
    L_kept = []
    for i in range(len(L_SCAN)):
        if abs(L_SCAN[i][2]-z) < tolerance:
            L_kept.append(L_SCAN[i])
    if show:
        v = pptk.viewer(L_kept)
        v.set(point_size=0.1)

    return L_kept

def calc_results(dif_red,show = True):
    crack_palette_png = show_img("crack palette.png","original crack palette")
    show_img(dif_red[89:313],"dif")
    plt.show()
    L_results = []
    L_count_red,L_count_white=[],[]
    count_white_on_palette = 0
    for n_crack in range(7):

        count_red = 0
        count_white = 0
        for i in range(89,313):
            for j in range(n_crack*138,n_crack*138+138):
                if list(dif_red[i,j]) == [255,0,0]:
                    count_red += 1
                if list(dif_red[i,j]) == [255,255,255]:
                    count_white += 1
                if list(crack_palette_png[i,j]) == [255,255,255]:
                    count_white_on_palette += 1
        
        L_results.append(100*count_red/count_white_on_palette)
        L_count_red.append(count_red)
        L_count_white.append(count_white)

    if show:
        plt.grid()
        plt.title("ratio of red pixels over white pixels for each crack")

        for i, j in zip(list(range(1,8)), L_results):
            plt.text(i, j+5, '{}%'.format(round(j,2)))

        plt.scatter(list(range(1,8)),L_results)
        plt.show()
        #count red
        plt.grid()
        plt.title("ratio of red pixels/wihte = errors")

        for i, j in zip(list(range(1,8)), np.array(L_count_red)/( np.array(L_count_red)+ np.array(L_count_white))):
            plt.text(i, j+5, '{}'.format(round(j,2)))

        plt.scatter(list(range(1,8)),np.array(L_count_red)/( np.array(L_count_red)+ np.array(L_count_white)))
        plt.show()
        #count white
        plt.grid()
        plt.title("ratio of white pixels/ red+white = correctly guessed")

        for i, j in zip(list(range(1,8)), np.array(L_count_white)/( np.array(L_count_red)+ np.array(L_count_white))):
            plt.text(i, j+5, '{}'.format(round(j,2)))

        plt.scatter(list(range(1,8)),np.array(L_count_white)/( np.array(L_count_red)+ np.array(L_count_white)))
        plt.show()



    return L_results



def raw_img_kernel_and_turn(img,tolerance_theta = 2,show = True):
    
    img_turn = img.copy()
    img_hough = img.copy()
    img_hough = np.uint8(img_hough)
    if show:
        print(img_hough)
    edges = cv.Canny(img_hough,50,150,apertureSize = 3)
    lines = cv.HoughLines(edges,1,np.pi/180,200)
    L_thetas = []


    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img_hough,(x1,y1),(x2,y2),(155,0,5),2)
        if abs((theta*180/np.pi) - 90) < tolerance_theta:
            L_thetas.append((theta*180/np.pi)-90)
    if show:
        print(L_thetas)

    rows,cols = img_turn.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),np.mean(L_thetas),1)
    img_turn = cv.warpAffine(img_turn,M,(cols,rows))

    img_turn = cv.warpAffine(img_turn, M, (img_turn.shape[1],
        img_turn.shape[0]), borderValue = 255)

    _,img_turn = cv.threshold(img_turn, 127,255 , cv.THRESH_TOZERO)

    cv.imwrite('houghlines3.jpg',img_hough)
    img_turn = cv.flip(img_turn, 1)
    if show:
        show_img(img_turn)

    return img_turn

def compress_list(values):
    result = []

    for key, group in itertools.groupby(range(len(values)), values.__getitem__):
        indices = list(group)

        if len(indices) > 1:
            result.append([indices[-1]-indices[0]+1, key])
        else:
            result.append([1, key])
    return result

def PC_to_image(L_kept,i_PC_to_do,show = True):
    
    L_x = np.array([L_kept[i][0] for i in range(len(L_kept))])
    L_y = np.array([L_kept[i][1] for i in range(len(L_kept))])


    height = max(L_x)-min(L_x)
    width = max(L_y)-min(L_y)

    L_x = L_x - min(L_x)
    L_y = L_y - min(L_y)

    L_x = L_x * 966 / width
    L_y = L_y * 400 / height

    if show:
        print(max(L_x))
        print(max(L_y))

    lim_1 =  max(round(max(L_x))+1,400)
    lim_2 = max(round(max(L_y))+1,966)

    output_image = np.ones((lim_1+1,lim_2+1))*255

    for i in range(len(L_kept)):
        output_image[round(L_x[i]),round(L_y[i])] = 0

    if show:
        show_img(output_image,"3DPC to img and no processing")
        
    mask = output_image.copy()


    if results[i_PC_to_do][2] == 1:
        compressed_slice = compress_list(output_image[:, 5])[1:-2]
        if show:
            print(len(output_image[:, 5]))
            print(output_image[:, 5])
    if results[i_PC_to_do][2] == 0:
        compressed_slice = compress_list(output_image[10])[1:-2]
        if show:
            print(len(output_image[10]))
            print(output_image[10])

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

def raw_img_kernel_and_turn_SIFT(img1, img2,SIFT =True, show = True):
    """
    Trouver la matrice d'homography entre deux arrays (images)
    à l'aide des points SIFT

    retourne H = la matrice OU False si rien trouvé
    
    img1 : array de la première image
    img2 : array de la deuxieme image
    """
    if SIFT:
        # define constants
        MIN_MATCH_COUNT = 10
        MIN_DIST_THRESHOLD = 0.98

        img1 = cv.flip(img1, 1)
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()



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
                if show:
                    print(Homography)
                
                img3 = cv.drawMatches(image1_8bit, kp1, image2_8bit, kp2, good[:10], image2_8bit, flags=2)
                show_img(img3)

                img1_SIFT = cv.warpAffine(img1, Homography, (img1.shape[1],
                img1.shape[0]), borderValue = 255)

                _,img1_SIFT = cv.threshold(img1_SIFT, 127,255 , cv.THRESH_TOZERO)


                #if abs(Homography[0][0]-1) > 0.1:
                    #return np.float32([[1,0,0],[0,1,0]])

                return img1_SIFT
            #sinon on renvoie False

            else:
                if show:
                    print("Not enough matches are found - {}/{}".format(len(good),
                    MIN_MATCH_COUNT))

                return False
        else:
            if show:
                print("Pas de point sift trouvé sur une des images")
            return False

    else:
        MIN_MATCH_COUNT = 3
        MIN_DIST_THRESHOLD = 0.8

        img1 = cv.flip(img1, 1)
        img1_untouched=img1.copy()
        img2 =  cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        ret,img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)


        kernel = np.ones((10,10), np.uint8)
        # Using cv.erode() method 
        img1 = cv.erode(img1, kernel) 
        img1 = cv.dilate(img1, kernel) 

        img2 = cv.erode(img2, kernel) 
        img2 = cv.dilate(img2, kernel) 
        connectivity = 4  



       

        image1_8bit = cv.normalize(img1, None, 0, 255,
        cv.NORM_MINMAX).astype('uint8')
        image2_8bit = cv.normalize(img2, None, 0, 255,
        cv.NORM_MINMAX).astype('uint8')

        #ret,image2_8bit = cv.threshold(image2_8bit,0,255,150)
        

        
        numLabels1, labels1, stats1, centroids1 = cv.connectedComponentsWithStats(image1_8bit, connectivity, cv.CV_32S)
        numLabels2, labels2, stats2, centroids2 = cv.connectedComponentsWithStats(image2_8bit, connectivity, cv.CV_32S)
        
        
        if show:
            print(stats1)
            print(stats2)

        stats2 = list(stats2)
        stats2 = list(map(list,stats2))

        stats1 = list(stats1)
        stats1 = list(map(list,stats1))

        from operator import itemgetter


        top_right_marker = max(stats2,key=itemgetter(4)) 
        stats2.pop(stats2.index(top_right_marker))
        bottom_right_marker = top_right_marker
        
        def box_to_4_points(mk1,mk2):
            return [[mk1[0],mk1[1]],[mk1[0]+mk1[2],mk1[1]],[mk1[0],mk1[1]+mk1[3]],[mk1[0]+mk1[2],mk1[1]+mk1[3]],
            [mk2[0],mk2[1]],[mk2[0]+mk2[2],mk2[1]],[mk2[0],mk2[1]+mk2[3]],[mk2[0]+mk2[2],mk2[1]+mk2[3]]]
        
        src_pts = box_to_4_points(top_right_marker,bottom_right_marker)
        

        top_right_marker = max(stats1,key=itemgetter(4)) 
        stats1.pop(stats1.index(top_right_marker))
        bottom_right_marker = top_right_marker
        
        dst_pts = box_to_4_points(top_right_marker,bottom_right_marker)


        if show:
            print(src_pts)
            print(dst_pts)

        for i in range(len(src_pts)):

            cv.circle(img1, tuple(src_pts[i]), 10, 125, 10)
            
        for i in range(len(dst_pts)):

            cv.circle(img2, tuple(dst_pts[i]), 10, 125, 10)

        show_img(img1)
        show_img(img2)

        Homography, _ = cv.estimateAffine2D(np.float32(dst_pts), np.float32(src_pts))
        if show:
            print(Homography)

        img1_SIFT = cv.warpAffine(img1_untouched, Homography, (img1_untouched.shape[1],
        img1_untouched.shape[0]), borderValue = 255)

        _,img1_SIFT = cv.threshold(img1_SIFT, 127,255 , cv.THRESH_TOZERO)


        return img1_SIFT

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return nodes[np.argmin(dist_2)]


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


    pts,img = coordinateStore.getpt(4,img)
    if show:
        print(pts)

    cv.imshow(windowname,img)
    cv.waitKey(0)

    return np.float32(pts)

def img_to_4_corners(img):
    img_8bit = np.uint8(img)
    im2, contours = cv.findContours(255-img_8bit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    for i in range(len(im2)):
        imgdraw = img.copy()
        for j in range(len(im2[i])):
            cv.circle(imgdraw, tuple(list(im2[i][j][0])), 10, 125, 2)

    pts=points_on_image(imgdraw)

    L_4_corners = []

    possible_points = [list(im2[i][j][0]) for j in range(len(im2[0]))]

    for i in range(len(pts)):
        L_4_corners.append(closest_node(pts[i], possible_points))

    L_4_corners = [list(L_4_corners[i]) for i in range(len(L_4_corners))]
    return L_4_corners


def click_on_corners(img1, img2, show = True):
    
    img1 = cv.flip(img1, 1)
    img1_untouched=img1.copy()
    img2 =  cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret,img2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)


    image1_8bit = cv.normalize(img1, None, 0, 255,
    cv.NORM_MINMAX).astype('uint8')
    image2_8bit = cv.normalize(img2, None, 0, 255,
    cv.NORM_MINMAX).astype('uint8')





    src_pts = img_to_4_corners(img1)

    

    dst_pts = [[0, 0], [965, 0], [0, 399], [965, 399]]
    print(src_pts)
    src_pts_ordered = []
    for i in range(len(dst_pts)):
        src_pts_ordered.append(list(closest_node(dst_pts[i],src_pts)))
        src_pts.remove(list(src_pts_ordered[-1]))

    print(src_pts_ordered)
    src_pts=src_pts_ordered
    if show:
        print(src_pts)
        print(dst_pts)

    for i in range(len(src_pts)):

        cv.circle(img1, tuple(src_pts[i]), 10, 125, 10)
        
    for i in range(len(dst_pts)):

        cv.circle(img2, tuple(dst_pts[i]), 10, 125, 10)

    show_img(img1,"img1 avec les cercles")
    show_img(img2,"img2 avec les cercles")

    Homography, _ = cv.estimateAffine2D(np.float32(src_pts), np.float32(dst_pts))
    if show:
        print(Homography)

    img1_corner = cv.warpAffine(img1_untouched, Homography, (img1_untouched.shape[1],
    img1_untouched.shape[0]), borderValue = 255)

    _,img1_corner = cv.threshold(img1_corner, 127,255 , cv.THRESH_TOZERO)

    if show:
        show_img(img1_corner,"img1_corner_turned")

    return img1_corner
# %%
print(df)

# %%
show = True
i_PC_to_do = 13
L_SCAN = import_my_3DPC(i_PC_to_do,file_paths,show = True)
crack_palette = show_img("crack palette.png","original crack palette")

L_kept = keep_top_z(L_SCAN,8.5,1,show = True)

# %%
img = PC_to_image(L_kept,i_PC_to_do,show = True)
# %%
img1_turned = click_on_corners(img,crack_palette,show = False)


dif_red = crack_palette.copy()

for i in range(len(dif_red)):#89,313):
    for j in range(len(dif_red[0])):
        if img1_turned[i,j] != dif_red[i,j,0]:
            dif_red[i,j] = [255,0,0]
show = True
if show:
    show_img(img1_turned,"img_turned")
    show_img(crack_palette,"original crack palette")
    show_img(dif_red,"difference between the 2")

L_res = calc_results(dif_red)
# %%


# %%





# %%


img = PC_to_image(L_kept,i_PC_to_do,show = True)

show_img(img)
# %%



# %%