#%%
import cv2 as cv
import numpy as np
img = cv.imread("images/eroded_3DPC.png")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/180,200)
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
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    print(theta)


img = cv.imread('messi5.jpg',0)
rows,cols = img.shape
# cols-1 and rows-1 are the coordinate limits.
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imwrite('houghlines3.jpg',img)

def manualdHomography(img1, img2):

    rows,cols = img1.shape
    theta_mean_rad = 0
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    lines = cv.HoughLines(edges,1,np.pi/180,200)

    for line in lines:
        rho,theta = line[0]
        theta_mean_rad+= theta

    theta_mean_rad = theta_mean_rad/len(lines)

    theta_deg = (theta_mean_rad*360/(2*np.pi)-90)
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),theta_deg,1)



    return M

# %%
