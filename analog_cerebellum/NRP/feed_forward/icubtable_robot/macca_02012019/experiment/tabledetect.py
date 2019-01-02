
import sys
sys.path.append("/usr/local/lib")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import cv2
import numpy as np


from matplotlib import pyplot as plt
'''
img = cv2.imread('icubtable4.png',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

def mostrarVentana (titulo, imagen):
    print('Mostrando imagen')
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.imshow(titulo,imagen)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
       cv2.destroyAllWindows()
'''

img = cv2.imread('icubtable4.png', 1)  # loading image in BGR
print( img.shape) #This should not print error response
'''
if not img is None and len(img.shape) == 3 and img.shape[2] == 3:
    blue_img, green_img, red_img = cv2.split(img)  # extracting red channel
    rbin, threshImg = cv2.threshold(red_img, 100, 200, cv2.THRESH_BINARY)  # thresholding
    #mostrarVentana('Binary image', threshImg)
    _,contours,hier = cv2.findContours(threshImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt)>5000:  # remove small areas like noise etc
            # compute the center of the contour
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            hull = cv2.convexHull(cnt)    # find the convex hull of contour
            hull = cv2.approxPolyDP(hull,0.1*cv2.arcLength(hull,True),True)
            if len(hull)==4:
                cv2.drawContours(img,[hull],0,(0,255,0),2)
                cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
                #cv2.putText(img, "center", (cX - 20, cY - 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


else:
    if img is None:
        print ("Sorry the image path was not valid")
    else:
        print ("Sorry the Image was not loaded in BGR; 3-channel format")
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt


import argparse

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect 

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    print(rect)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([   
    		[0, 0],
    		[maxWidth - 1, 0],
    		[maxWidth - 1, maxHeight - 1],
    		[0, maxHeight - 1]], dtype = "float32")
     
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return warped, dst

# Method to find the mid point
def midpoint(ptA, ptB):
    return [  (max(ptA[0], ptB[0]) - min(ptA[0] , ptB[0]) ) * 0.5, (max(ptA[1] , ptB[1]) - min(ptA[1] , ptB[1]) ) * 0.5 ]
    
    
def center_table(ptA, ptB):
    m = midpoint(ptA , ptB)
    x = int( min(ptA[0] , ptB[0]) + m[0] )
    print(x)
    y =  int( min(ptA[1],ptB[1])+ m[1])
    print(y)
    return  [ x , y]

 


# mask boundaries for red
# in OpenCV H has values from 0 to 180, S and V from 0 to 255. The red color, 
# in OpenCV, has the hue values approximately in the range of 0 to 10 and 160 to 180.
# the two range mask are merge to cover all the red shades
image_lower_bound_m1 = np.array([   0,  50,  50], dtype = "uint8")
image_upper_bound_m1 = np.array([  10, 255, 255], dtype = "uint8")
image_lower_bound_m2 = np.array([ 10,  50,  50], dtype = "uint8")
image_upper_bound_m2 = np.array([ 180, 255, 255], dtype = "uint8")
# read image
img_in = cv2.imread('icubtable4.png')
#img_in = cv2.imread('icubtable2.png')
#pts = np.array(eval(args["coords"]), dtype = "float32")
#img_in = cv2.imread('icubtable2.jpg')
#img_in = cv2.resize(img_in,(500,500))
# if nrp 
#cvbridge = CvBridge()
#img_in = cvbridge.imgmsg_to_cv2(img_in, "rgb8")
# converting from rgb to hsv

hsv_im   = cv2.cvtColor(img_in, cv2.COLOR_RGB2HSV)
# Red detection
mask1   = cv2.inRange(hsv_im, image_lower_bound_m1, image_upper_bound_m1)
mask2   = cv2.inRange(hsv_im, image_lower_bound_m2, image_upper_bound_m2)
mask    = cv2.bitwise_or(mask1, mask2)
#img_out = cv2.bitwise_and(img_in, img_in, mask=mask)


# join my masks
#mask = mask1+mask2

# set my output img to zero everywhere except my mask
output_img = img_in.copy()
output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
output_hsv = hsv_im.copy()
output_hsv[np.where(mask==0)] = 0


# Detect red blobs
r, g, b = cv2.split(output_img)

edges = cv2.Canny(output_img,100,200)
plt.subplot(121),plt.imshow(output_img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()




(_, cnts, _) = cv2.findContours(r.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
# loop over our contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
print("screenCnt"+str(screenCnt))

pts = []
border = [] 
for scrcn in range(0,len(screenCnt)):
    border.append([screenCnt[scrcn][0][0], screenCnt[scrcn][0][1]]) 
    cv2.circle(img_in, (screenCnt[scrcn][0][0], screenCnt[scrcn][0][1] ), 5, (255, 0, 0), -1)
    cv2.putText(img_in, "({},{})".format(screenCnt[scrcn][0][0], screenCnt[scrcn][0][1]), (int(screenCnt[scrcn][0][0] - 50), int(screenCnt[scrcn][0][1] - 10) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    pts.append(screenCnt[scrcn][0])
cv2.drawContours(img_in, [screenCnt], -1, (255, 0, 0), 4)

center1 = center_table(border[0],border[2])
center2 = center_table(border[1],border[3])
center  = center_table(center1, center2)

print("center",center)
cv2.circle(img_in, (center[0], center[1]), 5, (255, 0, 0), -1)
cv2.putText(img_in, "center({},{})".format(center[0], center[1]), (int(center[0] - 50), int(center[1] - 10) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) 
cv2.drawContours(img_in, [screenCnt], -1, (255, 0, 0), 4)
# apply the four point tranform to obtain a "birds eye view" of
# the image
warped, points = four_point_transform(img_in,np.array(pts))
#rect = order_points( np.array(pts) )
print(points)


#print(midpoint(points[0][0],points[1][0]))
cv2.imshow("image",img_in);
#cv2.imshow("image",warped);
while(0xff & cv2.waitKey(1) != ord('q')):pass 
cv2.destroyAllWindows();