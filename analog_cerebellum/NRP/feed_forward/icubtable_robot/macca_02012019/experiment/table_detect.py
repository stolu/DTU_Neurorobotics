import sys
sys.path.append("/usr/local/lib")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import cv2
import numpy as np


class detect_table:
    
    def __init__(self,labels):
        
        # mask boundaries for red
        # in OpenCV H has values from 0 to 180, S and V from 0 to 255. The red color, 
        # in OpenCV, has the hue values approximately in the range of 0 to 10 and 160 to 180.
        # the two range mask are merge to cover all the red shades
        self.image_lower_bound_m1 = np.array([   0,  50,  50], dtype = "uint8")
        self.image_upper_bound_m1 = np.array([  10, 255, 255], dtype = "uint8")
        self.image_lower_bound_m2 = np.array([  10,  50,  50], dtype = "uint8")
        self.image_upper_bound_m2 = np.array([ 180, 255, 255], dtype = "uint8")
        self.show_labels = labels
        self.show_contour = True        
        self.previous_screenCnt = [[[214, 196]],[[206, 201]],[[200, 202]], [[213, 206]]]

        self.contour_color = (0, 0, 255)
        self.previous_center = [120,160]
        self.max_offset = 150.
        self.debug = True
        self.table_border = np.float32( [ 300, 300 ,  300, 0 , 300, 300 , 0,300 ],  dtype='float32' )
        self.square_center = np.array( [[150, 125], ],  dtype='float32' )        
        self.rectified_center = []
        self.ball_pos = [0,0]
        self.rectified_target_pos = [0,0]
    def order_points(self,pts):
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
    
    def four_point_transform(self, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        
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
        #dst = np.array([   
        #		[0, 0],
        #		[maxWidth - 1, 0],
        #		[maxWidth - 1, maxHeight - 1],
        #		[0, maxHeight - 1]], dtype = "float32")
        dst = np.array([   
        		[0, 0],
        		[300 - 1, 0],
        		[300 - 1, 300 - 1],
        		[0, 300 - 1]], dtype = "float32") 
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst) 
        #warped = cv2.warpPerspective(self.image, M, (maxWidth, maxHeight))
        warped = cv2.warpPerspective(self.image, M, (300,300))
        # return the warped image
        return warped, dst, M
    
    # Method to find the mid point
    def midpoint(self, ptA, ptB):
        return [  (max(ptA[0], ptB[0]) - min(ptA[0] , ptB[0]) ) * 0.5, (max(ptA[1] , ptB[1]) - min(ptA[1] , ptB[1]) ) * 0.5 ]
        
        
    def center_table(self, ptA, ptB):
        m = self.midpoint(ptA , ptB)
        x = int( min(ptA[0] , ptB[0]) + m[0] )
        
        y =  int( min(ptA[1],ptB[1])+ m[1])
        
        return  [ x , y]    
    
    def find_center(self,img_in):
        self.image = img_in
        hsv_im   = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        # Red detection
        mask1   = cv2.inRange(hsv_im, self.image_lower_bound_m1, self.image_upper_bound_m1)
        mask2   = cv2.inRange(hsv_im, self.image_lower_bound_m2, self.image_upper_bound_m2)
        mask    = cv2.bitwise_or(mask1, mask2)
        #img_out = cv2.bitwise_and(img_in, img_in, mask=mask)
        
        
        # join my masks
        #mask = mask1+mask2
        
        # set my output img to zero everywhere except my mask
        output_img = self.image.copy()
        output_img[np.where(mask==0)] = 0
        if self.debug == True:
            print("\n Out image after mask "+str(output_img))
        # or your HSV image, which I *believe* is what you want
        output_hsv = hsv_im.copy()
        output_hsv[np.where(mask==0)] = 0
        
        
        # Detect red blobs
        r, g, b = cv2.split(output_img)
        
        edges = cv2.Canny(output_img,100,200)


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
                if self.debug == True:
                    print("\n len approx is 4")
                break

        if screenCnt == None: 
            screenCnt = self.previous_screenCnt
            if self.debug == True:
                print("\n screenCnt is None")
        pts = []
        border = [] 
        self.previous_screenCnt = screenCnt

        for scrcn in range(0,len(screenCnt)):
            border.append([screenCnt[scrcn][0][0], screenCnt[scrcn][0][1]]) 
            #cv2.circle(self.image, (screenCnt[scrcn][0][0], screenCnt[scrcn][0][1] ), 5, (255, 0, 0), -1)
            if self.debug == True:
                print("\n border "+str(border))

            if self.show_contour and self.show_labels == True:                
                cv2.putText(self.image, "({},{})".format(screenCnt[scrcn][0][0], screenCnt[scrcn][0][1]), (int(screenCnt[scrcn][0][0] - 50), int(screenCnt[scrcn][0][1] - 10) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.contour_color, 2)
            pts.append(screenCnt[scrcn][0])
        if self.debug == True:
            print("\n pts "+str(pts))

        
              
        # find center 
        center1 = self.center_table(border[0],border[2])
        center2 = self.center_table(border[1],border[3])
        center  = self.center_table(center1, center2)
        if self.debug == True:
                print("\n center1 "+str(center1))
                print("\n center2 "+str(center2))
                print("\n center "+str(center))
        error_center = [ center[0] - self.previous_center[0] , center[0] - self.previous_center[0] ]
        if self.debug == True:
                print("\n current center "+str(center)+" previous center "+str(self.previous_center))        
        if np.sqrt( np.sum([float(c)**2 for c in error_center]) ) >= self.max_offset :
            center = self.previous_center
        self.previous_center = center
        
        #cv2.circle(self.image, (center[0], center[1]), 5, self.contour_color, -1)
        #if self.show_labels == True:
        #    cv2.putText(self.image, "center({},{})".format(center[0], center[1]), (int(center[0] - 50), int(center[1] - 10) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.contour_color, 2) 
        if self.show_contour == True:
            cv2.drawContours(self.image, [screenCnt], -1, self.contour_color, 4)
        # apply the four point tranform to obtain a "birds eye view" of
        # the image
        
        inv_rectify_matrix = cv2.getPerspectiveTransform(  np.float32( [ [ 0, 0] ,  [ 300, 0] , [ 300, 300] , [ 0,300] ],  dtype='float32' ), np.array( border,  dtype='float32' ) )
        if self.debug == True:
                print("\n from square to "+str( inv_rectify_matrix ))
        
        warped, points, self.rectify_matrix = self.four_point_transform(np.array(pts))
        #self.rectify_matrix     = cv2.getPerspectiveTransform(np.float32( border,  dtype='float32' ),  np.float32( [ [0, 0, -5, -5] , [ 300, 0, 5,5 ], [300, 300, 25, 25 ],   [0,300, -25,-25] ],  dtype='float32' ))
        if self.debug == True:
                print("\n from pt to square "+str(self.rectify_matrix ))
        #xy_table_center_pos = [cv2.perspectiveTransform( self.square_center,  inv_rectify_matrix) ]
        xy_table_center_pos = cv2.perspectiveTransform( np.float32( [ ( (  (150,120) ), ) ]),  inv_rectify_matrix)  
        
        center = xy_table_center_pos[0][0]
        error_center = [ center[0] - self.previous_center[0] , center[0] - self.previous_center[0] ]
        if self.debug == True:
                print("\n current center "+str(center)+" previous center "+str(self.previous_center))        
        if np.sqrt( np.sum([float(c)**2 for c in error_center]) ) >= self.max_offset :
            center = self.previous_center
        self.previous_center = center
        
        cv2.circle(self.image, (center[0], center[1]), 5, self.contour_color, -1)
        if self.show_labels == True:
            cv2.putText(self.image, "center({},{})".format(center[0], center[1]), (int(center[0] - 50), int(center[1] - 10) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.contour_color, 2) 

        
        
       
        '''
        cv2.circle(self.image, (xy_table_center_pos[0][0][0], xy_table_center_pos[0][0][1]), 5, self.contour_color, -1)        
        
        if self.show_labels == True:
            cv2.putText(self.image, "center({},{})".format(xy_table_center_pos[0][0][0], xy_table_center_pos[0][0][1]), (int(xy_table_center_pos[0][0][0]- 50), int( xy_table_center_pos[0][0][1]- 10) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.contour_color, 2) 
                
        print("\n xy_table_center_pos "+str(xy_table_center_pos))
        print("\n xy_table_center_pos 2 "+str(xy_table_center_pos[0][0]))
        border2 = []
        print("\n border "+str(border))
        #warped, points = self.four_point_transform(np.array(pts))
        '''
        
        '''        
        print("\n points "+str(points))
        print("\n points[1][1] "+str(points[1][1]))
        #for p in range(0,4):
        border2 = [ [points[0][0], points[0][1]],
                   [points[1][0], points[1][1]],
                    [points[2][0], points[2][1]],
                    [points[3][0], points[3][1]]] 
        print("appending border2"+str(border2))
        #cv2.circle(self.image, (screenCnt[scrcn][0][0], screenCnt[scrcn][0][1] ), 5, (255, 0, 0), -1)
        #if self.show_contour and self.show_labels == True:                
        #    cv2.putText(warped, "({},{})".format(border2[0][0], border2[0][1]), (int(border2[1][0] - 50), int(border2[1][1] - 10) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.contour_color, 2)
        
        # find center 
        print("\n border2"+str(border2))
        print("\n points"+str(points))
        center1 = self.center_table(border2[0],border2[2])
        center2 = self.center_table(border2[1],border2[3])
        center  = self.center_table(center1, center2)
        print("\n center "+str(center))
        error_center = [ center[0] - self.previous_center[0] , center[0] - self.previous_center[0] ]
        if self.debug == True:
                print("\n current center "+str(center)+" previous center "+str(self.previous_center))        
        if np.sqrt( np.sum([float(c)**2 for c in error_center]) ) >= self.max_offset :
            center = self.previous_center
        self.previous_center = center
        
        cv2.circle(self.warped, (center[0], center[1]), 5, self.contour_color, -1)
        if self.show_labels == True:
            cv2.putText(self.warped, "center({},{})".format(center[0], center[1]), (int(center[0] - 50), int(center[1] - 10) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.contour_color, 2) 
        if self.show_contour == True:
            cv2.drawContours(self.warped, [screenCnt], -1, self.contour_color, 4)
        '''
        return center, warped

    def ball_pose(self, xy_ball_pos):
        rectified_target_pos = cv2.perspectiveTransform( np.float32( [ ( (  xy_ball_pos ), ) ]) , self.rectify_matrix)
        #print("\n rectified_target_pos "+str(rectified_target_pos))