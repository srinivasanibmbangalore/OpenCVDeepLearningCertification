import argparse
import os
import shutil
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

positions=[]
positions2=[]
count=0

# This is a callback to get the position of the image in the building
# on which image will be overlaid
def draw_circle(event,x,y,flags,param):
    global positions,count,bldg
    # If event is Left Button Click then store the coordinate in the lists
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(bldg,(x,y),2,(255,0,0),-1)
        positions.append([x,y])
        if(count!=3):
            positions2.append([x,y])
        elif(count==3):
            positions2.insert(2,[x,y])
        count+=1
    #print(positions2)


bldg = cv2.imread('building3.jpeg', 1)
family=cv2.imread('sanjana varun image.jpg',1)
#family = cv2.resize(family, (150,150), interpolation = cv2.INTER_AREA)
cv2.namedWindow("Bill Board")
# highgui function called when mouse events occur
#cv2.setMouseCallback("Blemish Removal Filter", acneRemover, img)
cv2.setMouseCallback('Bill Board',draw_circle)
k = 0
while k != 27:
    cv2.imshow("Bill Board", bldg)
    k = cv2.waitKey(20)
cv2.destroyAllWindows()

height, width = bldg.shape[:2]
h1,w1 = family.shape[:2]
print("Building Height Width",height,width)
print("Family Height Width",h1,w1)

pts1=np.float32([[0,0],[w1,0],[0,h1],[w1,h1]])
pts2=np.float32(positions2)
print(pts2)
print(pts1)
'''
Step-2: Now that we have the coordinates of the pictures.
We will calculate the homography matrix using the cv2.findHomography() function.
'''
h, mask = cv2.findHomography(pts1, pts2,cv2.RANSAC,7.0)
#h, mask = cv2.findHomography(pts1, pts2)

'''
Step-3: Now we will use cv2.warpPerspective() function which 
will take the inputs image1(family), 
homography matrix(h) and width and height of the building image, and 
will give the following output.
'''

im_out = cv2.warpPerspective(family, h, (bldg.shape[1],bldg.shape[0]))
bldgClone = bldg.copy()

#Black out the polygonal banner area in the virtual billboard image
cv2.fillConvexPoly(bldgClone, np.int32(positions2), 0, 16)
result2 = bldgClone + im_out
cv2.imwrite('final13.jpg',result2)
'''
mask2 = np.zeros(bldg.shape, dtype=np.uint8)

roi_corners2 = np.int32(positions2)

channel_count2 = bldg.shape[2]
ignore_mask_color2 = (255,)*channel_count2

cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)

mask2 = cv2.bitwise_not(mask2)
masked_image2 = cv2.bitwise_and(bldg, mask2)

#Using Bitwise or to merge the two images
final = cv2.bitwise_or(im_out, masked_image2)
cv2.imwrite('final21.jpg',final)
'''