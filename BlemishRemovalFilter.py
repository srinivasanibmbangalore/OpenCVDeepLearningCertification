#! /usr/bin/python

__author__ = "Srinivasan Subramaniam"

__email__ = "srinivasan.ibmbangalore@gmail.com"
__status__ = "Course-1 : Week 6 Project 2 : 12-Oct 2020"

import numpy as np
import cv2

def acneRemover(event,x,y,flags,userdata):
    global img,radius
    if event == cv2.EVENT_FLAG_LBUTTON:
        print(x,"  ",y," ",radius)
        acneLocation = (x,y)
        newX,newY = getBestPatch(x,y,radius)
        newP = img[newY: (newY+2*radius),newX:(newX+2*radius)]
        cv2.imwrite("npatch.png",newP)
        # Create a mask for the patch which is used as an input for seamless cloning
        mask= 255*np.ones(newP.shape,newP.dtype)
        img=cv2.seamlessClone(newP,img,mask,acneLocation,cv2.NORMAL_CLONE)
        cv2.imshow("Blemish Removal Filter",img)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.imshow("Blemish Removal Filter", img)


def getBestPatch(x,y,r):

    patches={}

    k1=getGradientForPatch(x+2*r,y)
    patches['k1']=(x+2*r,y,k1[0],k1[1])

    k2 = getGradientForPatch(x + 2 * r, y+r)
    patches['k2'] = (x + 2 * r, y+r, k2[0], k2[1])

    k3 = getGradientForPatch(x - 2 * r, y)
    patches['k3'] = (x - 2 * r, y , k3[0], k3[1])

    k4 = getGradientForPatch(x - 2 * r, y-r)
    patches['k4'] = (x - 2 * r, y-r, k4[0], k4[1])

    k5 = getGradientForPatch(x,y+2*r)
    patches['k5'] = (x,y+2*r, k5[0], k5[1])

    k6 = getGradientForPatch(x+r, y + 2 * r)
    patches['k6'] = (x+r, y + 2 * r, k6[0], k6[1])

    k7 = getGradientForPatch(x , y - 2 * r)
    patches['k7'] = (x , y - 2 * r, k7[0], k7[1])

    k8 = getGradientForPatch(x-r, y - 2 * r)
    patches['k8'] = (x-r, y - 2 * r, k8[0], k8[1])

    return findMinimum(patches)


def findMinimum(patches):

    lowX={}
    lowY={}
    ## Store the Sobel X and Sobel Y Gradient in a separate dictionary with the Same key as the input dictionary
    for key, (x,y,dx,dy) in patches.items():
        lowX[key]=dx
        lowY[key]=dy
    #Let's use the key parameter so that we can find the dictionary's key having the smallest value.
    dy_min_key= min(lowY.keys(),key=(lambda i: lowY[i] ))
    dx_min_key= min(lowX.keys(),key=(lambda i: lowX[i] ))

    if ( dy_min_key == dx_min_key ):
        return patches[dx_min_key][0],patches[dx_min_key][1]  ## Return the x and Y
    else:
        return patches[dx_min_key][0], patches[dx_min_key][1]  ## Return the x and Y. Other means




def getGradientForPatch(x,y):
    pImg=img[y:(y+2*radius),x:(x+2*radius)]
    return getGradient(pImg)

def getGradient(acne_img):
    sobelY=cv2.Sobel(acne_img,cv2.CV_64F,0,1,ksize=3)
    sobelX=cv2.Sobel(acne_img,cv2.CV_64F,1,0,ksize=3)

    # The sobelX and sobelY images are now of the floating
    # point data type -- we need to take care when converting
    # back to an 8-bit unsigned integer that we do not miss
    # any images due to clipping values outside the range
    # of [0, 255]. First, we take the absolute value of the
    # graident magnitude images, THEN we convert them back
    # to 8-bit unsigned integers
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    return np.mean(sobelX),np.mean(sobelY)


img = cv2.imread("blemish.png", 1)
radius=15 ##Patch of Radius 15 around the blemish that have to be removed

# Make a dummy image, will be useful to clear the drawing
cv2.namedWindow("Blemish Removal Filter")
# highgui function called when mouse events occur
cv2.setMouseCallback("Blemish Removal Filter", acneRemover, img)

k = 0
while k != 27:
    cv2.imshow("Blemish Removal Filter", img)
    k = cv2.waitKey(20)
cv2.destroyAllWindows()