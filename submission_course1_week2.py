#! /usr/bin/python

__author__ = "Srinivasan Subramaniam"

__email__ = "srinivasan.ibmbangalore@gmail.com"
__status__ = "Course-1 : Week 2 Assignment -1: 1-Sep 2020"

import cv2
import numpy as np

# Read the Source File
source = cv2.imread("smiling-man.jpeg", 1)
if ( source.any() == None):
    print("file not detected")
else:
    print("file detected")

cropping = False
cv2.namedWindow("Smiling Man")
x1, y1, x2, y2 = 0, 0, 0, 0

# Call Back function for trapping the mouse event
def mouse_capture(event, x, y, flags, userdata):
    try:
        global x1, y1, x2, y2, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = x, y, x, y
            cropping = True
            cv2.circle(source, (x1,y1), 1, (255, 0, 255), 2, cv2.LINE_AA)

        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping == True:
                x2,y2 = x,y
        elif event == cv2.EVENT_LBUTTONUP:
            x2,y2 = x,y
            cropping = False
            points = [(x1,y1),(x2,y2)]
            cv2.rectangle(source,points[0],points[1],(255,0,255),thickness=4,lineType=cv2.LINE_AA)
            if len(points)==2:
                print("cropping...")
                crop = source[points[0][1]:points[1][1],points[0][0]:points[1][0]]
                cv2.imwrite("smiling-man-face.jpeg",crop)
                cv2.imshow("Cropped", crop)

    except Exception as ex:
        print("Exception:",ex)


try:
    cv2.setMouseCallback("Smiling Man", mouse_capture)
    k = 0
    # loop until escape character is pressed
    while k!=27:

        dummy = source.copy()
        cv2.putText(source,'''Choose top left cornet and drag, 
                      Press ESC to exit and c to clear''' ,
              (10,20), cv2.FONT_HERSHEY_SIMPLEX,
              0.4,(10, 10, 10), 1 )


        if not cropping:
            cv2.imshow("Smiling Man", source)
        elif cropping:
            cv2.rectangle(dummy, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.imshow("Smiling Man", dummy)
        k = cv2.waitKey(20) & 0xFF

    cv2.destroyAllWindows()
except Exception as ex:
    print("Exception",ex)