# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
from os import path

def print_videoprops(cap):
    print("Frame per second is " + str(cap.get(cv2.CAP_PROP_FPS)))
    print("Height of Frame is " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Width of Frame is " + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Codec is " + str(cap.get(cv2.CAP_PROP_FOURCC)))

def getVideoWriter(cap):
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(frame_width)
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #outputFile = "D:\\OpenCVCertification\\Set1_Week2_Code_Data\\data\\videos\\chaplin_out.avi"
    outputFile = "D:\\OpenCVCertification\\Set1_Week2_Code_Data\\data\\videos\\chaplin_out.mp4"
    print(outputFile)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out=cv2.VideoWriter(outputFile,fourcc,10,(frame_width,frame_height))
    return out

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    a=path.exists("D:\\OpenCVCertification\\Set1_Week2_Code_Data\\data\\videos\\chaplin.mp4")
    print(a)
    cap=cv2.VideoCapture("D:\\OpenCVCertification\\Set1_Week2_Code_Data\\data\\videos\\chaplin.mp4")
    if (cap.isOpened() == False):
        print("Could not open the file")
    else:
        print("Located the video file")
    print_videoprops(cap)
    out=getVideoWriter(cap)
    while (cap.isOpened()):
        # capture frame by frame
        ret,frame=cap.read()
        if (ret == True):
            out.write(frame)
            cv2.imshow("Charlie Chaplin",frame)
            #wait for 25 seconds
            cv2.waitKey(25)
        else:
            break
    out.release()
    cap.release()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
