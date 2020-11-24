import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import time

labelsPath='coco.names'  # List of Class Labels
weightsPath='yolov3.weights' # Weights
configPath='yolov3.cfg' # Configuration File
soccerVideoPath='soccer-ball.mp4'
soccerOutputVideoPath='outputsoccer-ball.mp4'

# Initialize the parameters

'''
The YOLOv3 algorithm generates bounding boxes as the predicted detection outputs. 
Every predicted box is associated with a confidence score. 
In the first stage, all the boxes below the confidence threshold parameter 
are ignored for further processing.
'''

objectnessThreshold = 0.5 # Objectness threshold
confThreshold = 0.5       # Confidence threshold
nmsThreshold = 0.4        # Non-maximum suppression threshold
inpWidth = 416            # Width of network's input image
inpHeight = 416           # Height of network's input image

classes = None # Initialize the Number of Classes in the Object Detector
'''
Yolo V3 Pre-trained Model is used for Object Detection
The model pre-trained on COCO dataset is leveraged.
Classes relevant here are 'sports ball'.
Bounding box is put around this object post detection and tracked.
'''
def loadYoloNetwork(labelsPath,weightsPath,configPath):

    with open(labelsPath, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net

'''
Get the names of the output layers.The most salient feature of v3 
is that it makes detections at three different scales.
YOLO v3 makes prediction at three scales, which are precisely given by 
downsampling the dimensions of the input image by 32, 16 and 8 respectively.
We need to get the names
of those neural network layers who have only one output node
'''
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect (img,net):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(getOutputsNames(net))
    (left,top,right,bottom) = postprocess(frame, layerOutputs)
    end=time.time()
    print("Time taken for frame is ",(end-start))
    return (left,top,right,bottom)

def drawPred(img,classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        #label = 'Detection %s:%s' % (classes[classId], label)
        label="dtcn:"+label
        #label = 'Detection :%s' % (label)

    # Display the label at the top of the bounding box
    label = "dtcn:" + label
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    print("label is ",label)

'''
Non-Maxima Suppression (NMS) However, no matter which HOG + Linear SVM method you choose, 
you will (with almost 100% certainty) detect multiple bounding boxes surrounding 
the object in the image.While each detection may in fact be valid, 
I certainty don’t want my classifier to report to back to me saying that 
it found six faces when there is clearly only one face. 
Like I said, this is common “problem” when utilizing object detection methods.
YOLO does not apply non-maxima suppression for us, so we need to explicitly apply it.

Applying non-maxima suppression suppresses significantly overlapping bounding boxes, 
keeping only the most confident ones.

NMS also ensures that we do not have any redundant or extraneous bounding boxes.
'''

def postprocess(frame, outs):

    left1=0
    top1=0
    right1=0
    bottom1=0

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:  # Re
        # member Yolo processes in 3 Scales. This is the for loop for the 3 scales
        for detection in out:# The object detected has 5 parameters.
            if detection[4] > objectnessThreshold :
                scores = detection[5:]
                classId = np.argmax(scores)
                print('ClassId=',classId)
                if ( classId != 32):
                    continue
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2) # Note in Yolo bounding box coordinates are at center
                    top = int(center_y - height / 2) # Note in Yolo bounding box coordinates are at center.
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    #print(classId)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if (classIds[i] == 32): # Draw the bounding box only for the ball.
            drawPred(frame,classIds[i], confidences[i], left, top, left + width, top + height)
            left1 = left
            top1=top
            right1=left + width
            bottom1=top + height
            print('Final Class Id=',classIds[i])
    return (left1,top1,right1,bottom1)

net = loadYoloNetwork(labelsPath,weightsPath,configPath)
# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(soccerVideoPath)
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    print("[INFO] Total Number of Frame is ", total)
    print("[INFO] FPS is ", fps)
# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
# loop over frames from the video file stream
frmCntr=0
while True:
    # read the next frame from the file
    frmCntr+=1
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
	    break

	# if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities

    detect(frame,net)



    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(soccerOutputVideoPath, fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
#release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
