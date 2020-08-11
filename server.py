from yolov5.detect import Detector
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
from tracking import Sort

ap = argparse.ArgumentParser()
ap.add_argument("-mW", "--montageW", required=True, type=int,
	help="montage frame width")
ap.add_argument("-mH", "--montageH", required=True, type=int,
	help="montage frame height")
args = vars(ap.parse_args())

# window size
mW = args["montageW"]
mH = args["montageH"]

ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

detector = Detector() # select yolov5
tracker = Sort(15, 3)
font= cv2.FONT_HERSHEY_SIMPLEX
imageHub = imagezmq.ImageHub()
frameDict = {}
lastActive = {}
lastActiveCheck = datetime.now()

while True:
    # recive frame from client
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')
    
    if rpiName not in lastActive.keys():
        print("[INFO] receiving data from {}...".format(rpiName))
    
    lastActive[rpiName] = datetime.now()
    print(lastActive[rpiName])

    frame = imutils.resize(frame, width = 400)
    (h, w) = frame.shape[:2]

    tracking person 
    boxes,imgs = detector.detect(frame) # boxes : xmin,ymin,xmax,ymax
    boxes=np.array(boxes)
    if (len(boxes) != 0):
        boxes = boxes.astype(int)
        boxes = boxes + [-1,-1,1,1] 
        trackers = tracker.update(boxes)
        for idx, box in enumerate(trackers):
            id = str(box[4])
            box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]]
            frame=cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(100,255,100),1)
            frame = cv2.putText(frame,str(id),(box[0],box[1]),font,2,(0,200,100),2)

    #sending device name on the frame        
    cv2.putText(frame, rpiName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #update frame in frame dictionary
    frameDict[rpiName] = frame

    #build montages using image in the frame dictionary
    montages = build_montages(frameDict.values(), (w, h), (mW, mH))
    for (i, montage) in enumerate(montages):
        cv2.imshow('camera location({})'.format(i), montage)

    key = cv2.waitKey(1) & 0xFF

    if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
        for (rpiName, ts) in list(lastActive.items()):
            if (datetime.now() -ts).seconds > ACTIVE_CHECK_SECONDS:
                print("[INFO] lost connection to {}".format(rpiName))
                lastActive.pop(rpiName)
                frameDict.pop(rpiName)
    lastActiveCheck = datetime.now()

    if key == ord("q"):
        break

cv2.destroyAllWindows()

