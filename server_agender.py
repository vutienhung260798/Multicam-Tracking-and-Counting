from yolov5.detect import Detector
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
from tracking import Sort
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from mtcnn.mtcnn import MTCNN
import dlib

ap = argparse.ArgumentParser()
ap.add_argument("-mW", "--montageW", default = 2, type=int,
	help="montage frame width")
ap.add_argument("-mH", "--montageH", default= 2, type=int,
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
num_person = 0

margin = 0.3
img_size = 64
detector1 = MTCNN()
model = WideResNet(64, depth=16, k=8)()
model.load_weights('./pretrained_models/weights.29-3.76_utk.hdf5')
# detected = detector.detect_faces(input_img)

while True:
    # recive frame from client
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')
    
    if rpiName not in lastActive.keys():
        print("[INFO] receiving data from {}...".format(rpiName))
    
    lastActive[rpiName] = datetime.now()
    print(lastActive[rpiName])

    frame = imutils.resize(frame, width = 600)
    (h, w) = frame.shape[:2]

    #tracking person 
    boxes,imgs = detector.detect(frame) # boxes : xmin,ymin,xmax,ymax
    boxes=np.array(boxes)
    if (len(boxes) != 0):
        boxes = boxes.astype(int)
        boxes = boxes + [-1,-1,1,1] 
        trackers = tracker.update(boxes)
        for idx, box in enumerate(trackers):
            id = int(box[4])
            box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]]
            frame=cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(100,255,100),2)
            frame = cv2.putText(frame,'id: '+str(id),(box[0],box[1] - 5),font,0.8,(255, 0, 0),2)
            if id > num_person:
                num_person = id
            
            detected =detector1.detect_faces(frame[box[1]:box[3], box[0]:box[2], :])
            if len(detected) == 0:
                continue
            else:
                faces = np.empty((len(detected), 64, 64, 3))
                input_frame = frame[box[1]:box[3], box[0]:box[2], :]
                x1, y1, w1, h1 = detected[0]['box']
                x2, y2 = x1 + w1, y1 + h1
                xw1 = max(int(x1 - margin * w1), 0)
                yw1 = max(int(y1 - margin * h1), 0)
                xw2 = min(int(x2 + margin * w1), w - 1)
                yw2 = min(int(y2 + margin * h1), h - 1)
                # cv2.rectangle(frame, (box[0]+x1, box[1]+y1), (box[0]+x1+w1, box[1]+y1+h1), (255, 0, 0), 2)
                faces[0, :, :, :] = cv2.resize(input_frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (64, 64))
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
                agender = "{}, {}".format(int(predicted_ages[0]), "M" if predicted_genders[0][0] < 0.5 else "F")
                cv2.putText(frame, agender, (box[0],box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    label = 'num_person: ' + str(num_person)
    cv2.putText(frame, label, (10, h - 10) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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

