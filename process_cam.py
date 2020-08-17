from yolov5.tracking import Sort
import cv2
from yolov5.detect import Detector
import numpy as np
from agender.wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from contextlib import contextmanager

class ProcessCam:

    def __init__(self, rpiName, detector = Detector(), path_model = './agender/pretrained_models/weights.29-3.76_utk.hdf5'):
        self.rpiName = rpiName
        self.detector = detector
        self.tracker = Sort(15, 3)
        self.model = WideResNet(64, depth=16, k=8)()
        self.model.load_weights(path_model)
        self.num_person = 0

    def __agender(self, frame, box, detect_face, w, h, margin = 0.3):
        detected = detect_face.detect_faces(frame[box[1]:box[3], box[0]:box[2], :])
        # (w, h) = frame.shape[:2]
        if len(detected) != 0:
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
            results = self.model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            agender = "{}, {}".format(int(predicted_ages[0]), "M" if predicted_genders[0][0] < 0.5 else "F")
            cv2.putText(frame, agender, (box[0],box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    def __process(self, frame, detect_face):
        (h, w) = frame.shape[:2]
        boxes, imgs = self.detector.detect(frame)
        boxes = np.array(boxes)
        if len(boxes) != 0:
            boxes = boxes.astype(int)
            boxes = boxes + [-1, -1, 1, 1]
            trackers = self.tracker.update(boxes)
            for idx, box in enumerate(trackers):
                id =int(box[4])
                box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]]
                frame=cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(100,255,100),2)
                frame = cv2.putText(frame,str(id),(box[0],box[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),2)
                if id > self.num_person:
                    self.num_person = id
                self.__agender(frame, box, detect_face, w, h)

        label = 'num_person: ' + str(self.num_person)
        cv2.putText(frame, label, (10, h-10) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, str(self.rpiName), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frame = cv2.resize(frame, (600, 400))
        # frameDict[self.rpiName] = frame
    
    def run(self, frame, detect_face, frameDict):
        self.__process(frame, detect_face)
        frameDict[self.rpiName] = frame
