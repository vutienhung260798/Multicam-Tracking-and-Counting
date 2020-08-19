from yolov5.detect import Detector
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
from yolov5.tracking import Sort
from contextlib import contextmanager
from agender.wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from mtcnn.mtcnn import MTCNN
import dlib
from process_cam import *
import threading

class Server(object):

    def __init__(self, mW = 2, mH = 2, detector = Detector(), detect_face = MTCNN()):
        self.mW = mW
        self.mH = mH
        self.detector = detector
        self.detect_face = detect_face
        self.imageHub = imagezmq.ImageHub()
        self.process_cams = {}
        self.frameDict = {}
    
    def __recv_frame(self):
        while True:
            #recive frame from client
            (rpiName, frame) = self.imageHub.recv_image()
            self.imageHub.send_reply(b'OK')
            #create tracker model for each rpiName
            if rpiName not in self.process_cams.keys():
                self.process_cams[rpiName] = ProcessCam(rpiName, self.detector)
            
            # threading.Thread(target= self.process_cams[rpiName].process, args=(frame, self.frameDict)).start()
            # multiprocessing.Process(target = self.process_cams[rpiName].run, args = (frame,self.detect_face, self.frameDict,))
            self.process_cams[rpiName].run(frame, self.detect_face, self.frameDict)
            
            montages = build_montages(self.frameDict.values(), (600, 400), (self.mW, self.mH))
            
            for (i, montage) in enumerate(montages):
                cv2.imshow('camera location({})'.format(i), montage)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
    def run(self):
        self.__recv_frame()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    Server().run()
        