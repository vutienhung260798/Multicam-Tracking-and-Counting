import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.datasets import *
from utils.utils import *
from config import get_config
class Detector(object):
    def __init__(self):
        opt = get_config()
        self.img_size =opt.img_size
        weights= opt.weights
        self.device = opt.device
        self.model=torch.load(weights)['model'].float()
        self.model.to(self.device).eval()
        self.conf_thres=opt.conf_thres
        self.iou_thres=opt.iou_thres


    def detect(self,im0s):
        img = letterbox(im0s, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img,np.float32)  # uint8 to fp16/fp32
        img /= 255.0
        t = time.time()
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,fast=True)
            
        box_detects=[]
        ims=[]
        for i, det in enumerate(pred):  # detections per image
                im0 =im0s
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *x, conf, cls in det:
                     if(int(cls)==0): # person
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        ims.append(im0[c1[1]:c2[1],c1[0]:c2[0]])
                        top=c1[1]
                        left=c1[0]
                        right=c2[0]
                        bottom=c2[1]
                        box_detects.append(np.array([left,top, right,bottom]))
        return box_detects,ims


if __name__ == '__main__':

    detector=Detector()
    for path in glob.glob("*.jpg"):
        print(path)
        img=cv2.imread(path)
        boxes,ims=detector.detect(img)
        print(len(boxes))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for box,im in zip(boxes,ims):
            img =cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1,1)
            cv2.putText(img,"lpn",(box[0],box[1]),font,0.5,(255,0,0),1)
        cv2.imshow("image",img)
        cv2.waitKey(0)
