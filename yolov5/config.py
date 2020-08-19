from easydict import EasyDict as edict
import os
import torch

def get_config():
    conf=edict()
    conf.name="coco"
    conf.cur_dir=os.path.dirname(os.path.realpath(__file__))
    # conf.cfg=os.path.join(conf.cur_dir,'cfg',conf.name+'.cfg')
    conf.names=os.path.join(conf.cur_dir,'data',conf.name+'.names')
    conf.weights=os.path.join(conf.cur_dir,'weights',"yolov5m.pt")
    conf.img_size=608
    conf.conf_thres=0.5 #object confidence threshold
    conf.iou_thres=0.5 #IOU threshold for NMS
    conf.half = False 
    conf.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.view_img=False
    conf.save_txt=False
    conf.classes=None
    conf.agnostic_nms=False
    conf.augment=False
    return conf
