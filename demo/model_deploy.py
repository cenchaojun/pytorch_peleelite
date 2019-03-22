from __future__ import print_function

import numpy as np
import sys
import os
import torch
from load_param import load_param

#begin
dataset = 'VOC'
img_dim = 320
num_classes = (21, 81)[dataset == 'COCO']

sys.path.append('../')
from models import ATiny_pelee
net = ATiny_pelee.build_net(img_dim, num_classes)

#load parameter, but needn't do it in fact.
resume_net_path = os.path.join(
    '../weights','Atinypelee','SSD_Atinypelee_VOC_320','20190114','SSD_Atinypelee_VOC_epoches_680.pth')
if not os.path.exists(resume_net_path):
    print('pth file not exit!')
    exit()
load_param(net,resume_net_path)


