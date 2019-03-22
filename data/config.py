# config.py

# gets home dir cross platform
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# note: if you used our download scripts, this should be right
VOCroot = '/media/sdb/qyydata/data/VOCdevkit'  # path to VOCdevkit root dir
COCOroot = '/media/sdb/zldata/MS-COCO2014'


# Tinypelee CONFIGS
VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 2],

    'min_dim': 300,

    'steps': [8, 16, 30, 60, 100, 150],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

VOC_320 = {

    'feature_maps': [40, 20, 10, 5, 3, 2],

    'min_dim': 320,

    'steps': [8, 16, 32, 64, 107, 160],

    'min_sizes': [22.4, 48, 105.6, 163.2, 220.8, 278.4],

    'max_sizes': [48, 105.6, 163.2, 220.8, 278.4, 336],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

VOC_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 160],

    'min_sizes': [35.84, 76.8, 169.0, 261.1, 353.3, 445.4],

    'max_sizes': [76.8, 169.0, 261.1, 353.3, 445.4, 537.6],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 2],

    'min_dim': 300,

    'steps': [8, 16, 30, 60, 100, 150],

    'min_sizes': [21, 45, 99, 153, 207, 261],

    'max_sizes': [45, 99, 153, 207, 261, 315],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_320 = {

    'feature_maps': [40, 20, 10, 5, 3, 2],

    'min_dim': 320,

    'steps': [8, 16, 32, 64, 107, 160],

    'min_sizes': [22.4, 48, 105.6, 163.2, 220.8, 278.4],

    'max_sizes': [48, 105.6, 163.2, 220.8, 278.4, 336],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 160],

    'min_sizes': [35.84, 76.8, 169.0, 261.1, 353.3, 445.4],

    'max_sizes': [76.8, 169.0, 261.1, 353.3, 445.4, 537.6],

    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}
'''
# RFB_vgg ,RFB_E_vgg, RFB_mobile, SSD_vgg, SSD_peleenet
VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 107, 150],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

VOC_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],

    'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],

    'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [21, 45, 99, 153, 207, 261],

    'max_sizes': [45, 99, 153, 207, 261, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

COCO_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],

    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}
'''
COCO_mobile_300 = {
    'feature_maps': [19, 10, 5, 3, 2, 1],

    'min_dim': 300,

    'steps': [16, 32, 64, 100, 150, 300],

    'min_sizes': [45, 90, 135, 180, 225, 270],

    'max_sizes': [90, 135, 180, 225, 270, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

VOC_pelee_320 = {
    'feature_maps': [20, 20, 10, 5, 3, 1],

    'min_dim': 320,

    'steps': [16, 16, 32, 64, 107, 320],

    'min_sizes': [32, 64, 118.4, 172.8, 227.2, 281.6],

    'max_sizes': [64, 118.4, 172.8, 227.2, 281.6, 336],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

