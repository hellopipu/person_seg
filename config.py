##########################################
# @subject : Person segmentation         #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
import numpy as np


class Config():
    def __init__(self):
        self.PATH = "/data1/datasets/supervisely/"
        self.PATH_VOC = "/data1/datasets/VOCdevkit/VOC2012"
        self.PATH_COCO = "/data1/datasets/COCO2017"
        self.PATH_VIP = "/data1/datasets/VIP_Fine"
        self.PATH_ATR = "/data1/datasets/LIP/ATR/humanparsing"
        self.PATH_CHIP = "/data1/datasets/LIP/CIHP/instance-level_human_parsing"
        self.PATH_LIP = "/data1/datasets/LIP/LIP/trainval"
        self.PATH_MHP = "/data1/datasets/LV-MHP-v2/LV-MHP-v2"
        self.PATH_TRIMODAL = "/data1/datasets/trimodal"
        self.PATH_SIT = "/data1/datasets//SIT"
        self.split_train = ["seg__ds1", "seg__ds2", "seg__ds3", "seg__ds4", "seg__ds5", "seg__ds6", "seg__ds7",
                            "seg__ds8"]
        self.split_test = ["seg__ds9", "seg__ds10", "seg__ds11", "seg__ds12", "seg__ds13"]
        self.mean = np.array([[[0.485]], [[0.456]], [[0.406]]]).transpose(1, 2, 0)
        self.std = np.array([[[0.229]], [[0.224]], [[0.225]]]).transpose(1, 2, 0)
        self.BATCH_SIZE_TRAIN = 16  # 4
        self.BATCH_SIZE_VAL = 1
        self.BATCH_SIZE_TEST = 1
        self.EPOCH = 300
        self.WEIGHT_DECAY = 10 ** -7
        self.LR = 10 ** -3  # 0.001
