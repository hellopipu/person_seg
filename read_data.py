##########################################
# @subject : Person segmentation         #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
from torch.utils.data import Dataset
from config import *
import xml.dom.minidom
from util import *
from pycocotools.coco import COCO
import os
import glob

conf = Config()


class MyData(Dataset):
    def __init__(self, base_dir=conf.PATH, istrain='train', size=640):
        super(MyData, self).__init__()
        self.base_dir = base_dir
        self.istrain = istrain
        self.img_dir_super = []
        self.img_dir_voc = []
        self.imgIds = []
        self.img_dir_voc = []
        self.img_dir_vip = []
        self.img_dir_chip = []
        self.img_dir_atr = []
        self.img_dir_lip = []
        self.imgIds_other = []
        self.img_dir_mhp = []
        self.img_trimodal = []
        self.img_dir_sit = []
        self.size = size
        # SET TO USE DATASET
        SUPERVISELY = 1
        PASCAL = 1
        COCO = 1
        VIP = 1
        COCO_NEG = 1
        TRIMODAL = 1
        ######################
        #####supervisely dataset########################
        if SUPERVISELY:
            print("load supervisely datasets......")
            if self.istrain == "train" or self.istrain == "val":
                for i in conf.split_train:
                    path = os.path.join(self.base_dir, i, 'img')
                    self.img_dir_super.extend(glob.glob(path + '/*'))
                self.img_dir_super = split_trainval(self.img_dir_super, self.istrain)

            elif self.istrain == "test":
                for i in conf.split_test:
                    path = os.path.join(self.base_dir, i, 'img')
                    self.img_dir_super.extend(glob.glob(path + '/*'))
        #####VOC PASCAL dataset#########################
        if PASCAL:
            print("load VOC PASCAL datasets......")
            self.img_dir_voc = []
            self.seg_dir_voc = []
            if self.istrain == "train" or self.istrain == "val":
                path_voc = conf.PATH_VOC + "/ImageSets/Segmentation/train.txt"
            elif self.istrain == "test":
                path_voc = conf.PATH_VOC + "/ImageSets/Segmentation/val.txt"
            path_xml = conf.PATH_VOC + "/Annotations/"
            path_seg = conf.PATH_VOC + "/SegmentationClass/"
            path_img = conf.PATH_VOC + "/JPEGImages/"
            with open(path_voc, 'r') as file_to_read:
                line = file_to_read.readline()
                while line:
                    file_xml = path_xml + line.strip("\n") + ".xml"
                    dom = xml.dom.minidom.parse(file_xml)
                    cc = dom.getElementsByTagName('name')
                    for i in range(len(cc)):
                        if cc[i].firstChild.data == 'person':
                            self.img_dir_voc.append(path_img + line.strip("\n") + '.jpg')
                            break
                    line = file_to_read.readline()
            self.img_dir_voc = split_trainval(self.img_dir_voc, self.istrain)
        ##### COCO dataset###############################
        if COCO:
            print("load COCO datasets......")
            if self.istrain == "train" or self.istrain == "val":
                self.path_coco = conf.PATH_COCO + "/train2017/"
                path_coco_ann = conf.PATH_COCO + "/annotations/instances_train2017.json"
            elif self.istrain == "test":
                self.path_coco = conf.PATH_COCO + "/val2017/"
                path_coco_ann = conf.PATH_COCO + "/annotations/instances_val2017.json"
            self.coco = COCO(path_coco_ann)
            self.catIds = self.coco.getCatIds(catNms=["person"])
            self.imgIds = self.coco.getImgIds(catIds=self.catIds)
            self.imgIds = split_trainval(self.imgIds, self.istrain)

        ##### VIP dataset###############################
        if VIP:
            print("load VIP datasets......")
            self.img_dir_vip = []
            if self.istrain == "train" or self.istrain == "val":
                for line in open(conf.PATH_VIP + "/lists/trainval_id.txt", 'r'):
                    line = line.strip('\n')  # .replace('/','\\')
                    path = conf.PATH_VIP + "/Images/" + line + '.jpg'
                    self.img_dir_vip.append(path)
            elif self.istrain == "test":
                for line in open(conf.PATH_VIP + "/lists/test_id.txt", 'r'):
                    line = line.strip('\n')  # .replace('/', '\\')
                    path = conf.PATH_VIP + "/Images/" + line + '.jpg'
                    self.img_dir_vip.append(path)
            self.img_dir_vip = split_trainval(self.img_dir_vip, self.istrain)

        ##### ATR dataset###############################
        #        print("load ATR datasets......")
        #        self.img_dir_atr=[]
        #        path_str_img = next(os.walk(conf.PATH_ATR + "/JPEGImages"))[2]
        #       # print('haha',conf.PATH_ATR + "/JPEGImages")
        #        img_dir_atr_all = [conf.PATH_ATR + "/JPEGImages/" + p for p in path_str_img]
        #        trainval=split_trainval(img_dir_atr_all,'train')
        #        if self.istrain == "test":
        #            self.img_dir_atr=split_trainval(img_dir_atr_all,'test')
        #        else:
        #            self.img_dir_atr = split_trainval(trainval, self.istrain)
        ##### CHIP dataset###############################
        ##        print("load CHIP datasets......")
        ##        self.img_dir_chip=[]
        ##        if self.istrain == "train" or self.istrain == "val":
        ##            for line in  open(conf.PATH_CHIP+"/Training/train_id.txt", 'r'):
        ##                line=line.strip('\n')#.replace('/','\\')
        ##                path=conf.PATH_CHIP+"/Training/Images/"+line+'.jpg'
        ##                self.img_dir_chip.append(path)
        ##        elif self.istrain == "test":
        ##            for line in open(conf.PATH_CHIP+"/Validation/val_id.txt", 'r'):
        ##                line = line.strip('\n')#.replace('/', '\\')
        ##                path=conf.PATH_CHIP+"/Validation/Images/"+line+'.jpg'
        ##                self.img_dir_chip.append(path)
        ##        self.img_dir_chip = split_trainval(self.img_dir_chip, self.istrain)
        #        ##### LIP dataset###############################
        #        print("load LIP datasets......")
        #        self.img_dir_lip=[]
        #        self.label_dir_lip=[]
        #        if self.istrain == "train" :
        #            for line in  open(conf.PATH_LIP+"/train_id.txt", 'r'):
        #                line=line.strip('\n')#.replace('/','\\')
        #                path=conf.PATH_LIP+"/train_images/"+line+'.jpg'
        #                self.img_dir_lip.append(path)
        #                path_label=conf.PATH_LIP+"/TrainVal_parsing_annotations/train_segmentations/"+line+'.png'
        #                self.label_dir_lip.append(path_label)
        #            self.img_dir_lip = split_trainval(self.img_dir_lip, 'test')  #split val from test
        #            self.label_dir_lip = split_trainval(self.label_dir_lip, 'test')
        #        elif self.istrain == "test" or self.istrain == "val":
        #            for line in open(conf.PATH_LIP+"/val_id.txt", 'r'):
        #                line = line.strip('\n')#.replace('/', '\\')
        #                path=conf.PATH_LIP+"/val_images/"+line+'.jpg'
        #                self.img_dir_lip.append(path)
        #                path_label=conf.PATH_LIP+"/TrainVal_parsing_annotations/val_segmentations/"+line+'.png'
        #                self.label_dir_lip.append(path_label)
        #            if self.istrain=="val":
        #                self.img_dir_lip = split_trainval(self.img_dir_lip, "val")  #split val from test
        #                self.label_dir_lip = split_trainval(self.label_dir_lip, 'val')
        #            else:
        #                self.img_dir_lip = split_trainval(self.img_dir_lip, "train") #split val from test
        #                self.label_dir_lip = split_trainval(self.label_dir_lip, 'train')
        # ##### COCO negtive dataset###############################
        if COCO_NEG:
            print("load COCO negative datasets......")
            if self.istrain == "train" or self.istrain == "val":
                self.path_coco = conf.PATH_COCO + "/train2017/"
                path_coco_ann = conf.PATH_COCO + "/annotations/instances_train2017.json"
            elif self.istrain == "test":
                self.path_coco = conf.PATH_COCO + "/val2017/"
                path_coco_ann = conf.PATH_COCO + "/annotations/instances_val2017.json"
            self.coco = COCO(path_coco_ann)
            self.catIds = self.coco.getCatIds(catNms=["person"])
            self.imgIds_person = self.coco.getImgIds(catIds=self.catIds)
            self.imgIds_all = self.coco.getImgIds()
            self.imgIds_other = list(set(self.imgIds_all) - set(self.imgIds_person))
            if self.istrain == 'train':
                pass
            else:
                self.imgIds_other = []

        ##MHP V2 dataset
        #        print("load MHP datasets......")
        #        self.img_dir_mhp=[]
        #        self.line_mhp=[]
        #        if self.istrain == "train" or self.istrain == "val":
        #            self.anno_path_mhp = next(os.walk(conf.PATH_MHP + "/train/parsing_annos"))
        #            for line in  open(conf.PATH_MHP+"/list/train.txt", 'r'):
        #                line=line.strip('\n')#.replace('/','\\')
        #                self.line_mhp.append(line)
        #                path=conf.PATH_MHP+"/train/images/"+line+'.jpg'
        #                self.img_dir_mhp.append(path)
        #        elif self.istrain == "test":
        #            self.anno_path_mhp = next(os.walk(conf.PATH_MHP + "/val/parsing_annos"))
        #            for line in open(conf.PATH_MHP+"/list/val.txt", 'r'):
        #                line = line.strip('\n')#.replace('/', '\\')
        #                self.line_mhp.append(line)
        #                path=conf.PATH_MHP+"/val/images/"+line+'.jpg'
        #                self.img_dir_mhp.append(path)
        #        self.img_dir_mhp = split_trainval(self.img_dir_mhp, self.istrain)
        #        self.line_mhp=split_trainval(self.line_mhp,self.istrain)
        #        ##Trimodal dataset
        if TRIMODAL:
            print("load Trimodal dataset...")
            self.img_trimodal = []
            path1 = conf.PATH_TRIMODAL + "/Scene 1/rgbMasks"
            path2 = conf.PATH_TRIMODAL + "/Scene 2/rgbMasks"
            path3 = conf.PATH_TRIMODAL + "/Scene 3/rgbMasks"
            p1 = next(os.walk(path1))[2]
            p2 = next(os.walk(path2))[2]
            p3 = next(os.walk(path3))[2]
            for i in p1:
                self.img_trimodal.append(path1 + "/" + i)
            for i in p2:
                self.img_trimodal.append(path2 + "/" + i)
            for i in p3:
                self.img_trimodal.append(path3 + "/" + i)
            if self.istrain == 'train':
                pass
            else:
                self.img_trimodal = []

        #        ##SIT dataset
        #        print("load SIT dataset...")
        #        self.img_dir_sit=[]
        #        path_dis=conf.PATH_SIT+"/disaster/img"
        #        path_ran = conf.PATH_SIT + "/RANGE"
        #        path_sit= conf.PATH_SIT + "/Sitting/img"
        #        p1=next(os.walk(path_dis))[2]
        #        p2 = next(os.walk(path_ran))[2]
        #        p3 = next(os.walk(path_sit))[2]
        #        for i in p1:
        #            self.img_dir_sit.append(path_dis+"/"+i)
        #        for i in p2:
        #            pp=path_ran+"/"+i+"/d_images"
        #            ppp=next(os.walk(pp))[2]
        #            for j in ppp:
        #                self.img_dir_sit.append(pp+"/"+j)
        #        for i in p3:
        #            self.img_dir_sit.append(path_sit+"/"+i)
        #        if self.istrain=='train' :
        #            pass
        #        else:
        #            self.img_dir_sit=[]

        print("mode", self.istrain)
        self.num_super = len(self.img_dir_super)
        self.num_voc = len(self.img_dir_voc)
        self.num_coco = len(self.imgIds)
        self.num_vip = len(self.img_dir_vip)
        self.num_atr = len(self.img_dir_atr)
        self.num_chip = len(self.img_dir_chip)
        self.num_lip = len(self.img_dir_lip)
        self.num_coco_negtive = len(self.imgIds_other)
        self.num_mhp = len(self.img_dir_mhp)
        self.num_trimodal = len(self.img_trimodal)
        self.num_sit = len(self.img_dir_sit)
        print('SUPER', self.num_super)
        print('VOC  ', self.num_voc)
        print('COCO ', self.num_coco)
        print('VIP  ', self.num_vip)
        print("ATR  ", self.num_atr)
        print("CHIP ", self.num_chip)
        #        print("LIP  ", self.num_lip)
        print('COCO_negtive', len(self.imgIds_other))
        print("MHP", len(self.img_dir_mhp))
        print("TRIMODAL", len(self.img_trimodal))
        print("SIT", len(self.img_dir_sit))

    def __getitem__(self, index):
        # READ DIFFERENT DATASETS ###############################################################################################################
        # read supervisely
        if index < self.num_super:  ####
            img = cv2.imread(self.img_dir_super[index])
            r, c, _ = img.shape
            label = read_json(self.img_dir_super[index], r, c)
        # read voc
        elif index >= self.num_super and index < self.num_super + self.num_voc:  ####
            index = index - self.num_super
            img_path = self.img_dir_voc[index]
            seg_path = seg_path = img_path.replace('JPEGImages', 'SegmentationClass').replace('jpg',
                                                                                              'png')  # self.seg_dir_voc[index]
            img = cv2.imread(img_path)
            seg = cv2.imread(seg_path)
            label = np.zeros((seg.shape[0], seg.shape[1]))
            label[seg.sum(2) == 128 + 128 + 192] = 1
        # read COCO
        elif index >= self.num_super + self.num_voc and index < self.num_super + self.num_voc + self.num_coco:  ####
            index = index - self.num_super - self.num_voc
            img_file = self.coco.loadImgs(self.imgIds[index])[0]
            img = cv2.imread(self.path_coco + img_file['file_name'])
            r, c, _ = img.shape
            label = np.zeros((r, c))
            annIds = self.coco.getAnnIds(imgIds=img_file['id'], catIds=self.catIds, iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            #            ####################read coco mask ###########################
            for ann in anns:
                m = np.array(self.coco.annToMask(ann))
                label = label + m
            label[label != 0] = 1
        # read VIP
        elif index >= self.num_super + self.num_voc + self.num_coco and index < self.num_super + self.num_voc + self.num_coco + self.num_vip:
            index = index - (self.num_super + self.num_voc + self.num_coco)
            path_vip_img = self.img_dir_vip[index]
            path_vip_label = self.img_dir_vip[index].replace('Images', 'Categorys').replace('jpg', 'png')
            img = cv2.imread(path_vip_img)
            r, c, _ = img.shape
            label = np.zeros((r, c))
            seg = cv2.imread(path_vip_label)
            label[seg.sum(2) != 0] = 1
        # read ATR
        elif index >= self.num_super + self.num_voc + self.num_coco + self.num_vip and index < self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr:
            index = index - (self.num_super + self.num_voc + self.num_coco + self.num_vip)
            img = cv2.imread(self.img_dir_atr[index])
            seg = cv2.imread(
                self.img_dir_atr[index].replace('JPEGImages', 'SegmentationClassAug').replace('jpg', 'png'))
            r, c, _ = img.shape
            label = np.zeros((r, c))
            label[seg.sum(2) != 0] = 1
        # read CHIP
        elif index >= self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr and index < self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip:
            index = index - (self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr)
            path_chip_img = self.img_dir_chip[index]
            path_chip_label = self.img_dir_chip[index].replace('Images', 'Categories').replace('jpg', 'png')
            img = cv2.imread(path_chip_img)
            r, c, _ = img.shape
            label = np.zeros((r, c))
            seg = cv2.imread(path_chip_label)
            label[seg.sum(2) != 0] = 1
        # read LIP
        #        else:
        #            index=index-(self.num_super+self.num_voc+self.num_coco+self.num_vip+self.num_atr+self.num_chip)
        #            path_lip_img=self.img_dir_lip[index]
        #            path_lip_label=self.label_dir_lip[index]
        #            img=cv2.imread(path_lip_img)
        #            r, c, _ = img.shape
        #            label = np.zeros((r, c))
        #            seg=cv2.imread(path_lip_label)
        #            label[seg.sum(2) != 0] = 0.9
        #            label[label!=0.9]=0.1
        # read COCO_NEGTIVE
        elif index >= self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip and index < self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive:  ##### read COCO negtive
            index = index - self.num_super - self.num_voc - self.num_coco - self.num_vip - self.num_atr - self.num_chip
            img_file = self.coco.loadImgs(self.imgIds_other[index])[0]
            img_ = cv2.imread(self.path_coco + img_file['file_name'])
            img = img_.copy()
            r, c, _ = img.shape
            label_ = np.zeros((r, c))
            ####################read coco mask ###########################
            label = label_.copy()
        # read MHP
        elif index >= self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive and index < self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive + self.num_mhp:  # read  mhp
            index = index - (
                        self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive)
            path_mhp_img = self.img_dir_mhp[index]
            ppp = path_mhp_img[path_mhp_img.find('images') + 7:]
            img_index = ppp[:ppp.find('.jpg')]  # self.line_mhp[index]
            img = cv2.imread(path_mhp_img)
            r, c, _ = img.shape
            label = np.zeros((r, c))
            for i in self.anno_path_mhp[2]:
                if i.find(img_index + '_') == 0 and '_01.png' in i:
                    ii = i[0:i.find('_01.png')]
                    num_str = ii[ii.find('_') + 1:]
                    num = int(num_str)
                    label_ = cv2.imread(self.anno_path_mhp[0] + "/" + i)
                    label[label_.sum(2) != 0] = 1
                    for c in range(2, num + 1):
                        if c < 10:
                            cc = '0' + str(c)
                        else:
                            cc = str(c)
                        label_ = cv2.imread(self.anno_path_mhp[0] + "/" + ii + '_' + cc + '.png')
                        label[label_.sum(2) != 0] = 1
                    break
        # read TRIMODAL
        elif index >= self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive + self.num_mhp \
                and index < self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive + self.num_mhp + self.num_trimodal:
            index = index - (
                        self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive + self.num_mhp)
            path_trimodal = self.img_trimodal[index]
            label_ = cv2.imread(path_trimodal)
            r, c, _ = label_.shape
            label = np.zeros((r, c))
            img = cv2.imread(path_trimodal.replace('rgbMasks', 'SyncRGB', ).replace('.png', '.jpg'))
            label[label_.sum(2) != 0] = 1
        # read SIT
        elif index >= self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive + self.num_mhp + self.num_trimodal \
                and index < self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive + self.num_mhp + self.num_trimodal + self.num_sit:
            index = index - (
                        self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + self.num_coco_negtive + self.num_mhp + self.num_trimodal)
            path = self.img_dir_sit[index]
            img = cv2.imread(path)
            if 'RANGE' in path:
                label = scipy.io.loadmat(path.replace("d_images", "d_masks").replace('jpg', 'mat'))['M']

            else:
                label = scipy.io.loadmat(path.replace("img", "masks").replace('jpg', 'mat'))['M']
        else:  # MHP
            pass

        # augmentation ###############################################################################################################
        if self.istrain == "train":
            img, label = random_horizontal_flip_transform2(img, label)
            img, label = random_crop(img, label)
            img, label = randomRotate(img, label)
            img, label = random_rotate90_transform2(img, label, 0.1)
            img = randomHueSaturationValue(img, 0.3)
            img = random_clache(img, 0.3)
            img = random_gaussianblur(img, 0.3)
            img = random_gaussian_noise(img, 0.3)
            img = random_motionblur(img, 0.3)
        #        # resize to 256
        img, label = random_resize(img, label, self.size, 2)

        # show
        # import matplotlib.pyplot as plt
        # plt.imshow(BGR2RGB(img))
        # plt.show()

        # normalization ###############################################################################################################
        img = img / 255.
        img = (img - conf.mean) / conf.std
        img = img.transpose((2, 0, 1))
        label = label[np.newaxis, :].astype(np.float32)

        return img, label

    def __len__(self):
        return self.num_super + self.num_voc + self.num_coco + self.num_vip + self.num_atr + self.num_chip + \
               self.num_mhp + self.num_trimodal + self.num_sit + self.num_coco_negtive
