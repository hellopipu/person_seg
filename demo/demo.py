##########################################
# @subject : Person segmentation         #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
from read_data import *
from config import *
from models.linknet import *
from models.unet_plusplus import *

###############################################################################################################################
# You can choose the model here :
#                   MODEL                       INPUT_SIZE                 SPEED                 VAL iou on Supervisely
# 'unet'    :   unet++ model ,                   256 X 256                 24fps (CPU)                 0.56
# 'linknet' :   linknet_resnet18 model           640 X 640                100fps (P40)                 0.89
###############################################################################################################################
model_name = 'linknet'  # 'unet++'

if model_name == 'unet++':
    net = Unet_2D(3, 1, 'test')
    checkpoint = torch.load("../models/weights/unet++_11.pth", map_location='cpu')
    size = 256
else:  # 'linknet'
    net = LinkNet()
    checkpoint = torch.load("../models/weights/link_640_57.pth", map_location='cpu')
    size = 640
net.load_state_dict(checkpoint['net'])

# net.cuda()
print("start testing...")

# val###################
net.eval()

img = cv2.imread('pexels-photo-868704.png')
r, c, _ = img.shape
label = read_json('pexels-photo-868704.json', r, c)
img, label = random_resize(BGR2RGB(img), label, size, 2)
img = img / 255.
img = (img - conf.mean) / conf.std
img = img.transpose((2, 0, 1))
img = torch.from_numpy(img[np.newaxis, :].astype(np.float32))
label = torch.from_numpy(label[np.newaxis, np.newaxis, :].astype(np.float32))
print(img.shape)
print(label.shape)

output = net(img.float())  # .cuda()
output[output >= 0.5] = 1.
output[output != 1] = 0.
iou_ = iou(output.cpu(), label)

#############show the seg result ######################
if 1:
    import matplotlib.pyplot as plt

    plt.subplot(221)
    output = output[0, 0].detach().numpy()  # .detach().cpu()
    label = label[0, 0].numpy()
    img = (img[0].permute(1, 2, 0).detach().numpy() * conf.std + conf.mean)  # .detach().cpu()
    img = img * 255
    plt.imshow(np.uint8(img))
    plt.title("origin")
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(222)
    plt.imshow(label)
    plt.title("label")
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(223)
    mm = apply_mask(img, output, color=random_colors(1)[0])
    plt.imshow(np.uint8(mm))
    plt.title("origin+seg")
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(224)
    plt.imshow(output)
    plt.title("seg")

    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    # plt.savefig("./results/imgs/img_%d.png"%i)
    plt.show()

print(" test_iou : %.4f" % (iou_))
