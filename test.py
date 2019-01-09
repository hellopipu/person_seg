##########################################
# @subject : Person segmentation         #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
import torch.utils.data as Data
from read_data import *
from config import *
from models.unet_plusplus import *
from models.linknet import *

model_name = 'linknet'  # 'unet++'
if model_name == 'unet++':
    net = Unet_2D(3, 1, 'test')
    checkpoint = torch.load("../models/weights/unet++_11.pth", map_location='cpu')
    dataset_test = MyData(istrain="test", size=256)
else:  # 'linknet'
    net = LinkNet()
    checkpoint = torch.load("../models/weights/link_640_57.pth", map_location='cpu')
    dataset_test = MyData(istrain="test", size=640)
net.load_state_dict(checkpoint['net'])
loader_test = Data.DataLoader(dataset_test, batch_size=conf.BATCH_SIZE_TEST, shuffle=False)

net.cuda()
print("test_images", len(dataset_test))
print("start testing...")

# val###################
net.eval()
test_iou = 0
t1 = time.time()
for i, (img, label) in enumerate(loader_test):

    output = net(img.float()).cuda()
    output[output >= 0.5] = 1.
    output[output != 1] = 0.
    iou_ = iou(output.cpu(), label)
    test_iou += iou_
    print("img_%d: iou=%.4f" % (i, iou_))
    #############show
    isShow = True
    if isShow:
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
        # mm=cv2.addWeighted(np.uint8(img),0.8,np.uint8(output*255),0.2,0)
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

    del img, label
test_iou /= len(loader_test)
t2 = time.time()
print("speed: %.4f fps, test_iou : %.4f" % (len(dataset_test) / (t2 - t1), test_iou))
