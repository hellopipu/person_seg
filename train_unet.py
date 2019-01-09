##########################################
# @subject : Person segmentation         #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
import torch.utils.data as Data
from read_data import *
from util import *
from models.unet_plusplus import *
import tqdm

dataset_train = MyData(istrain="train", size=256)
dataset_val = MyData(istrain="val", size=256)
loader_train = Data.DataLoader(dataset_train, batch_size=conf.BATCH_SIZE_TRAIN, shuffle=True, drop_last=True,
                               num_workers=4)
loader_val = Data.DataLoader(dataset_val, batch_size=conf.BATCH_SIZE_VAL, shuffle=False, num_workers=1)

unet = Unet_2D(3, 1)
if 0:  # whether to load the pretrained model
    checkpoint = torch.load('/data1/codes/pseg_unet++/results/modified5_cc=16_256/unet++_10.pth',
                            map_location=lambda storage, loc: storage)
    unet.load_state_dict(checkpoint['net'])
    start_i = checkpoint['epoch'] + 1
    optimizer = torch.optim.Adam(unet.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY, amsgrad=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(0)
else:
    start_i = 0
    optimizer = torch.optim.Adam(unet.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY, amsgrad=True)
criterion = nn.BCELoss()
unet.cuda(0)
print("train_images", len(dataset_train))
print("val_images", len(dataset_val))
print("start training...")

# start training ###########################################################################
val_iou_all = 0
for epoch in range(start_i, conf.EPOCH):
    # step learning rate schedule
    if epoch == 20:
        adjust_learning_rate(optimizer, conf.LR * 0.5)
        print("LR change to 0.5*LR")
    elif epoch == 40:
        adjust_learning_rate(optimizer, conf.LR * 0.1)
        print("LR change to 0.1*LR")
    tq = tqdm.tqdm(total=len(loader_train) * conf.BATCH_SIZE_TRAIN)
    tq.set_description('epoch %d' % epoch)
    # train###############################################################################
    unet.train()
    for i, (img, label) in enumerate(loader_train):
        output0, output1, output2 = unet(img.float().cuda(0))
        loss = (criterion(output0, label.float().cuda(0)) + criterion(output1, label.float().cuda(0)) + criterion(
            output2, label.float().cuda(0))) / 3
        tq.update(conf.BATCH_SIZE_TRAIN)
        tq.set_postfix(loss='%.4f' % loss)
        optimizer.zero_grad()
        loss.backward()  #
        optimizer.step()
        train_loss += loss
    tq.close()
    train_loss /= len(loader_train)
    # val###############################################################################
    unet.eval()
    val_iou = 0
    val_iou1 = 0
    val_iou2 = 0
    for i, (img, label) in enumerate(loader_val):
        ###  unet++ has three outputs. output1,outpu2 will be pruned while testing
        output, output1, output2 = unet(img.float().cuda(0))
        ####0
        output[output >= 0.5] = 1
        output[output != 1] = 0
        iou_ = iou(output, label)
        val_iou += iou_
        ####1
        output1[output1 >= 0.5] = 1
        output1[output1 != 1] = 0
        iou_1 = iou(output1, label)
        val_iou1 += iou_1
        ####2
        output2[output2 >= 0.5] = 1
        output2[output2 != 1] = 0
        iou_2 = iou(output2, label)
        val_iou2 += iou_2

    val_iou /= len(loader_val)
    val_iou1 /= len(loader_val)
    val_iou2 /= len(loader_val)
    # if val iou improves , then save the model
    if val_iou_all <= val_iou:
        val_iou_all = val_iou
        state = {'net': unet.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, "./results/modified5_cc=16_256/unet++_%d.pth" % epoch)
    print("EPOCH %d : train_loss : %.4f , val_iou : %.4f , val_iou1 : %.4f, val_iou2 : %.4f" % (
    epoch, train_loss, val_iou, val_iou1, val_iou2))
