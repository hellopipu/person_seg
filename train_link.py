##########################################
# @subject : Person segmentation         #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
from models.linknet import *
import torch.utils.data as Data
from read_data import *
from util import *
from models.Bisenet import *
import time
import tqdm

dataset_train = MyData(istrain="train", size=640)
dataset_val = MyData(istrain="val", size=640)
loader_train = Data.DataLoader(dataset_train, batch_size=conf.BATCH_SIZE_TRAIN, shuffle=True, drop_last=True,
                               num_workers=4)
loader_val = Data.DataLoader(dataset_val, batch_size=conf.BATCH_SIZE_VAL, shuffle=False, num_workers=1)

linknet = LinkNet()
if 0:  # whether to load the pretrained model
    checkpoint = torch.load('/data1/codes/pseg_link/results/link1024_res34/link_1024_3__.pth',
                            map_location=lambda storage, loc: storage)
    linknet.load_state_dict(checkpoint['net'])
    start_i = checkpoint['epoch'] + 1
    optimizer = torch.optim.Adam(linknet.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY, amsgrad=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(1)
else:
    start_i = 0
    optimizer = torch.optim.Adam(linknet.parameters(), lr=conf.LR, weight_decay=conf.WEIGHT_DECAY, amsgrad=True)
criterion = nn.BCELoss()
linknet.cuda(1)
print("train_images", len(dataset_train))
print("val_images", len(dataset_val))
print("start training...")
val_iou_all = 0
for epoch in range(start_i, conf.EPOCH):
    if epoch == 20:
        adjust_learning_rate(optimizer, conf.LR * 0.5)
        print("LR change to 0.5*LR")
    elif epoch == 40:
        adjust_learning_rate(optimizer, conf.LR * 0.1)
        print("LR change to 0.1*LR")
    tq = tqdm.tqdm(total=len(loader_train) * conf.BATCH_SIZE_TRAIN)
    tq.set_description('epoch %d' % epoch)
    # train###############################################################################
    linknet.train()
    train_loss = 0
    for i, (img, label) in enumerate(loader_train):
        t1 = time.time()
        output = linknet(img.float().cuda(1))
        loss = criterion(output, label.float().cuda(1))
        tq.update(conf.BATCH_SIZE_TRAIN)
        tq.set_postfix(loss='%.4f' % loss)
        optimizer.zero_grad()
        loss.backward()  #
        optimizer.step()
        train_loss += loss
    tq.close()
    train_loss /= len(loader_train)
    #    del output,loss,img,label
    # val###############################################################################
    linknet.eval()
    val_iou = 0
    for i, (img, label) in enumerate(loader_val):
        output = linknet(img.float().cuda(1))
        output[output >= 0.5] = 1
        output[output != 1] = 0
        iou_ = iou(output, label)
        val_iou += iou_
    val_iou /= len(loader_val)
    #   del output,label,img
    # if val iou improves , then save the model
    if val_iou_all <= val_iou:
        val_iou_all = val_iou
        state = {'net': linknet.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, "./results/link640_res18/link_640_%d.pth" % epoch)
    print("EPOCH %d : train_loss : %.4f , val_iou : %.4f" % (epoch, train_loss, val_iou))
