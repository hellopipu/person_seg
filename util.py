##########################################
# @subject : Person segmentation         #
# @author  : perryxin                    #
# @date    : 2018.12.27                  #
##########################################
import numpy as np
import random
import cv2
import json
import zlib
import base64
from PIL import Image
import io
import colorsys


def split_trainval(img_dir, istrain):
    img_dir = sorted(img_dir)
    num = len(img_dir)
    np.random.seed(10)
    val_choice = np.random.choice(num, int(num / 10))
    val = [img_dir[i] for i in val_choice]
    if istrain == "val":
        img_dir = val
    elif istrain == "train":
        img_dir = list(set(img_dir) - set(val))
    else:
        pass
    return img_dir


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=1):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1.,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


def read_json(path, r=0, c=0):
    # image=cv2.imread(path)
    # r,c,_=image.shape
    label = np.zeros((r, c), np.uint8)
    path = path.replace(path[path.find('.'):], '.json').replace('img', 'ann')
    file = open(path)
    a = json.load(file)
    obj = a['objects']
    for oo in obj:
        if oo['classTitle'] == 'person_poly':
            exterior = oo['points']['exterior']
            interior = oo['points']['interior']
            for i in exterior:
                if len(i) == 2:
                    cv2.fillPoly(label, [np.array(exterior, np.int32)], 255)
                    break
                else:
                    cv2.fillPoly(label, [np.array(i, np.int32)], 255)
            for i in interior:
                if len(i) == 2:
                    cv2.fillPoly(label, [np.array(interior, np.int32)], 0)
                    break
                else:
                    cv2.fillPoly(label, [np.array(i, np.int32)], 0)
        elif oo['classTitle'] == 'person_bmp':
            mask = base64_2_mask(oo['bitmap']['data'])
            r_, c_ = mask.shape
            loc = oo['bitmap']['origin']
            temp_map = np.zeros((r, c), np.uint8)
            temp_map[loc[1]:loc[1] + r_, loc[0]:loc[0] + c_] = mask * 255
            label[temp_map == 255] = 255
    # plt.imshow(label[:,:,])
    # plt.show()
    return label


def BGR2RGB(im):
    im = im.copy()
    temp = im[:, :, 0].copy()
    im[:, :, 0] = im[:, :, 2].copy()
    im[:, :, 2] = temp
    return im


def dice_loss(m1, m2, is_average=True):
    num = m1.size(0)
    m1 = m1.view(num, -1)
    m2 = m2.view(num, -1)
    intersection = (m1 * m2)
    scores = (2. * intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    if is_average:
        score = scores.sum() / num
        return 1 - score
    else:
        return -scores


def iou(pred, target, n_classes=2):
    batch, _, _, _ = pred.shape
    pred = pred.view(batch, -1)
    target = target.view(batch, -1)
    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        mul = pred_inds.long() * target_inds.long()
        intersection = mul.sum(
            1)  # (pred_inds[target_inds]).long().sum(0).data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum(1) + target_inds.long().sum(1) - intersection
        kk = intersection.float() / (union + 10 ** -7).float()
    return np.array(kk.mean())


# def iou(pred, target, n_classes = 2):
#     ious = []
#     pred = pred.view(-1)
#     target = target.view(-1)
#
#     # Ignore IoU for background class ("0")
#     for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
#         pred_inds = pred == cls
#         target_inds = target == cls
#         intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
#         union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
#         if union == 0:
#             ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
#         else:
#             ious.append(float(intersection) / float(max(union, 1)))
#     return np.array(ious)
def random_Contrast_and_Brightness(img, alpha=1, beta=0, u=0.5):
    if random.random() < u:
        blank = np.zeros(img.shape, img.dtype)
        # dst = alpha * img + beta * blank
        if random.randint(0, 1) == 0:
            alpha = 1 - random.random() * 0.8
        else:
            alpha = random.random() * 4 + 1
        beta = -random.random() * 160 + 80  # -80~80
        img = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    return img


def clahe(img, clip_limit=2, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError('clahe supports only uint8 inputs')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img


def random_clache(img, u=0.5):
    if random.random() < u:
        limit = random.random() * 5
        img = clahe(img, clip_limit=limit)
    return img


def random_crop(image, label, u=0.5):
    if random.random() < u:
        scale = random.random() * 0.2
        h, w, _ = image.shape
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        image = image[crop_h:h - crop_h, crop_w:w - crop_w]
        label = label[crop_h:h - crop_h, crop_w:w - crop_w]
        image = cv2.resize(image, (w, h), cv2.INTER_AREA)  # np.fliplr(img) ##left-right
        label = cv2.resize(label, (w, h), cv2.INTER_NEAREST)
        label[label >= 0.5] = 1
        label[label != 1] = 0
    return image, label


def random_scale(image, label, u=0.5):
    if random.random() < u:
        scale = random.random() * 0.4 + 0.8
        h, w, _ = image.shape
        image = cv2.resize(image, (int(w * scale), int(h * scale)))  # np.fliplr(img) ##left-right
        label = cv2.resize(label, (int(w * scale), int(h * scale)))
        label[label >= 0.5] = 1
        label[label != 1] = 0
    return image, label


def random_resize(image, label, sz=256, u=0.5):
    if random.random() < u:
        image = cv2.resize(image, (sz, sz), cv2.INTER_AREA)  # np.fliplr(img) ##left-right
        label = cv2.resize(label, (sz, sz), interpolation=cv2.INTER_NEAREST)
        label[label >= 0.5] = 1
        label[label != 1] = 0
    return image, label


def random_resize_512(image, label, u=0.5):
    if random.random() < u:
        image = cv2.resize(image, (512, 512))  # np.fliplr(img) ##left-right
        label = cv2.resize(label, (512, 512))
        label[label >= 0.5] = 1
        label[label != 1] = 0
    return image, label


def random_horizontal_flip_transform2(image, label, u=0.5):
    if random.random() < u:
        image = cv2.flip(image, 1)  # np.fliplr(img) ##left-right
        label = cv2.flip(label, 1)
    return image, label


def random_vertical_flip_transform2(image, label, u=0.5):
    if random.random() < u:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
    return image, label


def random_rotate90_transform2(image, label, u=0.5):
    if random.random() < u:

        angle = random.randint(1, 3) * 90
        if angle == 90:
            image = image.transpose(1, 0, 2)  # (0,1,2)-->(1,0,2)     #cv2.transpose(img)
            image = cv2.flip(image, 1)  # right rotate 90
            label = label.transpose(1, 0)
            label = cv2.flip(label, 1)

        elif angle == 180:
            image = cv2.flip(image, -1)  # rotate 180,,-1 means flip l-r,flip u-d
            label = cv2.flip(label, -1)

        elif angle == 270:
            image = image.transpose(1, 0, 2)  # cv2.transpose(img)
            image = cv2.flip(image, 0)
            label = label.transpose(1, 0)  # cv2.transpose(img)
            label = cv2.flip(label, 0)
    return image, label


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def shift_hsv(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if dtype == np.uint8:
        img = img.astype(np.int32)
    hue, sat, val = cv2.split(img)
    hue = cv2.add(hue, hue_shift)
    hue = np.where(hue < 0, 180 - hue, hue)
    hue = np.where(hue > 180, hue - 180, hue)
    hue = hue.astype(dtype)
    sat = clip(cv2.add(sat, sat_shift), dtype, 255 if dtype == np.uint8 else 1.0)
    val = clip(cv2.add(val, val_shift), dtype, 255 if dtype == np.uint8 else 1.0)
    img = cv2.merge((hue, sat, val)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def randomHueSaturationValue(image, hue_shift_limit=(-5, 5),
                             sat_shift_limit=(-5, 5),
                             val_shift_limit=(-5, 5), u=0.5):
    if np.random.random() < u:
        h = -random.random() * 30 + 15
        s = -random.random() * 30 + 15
        v = -random.random() * 30 + 15
        image = shift_hsv(image, h, s, v)
    return image


def randomRotate(image, label, u=0.5, angle=0, scale=1):  # 1
    if random.random() < u:
        (h, w) = image.shape[:2]  # 2
        center = (w // 2, h // 2)  # 4
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.5, 1.5)
        # print(scale,angle)
        M = cv2.getRotationMatrix2D(center, angle, scale)  # 5

        image = cv2.warpAffine(image, M, (w, h))  # 6
        label = cv2.warpAffine(label, M, (w, h))  # 6

    return image, label  # 7


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def motion_blur(image, degree=15, angle=360):
    image = np.array(image)
    #
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def gaussian_noise(image, degree=None):
    row, col, ch = image.shape
    mean = 0
    if not degree:
        var = np.random.uniform(50, 200)
    else:
        var = degree
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    cv2.normalize(noisy, noisy, 0, 255, norm_type=cv2.NORM_MINMAX)
    noisy = np.array(noisy, dtype=np.uint8)
    return noisy


def random_gaussianblur(img, u=0.5):
    if random.random() < u:
        sz = 2 * random.randint(1, 7) + 1
        img = cv2.GaussianBlur(img, ksize=(sz, sz), sigmaX=0, sigmaY=0)
    return img


def random_gaussian_noise(img, u=0.5):
    if random.random() < u:
        img = gaussian_noise(img)
    return img


def random_motionblur(img, u=0.5):
    if random.random() < u:
        degree = 2 * random.randint(2, 7) + 1
        angle = random.randint(1, 360)
        img = motion_blur(img, degree, angle)
    return img


if __name__ == '__main__':
    I = cv2.imread("C:\\Users\\perryxin\\Downloads\\dataset\\supercisely\\seg__ds3\\img\\pexels-photo-93776.png")
    print("origin", I.shape)
    I = random_clache(I)
    print("rotate", I.shape)
    cv2.imshow("img", I)
    cv2.waitKey(0)
