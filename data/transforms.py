import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as VF

from tools import image_degradation, image_io


def normalize(data, alpha=1.0 / 255, beta=0):
    return data * alpha + beta


def anti_normalize(data, alpha=255.0, beta=0):
    return data * alpha + beta


def tensor2im(data, gray=False):
    data = anti_normalize(data)
    data = torch.clamp(data, 0, 255).to(torch.uint8)
    if data.ndim == 4:
        data = data.permute(0, 2, 3, 1)
        data = data.cpu().detach().numpy()
        output = []
        for i in range(data.shape[0]):
            output.append(data[i])
    elif data.ndim == 3:
        data = data.permute(1, 2, 0)
        data = data.cpu().detach().numpy()
        output = data
    else:
        raise ValueError(f'Unsupported tensor dimension: {data.ndim}')
    return output


def im2tensor(data, gray=False):
    data = normalize(data.astype(np.float32))
    data = torch.from_numpy(data)
    if data.ndim == 4:
        data = data.permute(0, 3, 1, 2)
    else:
        data = data.permute(2, 0, 1)
    return data


def imtensor2tensor(data, gray=False):
    data = normalize(data)
    if data.ndim == 4:
        data = data.permute(0, 3, 1, 2)
    else:
        data = data.permute(2, 0, 1)
    return data


class RandomTrans(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_transform_params(size=(512, 512), scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.), p=0.2):
        scale_value = np.random.uniform(scale[0], scale[1])
        ratio_value = np.random.uniform(ratio[0], ratio[1])
        if ratio_value > 1:
            height = int(size[0] * scale_value)
            width = int(size[1] * scale_value / ratio_value)
        else:
            height = int(size[0] * scale_value * ratio_value)
            width = int(size[1] * scale_value)

        top = int((size[0] - height) * np.random.random())
        left = int((size[1] - width) * np.random.random())

        flag = {
            'resized_crop': np.random.random() < p,
            'filp': np.random.random() < p,
            'color': np.random.random() < p,
        }
        value = {
            'finesize': size,
            'resized_crop': [top, left, height, width],
            'filp': 1,
            'color': {
                'brightness': 1 + np.random.uniform(-0.5, 0.5),
                'contrast': 1 + np.random.uniform(-0.3, 0.3),
                'saturation': 1 + np.random.uniform(-0.3, 0.3),
                'hue': np.random.uniform(-0.1, 0.1),
            }
        }

        return {'flag': flag, 'value': value}

    def forward(self, x, params):
        if params['flag']['resized_crop']:
            x = VF.resized_crop(x, *params['value']['resized_crop'], params['value']['finesize'], Image.BILINEAR)
        if params['flag']['filp']:
            x = VF.hflip(x)
        if params['flag']['color']:
            x = VF.adjust_brightness(x, params['value']['color']['brightness'])
            x = VF.adjust_contrast(x, params['value']['color']['contrast'])
            x = VF.adjust_saturation(x, params['value']['color']['saturation'])
            x = VF.adjust_hue(x, params['value']['color']['hue'])
        return x


class PreProcess(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_tensor = imtensor2tensor
        self.random_trans = RandomTrans()

    def forward(self, x, params=None, ran=False):
        x = self.to_tensor(x)
        if ran:
            x = self.random_trans(x, params)
        return x


def shuffledata(data, target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)


def random_transform_single_mask(img, out_shape):
    out_h, out_w = out_shape
    img = cv2.resize(img, (int(out_w * random.uniform(1.1, 1.5)), int(out_h * random.uniform(1.1, 1.5))))
    h, w = img.shape[:2]
    h_move = int((h - out_h) * random.random())
    w_move = int((w - out_w) * random.random())
    img = img[h_move:h_move + out_h, w_move:w_move + out_w]
    if random.random() < 0.5:
        if random.random() < 0.5:
            img = img[:, ::-1]
        else:
            img = img[::-1, :]
    if img.shape[0] != out_h or img.shape[1] != out_w:
        img = cv2.resize(img, (out_w, out_h))
    return img


def get_transform_params():
    crop_flag = True
    rotat_flag = np.random.random() < 0.2
    color_flag = True
    flip_flag = np.random.random() < 0.2
    degradate_flag = np.random.random() < 0.5
    flag_dict = {
        'crop': crop_flag,
        'rotat': rotat_flag,
        'color': color_flag,
        'flip': flip_flag,
        'degradate': degradate_flag,
    }

    crop_rate = [np.random.random(), np.random.random()]
    rotat_rate = np.random.random()
    color_rate = [
        np.random.uniform(-0.05, 0.05),
        np.random.uniform(-0.05, 0.05),
        np.random.uniform(-0.05, 0.05),
        np.random.uniform(-0.05, 0.05),
        np.random.uniform(-0.05, 0.05),
    ]
    flip_rate = np.random.random()
    degradate_params = image_degradation.get_random_degenerate_params(mod='weaker_2')
    rate_dict = {
        'crop': crop_rate,
        'rotat': rotat_rate,
        'color': color_rate,
        'flip': flip_rate,
        'degradate': degradate_params,
    }

    return {'flag': flag_dict, 'rate': rate_dict}


def random_transform_single_image(img, finesize, params=None, test_flag=False):
    if params is None:
        params = get_transform_params()

    if params['flag']['degradate']:
        img = image_degradation.degradate(img, params['rate']['degradate'])

    if params['flag']['crop']:
        h, w = img.shape[:2]
        h_move = int((h - finesize) * params['rate']['crop'][0])
        w_move = int((w - finesize) * params['rate']['crop'][1])
        img = img[h_move:h_move + finesize, w_move:w_move + finesize]

    if test_flag:
        return img

    if params['flag']['rotat']:
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 90 * int(4 * params['rate']['rotat']), 1)
        img = cv2.warpAffine(img, matrix, (w, h))

    if params['flag']['color']:
        img = image_io.color_adjust(
            img,
            params['rate']['color'][0],
            params['rate']['color'][1],
            params['rate']['color'][2],
            params['rate']['color'][3],
            params['rate']['color'][4],
        )

    if params['flag']['flip']:
        img = img[:, ::-1]

    if img.shape[0] != finesize or img.shape[1] != finesize:
        img = cv2.resize(img, (finesize, finesize))
        print('warning! shape error.')
    return img


def random_transform_pair_image(img, mask, finesize, test_flag=False):
    params = get_transform_params()
    img = random_transform_single_image(img, finesize, params)
    params['flag']['degradate'] = False
    params['flag']['color'] = False
    mask = random_transform_single_image(mask, finesize, params)
    return img, mask
