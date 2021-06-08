import os
import shutil
import time

import torch
from skimage import io
import numpy as np


def image_save_transform(tensor):
    image = tensor.to("cpu").clone().detach()
    # print(image.size())
    image = image.numpy()  # .squeeze()
    # print(image.shape)
    image = image.transpose(1, 2, 0)
    # print(image.shape)
    image = image * np.array((0.5, )) + np.array((0.5, ))
    image = image.clip(0, 1)
    return image


def save_image(epoch, data_loader, result_path, model):
    with torch.no_grad():
        inputs, _ = iter(data_loader).next()
        inputs = inputs.cuda()
        outputs = model(inputs)
        i = 0
        mask_result_path = result_path + '/mask'
        if not os.path.exists(mask_result_path):
            os.mkdir(mask_result_path)
        else:
            shutil.rmtree(mask_result_path)
            os.mkdir(mask_result_path)
        image_result_path = result_path + '/image'
        if not os.path.exists(image_result_path):
            os.mkdir(image_result_path)
        else:
            shutil.rmtree(image_result_path)
            os.mkdir(image_result_path)
        for output in outputs:
            image = (image_save_transform(output)*255).astype(np.uint8)
            name = str(epoch).zfill(5) + '_' + str(i).zfill(5) + '.jpg'
            io.imsave(image_result_path + '/' + name, image[:, :, :3])
            io.imsave(mask_result_path + '/' + name, image[:, :, 3:])
            i = i+1
        return total
