import os
import shutil
import time
import numpy as np
from skimage import io
import torch

from constants import GENERATOR_CHECKPOINT_PATH, \
    GENERATOR_RESULT_PATH
from utils.image import save_image, image_save_transform


def test(epoch, generator_model, data_loader):
    generator_path = GENERATOR_CHECKPOINT_PATH + '/' + str(epoch).zfill(5) + '.tar'
    checkpoint = torch.load(generator_path)
    print('INFO::Generator Path: ' + generator_path)
    generator_model.load_state_dict(checkpoint['model_state_dict'])
    # generator_
    e = checkpoint['epoch']
    print('INFO::Generator Epochs: ' + str(e))
    result_path = GENERATOR_RESULT_PATH + '/' + str(epoch).zfill(5)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    train_result_path = result_path + '/train'
    val_result_path = result_path + '/test'
    if not os.path.exists(train_result_path):
        os.mkdir(train_result_path)
    if not os.path.exists(val_result_path):
        os.mkdir(val_result_path)
    # save_image(e, val_data_loader, val_result_path, generator_model)
    total_time = 0
    i = 0
    tic = time.time()
    for inputs, reference in data_loader:
        inputs = inputs.cuda()
        start = time.time()
        outputs = generator_model(inputs)
        end = time.time()
        total = end - start
        total_time += total
        print('VERBOSE::Batch processing time is %.5f' % total)
        mask_result_path = result_path + '/mask'
        if not os.path.exists(mask_result_path):
            os.mkdir(mask_result_path)
        image_result_path = result_path + '/image'
        if not os.path.exists(image_result_path):
            os.mkdir(image_result_path)
        for output in outputs:
            image = (image_save_transform(output) * 255).astype(np.uint8)
            name = str(i).zfill(5) + '.jpg'
            io.imsave(image_result_path + '/' + name, image[:, :, :3])
            io.imsave(mask_result_path + '/' + name, image[:, :, 3:])
            i = i + 1
    toc = time.time()
    print('VERBOSE::Model Processing time is %.5f' % total_time)
    print('VERBOSE::Total Processing time is %.5f' % (toc - tic))

