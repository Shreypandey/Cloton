import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def inception_score_v2(images, batch_size):
    # list of scores
    scores = []

    # number of steps
    num_steps = int(math.ceil(float(len(images)) / float(batch_size)))

    # iterate over the images
    for i in range(num_steps):
        # mini-batch start and end index
        s = i * batch_size
        e = (i + 1) * batch_size

        # mini-batch images
        mini_batch = images[s:e]

        # mini-batch as Torch tensor with gradients
        batch = Variable(mini_batch)

        # apply a forward pass through inception network
        # skipping aux logits
        '''
         This network is unique because it has two output layers when training.
         The second output is known as an auxiliary output and is contained in the AuxLogits part of the network.
         The primary output is a linear layer at the end of the network.
         Note, when testing we only consider the primary output.
        '''
        s, _ = net(batch)

        # accumulate scores
        scores.append(s)

    # stack scores as tensor
    scores = torch.cat(scores, 0)

    # calculate inception score

    '''
    The formula for inception score
    IS(x) = E[ KL( P(y|x) || P(y)) ]
    x: generated images
    y: inception model classification distribution aka softmax
    '''

    # calculate p(y|x) across dimension 1
    # that is one row for each image
    p_yx = F.softmax(scores, 1)

    # calculate p(y) across dimension 0
    # that is one column for each class / label
    p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)

    # calculate KL divergence
    KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))

    # calculate mean aka expected of KL
    final_score = KL_d.mean()

    # return final score
    return final_score