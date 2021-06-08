import torch
from skimage import io
from torch.utils.data import Dataset


class FakeDataset(Dataset):
    def __init__(self, data_root, num, epoch=300, transform_input=None):
        self.num = num
        self.data_root = data_root
        self.epoch = epoch
        self.transform_input = transform_input

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        imres = io.imread(self.data_root + '/' + str(self.epoch).zfill(5) + '/image/' + str(idx).zfill(5) + '.jpg')#.transpose(2, 0, 1)
        if self.transform_input:
            imres = self.transform_input(imres)
        ret_val = imres
        return ret_val


