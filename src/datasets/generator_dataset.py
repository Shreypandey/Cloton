import torch
from skimage import io
from torch.utils.data import Dataset


class ClothDataset(Dataset):
    def __init__(self, data_root, num, mode='train', transform_input=None, transform_cloth=None):
        self.num = num
        self.data_root = data_root
        self.mode = mode
        self.transform_input = transform_input
        self.transform_cloth = transform_cloth

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        imres = io.imread(self.data_root + '/' + self.mode + '/images/' + str(idx).zfill(5) + '.jpg')#.transpose(2, 0, 1)
        imbody = io.imread(self.data_root + '/' + self.mode + '/inputs/' + str(idx).zfill(5) + '.tif').transpose(1, 2, 0)
        imcloth = io.imread(self.data_root + '/' + self.mode + '/clothes/' + str(idx+1).zfill(5) + '.jpg')
        imclothmask = io.imread(self.data_root + '/' + self.mode + '/cloth_masks/' + str(idx).zfill(5) + '.jpg')
        if self.transform_input:
            imres = self.transform_input(imres)
            imbody = self.transform_input(imbody)
            imclothmask = self.transform_input(imclothmask)
        if self.transform_cloth:
            imcloth = self.transform_cloth(imcloth)
        # print(imres.shape)
        # print(imcloth.shape)
        # print(imbody.shape)
        # print(imclothmask.shape)
        iminput = torch.cat((imbody, imcloth), 0)
        imres = torch.cat((imres, imclothmask), 0)
        ret_val = (iminput, imres)
        return ret_val


