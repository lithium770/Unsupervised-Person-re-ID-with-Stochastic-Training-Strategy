from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
from torch import randperm
from torch import tensor
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index


class UnsupervisedCamStylePreprocessor(Dataset):
    def __init__(self, dataset, root=None, camstyle_root=None, num_cam=6, transform=None, cam_label=-1):
        super(UnsupervisedCamStylePreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.num_cam = num_cam
        self.camstyle_root = camstyle_root
        self.cam_label = cam_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.cam_label > -1:
            sel_cam = tensor(self.cam_label)
        else:
            while True:
                sel_cam = randperm(self.num_cam)[0]
                if sel_cam != camid:
                    break
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        #if sel_cam == camid:
        img = Image.open(fpath).convert('RGB')
        if 'MSMT17' in fpath:
            aug_fname = osp.basename(fname)[:-4] + '_fake_' + str(sel_cam.numpy() + 1) + '.jpg'
        else:
            aug_fname = osp.basename(fname)[:-4] + '_fake_' + str(camid + 1) + 'to' + str(sel_cam.numpy() + 1) + '.jpg'
        fpath = osp.join(self.camstyle_root, aug_fname)
        trans_img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            trans_img = self.transform(trans_img)
        return img, trans_img, fname, pid, camid, index