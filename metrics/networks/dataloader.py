import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
from imageio import imread
import glob

class Dataset(data.Dataset) :
    def __init__(self, img_list, resize=True, load_size=(256, 176)):
        super(Dataset, self).__init__()
        self.img_list = img_list
        self.load_size = load_size
        self.resize = resize
    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_array = imread(img_path)
        if self.resize:
            img_array = cv2.resize(img_array, self.load_size[::-1]).astype(np.float32)
        return (torch.from_numpy(img_array).permute(2, 0, 1) / 255) * 2 -1
    def __len__(self):
        return len(self.img_list)

def get_dataloaders(real_path, gt_path, fake_path) :
    real_list = [os.path.join(real_path, filename) for filename in os.listdir(real_path)]
    gt_list, fake_list = preprocess_path_for_deform_task(gt_path, fake_path)
    real_dl = make_dataloader(Dataset(real_list))
    gt_dl = make_dataloader(Dataset(gt_list))
    fake_dl = make_dataloader(Dataset(fake_list, resize=False))

    return real_dl, gt_dl, fake_dl
def make_dataloader(dataloader, batchsize=32) :
    return  torch.utils.data.DataLoader(
            dataloader,
            batch_size=batchsize,
            num_workers=8,
            shuffle=False
        )

def preprocess_path_for_deform_task(gt_path, distorted_path):
    distorted_image_list = sorted(get_image_list(distorted_path))
    gt_list=[]
    distorated_list=[]

    for distorted_image in distorted_image_list:
        image = os.path.basename(distorted_image)
        image = image.split('_2_')[-1]
        image = image.split('_vis')[0] +'.png'
        gt_image = os.path.join(gt_path,  image)
        if not os.path.isfile(gt_image):
            print(gt_image)
            continue
        gt_list.append(gt_image)
        distorated_list.append(distorted_image)

    return gt_list, distorated_list


def get_image_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list'%flist)
    return []