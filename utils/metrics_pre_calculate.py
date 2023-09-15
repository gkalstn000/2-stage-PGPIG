import os
import pathlib
import torch
import numpy as np
from imageio import imread
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import glob
import argparse
import matplotlib.pyplot as plt
from inception import InceptionV3
# from scripts.PerceptualSimilarity.models import dist_model as dm
import lpips
import pandas as pd
import json
import imageio
from skimage.draw import disk, line_aa, polygon
import cv2

from tqdm import tqdm, trange
from options.test_options import TestOptions
from utils.metrics import FID, LPIPS, Reconstruction_Metrics
import models


if __name__ == "__main__":
    opt = TestOptions().parse()
    model = models.create_model(opt)

    fid = FID()
    lpips_obj = LPIPS()
    rec = Reconstruction_Metrics()

    real_path = '/datasets/msha/fashion/train_256'
    gt_path = '/datasets/msha/fashion/test_256'
    save_root = 'linear_sampling'

    real_list = [f'{real_path}/{filename}' for filename in os.listdir(real_path)]
    gt_list = [f'{gt_path}/{filename}' for filename in os.listdir(gt_path)]


    # FID calculate
    os.makedirs(os.path.join(save_root, 'fid'), exist_ok=True)
    path = pathlib.Path(real_path)
    files = list(path.glob('*.png'))
    imgs = np.array([(cv2.resize(imread(str(fn)).astype(np.float32), (176, 256))) for fn in tqdm(files, desc='loading real images to numpy')])
    imgs = imgs.transpose((0, 3, 1, 2))
    imgs = (imgs / 255 - 0.5) * 2 # normalize [0, 1]

    for step in trange(1, 21, desc= 'step calculating') :
        img_tensor = torch.from_numpy(imgs)
        sampling_step = torch.tensor([step for _ in range(img_tensor.size(0))])
        img_tensor = model.sample_image(img_tensor, sampling_step).numpy()
        img_tensor = (img_tensor - img_tensor.min((2, 3), keepdims=True)) / (img_tensor.max((2, 3), keepdims=True) - img_tensor.min((2, 3), keepdims=True))

        m, s = fid.calculate_activation_statistics(img_tensor, True)
        np.savez(os.path.join(save_root, 'fid', f'statics_{step}.npz'), mu=m, sigma=s)

    # Recon calculate
