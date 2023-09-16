import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
from imageio.v2 import imread
import glob
from metrics.networks import fid, inception, reconstruction
from metrics.networks.lpips import LPIPS
from options.train_options import TrainOptions
import models
from tqdm import tqdm, trange
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd
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
        else : # gen image
            img_array = img_array[:, 176*2:, :]
        return (torch.from_numpy(img_array).permute(2, 0, 1) / 255) * 2 -1
    def __len__(self):
        return len(self.img_list)

def get_dataloaders(real_path, gt_path, fake_path, batch_size) :
    real_list = get_image_list(real_path)
    gt_list, fake_list = preprocess_path_for_deform_task(gt_path, fake_path)
    real_dl = make_dataloader(Dataset(real_list), batch_size)
    gt_dl = make_dataloader(Dataset(gt_list), batch_size)
    fake_dl = make_dataloader(Dataset(fake_list, resize=False), batch_size)

    return real_dl, gt_dl, fake_dl


def make_dataloader(dataloader, batchsize=32):
    return torch.utils.data.DataLoader(
        dataloader,
        batch_size=batchsize,
        num_workers=8,
        shuffle=False
    )


def preprocess_path_for_deform_task(gt_path, distorted_path):
    distorted_image_list = sorted(get_image_list(distorted_path))
    gt_list = []
    distorated_list = []

    for distorted_image in distorted_image_list:
        image = os.path.basename(distorted_image)
        image = image.split('_2_')[-1]
        image = image.split('_vis')[0] + '.png'
        gt_image = os.path.join(gt_path, image)
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
    print('can not read files from %s return empty list' % flist)
    return []

if __name__ == "__main__":
    opt = TrainOptions().parse()
    model = models.create_model(opt)

    real_path = '/datasets/msha/fashion/train_256'
    gt_path = '/datasets/msha/fashion/test_256'
    fake_path = 'results/NTED'

    batch_size = 32
    real_dl, gt_dl, fake_dl = get_dataloaders(real_path, gt_path, fake_path, batch_size)

    # Loading models
    fid = fid.FID()
    print('load FID')
    rec = reconstruction.Reconstruction_Metrics()
    print('load rec')
    lpips = LPIPS()
    print('load LPIPS')

    score_dict = {'FID': [0]*20,
                  'LPIPS': [0]*20,
                  'PSNR': [0]*20,
                  'SSIM': [0]*20,
                  'SSIM_256': [0]*20}

    for step in range(1, 21):
        # Real FID Calulate
        sampling_step = torch.tensor([step for _ in range(batch_size)])

        fid_buffer = []
        npz_file = os.path.join(real_path, f'fid_static_{step}.npz')
        if os.path.exists(npz_file) :
            f = np.load(npz_file)
            m_real, s_real = f['mu'][:], f['sigma'][:]
            f.close()
        else :
            for real_batch in tqdm(real_dl, desc=f'Calculating {step}-step FID real statics...') :
                real_batch = model.sample_image(real_batch, sampling_step).cuda() * 0.5 + 0.5
                fid.model.eval()
                with torch.no_grad():
                    fid_score = fid.calculate_activation_statistics_of_images(real_batch)
                fid_buffer.append(fid_score)

            act = np.concatenate(fid_buffer, axis=0)
            m_real = np.mean(act, axis=0)
            s_real = np.cov(act, rowvar=False)
            np.savez(npz_file, mu=m_real, sigma=s_real)
        # Fake FID Calculate
        fid_buffer = []
        npz_file = os.path.join(fake_path, f'fid_static_{step}.npz')
        if os.path.exists(npz_file) :
            f = np.load(npz_file)
            m_fake, s_fake = f['mu'][:], f['sigma'][:]
            f.close()
        else :
            for fake_batch in tqdm(fake_dl, desc=f'Calculating {step}-step FID fake statics...') :
                fake_batch = fake_batch[:, :, :, 176 * (step-1): 176 * step].cuda() * 0.5 + 0.5
                fid.model.eval()
                with torch.no_grad():
                    fid_score = fid.calculate_activation_statistics_of_images(fake_batch)
                fid_buffer.append(fid_score)

            act = np.concatenate(fid_buffer, axis=0)
            m_fake = np.mean(act, axis=0)
            s_fake = np.cov(act, rowvar=False)
            np.savez(npz_file, mu=m_fake, sigma=s_fake)

        fid_value = fid.calculate_frechet_distance(m_real, s_real, m_fake, s_fake)
        score_dict['FID'][step-1] = fid_value

        # SSIM, PPSNR Calculate
        ssim_256_buffer = []
        ssim_buffer = []
        psnr_buffer = []
        npz_file = os.path.join(fake_path, f'recon_static_{step}.npz')
        if os.path.exists(npz_file):
            f = np.load(npz_file)
            psnr_buffer, ssim_256_buffer, ssim_buffer = f['psnr'], f['ssim_256'], f['ssim']
        else :
            for gt_batch, fake_batch in zip(tqdm(gt_dl, desc=f'Calculating {step}-step reconstruction score'), fake_dl) :
                fake_batch = fake_batch[:, :, :, 176 * (step-1): 176 * step].numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
                gt_batch = model.sample_image(gt_batch, sampling_step).numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
                for gt, fake in zip(gt_batch, fake_batch) :
                    psnr = compare_psnr(gt, fake, data_range=1)
                    ssim = compare_ssim(gt, fake, data_range=1, win_size=51, multichannel=True)
                    ssim_256 = compare_ssim(gt * 255, fake * 255,
                                            gaussian_weights=True, sigma=1.2,
                                            use_sample_covariance=False, multichannel=True,
                                            data_range=(fake * 255).max() - (fake * 255).min())

                    psnr_buffer.append(psnr)
                    ssim_buffer.append(ssim)
                    ssim_256_buffer.append(ssim_256)
            np.savez(npz_file, psnr=psnr_buffer, ssim=ssim_buffer, ssim_256=ssim_256_buffer)
        score_dict['SSIM_256'][step - 1] = np.round(np.mean(ssim_256_buffer), 6)
        score_dict['SSIM'][step - 1] = np.round(np.mean(ssim_buffer), 6)
        score_dict['PSNR'][step - 1] = np.round(np.mean(psnr_buffer), 6)

        # LPIPS Calculate
        lpips_buffer = []
        pt_file = os.path.join(fake_path, f'lpips_static_{step}.pt')
        if os.path.exists(pt_file) :
            lpips_buffer = torch.load(pt_file)
        else :
            for gt_batch, fake_batch in zip(tqdm(gt_dl, desc=f'Calculating {step}-step lpips score'), fake_dl) :
                fake_batch = fake_batch[:, :, :, 176 * (step - 1): 176 * step].cuda()
                gt_batch = model.sample_image(gt_batch, sampling_step).cuda()
                with torch.no_grad() :
                    lpips_buffer.append(lpips.model.forward(fake_batch, gt_batch))
            lpips_buffer = torch.cat(lpips_buffer,0).squeeze()
            torch.save(lpips_buffer, pt_file)
        lpips_value = lpips_buffer.mean().cpu().item()
        score_dict['LPIPS'][step-1] = lpips_value
        print(f'{step}-step\tFID: {fid_value},\t LPIPS: {lpips_value},\t SSIM_256: {np.round(np.mean(ssim_256_buffer), 6)}')

    df_score = pd.DataFrame.from_dict(score_dict).T
    df_score.columns = [f'step_{i}' for i in range(1, 21)]
    save_root = 'eval_results'
    save_name = fake_path.split('/')[-1]
    os.makedirs(save_root, exist_ok=True)
    df_score.to_csv(os.path.join(save_root, save_name+'.csv'))