import util.util
from options.test_options import TestOptions
import data
import models
import os
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
np.random.seed(1234)



opt = TestOptions().parse()
mode = 'train'
opt.phase = mode
dataloader_val = data.create_dataloader(opt, valid = True)
opt.phase = 'test'


test_images = []
with open(os.path.join(opt.dataroot, f'{mode}.lst'), 'r') as f:
    for lines in f:
        lines = lines.strip()
        if lines.endswith('.jpg'):
            test_images.append(lines)


n_samples = 100
interval = 10
alphas = torch.linspace(0, 1, interval + 2)[1:-1]
trans = T.ToPILImage()
save_root = 'custom/interpolation_results'
util.util.mkdirs(save_root)

index_pairs = np.unique(np.random.randint(0, len(test_images), size=(n_samples, 2)), axis=0)
name_pairs = [[test_images[from_], test_images[to_]] for from_, to_ in index_pairs]
dataloader_val.dataset.name_pairs = name_pairs

model = models.create_model(opt)
model.eval()
for i, data_i in enumerate(tqdm(dataloader_val)) :
    if i == n_samples - 1 : break
    texture, bone, ground_truth = model.preprocess_input(data_i)
    encoder = model.netG.z_encoder
    decoder = model.netG.decoder

    with torch.no_grad() :
        latent_src = encoder(texture)['noise']
        latent_tgt = encoder(ground_truth)['noise']
        interpolations = [latent_src] + [alpha * latent_src + (1 - alpha) * latent_tgt for alpha in alphas] + [latent_tgt]
        output = []
        for latent in interpolations :
            output.append(decoder(latent, [bone]))

    interpolation_result = torch.cat(output, -1)
    interpolation_result = (interpolation_result + 1) / 2
    for k in range(interpolation_result.size(0)) :
        img_path = data_i['path'][k]
        generated_image = trans(interpolation_result[k].cpu())
        generated_image.save(os.path.join(save_root, img_path))
