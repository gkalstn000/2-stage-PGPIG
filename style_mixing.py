import data
from options.test_options import TestOptions
import models
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from util.visualizer import Visualizer
from util import html, util
from collections import OrderedDict
import os


opt = TestOptions().parse()
dataloader = data.create_dataloader(opt)

model = models.create_model(opt)
model.eval()

if not opt.simple_test:
    visualizer = Visualizer(opt)
    # create a webpage that summarizes the all results
    web_dir = os.path.join(opt.results_dir, opt.id,
                           '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.id, opt.phase, opt.which_epoch))

trans = T.ToPILImage()
result_path = os.path.join(opt.results_dir, opt.save_id)
util.mkdirs(result_path)
# test
for i, data_i in enumerate(tqdm(dataloader)):
    fake_target = model(data_i, mode='mixing')
    fake_target = fake_target.clamp_(-1, 1)
    img_path = data_i['path']
    fake_target = (fake_target + 1) / 2
    for k in range(fake_target.shape[0]) :
        filename = img_path[k].replace('jpg', 'png')
        generated_image = trans(fake_target[k].cpu())
        generated_image.save(os.path.join(result_path, filename))
