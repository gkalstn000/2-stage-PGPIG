import torch
import torch.nn as nn
from models.dptn_networks.perceptual import PerceptualLoss
import models.dptn_networks as networks
import util.util as util
from models.dptn_networks import loss
from collections import defaultdict
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

class DPTNModel(nn.Module) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser
    def __init__(self, opt):
        super(DPTNModel, self).__init__()
        self.opt = opt
        self.min_size = (8, 5)
        self.load_size = opt.load_size
        self.step_size = opt.step_size

        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.GANloss = loss.GANLoss(opt.gan_mode).cuda()
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = loss.VGGLoss().cuda()
            self.CE = torch.nn.NLLLoss()
            self.Faceloss = PerceptualLoss(network= 'vgg19',
                                           layers= ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
                                           num_scales=1,
                                           ).cuda()


    def forward(self, data, mode):
        src_image, src_map, src_face, tgt_image, tgt_map, tgt_face = self.preprocess_input(data)

        if mode == 'generator':
            G_losses, sample = self.compute_generator_loss(src_image, src_map, src_face,
                                                           tgt_image, tgt_map, tgt_face)
            return G_losses, sample
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(src_image, src_map,
                                                     tgt_image, tgt_map)
            return d_loss
        elif mode == 'inference' :
            self.netG.eval()
            with torch.no_grad():
                sample = self.generate_fake_valid(src_image, src_map, tgt_image,  tgt_map)
            return sample
    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        if opt.isTrain and opt.continue_train :
            ckpt = util.load_network(opt.which_epoch, opt)
            optimizer_G.load_state_dict(ckpt['optG'])
            optimizer_D.load_state_dict(ckpt['optD'])

        return optimizer_G, optimizer_D

    def save(self, epoch, optG, optD):
        util.save_network(self.netG, self.netD, optG, optD, epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        # if opt.isTrain :
        #     print('load pre-trained step_dptn model')
        #     ckpt = torch.load('./checkpoints/pretrained_step_dptn.pth', map_location=lambda storage, loc: storage)
        #     netG.load_state_dict(ckpt['netG'])
        #     netD.load_state_dict(ckpt['netD'])
        #     print('load pre-trained step_dptn Done')

        if not opt.isTrain or opt.continue_train:
            ckpt = util.load_network(opt.which_epoch, opt)
            netG.load_state_dict(ckpt["netG"])
            if opt.isTrain:
                netD.load_state_dict(ckpt["netD"])
        return netG, netD
    def preprocess_input(self, data):
        source_image_, target_image_ = data['source_image'], data['target_image']
        source_skeleton_, target_skeleton_ = data['source_skeleton'], data['target_skeleton']
        source_face_, target_face_ = data['source_face_center'], data['target_face_center']

        if self.use_gpu():
            source_image_, target_image_ = data['source_image'].cuda(), data['target_image'].cuda()
            source_skeleton_, target_skeleton_ = data['source_skeleton'].cuda(), data['target_skeleton'].cuda()
            source_face_, target_face_ = data['source_face_center'].cuda(), data['target_face_center'].cuda()

        Is = torch.cat((source_image_, target_image_), 0)
        Bs = torch.cat((source_skeleton_, target_skeleton_), 0)
        Fs = torch.cat((source_face_, target_face_), 0)
        It = torch.cat((target_image_, source_image_), 0)
        Bt = torch.cat((target_skeleton_, source_skeleton_), 0)
        Ft = torch.cat((target_face_, source_face_), 0)

        return Is, Bs, Fs, It, Bt, Ft
    def backward_G_basic(self, fake_image, target_image, face, true_timestep, use_d):
        # Calculate reconstruction loss
        # Calculate GAN loss
        loss_ad_gen = None
        loss_step = None
        loss_face = None

        # loss_app_gen = self.L1loss(fake_image, target_image) * self.opt.lambda_rec
        cont, style = self.Vggloss(fake_image, target_image)
        loss_content_gen = cont * self.opt.lambda_content
        loss_style_gen = style * self.opt.lambda_style

        if use_d:
            D_fake, step_pred = self.netD(fake_image)
            loss_step = self.CE(step_pred, true_timestep.long())
            loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

            loss_face = self.Faceloss(
                util.crop_face_from_output(fake_image, face),
                util.crop_face_from_output(target_image, face))

        return loss_ad_gen, loss_style_gen, loss_content_gen, loss_face, loss_step
    def compute_generator_loss(self,
                               src_image, src_map, src_face,
                               tgt_image, tgt_map, tgt_face):
        self.netG.train()
        self.netD.train()

        G_losses = defaultdict(int)

        (gt_tgt_batch, fake_tgt_batch, gt_src_batch, fake_src_batch, gt_step_batch), sample = \
            self.generate_fake_train(src_image, src_map, tgt_image, tgt_map)
        tgt_face_batch = tgt_face.tile((self.step_size, 1))

        loss_ad_gen_t, loss_style_gen_t, loss_content_gen_t, loss_face_t, loss_step = self.backward_G_basic(fake_tgt_batch, gt_tgt_batch, tgt_face_batch, gt_step_batch, use_d=True)
        _, loss_style_gen_s, loss_content_gen_s, _, _ = self.backward_G_basic(fake_src_batch, gt_src_batch, None, None, use_d=False)
        # G_losses['L1_target'] = self.opt.t_s_ratio * loss_app_gen_t
        G_losses['GAN_target'] = loss_ad_gen_t * 0.5
        G_losses['VGG_target'] =  self.opt.t_s_ratio * (loss_style_gen_t + loss_content_gen_t)
        G_losses['Face_target'] = loss_face_t
        G_losses['Step_loss'] = loss_step * self.opt.lambda_step * 0.5
        # G_losses['L1_source'] = (1-self.opt.t_s_ratio) * loss_app_gen_s
        G_losses['VGG_source'] = (1-self.opt.t_s_ratio) * (loss_style_gen_s + loss_content_gen_s)

        return G_losses, sample
    def backward_D_basic(self, real, fake, step_true):
        # Real
        D_real, step_pred_true = self.netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        real_step = self.CE(step_pred_true, step_true) * self.opt.lambda_step
        # fake
        D_fake, step_pred_fake = self.netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        fake_step = self.CE(step_pred_fake, step_true) * self.opt.lambda_step

        # gradient penalty for wgan-gp
        gradient_penalty = 0
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = loss.cal_gradient_penalty(self.netD, real, fake.detach())

        return D_real_loss, real_step, D_fake_loss, fake_step, gradient_penalty
    def compute_discriminator_loss(self,
                                   src_image, src_map,
                                   tgt_image, tgt_map):
        self.netG.train()
        self.netD.train()

        D_losses = {}

        with torch.no_grad():
            (gt_tgt_batch, fake_tgt_batch, _, _, gt_step_batch), sample = \
                self.generate_fake_train(src_image, src_map, tgt_image, tgt_map)
            fake_tgt_batch.detach()

        D_real_loss, real_step, D_fake_loss, fake_step, gradient_penalty = self.backward_D_basic(gt_tgt_batch, fake_tgt_batch, gt_step_batch)
        D_losses['Real_loss'] = D_real_loss * 0.5
        D_losses['Real_step'] = real_step * 0.5
        D_losses['Fake_loss'] = D_fake_loss * 0.5
        D_losses['Fake_step'] = fake_step * 0.5
        if self.opt.gan_mode == 'wgangp':
            D_losses['WGAN_penalty'] = gradient_penalty

        return D_losses

    def generate_fake_one_step(self,
                               src_image, src_map, ref_timestep,
                               tgt_image, tgt_map, tgt_timestep):

        ref_image = self.sample_image(src_image, ref_timestep)
        intput_image = self.sample_image(tgt_image, tgt_timestep)

        fake_tgt, fake_src = self.netG(ref_image, src_map, ref_timestep, src_image,
                                       intput_image, tgt_map, tgt_timestep)

        return fake_tgt, fake_src,

    def generate_fake_valid(self,
                      src_image, src_map,
                      tgt_image, tgt_map):

        b, c, h, w = src_image.size()

        gt_tgts = []
        fake_tgts = []

        init_step = torch.tensor([0 for _ in range(b)])
        init_noise = torch.normal(mean=0, std=np.sqrt(0.1), size=(b, c, h, w)).to(src_image.device)
        xt = self.sample_image(src_image, init_step) + init_noise

        gt_tgts.extend([src_image.cpu(), tgt_map[:, :3].cpu()])
        fake_tgts.extend([xt.cpu(), tgt_map[:, :3].cpu()])

        for step in range(self.opt.step_size) :
            tgt_timestep = torch.tensor([step for _ in range(b)])
            ref_timestep = tgt_timestep + 1
            ref_image = self.sample_image(src_image, ref_timestep)

            xt, _ = self.netG(ref_image, src_map, ref_timestep, src_image,
                               xt, tgt_map, tgt_timestep)

            gt_tgt = self.sample_image(tgt_image, ref_timestep)

            gt_tgts.append(gt_tgt.cpu())
            fake_tgts.append(xt.cpu())

        gt_sample = torch.cat(gt_tgts, 3)
        fake_sample = torch.cat(fake_tgts, 3)
        sample = torch.cat([gt_sample, fake_sample], 2)

        return sample

    def generate_fake_train(self,
                      src_image, src_map,
                      tgt_image, tgt_map):

        b, c, h, w = src_image.size()

        gt_tgts = []
        fake_tgts = []
        gt_srcs = []
        fake_srcs = []
        gt_steps = []

        init_step = torch.tensor([0 for _ in range(b)])
        init_noise = torch.normal(mean=0, std=np.sqrt(0.1), size=(b, c, h, w)).to(src_image.device)
        xt = self.sample_image(src_image, init_step) + init_noise
        z = xt

        for step in range(self.opt.step_size) :
            tgt_timestep = torch.tensor([step for _ in range(b)])
            ref_timestep = tgt_timestep + 1
            ref_image = self.sample_image(src_image, ref_timestep)

            xt, fake_src = self.netG(ref_image, src_map, ref_timestep, src_image,
                               xt, tgt_map, tgt_timestep)

            gt_tgt = self.sample_image(tgt_image, ref_timestep)
            gt_src = self.sample_image(src_image, ref_timestep)

            gt_tgts.append(gt_tgt)
            fake_tgts.append(xt)
            gt_srcs.append(gt_src)
            fake_srcs.append(fake_src)
            gt_steps.append(tgt_timestep)

        gt_tgt_batch = torch.cat(gt_tgts, 0)
        fake_tgt_batch = torch.cat(fake_tgts, 0)
        gt_src_batch = torch.cat(gt_srcs, 0)
        fake_src_batch = torch.cat(fake_srcs, 0)
        gt_step_batch = torch.cat(gt_steps, 0).to(fake_tgt_batch.device)

        gt_sample = torch.cat(gt_tgts, 3).cpu().detach()
        gt_sample = torch.cat([tgt_map[:, :3].cpu(), gt_sample], 3)
        fake_sample = torch.cat(fake_tgts, 3).cpu().detach()
        fake_sample = torch.cat([z.cpu(), fake_sample], 3)
        sample = torch.cat([gt_sample, fake_sample], 2)

        return (gt_tgt_batch, fake_tgt_batch, gt_src_batch, fake_src_batch, gt_step_batch), sample

    def get_vis(self, true_list, fake_list):
        gt_vis = torch.cat(true_list, -1)
        fake_vis = torch.cat(fake_list, -1).detach()
        vis = torch.cat([gt_vis, fake_vis], -2)

        return vis

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def sample_timestep(self, b, tgt_timestep=None):
        step = torch.randint(0, self.step_size - 1, (b,))
        if tgt_timestep != None:
            exponential_distribution = dist.Exponential(1)
            step = exponential_distribution.sample((b,)) + tgt_timestep + 1
            step = torch.where(step > self.step_size, self.step_size, step)

        return step.int()
    def sample_image(self, images, step):
        min_h, min_w = self.min_size
        max_h, max_w = self.load_size
        downscale_size = torch.stack([self.exponential_sampling(min_h, max_h, step), self.exponential_sampling(min_w, max_w, step)], dim=1).tolist()

        result_batch = []
        for img, size in zip(images, downscale_size) :
            img_down = F.interpolate(img.unsqueeze(0), size = size, mode='bicubic', align_corners=True)
            img_up = F.interpolate(img_down, size = self.load_size, mode='bicubic', align_corners=True)
            result_batch.append(img_up)
        return torch.cat(result_batch, 0)

    def exponential_sampling(self, min_value, max_value, index):
        logspace_values = torch.logspace(torch.log10(torch.tensor(min_value)),
                                         torch.log10(torch.tensor(max_value)),
                                         self.step_size + 1)

        return torch.round(logspace_values[index.tolist()]).int()