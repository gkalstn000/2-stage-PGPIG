import torch
import torch.nn as nn
from models.dptn_networks.perceptual import PerceptualLoss
import models.dptn_networks as networks
import util.util as util
from models.dptn_networks import loss
from collections import defaultdict
import torch.nn.functional as F

class DPTNModel(nn.Module) :
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser
    def __init__(self, opt):
        super(DPTNModel, self).__init__()
        self.opt = opt
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
            self.Faceloss = PerceptualLoss(network= 'vgg19',
                                           layers= ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
                                           num_scales=1,
                                           ).cuda()
    def forward(self, data, mode):
        src_image, src_map, src_face, tgt_image, tgt_map, tgt_face = self.preprocess_input(data)
        if mode == 'generator':
            g_loss, sample = self.compute_generator_loss(src_image, src_map, src_face,
                                                         tgt_image, tgt_map, tgt_face)
            return g_loss, sample
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(src_image, src_map,
                                                     tgt_image, tgt_map)
            return d_loss
        elif mode == 'inference' :
            self.netG.eval()
            with torch.no_grad():
                src_fake, tgt_fake = self.netG(src_image, src_map, tgt_map)

            sample_src = torch.cat([src_image.cpu(), src_map[:, :3].cpu(), src_fake.cpu(), src_image.cpu()], 3)
            sample_tgt = torch.cat([src_image.cpu(), tgt_map[:, :3].cpu(), tgt_fake.cpu(), tgt_image.cpu()], 3)
            sample = torch.cat([sample_src, sample_tgt], 2)

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

    def backward_G_basic(self, fake_image, target_image, face, use_d):
        # Calculate reconstruction loss
        # Calculate GAN loss
        loss_ad_gen = None
        loss_step = None

        loss_app_gen = self.L1loss(fake_image, target_image) * self.opt.lambda_rec
        cont, style = self.Vggloss(fake_image, target_image)
        loss_content_gen = cont * self.opt.lambda_content
        loss_style_gen = style * self.opt.lambda_style
        loss_face = self.Faceloss(
            util.crop_face_from_output(fake_image, face),
            util.crop_face_from_output(target_image, face))
        if use_d:
            D_fake = self.netD(fake_image)
            loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g



        return loss_app_gen, loss_ad_gen, loss_style_gen, loss_content_gen, loss_face
    def compute_generator_loss(self,
                               src_image, src_map, src_face,
                               tgt_image, tgt_map, tgt_face):
        self.netG.train()
        self.netD.train()
        G_losses = defaultdict(int)
        src_fake, tgt_fake = self.netG(src_image, src_map, tgt_map)

        loss_app_gen_t, loss_ad_gen_t, loss_style_gen_t, loss_content_gen_t, loss_face_t = self.backward_G_basic(tgt_fake, tgt_image, tgt_face, use_d=True)
        loss_app_gen_s, _, loss_style_gen_s, loss_content_gen_s, loss_face_s = self.backward_G_basic(src_fake, src_image, src_face, use_d=False)
        G_losses['L1_target'] = self.opt.t_s_ratio * loss_app_gen_t
        G_losses['GAN_target'] = loss_ad_gen_t * 0.5
        G_losses['VGG_target'] =  self.opt.t_s_ratio * (loss_style_gen_t + loss_content_gen_t)
        G_losses['Face_target'] = loss_face_t
        G_losses['L1_source'] = (1-self.opt.t_s_ratio) * loss_app_gen_s
        G_losses['VGG_source'] = (1-self.opt.t_s_ratio) * (loss_style_gen_s + loss_content_gen_s)
        G_losses['Face_source'] = loss_face_s

        sample_src = torch.cat([src_image.cpu(), src_map[:, :3].cpu(), src_fake.detach().cpu(), src_image.cpu()], 3)
        sample_tgt = torch.cat([src_image.cpu(), tgt_map[:, :3].cpu(), tgt_fake.detach().cpu(), tgt_image.cpu()], 3)
        sample = torch.cat([sample_src, sample_tgt], 2)

        return G_losses, sample
    def backward_D_basic(self, real, fake):
        # Real
        D_real = self.netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = self.netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)

        # gradient penalty for wgan-gp
        gradient_penalty = 0
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = loss.cal_gradient_penalty(self.netD, real, fake.detach())

        return D_real_loss, D_fake_loss, gradient_penalty
    def compute_discriminator_loss(self,
                                   src_image, src_map,
                                   tgt_image, tgt_map):
        self.netG.train()
        self.netD.train()
        D_losses = {}
        with torch.no_grad():
            _, tgt_fake = self.netG(src_image, src_map, tgt_map)

        D_real_loss, D_fake_loss, gradient_penalty = self.backward_D_basic(tgt_image, tgt_fake)
        D_losses['Real_loss'] = D_real_loss * 0.5
        D_losses['Fake_loss'] = D_fake_loss * 0.5
        if self.opt.gan_mode == 'wgangp':
            D_losses['WGAN_penalty'] = gradient_penalty

        return D_losses

    def generate_fake(self,
                      src_image, src_map,
                      tgt_image, tgt_map,
                      is_train=True):

        b, c, h, w = src_image.size()

        gt_tgts = []
        gt_srcs = []
        fake_tgts = []
        fake_srcs = []

        xt = torch.randn(b, 3, h, w).to(src_image.device)
        for step in range(1, self.opt.step_size + 1) :
            gt_src = self.get_groundtruth(src_image, step)
            gt_tgt = self.get_groundtruth(tgt_image, step)

            xt, xs = self.netG(src_map,
                               tgt_map,
                               gt_src,
                               xt.detach(),
                               step)

            gt_tgts.append(gt_tgt)
            gt_srcs.append(gt_src)

            fake_tgts.append(xt)
            fake_srcs.append(xs)

        vis_tgt = self.get_vis(gt_tgts, fake_tgts)
        vis_src = self.get_vis(gt_srcs, fake_srcs)



        return (gt_tgts, gt_srcs), (fake_tgts, fake_srcs), (vis_tgt, vis_src)

    def get_vis(self, true_list, fake_list):
        gt_vis = torch.cat(true_list, -1)
        fake_vis = torch.cat(fake_list, -1).detach()
        vis = torch.cat([gt_vis, fake_vis], -2)

        return vis

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def get_groundtruth(self, img_tensor, step):

        total_step = self.opt.step_size
        if step == total_step : return img_tensor
        dstep = 255 // total_step
        destory_term = 255 - dstep * step

        img_tensor_denorm = (img_tensor + 1) / 2 * 255

        ground_truth = img_tensor_denorm // destory_term * destory_term

        ground_truth = (ground_truth / 255 * 2) - 1

        return ground_truth