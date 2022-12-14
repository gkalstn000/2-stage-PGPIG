import models
import torch


class Trainer() :
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.opt = opt
        self.model = models.create_model(opt)
        self.model = torch.nn.DataParallel(self.model, device_ids = opt.gpu_ids)
        # self.model = self.model.cuda()
        self.generated = None

        if opt.isTrain :
            self.optimizer_G, self.optimizer_D = \
                self.model.module.create_optimizers(opt)
            self.old_lr = opt.lr


    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, fake_t, fake_s = self.model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = (fake_t, fake_s)

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.model.module.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
