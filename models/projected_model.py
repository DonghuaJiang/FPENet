#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from .base_model import BaseModel
from .fs_networks_fix import Generator_Adain_Upsample
from pg_modules.projected_discriminator import ProjectedDiscriminator


class SSIMLoss(nn.Module):
    def __init__(self, data_range=(-1, 1)):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range

    def normalize(self, x):
        data_min, data_max = self.data_range
        x_norm = (x - data_min) / (data_max - data_min)
        return x_norm

    def forward(self, x, y):
        x_norm = self.normalize(x)
        y_norm = self.normalize(y)
        ssim_loss = 1 - ssim(x_norm, y_norm, data_range=1, nonnegative_ssim=True)
        return ssim_loss


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
        self.isTrain = opt.isTrain

        # Generator network
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=opt.Gdeep)
        self.netG.to("cuda:0")

        # Id network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint
        self.netArc = self.netArc.to("cuda:0")
        self.netArc.eval()
        self.netArc.requires_grad_(False)
        if not self.isTrain:
            pretrained_path = opt.checkpoints_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            return

        # Feature discriminator
        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        self.netD.to("cuda:0")

        if self.isTrain:
            # define loss functions
            self.criterionFeat = nn.L1Loss().to("cuda:0")
            self.criterionId = nn.MSELoss().to("cuda:0")
            self.criterionRec = nn.L1Loss().to("cuda:0")

            # initialize optimizers
            params = list(self.netG.parameters())                                          # optimizer G
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

            params = list(self.netD.parameters())                                          # optimizer D
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99), eps=1e-8)

        # load networks
        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
        torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch)
        self.save_network(self.netD, 'D', which_epoch)
        self.save_optim(self.optimizer_G, 'G', which_epoch)
        self.save_optim(self.optimizer_D, 'D', which_epoch)
        '''if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)'''

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


