import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

import torch.nn.functional as F

import os
import time 

import numpy as np

from visualization import board_add_image, board_add_images, save_images

class SkipConnectionBlock(nn.Module):
    def __init__(self, out_channels, in_channels_inner, in_channels_outer=None,
                 submodule=None, is_outermost=False, is_innermost=False, normalization_layer=nn.BatchNorm2d, use_dropout=False):
        super(SkipConnectionBlock, self).__init__()
        self.is_outermost = is_outermost
        use_bias = normalization_layer == nn.InstanceNorm2d

        if in_channels_outer is None:
            in_channels_outer = out_channels

        down_convolution = nn.Conv2d(in_channels_outer, in_channels_inner, kernel_size=4,
                                     stride=2, padding=1, bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = normalization_layer(in_channels_inner)
        up_relu = nn.ReLU(True)
        up_norm = normalization_layer(out_channels)

        if is_outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            up_convolution = nn.Conv2d(in_channels_inner * 2, out_channels,
                                       kernel_size=3, stride=1, padding=1, bias=use_bias)
            down_layers = [down_convolution]
            up_layers = [up_relu, upsample, up_convolution, up_norm]
            model_layers = down_layers + [submodule] + up_layers
        elif is_innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            up_convolution = nn.Conv2d(in_channels_inner, out_channels, kernel_size=3,
                                       stride=1, padding=1, bias=use_bias)
            down_layers = [down_relu, down_convolution]
            up_layers = [up_relu, upsample, up_convolution, up_norm]
            model_layers = down_layers + up_layers
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            up_convolution = nn.Conv2d(in_channels_inner*2, out_channels, kernel_size=3,
                                       stride=1, padding=1, bias=use_bias)
            down_layers = [down_relu, down_convolution, down_norm]
            up_layers = [up_relu, upsample, up_convolution, up_norm]

            if use_dropout:
                model_layers = down_layers + [submodule] + up_layers + [nn.Dropout(0.5)]
            else:
                model_layers = down_layers + [submodule] + up_layers

        self.model = nn.Sequential(*model_layers)

    def forward(self, x):
        #print(x.shape)
        if self.is_outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        


class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_downs, base_channels=64,
                 normalization_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetGenerator, self).__init__()
        # Construct U-Net structure
        unet_block = SkipConnectionBlock(
            out_channels=base_channels * 8, in_channels_inner=base_channels * 8, in_channels_outer=None, submodule=None, normalization_layer=normalization_layer, is_innermost=True)
        for i in range(num_downs - 5):
            unet_block = SkipConnectionBlock(
                out_channels=base_channels * 8,in_channels_inner= base_channels * 8, in_channels_outer=None, submodule=unet_block, normalization_layer=normalization_layer, use_dropout=use_dropout)
        unet_block = SkipConnectionBlock(
             out_channels=base_channels * 4,in_channels_inner= base_channels * 8, in_channels_outer=None, submodule=unet_block, normalization_layer=normalization_layer)
        unet_block = SkipConnectionBlock(
            out_channels=base_channels * 2,in_channels_inner=  base_channels * 4,  in_channels_outer=None, submodule=unet_block, normalization_layer=normalization_layer)
        unet_block = SkipConnectionBlock(
            out_channels=base_channels, in_channels_inner= base_channels * 2,  in_channels_outer=None, submodule=unet_block, normalization_layer=normalization_layer)
        unet_block = SkipConnectionBlock(
            out_channels=out_channels, in_channels_inner= base_channels, in_channels_outer=in_channels, submodule=unet_block, is_outermost=True, normalization_layer=normalization_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
    
def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()

def train_sam(config, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - config.keep_step) / float(config.decay_step + 1))

    for step in range(config.keep_step + config.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        pcm = inputs['parse_cloth_mask'].cuda()

        
        outputs = model(torch.cat([agnostic, c, cm], 1))  


        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, pcm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]  

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        
        loss_mask = criterionMask(m_composite, pcm)  
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % config.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step+1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % config.save_count == 0:
            save_checkpoint(model, os.path.join(
                config.checkpoint_dir, config.name, 'step_%06d.pth' % (step+1)))



def test_sam(config, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(config.checkpoint)
    
    
    save_dir = os.path.join(config.result_dir, config.name, config.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    p_rendered_dir = os.path.join(save_dir, 'p_rendered')
    if not os.path.exists(p_rendered_dir):
        os.makedirs(p_rendered_dir)
    m_composite_dir = os.path.join(save_dir, 'm_composite')
    if not os.path.exists(m_composite_dir):
        os.makedirs(m_composite_dir)
    im_pose_dir = os.path.join(save_dir, 'im_pose')
    if not os.path.exists(im_pose_dir):
        os.makedirs(im_pose_dir)
    shape_dir = os.path.join(save_dir, 'shape')
    if not os.path.exists(shape_dir):
        os.makedirs(shape_dir)
    im_h_dir = os.path.join(save_dir, 'im_h')
    if not os.path.exists(im_h_dir):
        os.makedirs(im_h_dir)  # for test data

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        
        outputs = model(torch.cat([agnostic, c, cm], 1))  
        p_rendered, m_composite = torch.split(outputs, 3, 1)


        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)

        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, 2*cm-1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        save_images(im_h, im_names, im_h_dir)
        save_images(shape, im_names, shape_dir)
        save_images(im_pose, im_names, im_pose_dir)
        save_images(m_composite, im_names, m_composite_dir)
        save_images(p_rendered, im_names, p_rendered_dir)  # For test data

        if (step+1) % config.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)