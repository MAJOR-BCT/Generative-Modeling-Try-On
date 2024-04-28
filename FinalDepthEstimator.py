import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .base_network import BaseModel

import sys
import functools
import random

from util import util
import PIL.Image as Image
import numpy as np

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return torch.nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    # Initialize network weights.
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

#for basis on num of GPUs
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class FDE(nn.Module):
    def __init__(self, in_channel, out_channel, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(FDE, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        
        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            norm_layer(self.ngf * 16)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 6, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input_data, inter_mode='bilinear'):
        x0 = self.l0(input_data)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u

def define_FDE(input_nc=4, output_nc=2, ngf=32, norm='instanc', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)

    net = FDE(input_nc, output_nc, ngf, norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)

class Sobel(nn.Module):
    """ Soebl operator to calculate depth grad. """

    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out
    

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, depth_pred, depth_gt):
        loss_depth = torch.log(torch.abs(depth_pred - depth_gt) + 1).mean()
        
        return loss_depth

class DepthGradLoss(nn.Module):
    def __init__(self):
        super(DepthGradLoss, self).__init__()

    def forward(self, depth_grad_pred, depth_grad_gt):
        depth_grad_gt_dx = depth_grad_gt[:, 0, :, :].unsqueeze(1)
        depth_grad_gt_dy = depth_grad_gt[:, 1, :, :].unsqueeze(1)
        depth_grad_pred_dx = depth_grad_pred[:, 0, :, :].unsqueeze(1)
        depth_grad_pred_dy = depth_grad_pred[:, 1, :, :].unsqueeze(1)
        
        loss_dx = torch.log(torch.abs(depth_grad_pred_dx - depth_grad_gt_dx) + 1).mean()
        loss_dy = torch.log(torch.abs(depth_grad_pred_dy - depth_grad_gt_dy) + 1).mean()
        
        loss_grad = loss_dx + loss_dy
    
        return loss_grad


# Final Depth Estimator Model
class FDEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
    
        parser.add_argument('--warproot', type=str, default='test_pairs')
        parser.add_argument('--input_gradient', action='store_true', default=True)
        parser.add_argument('--input_nc', type=int, default=8)
        parser.add_argument('--output_nc', type=int, default=2)
        parser.set_defaults(display_ncols=2)  
        parser.set_defaults(netD='basic')
        parser.add_argument('--input_nc_D', type=int, default=4)
        parser.add_argument('--add_grad_loss', action='store_true', default=True, )
        parser.add_argument('--lambda_depth', type=float, default=1.0)
        parser.add_argument('--lambda_grad', type=float, default=1.0)
        
        return parser

    def __init__(self, config):
        
        BaseModel.__init__(self, config)  # call the initialization method of BaseModel
        
        #taking the sobel of images as input 

        self.input_gradient = config.input_gradient
        
        if self.use_grad_loss:
            self.compute_grad = Sobel().to(self.device)

        self.loss_names = ['FDE', 'FDEpth', 'bdepth']
        if self.use_grad_loss:
            self.loss_names.extend(['fgrad', 'bgrad'])
        
        #  images to save and display
        self.visual_names = ['c', 'im_hhl', 'image_f_depth_initial', 'image_b_depth_initial']
        if self.input_gradient:
            self.visual_names.extend(['imhal_sobelx', 'imhal_sobely', 'c_sobelx', 'c_sobely'])
        self.visual_names.extend(['image_f_depth_pred','image_b_depth_pred', 'image_f_depth_diff', 'image_b_depth_diff'])
        if self.use_grad_loss:
            self.visual_names.extend(['fgrad_pred_x','fgrad_x', 'fgrad_pred_y', 'fgrad_y', 'fgrad_x_diff', 'fgrad_y_diff', 'bgrad_pred_x', 'bgrad_x', 'bgrad_pred_y', 'bgrad_y',  'bgrad_x_diff', 'bgrad_y_diff'])
        if self.use_normal_loss or self.use_gan_loss:
            self.visual_names.extend(['imfn_pred', 'imfn', 'imbn_pred', 'imbn', 'imfn_diff', 'imbn_diff'])
        
        self.model_names = ['FDE']
        

        # define networks; you can use config.isTrain to specify different behaviors for training and test.
        if self.input_gradient:
            config.input_nc += 4
        self.netFDE = define_FDE(config.input_nc, config.output_nc, config.ngf, config.norm, config.init_type, config.init_gain, self.gpu_ids)
        
        if self.isTrain:  # only defined during training time
            # define loss functions
            self.criterionDepth = DepthLoss().to(self.device)
            if self.use_grad_loss:
                self.criterionGrad = DepthGradLoss().to(self.device)
            
            # define and initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netFDE.parameters(), lr=config.lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G]
            
        
    def set_input(self, input):
        
        self.im_name = input['im_name']                             
        self.c_name = input['c_name']                              
        self.c = input['cloth'].to(self.device)                   
        self.im_hhl = input['head_hand_lower'].to(self.device)      
        self.image_f_depth_initial = input['initial_fdepth'].to(self.device) 
        self.image_b_depth_initial = input['initial_bdepth'].to(self.device) 
        if self.input_gradient:
            self.imhal_sobelx = input['imhal_sobelx'].to(self.device)  
            self.imhal_sobely = input['imhal_sobely'].to(self.device) 
            self.c_sobelx = input['cloth_sobelx'].to(self.device)     
            self.c_sobely = input['cloth_sobely'].to(self.device)     
        if self.isTrain:

            #with ground truth
            self.image_f_depth = input['person_FDEpth'].to(self.device)          
            self.image_b_depth = input['person_bdepth'].to(self.device)          
            if self.use_grad_loss:
                self.fgrad = self.compute_grad(self.image_f_depth) 
                self.bgrad = self.compute_grad(self.image_b_depth) 

    #image_f_depth is forward depth
    def forward(self):
        
        if self.input_gradient:
            self.input = torch.cat([self.image_f_depth_initial, self.image_b_depth_initial, self.c, self.im_hhl, self.c_sobelx, self.c_sobely, self.imhal_sobelx, self.imhal_sobely], 1)
        else:
            self.input = torch.cat([self.image_f_depth_initial, self.image_b_depth_initial, self.c, self.im_hhl], 1)
        outputs = self.netFDE(self.input)
        self.image_f_depth_pred, self.image_b_depth_pred= torch.split(outputs, [1,1], 1)
        self.image_f_depth_pred = torch.tanh(self.image_f_depth_pred)
        self.image_b_depth_pred = torch.tanh(self.image_b_depth_pred)

        if self.use_grad_loss:
            self.fgrad_pred = self.compute_grad(self.image_f_depth_pred)
            self.bgrad_pred = self.compute_grad(self.image_b_depth_pred)


    def backward_G(self):
        # for the loss of grad loss for depth
        self.loss_FDEpth = self.config.lambda_depth * self.criterionDepth(self.image_f_depth_pred, self.image_f_depth)
        self.loss_bdepth = self.config.lambda_depth * self.criterionDepth(self.image_b_depth_pred, self.image_b_depth)
        self.loss_FDE = self.loss_FDEpth + self.loss_bdepth

        if self.use_grad_loss:
            self.loss_fgrad = self.config.lambda_grad * self.criterionGrad(self.fgrad_pred, self.fgrad)
            self.loss_bgrad = self.config.lambda_grad * self.criterionGrad(self.bgrad_pred, self.bgrad)
            self.loss_FDE += self.loss_fgrad + self.loss_bgrad

        self.loss_FDE.backward()
    

    def optimize_parameters(self):
        
        self.forward() 
        self.optimizer_G.zero_grad() 
        self.backward_G()            
        self.optimizer_G.step()       
    
    def compute_visuals(self):
        
        self.image_f_depth_diff = self.image_f_depth_pred - self.image_f_depth
        self.image_b_depth_diff = self.image_b_depth_pred - self.image_b_depth
        if self.use_grad_loss:
            self.fgrad_pred_x = self.fgrad_pred[:,0,:,:].unsqueeze(1)
            self.fgrad_pred_y = self.fgrad_pred[:,1,:,:].unsqueeze(1)
            self.bgrad_pred_x = self.bgrad_pred[:,0,:,:].unsqueeze(1)
            self.bgrad_pred_y = self.bgrad_pred[:,1,:,:].unsqueeze(1)
            self.fgrad_x = self.fgrad[:,0,:,:].unsqueeze(1)
            self.fgrad_y = self.fgrad[:,1,:,:].unsqueeze(1)
            self.bgrad_x = self.bgrad[:,0,:,:].unsqueeze(1)
            self.bgrad_y = self.bgrad[:,1,:,:].unsqueeze(1)
            self.fgrad_x_diff = self.fgrad_pred_x - self.fgrad_x
            self.fgrad_y_diff = self.fgrad_pred_y - self.fgrad_y
            self.bgrad_x_diff = self.bgrad_pred_x - self.bgrad_x
            self.bgrad_y_diff = self.bgrad_pred_y - self.bgrad_y