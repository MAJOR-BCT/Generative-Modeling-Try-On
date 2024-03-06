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


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class FeatureExtraction(nn.Module):
    
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,  use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2**i * ngf if 2**i * ngf < 512 else 512
            out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        # init_weights(self.model, init_type='normal

    def forward(self, x):
        return self.model(x)

class DepthDec(nn.Module):
    """
    size: 32-32-32-64-128-256-512
    channel: in_nc-512-512-256-128-64-out_nc
    """
    def __init__(self, in_nc=1024, out_nc=2):
        super(DepthDec, self).__init__()
        self.upconv52 = nn.Conv2d(in_nc, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm52 = nn.InstanceNorm2d(512)
        self.upconv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.upnorm51 = nn.InstanceNorm2d(512)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upnorm4 = nn.InstanceNorm2d(256)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upnorm3 = nn.InstanceNorm2d(128)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upnorm2 = nn.InstanceNorm2d(64)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv1 = nn.Conv2d(64, out_nc, kernel_size=3, stride=1, padding=1)
        self.upnorm1 = nn.InstanceNorm2d(out_nc)

    def forward(self, x):
        x52up = F.relu_(self.upnorm52(self.upconv52(x)))	        
        x51up = F.relu_(self.upnorm51(self.upconv51(x52up)))	        
        x4up = F.relu_(self.upnorm4(self.upconv4(self.upsample4(x51up))))	        
        x3up = F.relu_(self.upnorm3(self.upconv3(self.upsample3(x4up))))	        
        x2up = F.relu_(self.upnorm2(self.upconv2(self.upsample2(x3up))))	        
        x1up = self.upnorm1(self.upconv1(self.upsample1(x2up)))	        

        return x1up

class IDE(nn.Module):
    def __init__(self, input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=5, 
                add_tps=True, add_depth=True, add_segmt=True, norm_layer=nn.InstanceNorm2d, use_dropout=False, device='cpu'):
        super(IDE, self).__init__()
        
        self.add_depth = add_depth
        
        self.extractionA = FeatureExtraction(input_nc_A, ngf, n_layers, norm_layer, use_dropout)
        self.extractionB = FeatureExtraction(input_nc_B, ngf, n_layers, norm_layer, use_dropout)
        if self.add_depth:
            self.depth_dec = DepthDec(in_nc=1024)

    def forward(self, inputA, inputB):
        """ 
            input A: agnostic (batch_size,12,512,320)
            input B: flat cloth mask(batch_size,1,512,320)
        """
        output = {'theta_tps':None, 'grid_tps':None, 'depth':None, 'segmt':None}
        featureA = self.extractionA(inputA) # featureA: size (batch_size,512,32,20)
        featureB = self.extractionB(inputB) # featureB: size (batch_size,512,32,20)
        if self.add_depth or self.add_segmt:
            featureAB = torch.cat([featureA, featureB], 1) # input for DepthDec and SegmtDec: (batch_size,1024,32,20)
            if self.add_depth:
                depth_pred = self.depth_dec(featureAB)
                output['depth'] = depth_pred
        
        return output


def define_IDE(input_nc_A=29, input_nc_B=3, ngf=64, n_layers=3, img_height=512, img_width=320, grid_size=5, add_tps=True, 
            add_depth=True, add_segmt=True, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    norm_layer = get_norm_layer(norm_type=norm)
    device = f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 else 'cpu'
    net = IDE(input_nc_A, input_nc_B, ngf, n_layers, img_height, img_width, grid_size, add_tps, add_depth, add_segmt, norm_layer, use_dropout, device)
    
    return init_net(net, init_type, init_gain, gpu_ids)

#Initial Depth Estimator Model 
class IDEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        
        # parser.set_defaults(dataset_mode='unaligned')  # You can rewrite default values for this model.
        parser.add_argument('--add_tps', action='store_true', default = True, help='if specified, add tps transformation')
        parser.add_argument('--add_depth', action='store_true', default = True, help='if specified, add depth decoder')
        # parser.add_argument('--use_featB_for_depth', action='store_true', help='if specified, use featB for depth')
        parser.add_argument('--add_segmt', action='store_true', default = True, help='if specified, add segmentation decoder')        
        parser.add_argument('--grid_size', type=int, default=3, help='size of the grid used to estimate TPS params.')                
        parser.add_argument('--input_nc_A', type=int, default=29, help='input nc of feature extraction A [11 for roi agnostic type | 29 for full agnostic type]')
        parser.add_argument('--input_nc_B', type=int, default=3, help='input nc of feature extraction B')
        parser.add_argument('--n_layers_feat_extract', type=int, default=3, help='# layers in feater extraction of IDE')
        parser.add_argument('--add_theta_loss', action='store_true', help='if specified, add theta loss')
        parser.add_argument('--add_grid_loss', action='store_true', help='if specified, add grid loss') 
        # parser.add_argument('--add_inlier_loss', action='store_true', help='if specified, add inlier loss') 
        # parser.add_argument('--dilation_filter', type=int, default=0, help='type of dilation filter [0: no filter | 1: 4-neighs | 2: 8-neighs]')
        parser.add_argument('--lambda_depth', type=float, default=1.0, help='weight of warp loss')
        # parser.add_argument('--lambda_inlier', action='store_true', help='weight of grid loss')
        
        return parser

    def __init__(self, config):
        
        BaseModel.__init__(self, config)  # call the initialization method of BaseModel
        
        self.add_depth = config.add_depth
        
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['IDE']
        if self.add_depth:
            self.loss_names.extend(['fdepth', 'bdepth'])
              
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['im_hhl','im_shape','pose']
        if self.add_depth:
            self.visual_names.extend(['fdepth_pred', 'fdepth_gt', 'fdepth_diff', 'bdepth_pred', 'bdepth_gt', 'bdepth_diff'])

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks. (you can use config.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.)
        self.model_names = ['IDE']
        
        # define networks; you can use config.isTrain to specify different behaviors for training and test.
        self.netIDE = define_IDE(config.input_nc_A, config.input_nc_B, config.ngf, config.n_layers_feat_extract, config.img_height, config.img_width, config.grid_size, config.add_tps, config.add_depth, config.add_segmt, config.norm, config.use_dropout, config.init_type, config.init_gain, self.gpu_ids)

        if self.isTrain:  # only defined during training time
            # define loss functions.
            self.criterionWarp = torch.nn.L1Loss()
            if self.add_depth:
                self.criterionDepth = torch.nn.L1Loss()

            # define and initialize configimizers. You can define one configimizer for each network.
            self.optimizer = torch.configim.Adam(self.netIDE.parameters(), lr=config.lr, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        
        self.im_name = input['im_name']                              
        self.c_name = input['c_name']                                                    
        self.agnostic = input['agnostic'].to(self.device)            
        self.c = input['cloth'].to(self.device)                      
        self.cm = input['cloth_mask'].to(self.device)                
        self.im = input['person'].to(self.device)                    
        self.im_shape = input['person_shape']                        
        self.im_hhl = input['head_hand_lower']                       
        self.pose = input['pose']                                    
        self.im_g = input['grid_image'].to(self.device)              
        self.im_c =  input['parse_cloth'].to(self.device)            
        self.segmt_gt = input['person_parse'].long().to(self.device) 
        if self.isTrain:
            self.fdepth_gt = input['person_fdepth'].to(self.device)      
            self.bdepth_gt = input['person_bdepth'].to(self.device)      
        
    def forward(self):
        # print(f'shape is {self.agnostic.shape}, {self.c.shape}')
        self.output = self.netIDE(self.agnostic, self.c)

        
        if self.output['depth'] is not None:
            self.fdepth_pred, self.bdepth_pred = torch.split(self.output['depth'], [1,1], 1)
            self.fdepth_pred = torch.tanh(self.fdepth_pred)
            self.bdepth_pred = torch.tanh(self.bdepth_pred)
            if self.isTrain:
                self.fdepth_diff = self.fdepth_pred - self.fdepth_gt # just for visual
                self.bdepth_diff = self.bdepth_pred - self.bdepth_gt # fust for visual
        
    def backward(self):
        self.loss_IDE = torch.tensor(0.0, requires_grad=True).to(self.device)
     
        if self.add_depth:
            self.loss_fdepth = self.config.lambda_depth * self.criterionDepth(self.fdepth_pred, self.fdepth_gt)
            self.loss_bdepth = self.config.lambda_depth * self.criterionDepth(self.bdepth_pred, self.bdepth_gt)
            self.loss_IDE += (self.loss_fdepth + self.loss_bdepth)

        self.loss_IDE.backward()
    
    def configimize_parameters(self):
        self.forward() 
        self.optimizer.zero_grad()  
        self.backward()              
        self.optimizer.step()       

    def compute_visuals(self):
        # convert raw scores to 1-channel mask that can be visualized
        # segmt_pred: size (batch_size, 1, 512, 320)
        if self.add_segmt:
            self.segmt_pred_vis = util.decode_labels(self.segmt_pred_argmax)
            self.segmt_gt_vis = util.decode_labels(self.segmt_gt)
