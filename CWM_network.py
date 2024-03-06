import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

import torch.nn.functional as F

import os
import time 

import numpy as np

from visualization import board_add_image, board_add_images, save_images

# for initializing weights based on the type of layer in the network 
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def initialize_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)

#depth is number of layers 
class ExtractFeatures(nn.Module):
    def __init__(self, input_channels, num_filters=64, depth=3, normalization_layer=nn.BatchNorm2d, dropout=False):
        super(ExtractFeatures, self).__init__()
        layers = [nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True), normalization_layer(num_filters)]
        for level in range(depth):
            current_filters = min(512, num_filters * 2**level)
            next_filters = min(512, num_filters * 2**(level + 1))
            layers.extend([
                nn.Conv2d(current_filters, next_filters, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                normalization_layer(next_filters)
            ])
        layers.extend([
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            normalization_layer(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ])

        self.layers = nn.Sequential(*layers)
        initialize_weights(self.layers, init_type='normal')

    def forward(self, input_tensor):
        # print(input_tensor.shape)
        return self.layers(input_tensor)

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) +
                         epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)

class CorrelateFeatures(nn.Module):
    def __init__(self):
        super(CorrelateFeatures, self).__init__()

    def forward(self, features_from_A, features_from_B):
        batch_size, channels, height, width = features_from_A.size()
        
        
        # print(f'\nSize of Input Features: {features_from_A.size()}\n')
        
        # Reshape the features to prepare for the correlation computation
        features_from_A = features_from_A.transpose(2, 3).contiguous().view(batch_size, channels, height * width)
        features_from_B = features_from_B.view(batch_size, channels, height * width).transpose(1, 2)
        
        # print(features_from_A.size())
        # print(features_from_B.size())
        
        # matrix multiplication to compute the correlation
        correlation = torch.bmm(features_from_B, features_from_A)
        
        
        # print(f'\nSize of Correlation Matrix: {correlation.size()}\n')
        
        # Reshape the correlation tensor desired output 
        correlation_tensor = correlation.view(batch_size, height, width, height * width).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class RegressFeatures(nn.Module):
    def __init__(self, in_channels=512, out_features=6, use_cuda=True):
        super(RegressFeatures, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64 * 4 * 3, out_features) # based on the network configuration
        self.activation = nn.Tanh()
        
        # for conversion to GPU compatible
        if use_cuda:
            self.to('cuda')

    def forward(self, x):
        
        # print(f'Input shape: {x.shape}')
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)  # Flattening
        x = self.fc(x)
        x = self.activation(x)
        return x


class TPSGrid(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TPSGrid, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(
            np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        
        
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size*grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(
                3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(
                3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()

    def forward(self, theta):
        # print("\nTheta shape:", theta.shape)
        warped_grid = self.apply_transformation(
            theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(
            Xmat-Xmat.transpose(0, 1), 2)+torch.pow(Ymat-Ymat.transpose(0, 1), 2)
        

        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat(
            (P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        
        
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]


        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand(
            (batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand(
            (batch_size, self.N, self.N)), Q_Y)
        


        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        

        # compute weights for affine part
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand(
            (batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand(
            (batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]

        
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(
            3).unsqueeze(4).expand(points[:, :, :, 0].size()+(1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(
            3).unsqueeze(4).expand(points[:, :, :, 1].size()+(1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation-P_X
            delta_Y = points_Y_for_summation-P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - \
                P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - \
                P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2)+torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand(
                (batch_size,)+points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand(
                (batch_size,)+points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)


class CWM(nn.Module):
    """ Geometric Matching Module
    """

    def __init__(self, config):
        super(CWM, self).__init__()
        self.extractionA = ExtractFeatures(
            input_channels=22, num_filters=64, depth=3, normalization_layer=nn.BatchNorm2d)
        self.extractionB = ExtractFeatures(
            input_channels=1, num_filters=64, depth=3, normalization_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = CorrelateFeatures()
        self.regression = RegressFeatures(
            in_channels=192, out_features=2*config.grid_size**2, use_cuda=True)
        self.gridGen = TPSGrid(
            config.fine_height, config.fine_width, use_cuda=True, grid_size=config.grid_size)

    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        # print(f'\n\nafter extration A x shape is {featureA.shape}\n\n')
        featureB = self.extractionB(inputB)
       # print(f'\n\nafter extration B x shape is {featureB.shape}\n\n')
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
      #  print(f'\n\nafter noemalization A and B x shape is{featureA.shape},  {featureB.shape}\n\n')
        
        correlation = self.correlation(featureA, featureB)
        
     #   print(f'\n\nafter Correlation x shape is {correlation.shape}\n\n')
        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        return grid, theta

#losses for the CWM, Grid loss (GIC)

#basic loss    
class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, x2):
        dt = torch.abs(x1 - x2)
        return dt


class DT2(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) +
                        torch.mul(y1 - y2, y1 - y2))
        return dt


class GicLoss(nn.Module):
    def __init__(self, config):
        super(GicLoss, self).__init__()
        self.dT = DT()
        self.config = config

    def forward(self, grid):
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]

        # for making sure the stretch of cloth doesn't deform the basic geometry of cloth 

        Gxcenter = Gx[:, 1:self.config.fine_height - 1, 1:self.config.fine_width - 1]
        Gxup = Gx[:, 0:self.config.fine_height - 2, 1:self.config.fine_width - 1]
        Gxdown = Gx[:, 2:self.config.fine_height, 1:self.config.fine_width - 1]
        Gxleft = Gx[:, 1:self.config.fine_height - 1, 0:self.config.fine_width - 2]
        Gxright = Gx[:, 1:self.config.fine_height - 1, 2:self.config.fine_width]

        Gycenter = Gy[:, 1:self.config.fine_height - 1, 1:self.config.fine_width - 1]
        Gyup = Gy[:, 0:self.config.fine_height - 2, 1:self.config.fine_width - 1]
        Gydown = Gy[:, 2:self.config.fine_height, 1:self.config.fine_width - 1]
        Gyleft = Gy[:, 1:self.config.fine_height - 1, 0:self.config.fine_width - 2]
        Gyright = Gy[:, 1:self.config.fine_height - 1, 2:self.config.fine_width]

        dtleft = self.dT(Gxleft, Gxcenter)
        dtright = self.dT(Gxright, Gxcenter)
        dtup = self.dT(Gyup, Gycenter)
        dtdown = self.dT(Gydown, Gycenter)

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))

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

def train_CWM(config, train_loader, model, board):
    model.cuda()
    model.train()
    print(f'\n\nMODEL INCOMING')
    print(model)
    
    print(f'\n\nMODEL Sakkyo')
    # criterion

    #for the warping cloth
    criterionL1 = nn.L1Loss()

    #for the grid warping loss
    gicloss = GicLoss(config)


    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - config.keep_step) / float(config.decay_step + 1))

    for step in range(config.keep_step + config.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        grid, theta = model(agnostic, cm)    # can be added c too for new training
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

        # loss for warped cloth
        Lwarp = criterionL1(warped_cloth, im_c)    
       
        
        # grid regularization loss
        Lgic = gicloss(grid)
        
        Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])

        # for getting appropriate scale of the loss, since metric is comparatively too small to bring improvements
        loss = Lwarp + 40 * Lgic    # total CWM loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % config.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('loss', loss.item(), step+1)
            board.add_scalar('40*Lgic', (40*Lgic).item(), step+1)
            board.add_scalar('Lwarp', Lwarp.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f, (40*Lgic): %.8f, Lwarp: %.6f' %
                  (step+1, t, loss.item(), (40*Lgic).item(), Lwarp.item()), flush=True)

        if (step+1) % config.save_count == 0:
            save_checkpoint(model, os.path.join(
                config.checkpoint_dir, config.name, 'step_%06d.pth' % (step+1)))
            


def test_cwm(config, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(config.checkpoint)
    name = config.name
    save_dir = os.path.join(config.result_dir, name, config.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    result_dir1 = os.path.join(save_dir, 'result_dir')
    if not os.path.exists(result_dir1):
        os.makedirs(result_dir1)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    if not os.path.exists(overlayed_TPS_dir):
        os.makedirs(overlayed_TPS_dir)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    if not os.path.exists(warped_grid_dir):
        os.makedirs(warped_grid_dir)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        shape_ori = inputs['shape_ori']  


        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        overlay = 0.7 * warped_cloth + 0.3 * im

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

        
        print(warped_cloth, im_names, warp_cloth_dir)
        
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                    0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)

        if (step+1) % config.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)