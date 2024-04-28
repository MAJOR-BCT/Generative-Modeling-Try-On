import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import util

class Visualizer():

    def __init__(self, config):
        
        self.config = config  
        self.use_tensorboard = not config.no_tensorboard
        self.win_size = config.display_winsize
        self.name = config.name
        self.saved = False
        self.ncols = config.display_ncols


        if self.use_tensorboard: 
            tensorboard_dir = os.path.join(config.checkpoints_dir, config.datamode, config.name, 'tensorboard')
            print('create tensorboard directory %s...' % tensorboard_dir)
            util.mkdir(tensorboard_dir)
            self.board = SummaryWriter(log_dir = tensorboard_dir)

        # logging file 
        self.log_name = os.path.join(config.checkpoints_dir, config.datamode, config.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch, total_iters):
        
        if self.use_tensorboard:# display images in tensorboard
            img_tensors = []
            img_tensors_list = []
            for visual_tensors in visuals.values():
                if len(img_tensors) < self.ncols:
                    img_tensors.append(visual_tensors)
                else:
                    img_tensors_list.append(img_tensors)
                    img_tensors = []
                    img_tensors.append(visual_tensors)
            img_tensors_list.append(img_tensors)

            self.board_add_images(self.board, 'Visuals', img_tensors_list, total_iters)

    def tensor_for_board(self, img_tensor):
        # map into [0,1]
        tensor = (img_tensor.clone()+1) * 0.5
        tensor.cpu().clamp(0,1)

        if tensor.size(1) == 1:
            tensor = tensor.repeat(1,3,1,1)

        return tensor

    def tensor_list_for_board(self, img_tensors_list):
        grid_h = len(img_tensors_list)
        grid_w = max(len(img_tensors)  for img_tensors in img_tensors_list)

        batch_size, channel, height, width = self.tensor_for_board(img_tensors_list[0][0]).size()
        canvas_h = grid_h * height
        canvas_w = grid_w * width
        canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
        for i, img_tensors in enumerate(img_tensors_list):
            for j, img_tensor in enumerate(img_tensors):
                offset_h = i * height
                offset_w = j * width
                tensor = self.tensor_for_board(img_tensor)
                canvas[:, :, offset_h : offset_h + height, offset_w : offset_w + width].copy_(tensor)

        return canvas
    
    def board_add_image(self, board, tag_name, img_tensor, step_count):
        tensor = self.tensor_for_board(img_tensor)

        for i, img in enumerate(tensor):
            self.board.add_image('%s/%03d' % (tag_name, i), img, step_count)

    def board_add_images(self, board, tag_name, img_tensors_list, step_count):
        tensor = self.tensor_list_for_board(img_tensors_list)

        for i, img in enumerate(tensor):
            self.board.add_image('%s/%03d' % (tag_name, i), img, step_count)

    def plot_current_losses(self, total_iters, losses):
        
        if self.use_tensorboard:
            for loss_name, loss_value in losses.items():
                self.board.add_scalar('Loss/'+loss_name, loss_value, total_iters)
            
        else:
            print('Plot failed')

    # losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  