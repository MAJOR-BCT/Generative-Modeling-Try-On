import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time

from datasets import PVTONDataset, PVTONDataLoader

from CWM_network import CWM, train_CWM
from SAM_network import UNetGenerator, train_sam

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="CWM")
    # parser.add_argument("--name", default="TOM")

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")

    parser.add_argument("--datamode", default="train")

    parser.add_argument("--stage", default="CWM")
    # parser.add_argument("--stage", default="SAM")

    parser.add_argument("--data_list", default="train_pairs.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    config = parser.parse_args()
    return config

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

def main():
    config = get_config()
    print(config)
    print("Start to train stage: %s, named: %s!" % (config.stage, config.name))

    # create dataset
    train_dataset = PVTONDataset(config)

    # create dataloader
    train_loader = PVTONDataLoader(config, train_dataset)

    # visualization
    if not os.path.exists(config.tensorboard_dir):
        os.makedirs(config.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(config.tensorboard_dir, config.name))

    # create model & train & save the final checkpoint
    if config.stage == 'CWM':
        print(f'Going inside {config.stage}')
        model = CWM(config)
        if not config.checkpoint == '' and os.path.exists(config.checkpoint):
            load_checkpoint(model, config.checkpoint)
        train_CWM(config, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            config.checkpoint_dir, config.name, 'CWM_final.pth'))
    elif config.stage == 'SAM':
        model = UNetGenerator(
            in_channels = 26, out_channels=4, num_downs=6, base_channels=64, normalization_layer=nn.InstanceNorm2d) 
        if not config.checkpoint == '' and os.path.exists(config.checkpoint):
            load_checkpoint(model, config.checkpoint)
        train_sam(config, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            config.checkpoint_dir, config.name, 'sam_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % config.stage)

    print('Finished training %s, named: %s!' % (config.stage, config.name))


if __name__ == "__main__":
    main()
