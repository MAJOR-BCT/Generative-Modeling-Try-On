import time
import torch

from base_dataset import create_dataset

from training_visualizer import Visualizer
from base_network import BaseOptions, create_model

class TrainModel(BaseOptions):
    

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # visualization parameters
        parser.add_argument('--no_tensorboard', action='store_true', help='if specifid, results will not be displayed on tensorboard')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--display_epoch_freq', type=int, default=5, help='frequency of showing training results on screen at the end of epoches')
        parser.add_argument('--display_ncols', type=int, default=3, help='if positive, display all images in a single tensorboard panel with certain number of images per row.')
        
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results at the end of iteritons')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count (>= 1), we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs_keep', type=int, default=50, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=50, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser

if __name__ == '__main__':
    train_start_time = time.time() # timer for training
    config = TrainModel().parse()   # get training options
    dataset = create_dataset(config)  # create a dataset given config.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    visualizer = Visualizer(config)   # create a visualizer that display/save images and plots

    model = create_model(config)      # create a model given config.model and other options
    model.setup(config)               # regular setup: load and print networks; create schedulers (if any)
    model.train()
    
    total_iters = 0                # the total number of training iterations
    for epoch in range(config.epoch_count, config.n_epochs_keep + config.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % config.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter += 1

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()

            if total_iters % config.print_freq == 0:    # print & plot training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / config.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % config.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if config.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        
        if epoch % config.display_epoch_freq == 0:   # display images on tensorboard
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters)
        if epoch % config.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, total iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        model.update_learning_rate()    # update learning rates at the end of every epoch.

        message = 'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, config.n_epochs_keep + config.n_epochs_decay, time.time() - epoch_start_time)
        print(message)
        with open(visualizer.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
    
    train_end_message = 'End of training \t Time Taken: %.3f hours' % ((time.time() - train_start_time)/3600.0)
    print(train_end_message)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s\n' % train_end_message)  # save the message
