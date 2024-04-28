import time
import torch

from base_dataset import create_dataset

from training_visualizer import Visualizer
from base_network import BaseOptions, create_model

class TrainModel(BaseOptions):
    

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # visualization parameters
        parser.add_argument('--no_tensorboard', action='store_true')
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--display_epoch_freq', type=int, default=5)
        parser.add_argument('--display_ncols', type=int, default=3)
        
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=500)
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true')
        parser.add_argument('--continue_train', action='store_true' )
        parser.add_argument('--epoch_count', type=int, default=1 )
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs_keep', type=int, default=50)
        parser.add_argument('--n_epochs_decay', type=int, default=50)
        # for ADAM optim
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--pool_size', type=int, default=50)
        parser.add_argument('--lr_policy', type=str, default='linear')
        parser.add_argument('--lr_decay_iters', type=int, default=50)

        self.isTrain = True
        return parser

if __name__ == '__main__':
    train_start_time = time.time() 
    config = TrainModel().parse()   
    dataset = create_dataset(config)  
    dataset_size = len(dataset)    
    print('The number of training images = %d' % dataset_size)
    visualizer = Visualizer(config)   

    model = create_model(config)     
    model.setup(config)               
    model.train()
    
    total_iters = 0                
    for epoch in range(config.epoch_count, config.n_epochs_keep + config.n_epochs_decay + 1):    
        epoch_start_time = time.time()  
        iter_data_time = time.time()    
        epoch_iter = 0                 
        
        for i, data in enumerate(dataset):  
            iter_start_time = time.time()  
            if total_iters % config.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter += 1

            model.set_input(data)         
            model.optimize_parameters()

            if total_iters % config.print_freq == 0:    # print & plot training losses 
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / config.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % config.save_latest_freq == 0:  
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if config.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        
        if epoch % config.display_epoch_freq == 0:
            # display images on tensorboard   
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters)
        if epoch % config.save_epoch_freq == 0:              
            print('saving the model at the end of epoch %d, total iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        model.update_learning_rate()    

        message = 'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, config.n_epochs_keep + config.n_epochs_decay, time.time() - epoch_start_time)
        print(message)
        with open(visualizer.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  
    
    train_end_message = 'End of training \t Time Taken: %.3f hours' % ((time.time() - train_start_time)/3600.0)
    print(train_end_message)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s\n' % train_end_message)  
