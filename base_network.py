import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from torch.optim import lr_scheduler
import importlib

from base_dataset import get_option_setter as data_get_option_setter
import util 
import argparse 
import os 

def models_get_option_setter(model_name):
    
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options

class BaseOptions():

    def __init__(self):
        
        self.initialized = False

    def initialize(self, parser):
       
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='depth_example',help='path to dataset')
        parser.add_argument('--datamode', type=str, default='aligned',help='dataset mode for MTM [unaligned | aligned]')
        parser.add_argument('--datalist', type=str, default='test_pairs', help='depth data list [train_pairs | test_pairs | test_score]')
        parser.add_argument('--name', type=str, default='MTM', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0,1,2, currently we only support single GPU, use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='pretrained', help='models are saved to opt.checkpoints_dir/opt.datamode/opt.name')
        # model parameters
        parser.add_argument('--model', type=str, default='MTM', help='chooses which model to use. [MTM | DRM | TFM]')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel | spectral_norm]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        # dataset parameters
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--img_height', type=int, default=512, help='img height in the dataset')
        parser.add_argument('--img_width', type=int, default=320, help='img width in the dataset')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size (suggested 8 for MTM, 4 for TFM)')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data (suggested 8 for MTM, 1 for TFM)')
        parser.add_argument('--no_pin_memory', action='store_true', help='if specified, not pin data to memory')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--display_winsize', type=int, default=512, help='display window size for HTML')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models_get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.datamode
        dataset_option_setter = data_get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args() # If work in IPython (Jupyter Notebook), you need to pass an empty string: parser.parse_args("")

    def print_options(self, opt):
    
        message = ''
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.datamode, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt



def get_scheduler(optimizer, config):
    
    if config.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.epoch_count - config.n_epochs_keep) / float(config.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs_keep, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler


class BaseModel(ABC):
    

    def __init__(self, config):
        
        self.config = config
        self.gpu_ids = config.gpu_ids
        self.isTrain = config.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(config.checkpoints_dir, config.datamode, config.name)  # save all the checkpoints to save_dir.
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  

    @staticmethod
    def modify_commandline_options(parser, is_train):
        
        return parser

    def setup(self, config):
        
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, config) for optimizer in self.optimizers]
        if not self.isTrain or config.continue_train:
            load_suffix = 'iter_%d' % config.load_iter if config.load_iter > 0 else config.epoch
            self.load_networks(load_suffix)
        self.print_networks(config.verbose)

    def train(self):
        
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
                
    def eval(self):
        
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        
        with torch.no_grad():
            self.forward()
            
    def compute_visuals(self):
        
        pass

    def update_learning_rate(self):
        
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.config.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                elif len(self.gpu_ids) == 1 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        
        
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose: # save detailed model info to the disk
                    os.makedirs(self.save_dir, exist_ok=True)
                    file_name = os.path.join(self.save_dir, 'model.txt')
                    with open(file_name, 'wt') as model_file: 
                        print(net, file=model_file)
                        model_file.write('\n[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6))
                    model_file.close()
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        
       

    def set_requires_grad(self, nets, requires_grad=False):
        
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad



def find_model_using_name(model_name):
    
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(config):
    model = find_model_using_name(config.model)
    instance = model(config)
    print("model [%s] was created" % type(instance).__name__)
    return instance
