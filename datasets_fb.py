# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json
import os 

#PVTON as Personal Virtual Try-On 
class PVTONDataset(data.Dataset):
    

    def __init__(self, config):
        super(PVTONDataset, self).__init__()
        # base setting
        self.config = config
        self.root = config.dataroot
        self.datamode = config.datamode  # train or test or self-defined
        self.stage = config.stage  # CWM or SAM
        self.data_list = config.data_list
        self.fine_height = config.fine_height
        self.fine_width = config.fine_width
        self.radius = config.radius
        self.data_path = osp.join(config.dataroot, config.datamode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(config.dataroot, config.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "PVTONDataset"

    def __getitem__(self, index):
        cloth_name = self.c_names[index]
        image_name = self.im_names[index]

        image_name = image_name.split('=')[0]
        cloth_name = cloth_name.split('=')[0]

        cloth_path_add = '=cloth_front.jpg'
        cloth_mask_path_add = '=cloth_front_mask.jpg'
        image_path_add = '=person_whole_front.png'
        image_parse_path_add = '=person_whole_front_label.png'
        pose_path_add = '=person_whole_front_keypoints.json'

        train_path = 'data/'

        if self.stage == 'CWM':
            cloth_path = os.path.join(train_path, "cloth", cloth_name+cloth_path_add)
            c = Image.open(cloth_path)

            cloth_mask_path = os.path.join(train_path, "cloth-mask", cloth_name+cloth_mask_path_add)
            cm = Image.open(cloth_mask_path).convert('L')
        else:
            # print(cloth_name)
            
            cloth_path = os.path.join(train_path, "warp-cloth", cloth_name+'.jpg')
            # print(cloth_path)
            c = Image.open(cloth_path)    

            cloth_mask_path = os.path.join(train_path, "warp-mask", cloth_name+'.jpg')
            # print(cloth_mask_path)
            cm = Image.open(cloth_mask_path).convert('L')    

        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)  # [0,1]
        cm.unsqueeze_(0)

        # person image
        
        image_path =os.path.join(train_path, "image", image_name+image_path_add)
        im = Image.open(image_path)
        im = self.transform(im)  # [-1,1]

        """
        LIP labels
        
        [(0, 0, 0),    # 0=Background
         (128, 0, 0),  # 1=Hat
         (255, 0, 0),  # 2=Hair
         (0, 85, 0),   # 3=Glove
         (170, 0, 51),  # 4=SunGlasses
         (255, 85, 0),  # 5=UpperClothes
         (0, 0, 85),     # 6=Dress
         (0, 119, 221),  # 7=Coat
         (85, 85, 0),    # 8=Socks
         (0, 85, 85),    # 9=Pants
         (85, 51, 0),    # 10=Jumpsuits
         (52, 86, 128),  # 11=Scarf
         (0, 128, 0),    # 12=Skirt
         (0, 0, 255),    # 13=Face
         (51, 170, 221),  # 14=LeftArm
         (0, 255, 255),   # 15=RightArm
         (85, 255, 170),  # 16=LeftLeg
         (170, 255, 85),  # 17=RightLeg
         (255, 255, 0),   # 18=LeftShoe
         (255, 170, 0)    # 19=RightShoe
         (170, 170, 50)   # 20=Skin/Neck/Chest (Newly added after running dataset_neck_skin_correction.py)
         ]
         """

        # load parsing image
        image_parse_path = os.path.join(train_path, "image-parse", image_name + image_parse_path_add)

        im_parse = Image.open(image_parse_path).convert('L')
        parse_array = np.array(im_parse)
        

        parse_shape = (parse_array > 0).astype(np.float32)

        if self.stage == 'CWM':
            parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(
                    np.float32) 
        else:
            parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) + \
                (parse_array == 12).astype(np.float32) + \
                (parse_array == 13).astype(np.float32) + \
                (parse_array == 16).astype(np.float32) + \
                (parse_array == 17).astype(
                np.float32) 

        parse_cloth = (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 7).astype(np.float32)    # upper-clothes labels

        # shape downsample
        parse_shape_ori = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape_ori.resize(
            (self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        
        parse_shape = parse_shape.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR)
        parse_shape_ori = parse_shape_ori.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR)
        
        shape_ori = self.transform(parse_shape_ori)  # [-1,1]
        shape = self.transform(parse_shape)  # [-1,1]
        phead = torch.from_numpy(parse_head)  # [0,1]
        # phand = torch.from_numpy(parse_hand)  # [0,1]
        pcm = torch.from_numpy(parse_cloth)  # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm)  
        im_h = im * phead - (1 - phead)  

        # load pose points
        pose_path = os.path.join(train_path, "pose", image_name + pose_path_add)


        def crop_pose(pose_arr):
            return np.concatenate((pose_arr[:15], pose_arr[16:19]), axis=0)


        with open(pose_path, 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
            pose_data = crop_pose(pose_data)
            # print(pose_data.shape )


        # print(f'Pose data ko shape {pose_data.shape}')
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        # print(f'\n\n {pose_map.shape}\n\n')
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx +
                                r, pointy+r), 'white', 'white')
                pose_draw.rectangle(
                    (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform(im_pose)

        # cloth-agnostic representation
        # print(shape.shape, im_h.shape, pose_map.shape )
        agnostic = torch.cat([shape, im_h, pose_map], 0)

        if self.stage == 'CWM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''

        pcm.unsqueeze_(0)  

        result = {
            'c_name':   cloth_name,     # cloth path 
            'im_name':  image_name,    # original image as G.T 
            'cloth':    c,          # cloth 
            'cloth_mask':     cm,   # cloth mask 
            'image':    im,         
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         
            'head': im_h,           
            'pose_image': im_pose,  
            'grid_image': im_g,     
            'parse_cloth_mask': pcm,     # for input of Style Aggregator Module
            'shape_ori': shape_ori,     # original body shape without resize
        }
        return result

    def __len__(self):
        return len(self.im_names)


class PVTONDataLoader(object):
    def __init__(self, config, dataset):
        super(PVTONDataLoader, self).__init__()

        if config.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=(
                train_sampler is None),
            num_workers=config.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="CWM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=320)
    parser.add_argument("--fine_height", type=int, default=512)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)

    config = parser.parse_args()
    dataset = PVTONDataset(config)
    data_loader = PVTONDataLoader(config, dataset)

    print('Size of the dataset: %05d, dataloader: %04d'
          % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()
