"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms
from config import *

n_sigma=2
args = dict(

    cuda=True,
    display=False,
    n_sigma=n_sigma,

    save=True,
    checkpoint_path='./pointTrack_weights/best_seed_model.pthCar',

    min_pixel=160,
    threshold=0.94,
    model={
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1],
            'input_channel': 3
        }
    },

    save_dir='./car_SE_val_prediction/',
    dataset= {
        'name': 'mots_cars_val',
        'kwargs': {
            'root_dir': kittiRoot,
            # 'type': 'train',
            'type': 'val',
            # 'size': 1000,
            'transform': my_transforms.get_transform([
                {
                    'name': 'LU_Pad',
                    'opts': {
                        'keys': ('mot_image', 'mot_instance','mot_label'),
                        'size': (384, 1248),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('mot_image', 'mot_instance','mot_label'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 32
    },

    max_disparity=192.0,
    with_uv=True
)


def get_args():
    return copy.deepcopy(args)
