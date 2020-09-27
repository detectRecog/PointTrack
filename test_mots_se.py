"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os, sys
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from config_mots import *
import torch
from datasets import get_dataset
from models import get_model
from utils.utils import Cluster, Visualizer
# from utils1.embeddings import Embedding
import numpy as np
from PIL import Image
import cv2
from file_utils import remove_key_word, save_pickle2, load_pickle, save_zipped_pickle
import shutil
import subprocess
from config import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 0
torch.manual_seed(seed)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

config_name = sys.argv[1]
args = eval(config_name).get_args()
max_disparity = args['max_disparity']

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# cluster module
cluster = Cluster()

xm = torch.linspace(0, 2, 2048).view(
    1, 1, -1).expand(1, 1024, 2048)
ym = torch.linspace(0, 1, 1024).view(
    1, -1, 1).expand(1, 1024, 2048)
xym = torch.cat((xm, ym), 0).cuda()


def prepare_img(image):
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, torch.Tensor):
        image.squeeze_()
        image = image.numpy()

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = image.transpose(1, 2, 0)
        return image


dColors = [(128, 0, 0), (170, 110, 40), (128, 128, 0), (0, 128, 128), (0, 0, 128), (230, 25, 75), (245, 130, 48)
        , (255, 225, 25), (210, 245, 60), (60, 180, 75), (70, 240, 240), (0, 130, 200), (145, 30, 180), (240, 50, 230)
        , (128, 128, 128), (250, 190, 190), (255, 215, 180), (255, 250, 200), (170, 255, 195), (230, 190, 255), (255, 255, 255)]

with torch.no_grad():
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)

    model.eval()

    if os.path.isdir(args['save_dir']):
        shutil.rmtree(args['save_dir'])
    os.mkdir(args['save_dir'])
    # try:
    for i, sample in enumerate(tqdm(dataset_it)):
        base = sample['mot_im_name'][0].replace('/', '_').replace('.png', '.pkl')
        middle = base[:-4].split('_')[-1]
        base = base.replace(middle, str(int(float(middle))))

        im = sample['mot_image']
        instances = sample['mot_instance'].squeeze(1)[0]
        w, h = sample['im_shape'][0].item(), sample['im_shape'][1].item()

        output = model(im)
        im = im[:, :, :h, :w]
        output = output[:, :, :h, :w]
        instances = instances[:h, :w]
        instance_map = cluster.cluster_mots_wo_points(output[0], threshold=args['threshold'],
                                                      min_pixel=args['min_pixel'],
                                                      with_uv=args['with_uv'], n_sigma=args["n_sigma"] if "n_sigma" in args.keys() else 1)

        instance_map_np = instance_map.numpy()
        # cv2.imwrite("/home/xubb/1.jpg", instance_map.numpy() * 50)
        save_pickle2(os.path.join(args['save_dir'], base), instance_map_np)

    # eval on args['save_dir']
    p = subprocess.run([pythonPath, "-u", "test_tracking.py", 'car_test_tracking_val'], stdout=subprocess.PIPE, cwd=rootDir)

    pout = p.stdout.decode("utf-8")
    # class_str = "Evaluate class: Cars"
    # pout = pout[pout.find(class_str):]
    # acc = pout[pout.find('all   '):][6:26].strip().split(' ')
    print(pout, '\n\n\n')
