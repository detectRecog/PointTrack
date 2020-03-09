"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os, sys
import shutil
import time
from config import *

os.chdir(rootDir)

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from config_mots import *
from criterions.mots_seg_loss import *
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer
from file_utils import remove_key_word
from random import random

torch.backends.cudnn.benchmark = True
config_name = sys.argv[1]

args = eval(config_name).get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")
# clustering
cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')

# train dataloader
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)

# set criterion
criterion = eval(args['loss_type'])(**args['loss_opts'])
criterion = torch.nn.DataParallel(criterion).to(device)

# resume
start_epoch = 0
best_iou = 0
best_seed = 10
max_disparity = args['max_disparity']
if 'resume_path' in args.keys() and args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    if 'start_epoch' in args.keys():
        start_epoch = args['start_epoch']
    elif 'epoch' in state.keys():
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 1
    # best_iou = state['best_iou']
    for kk in state.keys():
        if 'state_dict' in kk:
            state_dict_key = kk
            break
    new_state_dict = state[state_dict_key]
    if not 'state_dict_keywords' in args.keys():
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except:
            print('resume checkpoint with strict False')
            model.load_state_dict(new_state_dict, strict=False)
    else:
        new_state_dict = remove_key_word(state[state_dict_key], args['state_dict_keywords'])
        model.load_state_dict(new_state_dict, strict=False)
        print('resume checkpoint with strict False')
    try:
        logger.data = state['logger_data']
    except:
        pass

# set optimizer
if 'seed_only' in args.keys() and args['seed_only']:
    print('finetune SEED only')
    optimizer = torch.optim.Adam(model.module.decoders[1].parameters(), lr=args['lr'], weight_decay=1e-4)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)


def lambda_(epoch):
    return pow((1 - ((epoch) / args['n_epochs'])), 0.9)


if 'milestones' in args.keys():
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=0.1)
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_, )


def train(epoch):
    # define meters
    loss_meter = AverageMeter()
    loss_seed_meter = AverageMeter()

    # put model into training mode
    model.train()
    if 'fix_bn' in args.keys() and args['fix_bn']:
        print('BN Fixed!')
        # freeze bn if need
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):
        ims = sample['image']
        instances = sample['instance'].squeeze(1)
        class_labels = sample['label'].squeeze(1)

        output = model(ims)
        loss, seed_loss = criterion(output, instances, class_labels, **args['loss_w'], show_seed=True)

        loss = loss.mean()
        seed_loss = seed_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        loss_seed_meter.update(seed_loss.item())

    return loss_meter.avg, loss_seed_meter.avg


def val(epoch):
    # define meters
    loss_meter, iou_meter, loss_seed_meter = AverageMeter(), AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            ims = sample['image']
            instances = sample['instance'].squeeze(1)
            class_labels = sample['label'].squeeze(1)

            output = model(ims)
            loss, seed_loss = criterion(output, instances, class_labels, **args['loss_w'], iou=True,
                                        iou_meter=iou_meter, show_seed=True)
            loss = loss.mean()
            seed_loss = seed_loss.mean()

            loss_meter.update(loss.item())
            loss_seed_meter.update(seed_loss.item())

    return loss_meter.avg, iou_meter.avg, loss_seed_meter.avg


def save_checkpoint(state, is_best, val_iou, val_seed_loss, is_lowest=False, name='checkpoint.pth'):
    print('=> saving checkpoint')
    if 'save_name' in args.keys():
        file_name = os.path.join(args['save_dir'], args['save_name'])
    else:
        file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth' + str(val_iou)))
    if is_lowest:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_seed_model.pth' + str(val_seed_loss)))


for epoch in range(start_epoch, args['n_epochs']):

    print('Starting epoch {}'.format(epoch))
    if epoch > start_epoch:
        scheduler.step()
    else:
        val_loss, val_iou, val_seed_loss = val(epoch)
        print('===> val loss: {:.4f}, val iou: {:.4f}, val seed: {:.4f}'.format(val_loss, val_iou, val_seed_loss))

    train_loss, seed_loss = train(epoch)
    val_loss, val_iou, val_seed_loss = val(epoch)

    print('===> train loss: {:.4f}'.format(train_loss))
    print('===> seed loss: {:.4f}'.format(seed_loss))
    print('===> val loss: {:.4f}, val iou: {:.4f}, val seed: {:.4f}'.format(val_loss, val_iou, val_seed_loss))

    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.add('iou', val_iou)
    logger.plot(save=args['save'], save_dir=args['save_dir'])

    is_best = val_iou > best_iou
    best_iou = max(val_iou, best_iou)

    is_lowest = val_loss < best_seed
    best_seed = min(val_loss, best_seed)

    if args['save']:
        state = {
            'epoch': epoch,
            'best_iou': best_iou,
            'best_seed': best_seed,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(state, is_best, val_iou, val_loss, is_lowest=is_lowest)
