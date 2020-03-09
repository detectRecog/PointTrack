"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm
from config import *
import multiprocessing
from file_utils import *


def process(tup):
    image_path, instance_path, name = tup['img'], tup['inst'], tup['name']
    OBJ_ID = class_id
    # OBJ_ID = 26 if 'KINS' in instance_path else 1

    image = Image.open(image_path)
    instance = Image.open(instance_path)
    w, h = image.size

    instance_np = np.array(instance, copy=False)
    object_mask = np.logical_and(instance_np >= OBJ_ID * 1000, instance_np < (OBJ_ID + 1) * 1000)

    ids = np.unique(instance_np[object_mask])
    ids = ids[ids!= 0]

    # loop over instances
    for j, id in enumerate(ids):

        y, x = np.where(instance_np == id)
        ym, xm = np.mean(y), np.mean(x)

        ii = int(np.clip(ym-CROP_SIZE[0]/2, 0, h-CROP_SIZE[0]))
        jj = int(np.clip(xm-CROP_SIZE[1]/2, 0, w-CROP_SIZE[1]))

        im_crop = image.crop((jj, ii, jj + CROP_SIZE[1], ii + CROP_SIZE[0]))
        instance_crop = instance.crop((jj, ii, jj + CROP_SIZE[1], ii + CROP_SIZE[0]))

        iname = name + '_' + str(j) + '.png'
        im_crop.save(os.path.join(image_dir, iname))
        instance_crop.save(os.path.join(inst_dir, iname))


if __name__ == '__main__':

    # dst_kins = 'person_KINS'
    # dst_mots = 'motsPersonsTrain.pkl'
    # class_id = 2
    # CROP_SIZE = (160, 256)  # for person
    dst_kins = 'KINS'
    dst_mots = 'motsCarsTrain.pkl'
    class_id = 26
    CROP_SIZE=(160, 640) # h, w for car

    # initialize folders to save crops
    save_root = os.path.join(kittiRoot,'crop_' + dst_kins)
    mkdir_if_no(save_root)
    inst_dir = os.path.join(save_root, 'instances')
    mkdir_if_no(inst_dir)
    image_dir = os.path.join(save_root, 'images')
    mkdir_if_no(image_dir)

    # load images from KINS
    instance_list = make_dataset(os.path.join(kittiRoot, 'training/'+dst_kins), suffix='.png') + make_dataset(os.path.join(kittiRoot, 'testing/'+dst_kins), suffix='.png')
    image_list = [el.replace(dst_kins, 'image_2') for el in instance_list]
    save_list = ['KINS_' + el.split('/')[-1][:-4] for el in instance_list]
    # image_list, instance_list, save_list = [], [], []

    # load images from KITTI MOTS
    mots_instance_root = os.path.join(kittiRoot, 'instances')
    mots_image_root = os.path.join(kittiRoot, 'images')
    mots_persons = load_pickle(os.path.join(kittiRoot, dst_mots))
    mots_instance_list = [os.path.join(mots_instance_root, el) for el in mots_persons]
    mots_image_list =  [el.replace('instances', 'images') for el in mots_instance_list]
    mots_save_list = ['MOTS_' + el.split('/')[-1][:-4] for el in mots_instance_list]

    total_image_list = image_list + mots_image_list
    total_inst_list = instance_list + mots_instance_list
    total_save_list = save_list + mots_save_list
    infos = [{'img': total_image_list[el], 'inst': total_inst_list[el], 'name': total_save_list[el]} for el in range(len(total_image_list))]

    # process(infos[0])
    # for el in infos:
    #     process(el)
    pool = multiprocessing.Pool(processes=32)
    results = pool.map(process, infos)
    pool.close()
