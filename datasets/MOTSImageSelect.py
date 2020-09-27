"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import cv2
import numpy as np
import PIL.Image as Image
from file_utils import *
import multiprocessing
from config import *

SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
label_id=1


def count_similar(src, ids_list, s, e):
    res = []
    for i in range(s, e, 1):
        target = ids_list[i]
        count = 0
        for j in src:
            if j in target:
                count += 1
        res.append(count)
    return np.array(res)


def getPairs(id):
    label_root = kittiRoot + "instances/" + id
    image_root = label_root.replace('instances', 'images')

    image_list = make_dataset(image_root, suffix='.png')
    image_list.sort()
    label_list = make_dataset(label_root, suffix='.png')
    label_list.sort()

    imgs_list = []

    # filter out images with no cars
    for ind, image_path in enumerate(image_list):
        label_path = label_list[ind]
        label = np.array(Image.open(label_path))
        # for car
        mask = np.logical_and(label >= label_id * 1000, label < (label_id + 1) * 1000)

        obj_ids = np.unique(label[mask]).tolist()

        if len(obj_ids) < 1:
            continue

        imgs_list.append('/'.join(image_path.split('/')[-2:]))

    return imgs_list


pool = multiprocessing.Pool(processes=32)
pairs = pool.map(getPairs, SEQ_IDS_TRAIN)
pool.close()
allPairs = []
for pair in pairs:
    allPairs += pair
save_pickle2(kittiRoot + 'motsCarsTrain.pkl', allPairs)

pool = multiprocessing.Pool(processes=32)
pairs = pool.map(getPairs, SEQ_IDS_VAL)
pool.close()
allPairs = []
for pair in pairs:
    allPairs += pair
save_pickle2(kittiRoot + 'motsCarsTest.pkl', allPairs)

