"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import cv2
import cvbase as cvb
import os
import pycocotools.mask as maskUtils
from file_utils import mkdir_if_no
import multiprocessing
from PIL import Image
from config import kittiRoot


def make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict


''' A cityscapes example
        >>> from PIL import Image
        >>> import cv2
        >>> import numpy as np
        >>> a = np.array(Image.open('aachen_000001_000019_gtFine_instanceIds.png'))
        >>> np.unique(a)
        array([    0,     1,     3,     4,     5,     6,     7,     8,    11,
                  12,    13,    17,    19,    20,    21,    22,    23, 26000,
               26001, 26002, 26003, 26005, 26006, 26007, 26008, 26009, 26010,
               26011, 26012, 26013, 26014], dtype=int32)
        '''


def plot_img_id(img_id):
    img_name = imgs_dict[img_id]

    img_path = os.path.join(src_img_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    anns_ = anns_dict[img_id]

    inmodal = np.zeros((height, width), dtype=np.int16)
    count = 0
    base = 26000
    not_car = 0
    for ann in anns_:
        if ann['category_id'] in [4, 5, 6, 7, 8]:
            count += 1
            if ann['category_id'] == 4:
                inmodal_ann_mask = maskUtils.decode(ann['inmodal_seg'])
                inmodal[inmodal_ann_mask > 0] = base + count
            else:
                inmodal_ann_mask = maskUtils.decode(ann['inmodal_seg'])
                inmodal[inmodal_ann_mask > 0] = not_car + count

    if count > 0:
        Image.fromarray(inmodal).save(save_path + img_name)


if __name__ == '__main__':
    src_img_path = kittiRoot + "training/image_2"
    src_gt7_path = kittiRoot + "kitti_instances_train.json"
    save_path = kittiRoot + "training/KINS/"
    mkdir_if_no(save_path)

    anns = cvb.load(src_gt7_path)
    imgs_info = anns['images']
    anns_info = anns["annotations"]
    imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
    allImgs = list(anns_dict.keys())
    # plot_img_id(allImgs[0])
    pool = multiprocessing.Pool(processes=32)
    results = pool.map(plot_img_id, allImgs)
    pool.close()

    src_img_path = kittiRoot + "testing/image_2"
    src_gt7_path = kittiRoot + "kitti_instances_val.json"
    save_path = kittiRoot + "/testing/KINS/"
    mkdir_if_no(save_path)

    anns = cvb.load(src_gt7_path)
    imgs_info = anns['images']
    anns_info = anns["annotations"]
    imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
    allImgs = list(anns_dict.keys())
    # plot_img_id(allImgs[0])
    pool = multiprocessing.Pool(processes=32)
    results = pool.map(plot_img_id, allImgs)
    pool.close()