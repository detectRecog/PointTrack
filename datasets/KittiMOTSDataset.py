"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
import random
import numpy as np
import math
from PIL import Image
from skimage.segmentation import relabel_sequential
import torch
from torch.utils.data import Dataset
import cv2
from config import *
from file_utils import *
from utils.mots_util import *
import torch.nn.functional as F


class MOTSTest(Dataset):
    SEQ_IDS_TEST = ["%04d" % idx for idx in range(29)]
    TIMESTEPS_PER_SEQ_TEST = {'0000': 465, '0015': 701, '0017': 305, '0003': 257, '0001': 147, '0018': 180, '0005': 809,
                              '0022': 436, '0021': 203, '0023': 430, '0012': 694, '0008': 165, '0009': 349, '0020': 173,
                              '0016': 510, '0013': 152, '0004': 421, '0028': 175, '0024': 316, '0019': 404, '0026': 170,
                              '0007': 215, '0014': 850, '0025': 176, '0027': 85, '0011': 774, '0010': 1176, '0006': 114,
                              '0002': 243}

    def __init__(self, root_dir='./', type="train", class_id=26, size=None, transform=None, batch=False, batch_num=8):

        print('Kitti Dataset created')
        self.batch = batch
        self.batch_num = batch_num

        self.class_id = class_id
        self.size = size
        self.transform = transform

        self.mots_image_root = os.path.join(kittiRoot, 'testing/image_02')
        self.timestamps = self.TIMESTEPS_PER_SEQ_TEST

        self.mots_car_pairs = []
        for subdir in self.SEQ_IDS_TEST:
            image_list = sorted(make_dataset(os.path.join(self.mots_image_root, subdir), suffix='.png'))
            image_list = ['/'.join(el.split('/')[-2:]) for el in image_list]
            self.mots_car_pairs += image_list

        self.mots_num = len(self.mots_car_pairs)
        self.mots_class_id = 2

    def __len__(self):

        return self.mots_num if self.size is None else self.size

    def get_data_from_mots(self, index):
        # random select and image and the next one
        # index = random.randint(0, self.mots_num - 1)
        path = self.mots_car_pairs[index]
        sample = {}
        image = Image.open(os.path.join(self.mots_image_root, path))

        sample['mot_image'] = image
        sample['mot_im_name'] = path
        sample['im_shape'] = image.size

        return sample

    def __getitem__(self, index):
        sample = self.get_data_from_mots(index)

        # transform
        if(self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample


class MOTSCarsVal(Dataset):
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]

    def __init__(self, root_dir='./', type="train", class_id=26, size=None, transform=None, batch=False, batch_num=8):

        print('Kitti Dataset created')
        type = 'training' if type in 'training' else 'testing'
        self.type = type
        self.sequence = self.SEQ_IDS_TRAIN if type == 'training' else self.SEQ_IDS_VAL
        self.batch = batch
        self.batch_num = batch_num

        self.class_id = class_id
        self.size = size
        self.transform = transform

        self.mots_instance_root = os.path.join(kittiRoot, 'instances')
        self.mots_image_root = os.path.join(kittiRoot, 'images')

        self.mots_car_pairs = []
        for subdir in self.sequence:
            instance_list = sorted(make_dataset(os.path.join(self.mots_instance_root, subdir), suffix='.png'))
            instance_list = ['/'.join(el.split('/')[-2:]) for el in instance_list]
            for i in instance_list:
                self.mots_car_pairs.append(i)

        self.mots_num = len(self.mots_car_pairs)
        self.mots_class_id = 1

    def __len__(self):

        return self.mots_num if self.size is None else self.size

    def get_data_from_mots(self, index):
        path = self.mots_car_pairs[index]
        sample = {}
        image = Image.open(os.path.join(self.mots_image_root, path))
        # load instances
        instance = Image.open(os.path.join(self.mots_instance_root, path))
        instance, label = self.decode_mots(instance, self.mots_class_id)  # get semantic map and instance map
        sample['mot_image'] = image
        sample['mot_instance'] = instance
        sample['mot_label'] = label
        sample['mot_im_name'] = path
        sample['im_shape'] = image.size

        return sample

    def __getitem__(self, index):
        sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample

    @classmethod
    def decode_mots(cls, pic, class_id=None):
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if class_id is not None:
            mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
            if mask.sum() > 0:
                # keep the instance ids for tracking
                # +1 because 1000 is also a valid car, inst id >= 1
                instance_map[mask] = (pic[mask] + 1) % 1000
                class_map[mask] = 1

            # assign dontcare area to -2
            mask_others = pic == 10000
            if mask_others.sum() > 0:
                class_map[mask_others] = -2
        else:
            for i, c in enumerate(cls.class_ids):
                mask = np.logical_and(pic >= c * 1000, pic < (c + 1) * 1000)
                if mask.sum() > 0:
                    ids, _, _ = relabel_sequential(pic[mask])
                    instance_map[mask] = ids + np.amax(instance_map)
                    class_map[mask] = i + 1

        return Image.fromarray(instance_map), Image.fromarray(class_map)


class MOTSTrackCarsValOffset(Dataset):
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}
    SEQ_IDS_TEST = ["%04d" % idx for idx in range(29)]
    TIMESTEPS_PER_SEQ_TEST = {'0000': 465, '0015': 701, '0017': 305, '0003': 257, '0001': 147, '0018': 180, '0005': 809,
                              '0022': 436, '0021': 203, '0023': 430, '0012': 694, '0008': 165, '0009': 349, '0020': 173,
                              '0016': 510, '0013': 152, '0004': 421, '0028': 175, '0024': 316, '0019': 404, '0026': 170,
                              '0007': 215, '0014': 850, '0025': 176, '0027': 85, '0011': 774, '0010': 1176, '0006': 114,
                              '0002': 243}

    def __init__(self, root_dir='./', type="train", num_points=250, transform=None, random_select=False, az=False,
                 border=False, env=False, gt=True, box=False, test=False, category=False, ex=0.2):

        print('MOTS Dataset created')
        type = 'training' if type in 'training' else 'testing'
        self.type = type
        assert self.type == 'testing'

        self.transform = transform
        if not test:
            ids = self.SEQ_IDS_VAL
            timestamps = self.TIMESTEPS_PER_SEQ
            self.image_root = os.path.join(kittiRoot, 'images')
            self.mots_root = os.path.join(systemRoot, 'SpatialEmbeddings/car_SE_val_prediction')
        else:
            ids = self.SEQ_IDS_TEST
            timestamps = self.TIMESTEPS_PER_SEQ_TEST
            self.image_root = os.path.join(kittiRoot, 'testing/image_02/')
            self.mots_root = os.path.join(systemRoot, 'SpatialEmbeddings/car_SE_test_prediction')

        print('use ', self.mots_root)
        self.batch_num = 2

        self.mots_car_sequence = []
        for valF in ids:
            nums = timestamps[valF]
            for i in range(nums):
                pklPath = os.path.join(self.mots_root, valF + '_' + str(i) + '.pkl')
                if os.path.isfile(pklPath):
                    self.mots_car_sequence.append(pklPath)

        self.real_size = len(self.mots_car_sequence)
        self.mots_num = len(self.mots_car_sequence)
        self.mots_class_id = 1
        self.expand_ratio = ex
        self.vMax, self.uMax = 375.0, 1242.0
        self.num_points = num_points
        self.env_points = 200
        self.random = random_select
        self.az = az
        self.border = border
        self.env = env
        self.box = box
        self.offsetMax = 128.0
        self.category = category
        self.category_embedding = np.array(category_embedding, dtype=np.float32)
        print(self.mots_root)

    def __len__(self):
        return self.real_size

    def get_crop_from_mask(self, mask, img, label):
        label[mask] = 1
        vs, us = np.nonzero(mask)
        h, w = mask.shape
        v0, v1 = vs.min(), vs.max() + 1
        vlen = v1 - v0
        u0, u1 = us.min(), us.max() + 1
        ulen = u1 - u0
        # enlarge box by 0.2
        v0 = max(0, v0 - int(self.expand_ratio * vlen))
        v1 = min(v1 + int(self.expand_ratio * vlen), h - 1)
        u0 = max(0, u0 - int(self.expand_ratio * ulen))
        u1 = min(u1 + int(self.expand_ratio * ulen), w - 1)
        return mask[v0:v1, u0:u1], img[v0:v1, u0:u1], label[v0:v1, u0:u1], (v0, u0)

    def get_xyxy_from_mask(self, mask):
        vs, us = np.nonzero(mask)
        y0, y1 = vs.min(), vs.max()
        x0, x1 = us.min(), us.max()
        return [x0/self.uMax, y0/self.vMax, x1/self.uMax, y1/self.vMax]

    def get_data_from_mots(self, index):
        # random select and image and the next one
        path = self.mots_car_sequence[index]
        instance_map = load_pickle(path)
        subf, frameCount = os.path.basename(path)[:-4].split('_')
        imgPath = os.path.join(self.image_root, subf, '%06d.png' % int(float(frameCount)))
        img = cv2.imread(imgPath)

        sample = {}
        sample['name'] = imgPath
        sample['masks'] = []
        sample['points'] = []
        sample['envs'] = []
        sample['xyxys'] = []
        inds = np.unique(instance_map).tolist()[1:]
        label = (instance_map > 0).astype(np.uint8) * 2
        for inst_id in inds:
            mask = (instance_map == inst_id)
            sample['xyxys'].append(self.get_xyxy_from_mask(mask))
            sample['masks'].append(np.array(mask)[np.newaxis])
            mask, img_, maskX, sp = self.get_crop_from_mask(mask, img, label.copy())
            # fg/bg ratio
            ratio = 2.0
            # ratio = max(mask.sum() / (~mask).sum(), 2.0)
            bg_num = int(self.num_points / (ratio + 1))
            fg_num = self.num_points - bg_num

            vs_, us_ = np.nonzero(mask)
            vc, uc = vs_.mean(), us_.mean()

            vs = (vs_ - vc) / self.offsetMax
            us = (us_ - uc) / self.offsetMax
            rgbs = img_[mask] / 255.0
            pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
            choices = np.random.choice(pointUVs.shape[0], fg_num)
            points_fg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
            points_fg = np.concatenate(
                [points_fg, np.zeros((points_fg.shape[0], points_fg.shape[1], 3), dtype=np.float32)], axis=-1)

            if (~mask).sum() == 0:
                points_bg = np.zeros((1, bg_num, 8), dtype=np.float32)
            else:
                vs, us = np.nonzero(~mask)
                vs = (vs - vc) / self.offsetMax
                us = (us - uc) / self.offsetMax
                rgbs = img_[~mask] / 255.0
                cats = maskX[~mask]
                cat_embds = self.category_embedding[cats]
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis], cat_embds], axis=1)
                choices = np.random.choice(pointUVs.shape[0], bg_num)
                points_bg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
            sample['points'].append(np.concatenate([points_fg, points_bg], axis=1))
            sample['envs'].append(fg_num)

        if len(sample['points']) > 0:
            sample['points'] = np.concatenate(sample['points'], axis=0)
            sample['masks'] = np.concatenate(sample['masks'], axis=0)
            sample['envs'] = np.array(sample["envs"], dtype=np.int32)
            sample['xyxys'] = np.array(sample["xyxys"], dtype=np.float32)
        return sample

    def __getitem__(self, index):
        # select nearby images from mots
        sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample


class MOTSTrackCarsTrain(Dataset):
    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}

    def __init__(self, root_dir='./', type="train", num_points=250, transform=None, random_select=False, batch_num=8,
                 shift=False, size=3000, sample_num=30, nearby=1, trainval=False, category=False):
        print('MOTS Dataset created')
        type = 'training' if type in 'training' else 'testing'
        if trainval:
            self.squence = self.SEQ_IDS_TRAIN + self.SEQ_IDS_VAL
            print("train with training and val set")
        else:
            self.squence = self.SEQ_IDS_TRAIN if type == 'training' else self.SEQ_IDS_VAL
        self.type = type

        self.transform = transform
        db_dir = kittiRoot + 'ImgTrackEnvDB/'
        self.dbDict = {}
        for id in self.squence:
            image_root = kittiRoot + "images/" + id
            image_list = make_dataset(image_root, suffix='.png')
            image_list.sort()
            infos = {}
            for ind, image_path in enumerate(image_list):
                pkl_path = os.path.join(db_dir, id + '_' + str(ind) + '.pkl')
                if os.path.isfile(pkl_path):
                    infos[ind] = load_pickle(pkl_path)
            self.dbDict[id] = infos

        self.mots_car_instances = self.getInstanceFromDB(self.dbDict)
        print('dbDict Loaded, %s instances' % len(self.mots_car_instances))
        self.batch_num = batch_num

        self.inst_names = list(self.mots_car_instances.keys())
        self.inst_num = len(self.inst_names)
        self.real_size = size
        self.mots_class_id = 1
        self.vMax, self.uMax = 375.0, 1242.0
        self.offsetMax = 128.0
        self.num_points = num_points
        self.random = random_select
        self.shift = shift
        self.frequency = 1
        self.sample_num = sample_num
        self.nearby = nearby
        self.category = category
        self.category_embedding = np.array(category_embedding, dtype=np.float32)

    def getInstanceFromDB(self, dbDict):
        allInstances = {}
        for k, fs in dbDict.items():
            # current video k
            # num_frames = self.TIMESTEPS_PER_SEQ[k]
            if not k in self.squence:
                continue
            for fi, f in fs.items():
                frameCount = fi
                for inst in f:
                    inst_id = k + '_' + str(inst['inst_id'])
                    newDict = {'frame': frameCount, 'sp': inst['sp'], 'img': inst['img'], 'mask': inst['mask'], 'maskX': inst['maskX']}
                    if not inst_id in allInstances.keys():
                        allInstances[inst_id] = [newDict]
                    else:
                        allInstances[inst_id].append(newDict)
        return allInstances

    def __len__(self):
        return self.real_size

    def get_data_from_mots(self, index):
        # sample ? instances from self.inst_names
        inst_names_inds = random.sample(range(len(self.inst_names)), self.sample_num)
        inst_names = [self.inst_names[el] for el in inst_names_inds]
        pickles = [self.mots_car_instances[el] for el in inst_names]

        sample = {}
        sample['mot_im_name0'] = index
        sample['points'] = []
        sample['labels'] = []
        sample['imgs'] = []
        sample['inds'] = []
        sample['envs'] = []
        sample['xyxys'] = []
        for pind, pi in enumerate(pickles):
            inst_id = pind + 1
            inst_length = len(pi)
            if inst_length > 2:
                mid = random.choice(range(1, inst_length - 1))
                nearby = random.choice(range(1, self.nearby+1))
                start, end = max(0, mid - nearby), min(inst_length - 1, mid + nearby)
                pis = [pi[start], pi[mid], pi[end]]
            else:
                start, end = 0, 1
                pis = pi[start:end + 1]
            for ii, inst in enumerate(pis):
                img = inst['img']
                mask = inst['mask'].astype(np.bool)
                maskX = inst['maskX']
                sp = inst['sp']
                assert (~mask).sum() > 0

                ratio = 2.0
                bg_num = int(self.num_points / (ratio + 1))
                fg_num = self.num_points - bg_num

                # get center
                vs_, us_ = np.nonzero(mask)
                vc, uc = vs_.mean(), us_.mean()

                vs, us = np.nonzero(~mask)
                vs = (vs - vc) / self.offsetMax
                us = (us - uc) / self.offsetMax
                rgbs = img[~mask] / 255.0
                if self.shift:
                    # us += (random.random() - 0.5) * 0.05  # -0.025~0.025
                    vs += np.random.normal(0, 0.001, size=vs.shape)  # random jitter
                    us += np.random.normal(0, 0.001, size=us.shape)  # random jitter
                cats = maskX[~mask]
                cat_embds = self.category_embedding[cats]
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis], cat_embds], axis=1)
                choices = np.random.choice(pointUVs.shape[0], bg_num)
                points_bg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)

                vs = (vs_ + sp[0]) / self.vMax
                us = (us_ + sp[1]) / self.uMax  # to compute the bbox position
                sample['xyxys'].append([us.min(), vs.min(), us.max(), vs.max()])

                vs = (vs_ - vc) / self.offsetMax
                us = (us_ - uc) / self.offsetMax
                rgbs = img[mask.astype(np.bool)] / 255.0
                if self.shift:
                    # us += (random.random() - 0.5) * 0.05  # -0.025~0.025
                    vs += np.random.normal(0, 0.001, size=vs.shape)  # random jitter
                    us += np.random.normal(0, 0.001, size=us.shape)  # random jitter
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
                choices = np.random.choice(pointUVs.shape[0], fg_num)
                pointUVs = np.concatenate([rgbs, vs[:, np.newaxis], us[:, np.newaxis]], axis=1)
                points_fg = pointUVs[choices][np.newaxis, :, :].astype(np.float32)
                points_fg = np.concatenate(
                    [points_fg, np.zeros((points_fg.shape[0], points_fg.shape[1], 3), dtype=np.float32)], axis=-1)

                sample['points'].append(np.concatenate([points_fg, points_bg], axis=1))
                sample['labels'].append(np.array(inst_id)[np.newaxis])
                sample['envs'].append(fg_num)

        sample['points'] = np.concatenate(sample['points'], axis=0)
        sample['envs'] = np.array(sample["envs"], dtype=np.int32)
        sample['labels'] = np.concatenate(sample['labels'], axis=0)
        sample['xyxys'] = np.array(sample["xyxys"], dtype=np.float32)
        return sample

    def __getitem__(self, index):
        # select nearby images from mots
        while 1:
            try:
                sample = self.get_data_from_mots(index)
                break
            except:
                pass
        # sample = self.get_data_from_mots(index)

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)
            return sample
        else:
            return sample


class MOTSCars(Dataset):

    SEQ_IDS_TRAIN = ["%04d" % idx for idx in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
    SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 6, 7, 8, 10, 13, 14, 16, 18]]
    TIMESTEPS_PER_SEQ = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
                         "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
                         "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}

    def __init__(self, root_dir='./', type="train", size=None, transform=None, kins=True, kins_only=False):

        print('Kitti Dataset created')
        self.class_id = 26
        self.type = type

        if type == 'crop':
            self.image_list = make_dataset(os.path.join(kittiRoot,'crop_KINS', 'images'), suffix='.png')
            self.instance_list = [el.replace('images', 'instances') for el in self.image_list]
        else:
            type = 'training' if type in 'training' else 'testing'
            if kins and type == 'training':
                self.image_index = self._load_image_set_index_new('training') + self._load_image_set_index_new('testing')
                self.clean_kins_inst_file = os.path.join(root_dir, 'KINSCarValid.pkl')
                if not os.path.isfile(self.clean_kins_inst_file):
                    self.instance_list = make_dataset(os.path.join(root_dir, 'training/KINS/'), suffix='.png') + make_dataset(os.path.join(root_dir, 'testing/KINS/'), suffix='.png')
                    self.instance_list = leave_needed(self.instance_list, self.class_id) # 14908 -> 13997
                    save_pickle2(self.clean_kins_inst_file, self.instance_list)
                else:
                    self.instance_list = load_pickle(self.clean_kins_inst_file)
                # get image and instance list
                image_list = [el.replace('KINS', 'image_2') for el in self.instance_list]
                self.image_list = image_list
            else:
                self.instance_list, self.image_list = [], []

            if not kins_only:
                self.mots_instance_root = os.path.join(kittiRoot, 'instances')
                self.mots_image_root = os.path.join(kittiRoot, 'images')
                assert os.path.isfile(os.path.join(kittiRoot, 'motsCarsTrain.pkl'))

                if type == 'training':
                    self.mots_persons = load_pickle(os.path.join(kittiRoot, 'motsCarsTrain.pkl'))
                    self.mots_instance_list = [os.path.join(self.mots_instance_root, el) for el in self.mots_persons]
                    self.mots_image_list = [el.replace('instances', 'images') for el in self.mots_instance_list]
                    self.image_list += self.mots_image_list
                    self.instance_list += self.mots_instance_list
                else:
                    self.mots_persons = load_pickle(os.path.join(kittiRoot, 'motsCarsTest.pkl'))
                    self.mots_instance_list = [os.path.join(self.mots_instance_root, el) for el in self.mots_persons]
                    self.mots_image_list = [el.replace('instances', 'images') for el in self.mots_instance_list]
                    self.image_list = self.mots_image_list
                    self.instance_list = self.mots_instance_list
        self.real_size = len(self.image_list)
        self.size = size
        self.transform = transform

    def _load_image_set_index_new(self, type):
        if type == 'training':
            train_set_file = open(rootDir + 'datasets/splits/train.txt', 'r')
            image_index = train_set_file.read().split('\n')
        else:
            val_set_file = open(rootDir + 'datasets/splits/val.txt', 'r')
            image_index = val_set_file.read().split('\n')
        return image_index

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def get_data_from_kins(self, index):
        if self.type == 'crop':
            index = random.randint(0, self.real_size - 1)
        sample = {}
        image = Image.open(self.image_list[index])
        # load instances
        instance = Image.open(self.instance_list[index])
        instance, label = self.decode_instance(instance, self.instance_list[index])  # get semantic map and instance map
        sample['image'] = image
        sample['im_name'] = self.image_list[index]
        sample['instance'] = instance
        sample['label'] = label

        return sample

    def __getitem__(self, index):
        # select two images from kins
        while 1:
            try:
                sample = self.get_data_from_kins(index)

                # transform
                if (self.transform is not None):
                    sample = self.transform(sample)
                    return sample
                else:
                    return sample
            except:
                pass

    def decode_instance(self, pic, path):
        if self.type == 'crop':
            class_id = 1 if 'MOTS' in path else 26
        else:
            class_id = 26 if 'KINS' in path else 1
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
        if self.type=='crop':
            assert mask.sum() > 0
        if mask.sum() > 0:
            ids, _, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids
            class_map[mask] = 1

        # assign vehicles but not car to -2, dontcare
        mask = np.logical_and(pic > 0, pic < 1000)
        mask_others = (pic == 10000) & mask
        if mask_others.sum() > 0:
            class_map[mask_others] = -2

        return Image.fromarray(instance_map), Image.fromarray(class_map)
