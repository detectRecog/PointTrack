"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageStat
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import torch


class AdjustBrightness(object):

    def __init__(self, keys=[]):
        self.keys = keys

    def __call__(self, sample):
        for idx, k in enumerate(self.keys):
            assert (k in sample)

            image = sample[k]
            temp = image.convert('L')
            stat = ImageStat.Stat(temp)
            brightness = (stat.mean[0] / 255)

            # Think this makes more sense
            enhancer = ImageEnhance.Brightness(image)
            if 0.3 < brightness < 0.7:
                t = random.random()
                if t < 0.25:
                    image = enhancer.enhance(0.1 * random.randint(15, 20))
                elif t < 0.5:
                    image = enhancer.enhance(0.1 * random.randint(5, 8))
                else:
                    pass

            sample[k] = image

        return sample


class Flip(object):

    def __init__(self, keys=[]):
        self.keys = keys

    def __call__(self, sample):
        if random.random() > 0.5:
            for idx, k in enumerate(self.keys):
                assert (k in sample)

                sample[k] = torch.flip(sample[k], [-1])
                if k == 'part0':
                    bg_mask = sample[k] == 0
                    sample[k] = 1.0 - sample[k]
                    sample[k][bg_mask] = 0
            sample['Flip'] = 1
        else:
            sample['Flip'] = 0
        return sample


class CropRandomObject:

    def __init__(self, keys=[],object_key="instance", size=100):
        self.keys = keys
        self.object_key = object_key
        self.size = size

    def __call__(self, sample):

        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]
        
        if unique_objects.size > 0:
            random_id = np.random.choice(unique_objects, 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)
            
            i = int(np.clip(ym-self.size[1]/2, 0, h-self.size[1]))
            j = int(np.clip(xm-self.size[0]/2, 0, w-self.size[0]))

        else:
            i = random.randint(0, h - self.size[1])
            j = random.randint(0, w - self.size[0])

        for k in self.keys:
            assert(k in sample)

            sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        return sample


class LU_Pad(object):
    # pad at the right and the bottom
    def __init__(self, keys=[], size=100):
        self.keys = keys
        self.size = size

    def __call__(self, sample):

        for k in self.keys:

            assert(k in sample)

            w, h = sample[k].size

            padding = (0,0,self.size[1]-w, self.size[0]-h)

            sample[k] = F.pad(sample[k], padding, padding_mode='edge')

        sample['start_point'] = torch.FloatTensor([0,0])  # y0, x0
        sample['x_diff'] = torch.FloatTensor([0])
        return sample


class RandomCrop(T.RandomCrop):

    def __init__(self, keys=[], size=100):

        super().__init__(size)
        self.keys = keys

    def __call__(self, sample):

        params = None

        for k in self.keys:

            assert(k in sample)

            if params is None:
                params = self.get_params(sample[k], self.size)

            sample[k] = F.crop(sample[k], *params)

        return sample

class RandomRotation(T.RandomRotation):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.resample, collections.Iterable):
            assert(len(keys) == len(self.resample))

    def __call__(self, sample):

        angle = self.get_params(self.degrees)

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            resample = self.resample
            if isinstance(resample, collections.Iterable):
                resample = resample[idx]

            sample[k] = F.rotate(sample[k], angle, resample,
                                 self.expand, self.center)

        return sample


class Resize(T.Resize):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.interpolation, collections.Iterable):
            assert(len(keys) == len(self.interpolation))

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, collections.Iterable):
                interpolation = interpolation[idx]

            sample[k] = F.resize(sample[k], self.size, interpolation)

        return sample


class ToTensor(object):

    def __init__(self, keys=[], type="float"):

        if isinstance(type, collections.Iterable):
            assert(len(keys) == len(type))

        self.keys = keys
        self.type = type

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            sample[k] = F.to_tensor(sample[k])

            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]

            if t == torch.ByteTensor or t == torch.LongTensor:
                sample[k] = sample[k]*255

            sample[k] = sample[k].type(t)

        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']
        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)
