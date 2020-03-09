"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import pickle
import gzip
import json
import os
import sys
import torch
import shutil


def remove_key_word(previous_dict, keywords):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in previous_dict.items():
        if_exist_keyword = [1 if el in k else 0 for el in keywords]
        if sum(if_exist_keyword) == 0:
            new_state_dict[k] = v
    return new_state_dict


def load_weights_from_data_parallel(model_path, net):
    if not os.path.isfile(model_path):
        print('%s not found' % model_path)
        exit(0)
    else:
        print('Load from %s' % model_path)
    previous_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in previous_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=True)
    return net


def remove_module_in_dict(loaded_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in loaded_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def load_weights(model_path, net, strict=True):
    if not os.path.isfile(model_path):
        print('%s not found' % model_path)
        exit(0)
    else:
        print('Load from %s' % model_path)
    previous_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(previous_dict, strict=strict)
    return net


def remove_and_mkdir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(path, 'removed')
    os.makedirs(path)


def mkdir_if_no(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)
    return


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object


def save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def save_pickle2(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_json(filename):
    return json.load(open(filename, 'r'))


def save_json(filename, res):
    json.dump(res, open(filename, 'w'))


def is_image_file(filename, suffix=None):
    if suffix is not None:
        IMG_EXTENSIONS = [suffix]
    else:
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.npz'
        ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, suffix=None, max=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, suffix):
                path = os.path.join(root, fname)
                images.append(path)
    if max is not None:
        return images[:max]
    else:
        return images


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element


def write_ply(filename, points=None, mesh=None, as_text=False):
    """
    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
    Returns
    -------
    boolean
        True if no problems
    """
    if not filename.endswith('ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']

        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))

        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                          encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                        encoding='ascii')

    else:
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True