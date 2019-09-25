import os
import sys
import h5py
import nibabel as nb
import numpy as np
import torch
import torch.utils.data as data


class ImdbData(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        #print("Index = ", index)
        img = torch.from_numpy(self.x[index])
        label = torch.from_numpy(self.y[index])
        return img, label

    def __len__(self):
        return len(self.y)


def get_imdb_dataset(data_params):

    data_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_file']), 'r')
    label_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_label_file']), 'r')

    data_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
    label_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')

    return ImdbData(data_train['data'], label_train['label']), ImdbData(data_test['data'], label_test['label'])


def load_file_paths(data_dir, label_dir):
    """
    Returns list of lists containing paths to T1.mgz and CP-ROI.mgz
    [['/Users/sq566/case1/T1.mgz', '/Users/sq566/case1/CP-ROI.mgz'],
    ['/Users/sq566/case2/T1.mgz', '/Users/sq566/case2/CP-ROI.mgz']]
    """
    volume_to_use = [name for name in os.listdir(data_dir)]
    # List of list
    file_path = [
        [
            os.path.join(data_dir, vol, 'T1.mgz'),
            os.path.join(label_dir, vol, 'CP-ROI.mgz')
        ] for vol in volume_to_use
    ]

    return file_path


def load_data(file_path):
    volume_nifty, labemap_nifty = nb.load(file_path[0]), nb.load(file_path[1])
    volume, labelmap = volume_nifty.get_fdata(), labemap_nifty.get_fdata()
    p = np.percentile(volume, 99)
    vol_data = volume / p
    vol_data[vol_data > 1] = 1
    vol_data[vol_data < 0] = sys.float_info.epsilon
    labelmap[labelmap > 0.0] = 1
    print(vol_data.shape)
    print(labelmap.shape)
    return vol_data, labelmap, volume_nifty.header


def load_and_preprocess(file_path):
    volume, labelmap, header = load_data(file_path)
    return volume, labelmap, header


def load_dataset(file_paths):

    volume_list, labelmap_list, headers = [], [], []

    for file_path in file_paths:
        volume, labelmap, header = load_and_preprocess(file_path)

        # Appending 3D numpy array to list
        volume_list.append(volume)
        labelmap_list.append(labelmap)
        headers.append(header)

    return volume_list, labelmap_list, headers
