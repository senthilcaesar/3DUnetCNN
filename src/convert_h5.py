import argparse
import os
import data_utils as du
import numpy as np
import h5py


def apply_split(data_split, data_dir, label_dir):
    """
    Randomly splits the data in training and test set
    """
    file_paths = du.load_file_paths(data_dir, label_dir)
    print("Total no of volumes to process: %d" % len(file_paths))
    train_ratio, test_ratio = data_split.split(",")
    train_len = int((int(train_ratio) / 100.0) * len(file_paths))

    train_idx = np.random.choice(len(file_paths), train_len, replace=False)
    test_idx = np.array([i for i in range(len(file_paths)) if i not in train_idx])
    train_file_paths = [file_paths[i] for i in train_idx]
    test_file_paths = [file_paths[i] for i in test_idx]
    return train_file_paths, test_file_paths


def _write_h5(data, label, f, mode):

    n_slices, h, w = data[0].shape
    with h5py.File(f[mode]['data'], "w") as data_handle:
        # -1 simply means that it is an unknown dimension and we want numpy to figure it out
        data_handle.create_dataset("data", data=np.concatenate(data).reshape(-1, h, w))
    with h5py.File(f[mode]['label'], "w") as label_handle:
        label_handle.create_dataset("label", data=np.concatenate(label).reshape(-1, h, w))


def convert_h5(data_dir, label_dir, data_split, f):

    if data_split:
        train_file_paths, test_file_paths = apply_split(data_split, data_dir, label_dir)
    else:
        raise ValueError('Please provide the split ratio')

    print("Training dataset size: ", len(train_file_paths))
    print("Testing dataset size: ", len(test_file_paths))

    # data_train = list of 3D numpy array of training volumes
    # label_train = list of 3D numpy array of training labels
    # _ = list of header of training volumes
    print("Loading and pre-processing Training data...")
    data_train, label_train, _ = du.load_dataset(train_file_paths)
    _write_h5(data_train, label_train, f, mode="train")

    print("Loading and pre-processing Testing data...")
    data_test, label_test, _ = du.load_dataset(test_file_paths)
    _write_h5(data_test, label_test, f, mode="test")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--label_dir')
    parser.add_argument('--data_split')
    parser.add_argument('--destination_folder')

    args = parser.parse_args()

    f = {
        'train': {
            "data": os.path.join(args.destination_folder, "Data_train.h5"),
            "label": os.path.join(args.destination_folder, "Label_train.h5")
        },
        'test': {
            "data": os.path.join(args.destination_folder, "Data_test.h5"),
            "label": os.path.join(args.destination_folder, "Label_test.h5")
        }
    }

    convert_h5(args.data_dir, args.label_dir, args.data_split, f)
