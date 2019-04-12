import numpy as np
import nibabel as nib

def normalize_data(data, mean, std):
    data -= mean
    data /= std
    return data

def normalize_data_storage(data_storage):
    modality='/t1ce.nii.gz'
    means = list()
    stds = list()
    for subject in case_arr:
        img = nib.load(subject+modality)
        imgU16 = img.get_data().astype(np.int16)
        means.append(imgU16.mean(axis=(0, 1, 2)))
        stds.append(imgU16.std(axis=(0, 1, 2)))

    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    count = 0
    for subject in case_arr:
        img = nib.load(subject+modality)
        data = img.get_data().astype(np.float64)
        data_n = normalize_data(data, mean, std)
        data_n[data_n < 0.0] = 0;
        print('Case ' + str(count) + ' done')
        image = nib.Nifti1Image(data_n, img.affine, img.header)
        nib.save(image , subject+ '/' + 't1ce_n.nii.gz')
        count = count + 1

cases = '/rfanfs/pnl-zorro/home/sq566/pycharm/brats/3DUnetCNN/brats/data/original/case.txt'
with open(cases) as f:
    case_arr = f.read().splitlines()

normalize_data_storage(case_arr)
