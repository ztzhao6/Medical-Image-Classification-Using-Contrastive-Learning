import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from skimage import morphology, transform
import nrrd


def delete_useless_area(data_array):
    connect_areas = morphology.label(data_array, connectivity=3)

    sum_tmp = []
    j = 1
    while True:
        tmp = np.sum(connect_areas == j)
        if tmp == 0:
            break
        else:
            sum_tmp.append(tmp)
            j = j + 1

    label = np.zeros(data_array.shape, dtype=np.int16)
    num = np.argmax(sum_tmp) + 1
    label[connect_areas == num] = 1
    return label


def enlarge(data, expected_layer):
    z, x, y = data.shape
    layer_num = [expected_layer // z + 1] * z
    p1 = 0
    p2 = z - 1
    flag = 1
    while sum(layer_num) != expected_layer:
        if flag == 1:
            layer_num[p1] -= 1
            p1 += 1
            flag = 2
        elif flag == 2:
            layer_num[p2] -= 1
            p2 -= 1
            flag = 1

    new_data = np.zeros((expected_layer, x, y), dtype=np.float64)
    index = 0
    for old_index, layer in enumerate(layer_num):
        while layer != 0:
            new_data[index] = data[old_index, :, :]
            layer -= 1
            index += 1
    return new_data


def shrink(data, cut_ratio):
    z, x, y = data.shape
    expected_layer = len(cut_ratio)
    new_data = np.zeros((expected_layer, x, y), dtype=np.float64)

    for index, cut in enumerate(cut_ratio):
        new_data[index] = data[int(z * cut / expected_layer), :, :]
    return new_data


# kits_origin
data_root = "F:/Data/kits_origin/1. origin_data/"
save_processed_root = "F:/Data/kits_origin/3. processed_data/1. deepnet_input_data/"
expected_layer = 10
expectedXY = 64
cut_ratio = [1, 2, 3, 4, 4.5, 5.5, 6, 7, 8, 9]

for data_name in os.listdir(data_root):
    data = nib.load(data_root + data_name + "/imaging.nii.gz").get_fdata()
    label = nib.load(data_root + data_name + "/segmentation.nii.gz").get_data()

    # process label
    label[label == 1] = 0
    label[label == 2] = 1
    label = delete_useless_area(label)

    # process data
    z, x, y = np.where(label > 0)
    data = data[min(z):max(z) + 1, min(x):max(x) + 1, min(y):max(y) + 1]
    data = data + 1000
    data[data < 800] = 800
    data[data > 1300] = 1300
    if data.shape[0] < expected_layer:
        data = enlarge(data, expected_layer)
    elif data.shape[0] > expected_layer:
        data = shrink(data, cut_ratio)

    data = transform.resize(data, (expected_layer, expectedXY, expectedXY), anti_aliasing=True)

    # save data
    nrrd.write(save_processed_root + data_name + ".nrrd", data)


# seu_kidney
data_root = "F:/Data/seu_kidney/3. processed_data/1. origin_reformat_data/data/"
label_root = "F:/Data/seu_kidney/3. processed_data/1. origin_reformat_data/label/"
data_names = [data_name.split(".")[0] for data_name in os.listdir(data_root)
              if data_name[-3:] == "mhd"]
save_processed_root = "F:/Data/seu_kidney/5. group_data/classification_data/"
expected_layer = 30
expectedXY = 64
cut_ratio = [4, 5, 6, 7, 8, 9, 10, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 16, 16.5, 17, 17.5,
             18, 18.5, 19, 20, 21, 22, 23, 24, 25, 26]

for data_name in data_names:
    data = sitk.ReadImage(data_root + data_name + ".mhd")
    data = sitk.GetArrayFromImage(data)
    label = sitk.ReadImage(data_root + data_name + "_Label.mhd")
    label = sitk.GetArrayFromImage(label)
    label = delete_useless_area(label)

    # process data
    z, x, y = np.where(label > 0)
    data = data[min(z):max(z) + 1, min(x):max(x) + 1, min(y):max(y) + 1]
    data[data < 800] = 800
    data[data > 1300] = 1300
    if data.shape[0] < expected_layer:
        data = enlarge(data, expected_layer)
    elif data.shape[0] > expected_layer:
        data = shrink(data, cut_ratio)

    data = transform.resize(data, (expected_layer, expectedXY, expectedXY), anti_aliasing=True)

    # save data
    nrrd.write(save_processed_root + data_name + ".nrrd", data)