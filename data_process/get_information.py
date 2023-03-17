import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from skimage import morphology


def get_shape(data_array):
    z, x, y = np.where(data_array > 0)
    return max(z) - min(z) + 1, max(x) - min(x) + 1, max(y) - min(y) + 1


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


# kits_origin
# data_root = "F:/Data/kits_origin/1. origin_data/"
#
# for data_name in os.listdir(data_root):
#     data = nib.load(data_root + data_name + "/imaging.nii.gz")
#     label = nib.load(data_root + data_name + "/segmentation.nii.gz")
#     header = data.header
#
#     data = data.get_fdata()
#     label = label.get_data()
#
#     label[label == 1] = 0
#     label[label == 2] = 1
#     label = delete_useless_area(label)
#
#     print(data_name, data.shape, header.get_zooms(), np.max(data), np.min(data),
#           np.max(data[label == 1]), np.min(data[label == 1]), get_shape(label))


# seu_kidney_107
data_root = "F:/Data/seu_kidney/3. processed_data/2. self_labeled_data/"
data_names = [data_name[:-4] for data_name in os.listdir(data_root + "data/")
              if data_name[-3:] == "mhd"]
for data_name in data_names:
    data = sitk.ReadImage(data_root + "data/" + data_name + ".mhd")
    space_z, space_y, space_x = data.GetSpacing()
    data = sitk.GetArrayFromImage(data)

    label = sitk.ReadImage(data_root + "label/" + data_name + "_Label.mhd")
    label = sitk.GetArrayFromImage(label)
    label = delete_useless_area(label)

    print(data_name, data.shape, space_z, space_y, space_x, np.max(data), np.min(data),
          np.max(data[label == 1]), np.min(data[label == 1]), get_shape(label))
