import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import nrrd


class RadioDataset(Dataset):
    def __init__(self, csv_file):
        self.radio_frame = np.array(pd.read_csv(csv_file))

    def __len__(self):
        return len(self.radio_frame)

    def __getitem__(self, idx):
        radio_data = self.radio_frame[idx, 2:]
        radio_data = radio_data.astype(np.float32)

        label = self.radio_frame[idx, 1]

        return (radio_data, label)


def read_image(data_path):
    filetype = data_path.split(".")[-1]
    if filetype == "nrrd":
        img_data, header = nrrd.read(data_path)
        return img_data
    elif filetype == "PNG":
        with open(data_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = np.array(img)
            img = img.transpose((2, 0, 1))
            img = img.astype("float32")
            return img


class ImageDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None, self_supervised=False):
        self.data_root = data_root
        self.radio_frame = np.array(pd.read_csv(csv_file))
        self.transform = transform
        self.self_supervised = self_supervised

    def __len__(self):
        return len(self.radio_frame)

    def __getitem__(self, idx):
        data_name = self.radio_frame[idx, 0]
        label = self.radio_frame[idx, 1]
        data_path = os.path.join(self.data_root, data_name)
        image_data = read_image(data_path)

        if self.transform is not None:
            img = self.transform(torch.from_numpy(image_data))

        if self.self_supervised:
            img2 = self.transform(torch.from_numpy(image_data))
            img = torch.cat([img, img2], dim=0)

        img = img.float()

        if self.self_supervised:
            return (img, label, idx)
        else:
            return (img, label)


class MixDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None):
        self.data_root = data_root
        self.radio_frame = np.array(pd.read_csv(csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.radio_frame)

    def __getitem__(self, idx):
        radio_data = self.radio_frame[idx, 2:]
        radio_data = radio_data.astype(np.float32)

        data_name = self.radio_frame[idx, 0]
        label = self.radio_frame[idx, 1]
        data_path = os.path.join(self.data_root, data_name)
        image_data = read_image(data_path)

        if self.transform is not None:
            image_data = self.transform(torch.from_numpy(image_data))

        image_data = image_data.float()
        
        return (image_data, radio_data, label, idx)