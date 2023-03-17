'''
columns_name = ['Image', 'Label', features]
normalize data
'''

import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


def parse_option():
    radio_features_path = "F:/Data/kits_origin/classification_features.CSV"
    save_path = "F:/Data/kits_origin/norm_classification_features.CSV"

    scaler = "MinMaxScaler"

    parser = argparse.ArgumentParser()
    parser.add_argument("--radio_features_path", type=str, default=radio_features_path,
                        help="radio_features_path")
    parser.add_argument("--save_path", type=str, default=save_path,
                        help="save_path")
    parser.add_argument("--scaler", type=str, default=scaler,
                        choices=["StandardScaler", "MinMaxScaler"])

    opt = parser.parse_args()
    return opt


def normalize_radio_data(args):
    data = pd.read_csv(args.radio_features_path)

    image_label = data.iloc[:, :2]
    features = data.iloc[:, 2:]
    columns_names = np.array(data.columns.tolist())

    if args.scaler == "StandardScaler":
        scaler = StandardScaler()
    elif args.scaler == "MinMaxScaler":
        scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # save csv
    data = np.hstack((image_label, features))
    save_frame = pd.DataFrame(data, columns=columns_names)
    save_frame.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    args = parse_option()
    normalize_radio_data(args)