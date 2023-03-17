import numpy as np
import os
from PIL import Image
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def parse_option():

    csv_path = "F:/select_classify_handcrafted_features.CSV"
    save_split_root = "F:/"
    n_splits = 5
    test_size = 0.2

    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--csv_path", type=str, default=csv_path)
    parser.add_argument("--save_split_root", type=str, default=save_split_root)
    parser.add_argument("--n_splits", type=int, default=n_splits, help="n split")
    parser.add_argument("--test_size", type=float, default=test_size, help="test size")

    opt = parser.parse_args()

    return opt


def split_data(arg):
    data_frame = pd.read_csv(arg.csv_path)
    sss = StratifiedShuffleSplit(n_splits=arg.n_splits, test_size=arg.test_size, random_state=20)
    split_num = 0
    columns_name = data_frame.columns.tolist()
    for train_index, test_index in sss.split(data_frame['Image'], data_frame['Label']):
        X_train_information = data_frame.iloc[train_index, :]
        X_test_information = data_frame.iloc[test_index, :]

        # X_train_information = np.column_stack([X_train, y_train])
        X_train_csv = pd.DataFrame(X_train_information, columns=columns_name)
        X_train_csv.to_csv(arg.save_split_root + 'train_' + str(split_num) + '.CSV', index=False)

        # X_test_information = np.column_stack([X_test, y_test])
        X_test_csv = pd.DataFrame(X_test_information, columns=columns_name)
        X_test_csv.to_csv(arg.save_split_root + 'test_' + str(split_num) + '.CSV', index=False)
        split_num += 1


if __name__ == '__main__':
    args = parse_option()
    split_data(args)