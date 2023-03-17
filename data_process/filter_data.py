'''
filter row_data by filter_csv's Image to filtered_csv
'''

import argparse
import pandas as pd


def parse_option():
    filter_csv_path = "F:/12-24-3DCode/self_supervised_list.CSV"
    filtered_csv_path = "F:/12-24-3DCode/hand_features2D+3Dshape.CSV"
    save_csv_path = "F:/12-24-3DCode/self_supervised_features.CSV"

    parser = argparse.ArgumentParser()
    parser.add_argument("--filter_csv_path", type=str, default=filter_csv_path,
                        help="filter_csv_path")
    parser.add_argument("--filtered_csv_path", type=str, default=filtered_csv_path,
                        help="filtered_csv_path")
    parser.add_argument("--save_csv_path", type=str, default=save_csv_path,
                        help="save_csv_path")

    opt = parser.parse_args()
    return opt

def csv_row_select(all_data_path, select_data_path):
    all_data = pd.read_csv(all_data_path)
    select_data = pd.read_csv(select_data_path)
    select_data_names = list(select_data["Image"])

    drop_list = []
    for i in range(0, len(all_data)):
        if all_data.iloc[i]["Image"] not in select_data_names:
            drop_list.append(i)
    precessed_data = all_data.drop(drop_list)
    return precessed_data

def main():
    args = parse_option()

    save_csv_data = csv_row_select(args.filtered_csv_path, args.filter_csv_path)
    save_csv_frame = pd.DataFrame(save_csv_data)
    save_csv_frame.to_csv(args.save_csv_path, index=False)


if __name__ == '__main__':
    main()