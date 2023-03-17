import os
import radiomics
from radiomics import featureextractor
import collections
import csv


# TNSCUI
# setting_path = "./extract_features_settings/TNSCUI_settings.yaml"
# data_root = "F:/Data/TNSCUI/1. origin_data/image/"
# label_root = "F:/Data/TNSCUI/1. origin_data/mask/"
# output_file = "F:/Data/TNSCUI/4. handcrafted_features/handcrafted_features.CSV"
#
# data_names = os.listdir(data_root)
#
# extractor = featureextractor.RadiomicsFeatureExtractor(setting_path)
# headers = None
# for i in range(len(data_names)):
#     featureVector = collections.OrderedDict()
#     featureVector['Image'] = data_names[i]
#
#     imageFilePath = data_root + data_names[i]
#     maskFilePath = label_root + data_names[i]
#
#     # Calculating features
#     featureVector.update(extractor.execute(imageFilePath, maskFilePath))
#
#     with open(output_file, 'a') as outputFile:
#         writer = csv.writer(outputFile, lineterminator='\n')
#         if headers is None:
#             headers = list(featureVector.keys())
#             writer.writerow(headers)
#         row = []
#         for h in headers:
#             row.append(featureVector.get(h, "N/A"))
#         writer.writerow(row)


# kits_origin
# setting_path = "./extract_features_settings/kits_origin_2D_settings.yaml"
# data_root = "F:/Data/kits_origin/1. origin_data/"
# label_root = "F:/Data/kits_origin/1. origin_data/"
# output_file = "F:/Data/kits_origin/4. handcrafted_features/handcrafted_features.CSV"
#
# data_names = os.listdir(data_root)
#
# extractor = featureextractor.RadiomicsFeatureExtractor(setting_path)
# headers = None
# for i in range(len(data_names)):
#     featureVector = collections.OrderedDict()
#     featureVector['Image'] = data_names[i]
#
#     imageFilePath = data_root + data_names[i] + "/imaging.nii.gz"
#     maskFilePath = label_root + data_names[i] + "/segmentation.nii.gz"
#
#     # Calculating features
#     featureVector.update(extractor.execute(imageFilePath, maskFilePath))
#
#     with open(output_file, 'a') as outputFile:
#         writer = csv.writer(outputFile, lineterminator='\n')
#         if headers is None:
#             headers = list(featureVector.keys())
#             writer.writerow(headers)
#         row = []
#         for h in headers:
#             row.append(featureVector.get(h, "N/A"))
#         writer.writerow(row)


# seu_kindey
setting_path = "./extract_features_settings/seu_kidney_3D_settings.yaml"
data_root = "F:/Data/seu_kidney/3. processed_data/2. self_labeled_data/data/"
label_root = "F:/Data/seu_kidney/3. processed_data/2. self_labeled_data/label/"
output_file = "F:/Data/seu_kidney/4. handcrafted_features/handcrafted_features_self.CSV"

data_names = [name.split(".")[0] for name in os.listdir(data_root) if name.split('.')[1] == 'mhd']

extractor = featureextractor.RadiomicsFeatureExtractor(setting_path)
headers = None
for i in range(len(data_names)):
    featureVector = collections.OrderedDict()
    featureVector['Image'] = data_names[i]

    imageFilePath = data_root + data_names[i] + ".mhd"
    maskFilePath = label_root + data_names[i] + "_Label.mhd"

    # Calculating features
    featureVector.update(extractor.execute(imageFilePath, maskFilePath))

    with open(output_file, 'a') as outputFile:
        writer = csv.writer(outputFile, lineterminator='\n')
        if headers is None:
            headers = list(featureVector.keys())
            writer.writerow(headers)
        row = []
        for h in headers:
            row.append(featureVector.get(h, "N/A"))
        writer.writerow(row)