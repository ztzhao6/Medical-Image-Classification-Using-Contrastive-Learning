import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import argparse


def parse_option():
    train_path = ""
    result_root = ""
    # scoring = "f1_weighted"
    scoring = "f1"

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default=train_path, help="train path")
    parser.add_argument("--result_root", type=str, default=result_root, help="result root")
    parser.add_argument("--scoring", type=str, default=scoring, help="scoring")

    opt = parser.parse_args()

    return opt


def score_list(truth_labels, pred_labels, save_folder, i):
    file = open(save_folder + "result.txt", "a")
    file.write("---------  " + str(i) + "  ---------\n")
    file.write(classification_report(truth_labels, pred_labels, digits=4))
    file.write(str(confusion_matrix(truth_labels, pred_labels)))
    file.write("\n")
    file.close()


def model_pipeline():
    pipe = Pipeline([
        # ('scaler', MinMaxScaler()),
        ('univariate_select', SelectKBest(k=300)),
        ('feature_select', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=100000))),
        ('classify', LinearSVC(penalty="l1", dual=False, max_iter=100000))
    ])

    return pipe


# def get_results(args):
#     data_frame = pd.read_csv(args.train_path)
#     X = np.array(data_frame.drop(['Image', 'Label'], axis=1, inplace=False))
#     y = np.array(data_frame['Label'])
#
#     sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
#     i = 0
#     for train_index, test_index in sss.split(X, y):
#         X_train = X[train_index, :]
#         y_train = y[train_index]
#         X_test = X[test_index, :]
#         y_test = y[test_index]
#
#         pipe = model_pipeline()
#         pipe.fit(X_train, y_train)
#         # y_train_pred = pipe.predict(X_train)
#         y_test_pred = pipe.predict(X_test)
#
#         score_list(y_test, y_test_pred, args.result_root, i)
#
#         i += 1


def get_results(train_path_list, test_path, result_root):
    for i in range(len(train_path_list)):
        train_data = pd.read_csv(train_path_list[i])
        test_data = pd.read_csv(test_path)
        X_train = np.array(train_data.drop(['Image', 'Label'], axis=1, inplace=False))
        y_train = np.array(train_data['Label'])
        X_test = np.array(test_data.drop(['Image', 'Label'], axis=1, inplace=False))
        y_test = np.array(test_data['Label'])

        pipe = model_pipeline()
        pipe.fit(X_train, y_train)
        # y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        score_list(y_test, y_test_pred, result_root, i)



def main():
    # args = parse_option()
    # get_results(args)
    train_path_list = ["F:/Code/1-21-FinalCode/Experiments/Settings/kits/grad/radiomics/train_4.CSV"]
    test_path = "F:/Code/1-21-FinalCode/Experiments/Settings/kits/grad/radiomics/test_4.CSV"
    result_root = "F:/Code/1-21-FinalCode/Experiments/Settings/kits/grad/radiomics/"
    get_results(train_path_list, test_path, result_root)


if __name__ == '__main__':
    main()