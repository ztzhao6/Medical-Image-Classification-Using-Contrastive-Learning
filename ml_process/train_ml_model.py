import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
import argparse


def parse_option():
    train_path = "F:/12-24-3DCode/classification_features_4.CSV"
    result_path = "F:/12-24-3DCode/Experiments/HandFeatures/model_origin_2D_4.CSV"
    scoring = "f1_weighted"  # roc_auc / f1

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default=train_path, help="train csv path")
    parser.add_argument("--result_path", type=str, default=result_path, help="result path")
    parser.add_argument("--scoring", type=str, default=scoring, help="scoring")
    opt = parser.parse_args()

    return opt


def get_results(args):
    data_frame = pd.read_csv(args.train_path)
    X = data_frame.drop(['Image', 'Label'], axis=1, inplace=False)
    y = data_frame['Label']

    # pipeline definition
    pipe = Pipeline([
        ('scaler', None),
        ('univariate_select', SelectKBest(k=300)),
        ('feature_select', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=100000))),
        ('classify', LinearSVC(penalty="l1", dual=False, max_iter=100000))
    ])

    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'feature_select__estimator__C': [0.1, 0.5, 1, 2],
        'classify__C': [0.1, 0.5, 1, 2]
    }

    # train
    grid = GridSearchCV(pipe, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=10),
                        n_jobs=-3, param_grid=param_grid, iid=False, scoring=args.scoring,
                        refit=True)

    grid.fit(X, y)

    results = pd.DataFrame(grid.cv_results_)
    results.to_csv(args.result_path, index=False)


def main():
    args = parse_option()
    get_results(args)


if __name__ == '__main__':
    main()