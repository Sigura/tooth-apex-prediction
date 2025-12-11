import os

import catboost
import pandas as pd
import numpy as np
#import math
import random
from tabulate import tabulate

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

from txparser.consts import (
    baseline_features as features,
    baseline_y_columns as y_columns,
    axes_xyz,
)

from txparser.utils import (
    write_jsonp as write_json
)

import argparse

y_columns = {
    i: [f'origin_{i}_0', f'origin_{i}_1', f'origin_{i}_2']
    for i in range(1, 33)
}

seed = 42
np.random.seed(seed)
random.seed(seed)

# parts = 10
# val_parts = 1
# na = 0

def prediction_interval(y_true, y_pred, confidence_level=0.75):
    """
    Calculate a prediction interval for a list of predictions.

    Parameters:
        y_true (list or numpy array): ground truth values
        y_pred (list or numpy array): predicted values
        confidence_level (float): confidence level for the prediction interval

    Returns:
        (tuple): lower and upper bounds of the prediction interval
    """
    from scipy.stats import norm

    if isinstance(y_true, pd.DataFrame):
        y_true.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)

        y_true, y_pred = y_true.align(y_pred)

    # Calculate residuals
    residuals = np.subtract(y_true, y_pred)

    # Calculate standard deviation of residuals
    sigma = np.std(residuals)

    # Calculate z-score for the specified confidence level
    z_score = norm.ppf((1 + confidence_level) / 2)

    # Calculate the prediction interval
    interval_radius = z_score * sigma

    return interval_radius


def read_dataset(args, test_size=0.1):
    from sklearn.model_selection import train_test_split

    #x_train, x_val = None, None

    df = pd.read_csv(args.data_file)
    if args.val_file is None:
        x_train, x_val = train_test_split(df, test_size=test_size, random_state=seed)
    else:
        x_train, x_val = df, pd.read_csv(args.val_file)

    return x_train, x_val


def split_dataset(train, val, idx, axis):
    column = y_columns[idx][axis]

    X_train = train[~train[column].isna()][features].copy()
    y_train = train[~train[column].isna()][column].copy()
    X_val = val[~val[column].isna()][features].copy()
    y_val = val[~val[column].isna()][column].copy()


    return X_train, X_val, y_train, y_val


def inference(y_val, pred):
    rmse = (np.sqrt(mean_squared_error(y_val, pred)))
    r2 = r2_score(y_val, pred)
    pi = prediction_interval(y_val, pred)

    return rmse, r2, pi


def train(args, train_ds, val_ds, idx, axis):
    # by Dudnik Artur
    X_train, X_val, y_train, y_val = split_dataset(train_ds, val_ds, idx, axis)

    norm = train_ds['norm'].mean()

    print('##', idx, axes_xyz[axis], len(X_train.index), '/', len(X_val.index), f'({len(list(X_train))})')

    model = catboost.CatBoostRegressor(
        loss_function='RMSE',
        cat_features=['jaw'],
        task_type='GPU',
        devices='0',
        iterations=args.iterations,
        random_seed=seed,
    )

    model.fit(X_train, y_train)

    model.save_model(os.path.join('models', str(idx) + '-' + str(axis) + '-' + args.model_file))

    pred = model.predict(X_val)

    rmse, r2, pi = inference(y_val, pred)

    print(idx, axis, "testing performance")
    print("RMSE: {:.2f}".format(rmse*norm))
    print("R2  : {:.2f}".format(r2))
    print("Prediction interval (75%): {:.2f}\n".format(pi*norm))

    write_json(
        os.path.join('models', f'{idx}-{axis}-{args.model_file}.json'),
        {
            'features': features,
            'y_columns': y_columns,
            'iterations': args.iterations,
            'metrics': {
                'RMSE': rmse,
                'R2': r2,
            }
        }
    )

    return [str(idx), axes_xyz[axis], rmse * norm, r2, pi*norm]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-file', type=str, default='baseline/data/all.csv',
                        help="Dataset file. 'data/all.csv' by default")
    parser.add_argument('--val-file', type=str, default=None,
                        help="Dataset file. 'data/all.csv' by default")
    parser.add_argument('--model-file', type=str, default='root.model',
                        help="Model file out. 'root.model' by default")
    parser.add_argument('--iterations', type=int, default=500,
                        help="Iterations")
    parser.add_argument('--rebuild-split', action='store_true', default=False,
                        help="Will rewrite train/val csv")

    args = parser.parse_args()

    train_ds, val_ds = read_dataset(args)
    table = []

    for idx in range(1, 17):
        for axis in range(0, 3):
            row = train(args, train_ds, val_ds, idx, axis)
            table.append(row)

    data = np.asarray([[int(row[0]), row[2], row[3], row[4], axes_xyz.index(row[1])] for row in table])
    rmse, r2, pi = data[(data[:, 0] > 1) & (data[:, 0] < 16)][:, [1,2,3]].mean(axis=0)
    rmsex, r2x, pix = data[(data[:, 0] > 1) & (data[:, 0] < 16) & (data[:, 4] == 0)][:, [1,2,3]].mean(axis=0)
    rmsey, r2y, piy = data[(data[:, 0] > 1) & (data[:, 0] < 16) & (data[:, 4] == 1)][:, [1,2,3]].mean(axis=0)
    rmsez, r2z, piz = data[(data[:, 0] > 1) & (data[:, 0] < 16) & (data[:, 4] == 2)][:, [1,2,3]].mean(axis=0)

    table.append(['=>', 'x', rmsex, r2x, pix])
    table.append(['=>', 'y', rmsey, r2y, piy])
    table.append(['=>', 'z', rmsez, r2z, piz])
    table.append(['=>', 'e²', rmse, r2, (pix**2 + piy**2 + piz**2)**0.5])

    print('== Teeth acc')

    print(tabulate(table, headers=['##','axis', 'RMSE', 'R2', 'PI(75%)'], tablefmt='github'))

