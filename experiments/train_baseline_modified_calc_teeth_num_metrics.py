import os

import catboost
import pandas as pd
import numpy as np
import math
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

parts = 10
val_parts = 1
na = 0

tooth_type_mapping = {
    1: {'upper': 'UR8', 'lower': 'LL8'},
    2: {'upper': 'UR7', 'lower': 'LL7'},
    3: {'upper': 'UR6', 'lower': 'LL6'},
    4: {'upper': 'UR5', 'lower': 'LL5'},
    5: {'upper': 'UR4', 'lower': 'LL4'},
    6: {'upper': 'UR3', 'lower': 'LL3'},
    7: {'upper': 'UR2', 'lower': 'LL2'},
    8: {'upper': 'UR1', 'lower': 'LL1'},
    9: {'upper': 'UL1', 'lower': 'LR1'},
    10: {'upper': 'UL2', 'lower': 'LR2'},
    11: {'upper': 'UL3', 'lower': 'LR3'},
    12: {'upper': 'UL4', 'lower': 'LR4'},
    13: {'upper': 'UL5', 'lower': 'LR5'},
    14: {'upper': 'UL6', 'lower': 'LR6'},
    15: {'upper': 'UL7', 'lower': 'LR7'},
    16: {'upper': 'UL8', 'lower': 'LR8'},
}


def get_tooth_type(idx, jaw_value):
    """
    Determine tooth type (UR5, LL7) based on idx and jaw identifier.
    
    """
    if idx not in tooth_type_mapping:
        return None
    
    if jaw_value == 0:  # maxilla (upper jaw)
        return tooth_type_mapping[idx]['upper']
    elif jaw_value == 1:  # mandible (lower jaw)
        return tooth_type_mapping[idx]['lower']
    else:
        return None


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

    x_train, x_val = None, None
    

    df = pd.read_csv(args.data_file)
    if args.val_file is None:
        x_train, x_val = train_test_split(df, test_size=test_size, random_state=seed)
    else:
        x_train, x_val = df, pd.read_csv(args.val_file)
    x_val.to_csv(os.path.join(os.path.dirname(args.data_file), "val.csv"))
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
            'mertics': {
                'RMSE': rmse,
                'R2': r2,
            }
        }
    )
    
    # Get the jaw values from validation set to determine tooth types(numbering)
    column = y_columns[idx][axis]
    val_subset = val_ds[~val_ds[column].isna()].copy()
    jaw_values = val_subset['jaw'].values
    
   
    tooth_type_metrics = []
    for jaw_type in [0, 1]:  
        jaw_mask = jaw_values == jaw_type
        
        if jaw_mask.sum() > 0:
            tooth_type = get_tooth_type(idx, jaw_type)
            
            
            y_val_tooth = y_val.values[jaw_mask]
            pred_tooth = pred[jaw_mask]
            
            if len(y_val_tooth) > 0:
                # Calculate metrics for this specific tooth type
                rmse_tooth, r2_tooth, pi_tooth = inference(y_val_tooth, pred_tooth)
                
                tooth_type_metrics.append([
                    tooth_type,
                    axes_xyz[axis],
                    rmse_tooth * norm,
                    r2_tooth,
                    pi_tooth * norm
                ])
    
    
    return [str(idx), axes_xyz[axis], rmse * norm, r2, pi*norm], tooth_type_metrics


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
    
    
    tooth_type_table = []

    for idx in range(1, 17):
        for axis in range(0, 3):

            row, tooth_metrics = train(args, train_ds, val_ds, idx, axis)

            table.append(row)
            tooth_type_table.extend(tooth_metrics)

    
    data = np.asarray([[int(row[0]), row[2], row[3], row[4], axes_xyz.index(row[1])] for row in table])
    rmse, r2, pi = data[(data[:, 0] > 1) & (data[:, 0] < 16)][:, [1,2,3]].mean(axis=0)
    rmsex, r2x, pix = data[(data[:, 0] > 1) & (data[:, 0] < 16) & (data[:, 4] == 0)][:, [1,2,3]].mean(axis=0)
    rmsey, r2y, piy = data[(data[:, 0] > 1) & (data[:, 0] < 16) & (data[:, 4] == 1)][:, [1,2,3]].mean(axis=0)
    rmsez, r2z, piz = data[(data[:, 0] > 1) & (data[:, 0] < 16) & (data[:, 4] == 2)][:, [1,2,3]].mean(axis=0)

    table.append(['=>', 'x', rmsex, r2x, pix])
    table.append(['=>', 'y', rmsey, r2y, piy])
    table.append(['=>', 'z', rmsez, r2z, piz])
    table.append(['=>', 'e²', rmse, r2, (pix**2 + piy**2 + piz**2)**0.5])

    print('== Teeth acc (by idx)')
    print(tabulate(table, headers=['##','axis', 'RMSE', 'R2', 'PI(75%)'], tablefmt='github'))
    
    # calculate all metrics based on tooth type
    print('\n== Teeth acc (by tooth type)')
    tooth_order = ['UR8', 'UR7', 'UR6', 'UR5', 'UR4', 'UR3', 'UR2', 'UR1',
                   'UL1', 'UL2', 'UL3', 'UL4', 'UL5', 'UL6', 'UL7', 'UL8',
                   'LL8', 'LL7', 'LL6', 'LL5', 'LL4', 'LL3', 'LL2', 'LL1',
                   'LR1', 'LR2', 'LR3', 'LR4', 'LR5', 'LR6', 'LR7', 'LR8']
    
    # 999 - if there is undefind tooth number in tooth_order order sort it to the end
    tooth_type_table_sorted = sorted(tooth_type_table, 
                                     key=lambda x: (tooth_order.index(x[0]) if x[0] in tooth_order else 999, 
                                                   axes_xyz.index(x[1])))
    
    
    tooth_data = np.asarray([
        [row[0], row[2], row[3], row[4], axes_xyz.index(row[1])]
        for row in tooth_type_table_sorted
    ], dtype=object)
    
    
    final_tooth_table = tooth_type_table_sorted.copy()
    
    
    if len(tooth_data) > 0:
        for axis_idx, axis_name in enumerate(axes_xyz):
            axis_mask = tooth_data[:, 4] == axis_idx
            if axis_mask.sum() > 0:
                axis_metrics = tooth_data[axis_mask][:, [1, 2, 3]].astype(float)
                avg_rmse = axis_metrics[:, 0].mean()
                avg_r2 = axis_metrics[:, 1].mean()
                avg_pi = axis_metrics[:, 2].mean()
                final_tooth_table.append(['=>', axis_name, avg_rmse, avg_r2, avg_pi])
        
        
        all_metrics = tooth_data[:, [1, 2, 3]].astype(float)
        rmse_avg = all_metrics[:, 0].mean()
        r2_avg = all_metrics[:, 1].mean()
        
       
        pi_x = tooth_data[tooth_data[:, 4] == 0][:, 3].astype(float).mean() if (tooth_data[:, 4] == 0).sum() > 0 else 0
        pi_y = tooth_data[tooth_data[:, 4] == 1][:, 3].astype(float).mean() if (tooth_data[:, 4] == 1).sum() > 0 else 0
        pi_z = tooth_data[tooth_data[:, 4] == 2][:, 3].astype(float).mean() if (tooth_data[:, 4] == 2).sum() > 0 else 0
        pi_e2 = (pi_x**2 + pi_y**2 + pi_z**2)**0.5
        
        final_tooth_table.append(['=>', 'e²', rmse_avg, r2_avg, pi_e2])
    
    print(tabulate(final_tooth_table, headers=['Tooth Type','axis', 'RMSE', 'R2', 'PI(75%)'], tablefmt='github'))