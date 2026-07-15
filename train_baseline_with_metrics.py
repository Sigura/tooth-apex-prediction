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



def compute_mre(residuals_per_axis, norm_factor):
    """
    Compute Mean Radial Error from per-axis residuals.

    Parameters:
        residuals_per_axis: dict {0: array, 1: array, 2: array}
        norm_factor: float — scaling factor to convert back to mm

    Returns:
        mre_mean, mre_std (in mm)
    """
    euclidean_distances = np.sqrt(
        residuals_per_axis[0] ** 2 +
        residuals_per_axis[1] ** 2 +
        residuals_per_axis[2] ** 2
    ) * norm_factor

    return np.mean(euclidean_distances), np.std(euclidean_distances)


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
    
   
    residuals = np.array(y_val) - np.array(pred)
    val_indices = np.array(y_val.index)
    
    
    return [str(idx), axes_xyz[axis], rmse * norm, r2, pi*norm], tooth_type_metrics, residuals, val_indices, jaw_values, norm



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

    
    tooth_residuals = {}       # {idx: {axis: (residuals, val_indices, norm)}}
    tooth_type_residuals = {}  # {tooth_type: {axis: (residuals, norm)}} for MRE by tooth type
    

    for idx in range(1, 17):
        
        tooth_residuals[idx] = {}
       
        for axis in range(0, 3):

           
            row, tooth_metrics, residuals, val_indices, jaw_values, norm_factor = train(args, train_ds, val_ds, idx, axis)
        

            table.append(row)
            tooth_type_table.extend(tooth_metrics)

            
            tooth_residuals[idx][axis] = (residuals, val_indices, norm_factor)

            # residuals for MRE по tooth type 
            for jaw_type in [0, 1]:
                jaw_mask = jaw_values == jaw_type
                if jaw_mask.sum() > 0:
                    tt = get_tooth_type(idx, jaw_type)
                    if tt not in tooth_type_residuals:
                        tooth_type_residuals[tt] = {}
                    tooth_type_residuals[tt][axis] = (residuals[jaw_mask], norm_factor)
            

    
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


    sel = data[(data[:, 0] > 1) & (data[:, 0] < 16)]
    rmse_s, r2_s, pi_s = sel[:, [1, 2, 3]].std(axis=0, ddof=1)
    rmsex_s, r2x_s, pix_s = sel[sel[:, 4] == 0][:, [1, 2, 3]].std(axis=0, ddof=1)
    rmsey_s, r2y_s, piy_s = sel[sel[:, 4] == 1][:, [1, 2, 3]].std(axis=0, ddof=1)
    rmsez_s, r2z_s, piz_s = sel[sel[:, 4] == 2][:, [1, 2, 3]].std(axis=0, ddof=1)

    print('\n== mean ± std (ddof=1), excl. 3rd molars  [Table 2 / Table 3]')
    print(f'x : RMSE {rmsex:.4f} ± {rmsex_s:.4f} | R2 {r2x:.4f} ± {r2x_s:.4f} | PI {pix:.4f} ± {pix_s:.4f}')
    print(f'y : RMSE {rmsey:.4f} ± {rmsey_s:.4f} | R2 {r2y:.4f} ± {r2y_s:.4f} | PI {piy:.4f} ± {piy_s:.4f}')
    print(f'z : RMSE {rmsez:.4f} ± {rmsez_s:.4f} | R2 {r2z:.4f} ± {r2z_s:.4f} | PI {piz:.4f} ± {piz_s:.4f}')
    print(f'e²: RMSE {rmse:.4f} ± {rmse_s:.4f} | R2 {r2:.4f} ± {r2_s:.4f} | PI(eucl) {(pix**2 + piy**2 + piz**2)**0.5:.4f}')

   
    mre_table = []
    for idx in range(1, 17):
        if idx in tooth_residuals and len(tooth_residuals[idx]) == 3:
            common = sorted(
                set(tooth_residuals[idx][0][1]) &
                set(tooth_residuals[idx][1][1]) &
                set(tooth_residuals[idx][2][1])
            )
            if len(common) > 0:
                axis_res = {}
                for ax in range(3):
                    res, indices, nf = tooth_residuals[idx][ax]
                    mapping = dict(zip(indices, res))
                    axis_res[ax] = np.array([mapping[i] for i in common])

                mre_mean, mre_std = compute_mre(axis_res, nf)
                mre_table.append([str(idx), mre_mean, mre_std, len(common)])

    print('\n== MRE (Mean Radial Error) by tooth idx')
    print(tabulate(
        mre_table,
        headers=['Tooth', 'MRE (mm)', 'STD (mm)', 'N'],
        tablefmt='github',
        floatfmt='.2f'
    ))

    mre_arr = np.array(mre_table)
    mask = (mre_arr[:, 0].astype(int) > 1) & (mre_arr[:, 0].astype(int) < 16)
    if mask.any():
        filtered_mre = mre_arr[mask, 1].astype(float)
        print(f"\nOverall MRE (excl. 3rd molars): {filtered_mre.mean():.2f} ± {filtered_mre.std():.2f} mm")
   
    tooth_order = ['UR8', 'UR7', 'UR6', 'UR5', 'UR4', 'UR3', 'UR2', 'UR1',
                   'UL1', 'UL2', 'UL3', 'UL4', 'UL5', 'UL6', 'UL7', 'UL8',
                   'LL8', 'LL7', 'LL6', 'LL5', 'LL4', 'LL3', 'LL2', 'LL1',
                   'LR1', 'LR2', 'LR3', 'LR4', 'LR5', 'LR6', 'LR7', 'LR8']

    mre_type_table = []
    for tt in tooth_order:
        if tt in tooth_type_residuals and len(tooth_type_residuals[tt]) == 3:
            min_len = min(len(tooth_type_residuals[tt][ax][0]) for ax in range(3))
            if min_len > 0:
                axis_res = {}
                for ax in range(3):
                    axis_res[ax] = tooth_type_residuals[tt][ax][0][:min_len]
                nf = tooth_type_residuals[tt][0][1]

                mre_mean, mre_std = compute_mre(axis_res, nf)
                mre_type_table.append([tt, mre_mean, mre_std, min_len])

    print('\n== MRE (Mean Radial Error) by tooth type')
    print(tabulate(
        mre_type_table,
        headers=['Tooth Type', 'MRE (mm)', 'STD (mm)', 'N'],
        tablefmt='github',
        floatfmt='.2f'
    ))
    
    
    # calculate all metrics based on tooth type
    print('\n== Teeth acc (by tooth type)')
    
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


    functional_groups = {
        'Incisors':  ['UR1', 'UR2', 'UL1', 'UL2', 'LL1', 'LL2', 'LR1', 'LR2'],
        'Canines':   ['UR3', 'UL3', 'LL3', 'LR3'],
        'Premolars': ['UR4', 'UR5', 'UL4', 'UL5', 'LL4', 'LL5', 'LR4', 'LR5'],
        'Molars':    ['UR6', 'UR7', 'UL6', 'UL7', 'LL6', 'LL7', 'LR6', 'LR7'],
    }
    mre_groups_idx = {
        'Incisors':  [7, 8, 9, 10],
        'Canines':   [6, 11],
        'Premolars': [4, 5, 12, 13],
        'Molars':    [2, 3, 14, 15],
    }
    mre_by_idx = {int(r[0]): float(r[1]) for r in mre_table}

    group_table = []
    for g, types in functional_groups.items():
        sub = np.array([[r[2], r[3], r[4]] for r in tooth_type_table if r[0] in types], dtype=float)
        m, s = sub.mean(axis=0), sub.std(axis=0, ddof=1)
        mre_vals = np.array([mre_by_idx[i] for i in mre_groups_idx[g] if i in mre_by_idx], dtype=float)
        mre_str = f'{mre_vals.mean():.2f} ± {mre_vals.std(ddof=1):.2f}' if len(mre_vals) > 1 else f'{mre_vals.mean():.2f}'
        group_table.append([
            g,
            f'{m[0]:.4f} ± {s[0]:.4f}',
            f'{m[1]:.4f} ± {s[1]:.4f}',
            f'{m[2]:.4f} ± {s[2]:.4f}',
            mre_str,
        ])

    print('\n== [Table 4] Functional tooth groups: mean ± std (ddof=1), excl. 3rd molars')
    print(tabulate(
        group_table,
        headers=['Group', 'RMSE', 'R2', 'PI(75%)', 'MRE (mm)'],
        tablefmt='github'
    ))