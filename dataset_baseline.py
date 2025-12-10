import json
import os
import pandas as pd
import numpy as np
import psutil
from tqdm import tqdm
from multiprocessing import Pool, freeze_support

import argparse

from txparser.transforms import (
    apply_tr_vec,
    flip_idx,
    NotEnoughPoints,
    norm_points,
    x_flip,
    get_teeth_transform,
)

class FileDoesNotExists(Exception):
    pass

class Dataset:

    def __init__(self, dataset_root: str, raw_folder: str):
        self._dataset_folder = os.path.join(dataset_root)
        self._root = os.path.join(dataset_root, raw_folder)

        self._files = [
            x for x in os.listdir(self._root) \
                if not x.endswith('-asis.json') and x.endswith('.json')
        ]

    def _make_template(self, num=32) -> dict:
        result = {'file_id': '', 'norm': 1, 'jaw': 0}

        for fidx in range(3):
            result.update({f'origin_{idx}_{fidx}': None for idx in range(1, num+1)})

        return result


    def _make_row(self, norm, jaw, file_id, teeth, gap, flip=False, last_phase=None):
        row = self._make_template()

        row['file_id'] = file_id
        row['norm'] = norm
        row['jaw'] = jaw

        # row.update({key: val if val != 'nan' else '' for key, val in user_data.items()})

        for tid, tooth in teeth.items():

            key = str(int(tid) + gap)

            for lbl in ['origin', 'center']:
                for idx, val in enumerate(tooth[lbl]):
                    row[f'{lbl}_{key}_{idx}'] = val


            if last_phase is None or key not in last_phase:
                continue

        return row


    def _tr_teeth(self, teeth, trs):
        result = {}

        for idx, tooth in teeth.items():

            if tooth is None:
                continue

            cs = tooth['cs'] if 'cs' in tooth else tooth

            result[idx] = {
                'center': apply_tr_vec(
                    np.array([tooth['center']]) \
                , trs)[0],
                'origin': apply_tr_vec(
                    np.array([cs['origin']]) \
                , trs)[0]
            }

        return result


    def _get_teeth(self, path):
        last_phase = {}
        teeth = {}
        maxilla_normalization = []
        mandible_normalization = []

        with open(path, 'r') as file:
            data = json.load(file)

            if 'teeth' not in data or 'phases' not in data or len(data['phases']) < 2:
                return None, None, None, None

            pnames = [x['name'] for x in data['phases']]

            if ('Alignment' not in pnames[1]) \
                and ('Expansion' not in pnames[1]) \
                and any([True for x in data['phases'] if x['name'] == 'Space closure' and 'movements' in x]) \
                and 'movements' in data['phases'][-1]:
                return None, None, None, None

            teeth = data['teeth']
            last_phase = get_teeth_transform(data, 'maxilla_frames')
            last_phase.update(get_teeth_transform(data, 'mandible_frames'))

            # up 1-16
            maxilla_normalization = np.array_split(np.array(data['maxilla_normalization']), 4)
            # down 17-32
            mandible_normalization = np.array_split(np.array(data['mandible_normalization']), 4)

            last_phase = {
                idx: tooth \
                    for idx, tooth in last_phase.items()
                    if tooth is not None and 'movements' in tooth
            }

        return teeth, last_phase, mandible_normalization, maxilla_normalization


    def _read_file(self, file_name):
        path = os.path.join(self._root, file_name)
        file_id = file_name.split('.')[0]

        teeth, last_phase, mandible_normalization, maxilla_normalization = self._get_teeth(path)

        if teeth is None:
            # print('teeth not found')
            return None

        up = {idx: tooth for idx, tooth in teeth.items() if int(idx) < 17}
        down = {idx: tooth for idx, tooth in teeth.items() if int(idx) > 16}

        steps = [
            [0, up, maxilla_normalization, 0],
            [1, down, mandible_normalization, -16],
        ]

        result = []

        for jaw, teeth, jaw_norm, gap in steps:
            flip_tr = x_flip.copy()

            try:
                norm_teeth = self._tr_teeth(teeth, [jaw_norm])
                centers = {str(gap + int(idx)): tooth['center'] for idx, tooth in norm_teeth.items()}

                centers, norm, trs = norm_points(centers, jaw)

            except NotEnoughPoints as e:
                # print(e)
                return None

            nteeth = self._tr_teeth(norm_teeth, trs)
            fliped = flip_idx(self._tr_teeth(nteeth, [flip_tr]))

            row = self._make_row(
                norm,
                jaw,
                file_id,
                nteeth,
                gap=gap,
                last_phase=last_phase,
            )

            row_fliped = self._make_row(
                norm,
                jaw,
                file_id + '-flip',
                fliped,
                gap=gap,
                flip=True,
                last_phase=flip_idx(last_phase),
            )


            result += [row, row_fliped]

        return result


    def read(self):
        result = []
        cpu_count = psutil.cpu_count(logical=False)

        p = Pool(cpu_count)

        files = [x for x in self._files]

        print('pid:', os.getpid(), 'all:', len(self._files), 'left:', len(files) )

        loop = tqdm(p.imap(self._read_file, files), total=len(files), desc='Collect')

        for i, rows in enumerate(loop):
            if rows is not None:
                result += rows

        df = pd.DataFrame(result)

        return df


if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='/mnt/ml2datasets/treatment-planning/',
                        help="Dataset path if it is not default")
    parser.add_argument('--raw_data', type=str, default='/mnt/ml2datasets/treatment-planning/transforms',
                        help="Dataset path if it is not default")
    parser.add_argument('--data_file', type=str, default='baseline/data/all.csv',
                        help="Dataset file. 'baseline/data/all.csv' by default")

    args = parser.parse_args()

    dataset = Dataset(args.dataset_path, args.raw_data)

    df = dataset.read()

    df.to_csv(args.data_file, index=False)

    print('Dataset len', int(len(df.index)/2))
