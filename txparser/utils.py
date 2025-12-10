import os
import json
import numpy as np
import zipfile as zf

from plotly.utils import PlotlyJSONEncoder
from jsonpath_ng import jsonpath
from jsonpath_ng.ext import parse

phase_types = {
    'Default': 'Initial',
    '': 'Initial'
}

best_doctors = ['Aamodt', 'Purnomo']


def read_list(path):
    with open(path, 'r') as file:
        return [x[:-1] for x in file.readlines() if len(x[:-1]) > 0]

def write_json(path, data, sort_keys=True, indent=None):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, cls=PlotlyJSONEncoder, sort_keys=sort_keys, indent=indent)


def write_dict_jsonp(path, data, sort_keys=True):
    keys = data.keys()
    len_ = len(keys)
    with open(path, 'w') as json_file:
        json_file.write('{')
        for i, key in enumerate(keys):
            row_str = json.dumps(data[key], cls=PlotlyJSONEncoder, sort_keys=sort_keys)
            if i > 0:
                json_file.write(f' ')
            json_file.write(f'{json.dumps(str(key))}: {row_str}')
            if i + 1 < len_:
                json_file.write(',\n')
        json_file.write('}')


def write_iter_jsonp(path, iterator, len_, sort_keys=True):
    with open(path, 'w') as json_file:
        json_file.write('[')
        i = 0
        for obj in iterator:
            obj_str = json.dumps(obj, cls=PlotlyJSONEncoder, sort_keys=sort_keys)
            if i > 0:
                json_file.write(f' ')
            json_file.write(f'{obj_str}')
            i += 1
            if i < len_:
                json_file.write(f',\n')
        json_file.write(']')


def write_jsonp(path, data, sort_keys=True):

    if isinstance(data, dict):
        return write_dict_jsonp(path, data, sort_keys)

    iterator = None

    try:
        iterator = iter(data)
        write_iter_jsonp(path, iterator, len(data), sort_keys)
    except TypeError:
        return write_json(path, data, sort_keys)


def write(path, data):
    with open(path, 'wb') as file:
        file.write(data)

def expOrDef(path, default = None):
    exp = parse(path)

    def match(obj):
        match = exp.find(obj)
        # print(match)
        return [x.value for x in match] if len(match) > 0 else default

    return match


def expOneOrDef(path, default = None):
    exp = parse(path)

    def match(obj):
        match = exp.find(obj)

        return match[0].value if len(match) > 0 else default

    return match


def clean_value(val):
    if val is None or val == '' or val == 'None':
        return None

    if not np.isscalar(val) and len(val) == 1 and (val[0] is None or val[0] == 'None' or val[0] == 'Other'):
        return None

    if not np.isscalar(val):
        return '; '.join([str(x) for x in val if x is not None and x != 'None' and x != 'Other'])

    return val


def extract_file(src, files=[]):
    result = dict()
    try:
        with zf.ZipFile(src, 'r') as zip_ref:
            filtered_files = list([x for x in zip_ref.namelist() if not any(files) or x in files])

            for name in filtered_files:
                result[name] = zip_ref.read(name)
                # zip_ref.extract(name, target)
    except zf.BadZipFile as e:
        print('File is not a zip file', src)
    except FileNotFoundError as e:
        print('File not found', src)

    return result


def read_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)


def in_file(path, pattern):

    if not os.path.exists(path):
        return False

    with open(path) as f:
        content = f.read()

        return any([x in content for x in pattern])


def rename_types(phase_type):
    return phase_type if phase_type not in phase_types else phase_types[phase_type]
