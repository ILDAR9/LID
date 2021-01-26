import itertools as it
import os
import os.path
from multiprocessing import Pool

import numpy as np
import yaml

HERE: str = os.path.abspath(os.path.dirname(__file__))

LABELS = ["ru", "en", "de"]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_arguments(parser):
    a = parser.add_argument

    a('--store', dest='store', default='/Users/a18180846/projects/data', help='%(default)s')
    a('--model_store', type=str, default='./models', help='Model store')
    a('--settings', dest='settings_path', help='experiment settings yam file')


def parse_dimensions(s):
    pieces = s.split('x')
    return tuple(int(d) for d in pieces)


def to_bool(s) -> bool:
    if str(s).lower() in ['1', 'true']:
        return True
    elif str(s).lower() in ['0', 'false']:
        return False
    raise ValueError(f"Non boolean variable: '{s}'")


parsers = {
    'pool': parse_dimensions,
    'kernel': parse_dimensions,
    'conv_size': parse_dimensions,
    'downsample_size': parse_dimensions,
    'augment': to_bool,
    'noise_reduction': to_bool,
    'use_strides': to_bool
}


def load_yaml(path):
    assert path.endswith('.yml')
    with open(path, 'r') as config_file:
        settings = yaml.safe_load(config_file.read())
    # value type checking
    for k in settings:
        if k in parsers:
            settings[k] = parsers[k](settings[k])
    return settings


def load_config(config_fname: str):
    p = os.path.join(HERE, "configs")
    p = os.path.abspath(p)
    log_config_path = os.path.join(p, config_fname)
    return load_yaml(log_config_path)


def load_settings(conf_path: str, default_conf_name: str):
    experiment_conf = load_yaml(conf_path)
    default = load_config(default_conf_name)
    feature_settings = dict()
    for k in default.keys():
        feature_settings[k] = experiment_conf.get(k, default[k])
    return feature_settings


def parallelize_dataframe(df, func, settings, n_cores):
    assert n_cores >= 1 and n_cores <= 5
    if n_cores == 1:
        func((df, settings))
        return
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    pool.map(func, list(zip(df_split, it.repeat(settings))))
    pool.close()
    pool.join()
