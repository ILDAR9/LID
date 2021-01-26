import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lid import common

sklearn_seed = 777


def prepare_preprocessed_dataframe(data_fld):
    """
    Prepare dataframe from preprocessed files
    """
    settings = common.load_config('preprocess.yml')
    preprocess_fld = settings['preprocessed_out']
    path_list = []
    label_list = []
    duration_list = []
    count_error = 0
    for lang in ['ru', 'en', 'de']:
        lang_prep_fld = os.path.join(data_fld, lang, preprocess_fld)
        npy_names = [_ for _ in os.listdir(lang_prep_fld) if _.endswith('.npy')]
        path_list += npy_names
        label_list += [lang] * len(npy_names)
        cur_durations = []
        for fname in tqdm(npy_names):
            try:
                x = np.load(os.path.join(lang_prep_fld, fname))
                cur_durations.append(x.shape[0])
            except ValueError as e:
                count_error += 1
                print(lang, fname, e)
                cur_durations.append(0)

        duration_list += cur_durations
    print(f"Can not load {count_error} files.")

    sr_per_ms = settings['sample_rate'] // 1000
    d = {'path': path_list,
         'label': label_list,
         'duration': duration_list}
    df = pd.DataFrame(d)
    df['duration_ms'] = df.duration.apply(lambda size: size // sr_per_ms)
    df.drop('duration', inplace=True, axis=1)
    return df


def prepare_featurized_dataframe(store_path, feature_fld):
    """
    Prepare dataframe from featurized files
    """
    path_list = []
    label_list = []
    mel_count_list = []
    count_error = 0
    for lang in ['ru', 'en', 'de']:
        lang_feat_fld = os.path.join(store_path, lang, feature_fld)
        npz_names = [p for p in os.listdir(lang_feat_fld) if p.endswith('.npz') and '.aug_' not in p]
        path_list += npz_names
        label_list += [lang] * len(npz_names)
        cur_mel_counts = []
        for fname in tqdm(npz_names):
            try:
                mels = np.load(os.path.join(lang_feat_fld, fname))['arr_0']
                cur_mel_counts.append(mels.shape[1])
            except ValueError as e:
                count_error += 1
                print(lang, fname, e)
                cur_mel_counts.append(0)

        mel_count_list += cur_mel_counts
    if count_error > 0:
        print(f"Can not load {count_error} files.")

    d = {'path': path_list,
         'label': label_list,
         'mel_count': mel_count_list}
    df = pd.DataFrame(d)

    # combine duration_ms information

    return df


def load_training_data(settings):
    df = pd.read_csv(os.path.join(common.HERE, os.pardir, 'data', settings['feature_out'] + '.csv'))
    df['Y'] = df.label.apply(common.LABELS.index)
    df['fullpath'] = df.apply(lambda row: os.path.join(settings['store'], row.label, settings['feature_out'], row.path), axis=1)
    train_df, val_df = train_test_split(df, train_size=settings['train_part'], stratify=df.Y, random_state=sklearn_seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_validation_data(setting_path, user_settings):
    settings_feat = common.load_settings(setting_path, 'feature.yml')
    settings_train = common.load_settings(setting_path, 'train.yml')
    settings = dict(settings_feat, **settings_train)
    settings['store'] = user_settings['store']
    train_df, val_df = load_training_data(settings)
    del settings_feat, settings_train
    val_df['npy'] = val_df.apply(lambda row: os.path.join(settings['store'], row.label, 'preprocess', row.path.replace('npz', 'npy')), axis=1)
    val_df['mp3'] = val_df.apply(lambda row: os.path.join(settings['store'], row.label, 'clips', row.path.replace('npz', 'mp3')), axis=1)
    print(val_df.shape)
    print('Columns', val_df.columns)
    return val_df


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Prepare training data
    data_fld = '/Users/a18180846/projects/data'
    feature_fld = 'features_mel32hop512'
    df_feat = prepare_featurized_dataframe(data_fld, feature_fld)
    print(df_feat.head())
    print(df_feat.label.value_counts())
    df_feat.mel_count.hist(by=df_feat.label)
    plt.show()
    df_feat[['path', 'label', 'mel_count']].to_csv(os.path.join(common.HERE, os.pardir, 'data', feature_fld + '.csv'), index=False)
