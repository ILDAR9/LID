import csv

import keras
import numpy as np
import pandas as pd
from lid import common
import os
import re

def setup_keras():
    """Configure Keras Backend"""
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def history_dataframe(h):
    data = {}
    data['epoch'] = h.epoch
    for k, v in h.history.items():
        data[k] = v
    df = pd.DataFrame(data)
    return df


class LogCallback(keras.callbacks.Callback):
    def __init__(self, log_path, score_epoch):
        super().__init__()

        self.log_path = log_path
        self.score = score_epoch

        self._log_file = None
        self._csv_writer = None

    def __del__(self):
        if self._log_file:
            self._log_file.close()

    def write_entry(self, epoch, data):
        data = data.copy()

        if not self._csv_writer:
            # create writer when we know what fields
            self._log_file = open(self.log_path, 'w')
            fields = ['epoch'] + sorted(data.keys())
            self._csv_writer = csv.DictWriter(self._log_file, fields)
            self._csv_writer.writeheader()

        data['epoch'] = epoch
        self._csv_writer.writerow(data)
        self._log_file.flush()  # ensure data hits disk

    def on_epoch_end(self, epoch, logs=None):
        logs = logs.copy()

        more = self.score()  # uses current model
        for k, v in more.items():
            logs[k] = v

        self.write_entry(epoch, logs)


def batch_generator(X, Y, loader, batchsize, n_classes=3):
    """
    Keras generator for infinite lazy-loading data based on a pandas.DataFrame and uniform selection

    X: features ndarray
    Y: target column
    loader: batches
    """

    while True:
        idx = np.random.choice(len(X), size=batchsize, replace=False)
        x = [loader(fpath) for fpath in X.iloc[idx]]
        assert all(map(None.__ne__, x)), x[:3]
        x = np.array(x, dtype=np.float)
        y = keras.utils.to_categorical(Y.iloc[idx], num_classes=n_classes)
        batch = (x, y)
        yield batch


def dump_validation_data(val_gen):
    # ToDo may be remove
    Xs = []
    Ys = []

    i = 0
    for batch in val_gen:
        X, y = batch
        Xs.append(X)
        Ys.append(y)
        if i < 4:
            break
        i += 1

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    assert Xs.shape[0] > 2
    assert Xs.shape[0] == Ys.shape[0]

    np.savez('./data/test_data.npz', x_test=Xs, y_test=Ys)


def predict_voted(model, samples_df, loader, sample_featurizer, window_frames, method='mean', overlap=0.5, seconds_limit=None):
    out_mean = []
    out_majority = []
    for _, sample in samples_df.iterrows():
        windows = sample_featurizer.load_windows(sample, loader, overlap=overlap, window_frames=window_frames, seconds_limit=seconds_limit)
        inputs = np.stack(windows)

        predictions = model.predict(inputs)

        p = np.mean(predictions, axis=0)
        assert len(p) >= 3
        out_mean.append(p)

        votes = np.argmax(predictions, axis=1)
        p = np.bincount(votes, minlength=3) / len(votes)
        out_majority.append(p)

    out_mean = np.stack(out_mean)
    out_majority = np.stack(out_majority)
    assert len(out_mean.shape) == 2, out_mean.shape
    assert out_mean.shape[1] == 3, out_mean.shape
    assert out_majority.shape[1] == 3, out_majority.shape

    return out_mean, out_majority