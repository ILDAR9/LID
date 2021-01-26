import argparse
import os
from collections import namedtuple

import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from lid import common, preprocessing, featurization


class LID:
    Sample = namedtuple('Sample', 'mel_count fullpath')

    def __init__(self, settings_path, user_settings):
        """
        1. preprocessor; 2. featurizer; 3. settings; 4. load model; 5. loader
        """
        settings = common.load_settings(settings_path, default_conf_name='preprocess.yml')
        self.preprocessor = preprocessing.AudioPreprocessor(self._apply_user_settings(settings, user_settings))
        del settings

        settings_feature = common.load_settings(settings_path, default_conf_name='feature.yml')
        self.featurizer = featurization.AudioFeature(self._apply_user_settings(settings_feature, user_settings))

        settings_train = common.load_settings(settings_path, default_conf_name='train.yml')
        settings_model = common.load_settings(settings_path, default_conf_name=settings_train['model_conf'])
        settings = dict(settings_train, **settings_model)
        settings.update(settings_feature)
        self.settings = self._apply_user_settings(settings, user_settings)

        # Prepare predictor
        self.model = keras.models.load_model(self.best_model_fname)
        self.loader = lambda mels, start_time=None: self.featurizer.load_sample(mels, window_frames=self.settings['frames'],
                                                                                start_time=start_time, normalize=self.settings['normalize'])

    @staticmethod
    def _apply_user_settings(settings, user_settings):
        return {k: user_settings.get(k, settings[k]) for k in settings}

    @property
    def best_model_fname(self):
        prefix_name = f"model-{self.settings['model']}_hop{self.settings['hop_length']}"
        model_store = os.path.join(common.HERE, os.pardir, 'models')
        model_fld_names = [n for n in os.listdir(model_store) if n.startswith(prefix_name)]

        # Choose best weights
        w_fnames = []
        for model_fld in model_fld_names:
            w_fnames += [os.path.join(model_fld, n) for n in os.listdir(os.path.join(model_store, model_fld)) if n.endswith('.hdf5')]

        if len(w_fnames) == 0:
            raise FileNotFoundError(f'No any trained weights found for the model: {prefix_name}')

        best_model_weght_fname = max(w_fnames, key=lambda n: re.search(r'-v(\d.\d{2})\.t', n).group(1))

        return os.path.join(model_store, best_model_weght_fname)

    def predict_voted(self, row_or_mels):
        if type(row_or_mels) == np.ndarray:
            sample = self.Sample(row_or_mels.shape[1], row_or_mels)
        else:
            sample = row_or_mels

        windows = self.featurizer.load_windows(sample, self.loader, overlap=self.settings['voting_overlap'], window_frames=self.settings['frames'],
                                               seconds_limit=self.settings['voted_seconds'])
        inputs = np.stack(windows)

        #     predictions = model.predict(inputs)
        predictions = self.model.predict(inputs)[:-1]
        if self.settings['voting'] == 'mean':
            p = np.mean(predictions, axis=0)
        elif self.settings['voting'] == 'majority':
            votes = np.argmax(predictions, axis=1)
            p = np.bincount(votes, minlength=3) / len(votes)
        else:
            raise NotImplementedError(f"There is not such voting method: {self.settings['voting']}")
        return p, predictions

    def predict(self, x, sr, visualize=False):
        # noise_reduction & resample
        x_preprocessed = self.preprocessor.audio_strip(x, sr)
        # ssft & mel coefficients
        mels = self.featurizer.compute_mels(x_preprocessed)
        if visualize:
            librosa.display.specshow(librosa.power_to_db(mels, ref=np.max))
            plt.show()
        # generated windows features
        p, predictions = self.predict_voted(mels)

        return (common.LABELS[np.argmax(p)], np.max(p)), list(map(common.LABELS.__getitem__, np.argmax(predictions, axis=1)))

    def predict_sample(self, sample_row):
        p, _ = self.predict_voted(sample_row)
        return np.argmax(p)

    def evaluate(self, data):
        y_true = data.Y.tolist()
        y_pred = [self.predict_sample(row) for _, row in tqdm(data.iterrows(), total=data.shape[0])]
        report = classification_report(y_true, y_pred, target_names=common.LABELS)
        print(report)
        confusion = confusion_matrix(y_true, y_pred)
        return confusion


def parse(args):
    parser = argparse.ArgumentParser(description='Test trained models')
    a = parser.add_argument

    common.add_arguments(parser)

    a('--run', dest='run', default='', help='%(default)s')
    a('--out', dest='results_dir', default='./data/results', help='%(default)s')

    parsed = parser.parse_args(args)

    return parsed


def main():
    setting_path = r"experiments/ldcnn32mel512hop.yml"
    user_settings = {'vad_level': 1, 'noise_reduction': False,
                     'voted_seconds': 10, 'voting_overlap': 0.3, 'voting': 'majority'}
    predictor = LID(setting_path, user_settings)
    audio_path = '/Users/a18180846/projects/data/ru/clips/common_voice_ru_19787725.mp3'
    x, sr = librosa.load(audio_path)
    label, window_labels = predictor.predict(x, sr, False)
    print(label, window_labels)


if __name__ == '__main__':
    main()
