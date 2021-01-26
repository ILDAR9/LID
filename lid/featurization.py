import argparse
import collections
import os.path
import warnings

import librosa
import numpy as np
import pandas as pd
from librosa.feature import melspectrogram
from tqdm import tqdm

from lid import common
from lid.augmentation import AudioAugmentation


class AudioFeature:

    def __init__(self, settings):
        self.SAMPLE_RATE = settings['sample_rate']
        self.n_fft = settings['n_fft']
        self.n_mels = settings['n_mels']
        self.hop_length = settings['hop_length']
        self.fmin = settings['fmin']
        self.fmax = settings['fmax']
        self.augmentations = settings['augmentations']
        self.AUGMENTATION_VARIATIONS = np.array(['ts1', 'ps1', 'ts2', 'ps2', 'ts3', 'ps3', 'ts4', 'ps4', 'ts5', 'ts6', 'ts7', 'ts8'])
        self.Sample = collections.namedtuple('Sample', 'start end slice_file_name')

    def zero_crossing(self, x):
        """x is expected to be 1 second length"""
        return librosa.zero_crossings(x, pad=False)

    def compute_mels(self, x):
        mels = melspectrogram(x, sr=self.SAMPLE_RATE,
                              n_mels=self.n_mels,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              fmin=self.fmin,
                              fmax=self.fmax)
        return mels

    def load_sample(self, sample_path_or_mels, window_frames, augment=False, start_time=None, normalize='meanstd'):
        if type(sample_path_or_mels) is str:
            sample_path = sample_path_or_mels
            aug = None
            orig_path = sample_path
            if augment and self.augmentations > 0:
                # Choose a random augmentation
                aug = np.random.randint(-1, self.augmentations)
                if aug == -1:
                    aug = None
            if aug is not None:
                sample_path = sample_path.replace('npz', f'aug_{self.AUGMENTATION_VARIATIONS[aug]}.npz')

            # Choose a random mel
            try:
                mels = np.load(sample_path)['arr_0']
            except FileNotFoundError:
                # warnings.warn(f"Not found augmented thus load original: {sample_path}")
                mels = np.load(orig_path)['arr_0']
            assert mels.shape[0] == self.n_mels, mels.shape
        else:
            mels = sample_path_or_mels

        if start_time is None:
            # Sample a window in time randomly
            min_start = max(0, mels.shape[1] - window_frames)
            if min_start == 0:
                start_mel = 0
            else:
                start_mel = np.random.randint(0, min_start)
        else:
            start_mel = int(start_time * (self.SAMPLE_RATE / self.hop_length))

        end = start_mel + window_frames
        mels = mels[:, start_mel:end]

        # Normalize the window
        if mels.shape[1] > 0:
            if normalize == 'max':
                mels /= (np.max(mels) + 1e-9)
                mels = librosa.core.power_to_db(mels, top_db=80)
            elif normalize == 'meanstd':
                mels = librosa.core.power_to_db(mels, top_db=80)
                mels -= np.mean(mels)
                mels /= (np.std(mels) + 1e-9)
            else:
                mels = librosa.core.power_to_db(mels, top_db=80, ref=0.0)
        else:
            print('Warning: Sample {} with start_time {} has 0 length'.format(sample_path, start_time))

        # Pad to standard size
        if window_frames is None:
            padded = mels
        else:
            padded = np.full((self.n_mels, window_frames), 0.0, dtype=float)
            inp = mels[:, 0:min(window_frames, mels.shape[1])]
            padded[:, 0:inp.shape[1]] = inp

        # add 1 channel dimension
        data = np.expand_dims(padded, -1)
        return data

    def sample_windows(self, length, window_frames, overlap=0.5, start=0):
        """Split @samples into a number of windows of samples
        with length @frame_samples * @window_frames
        """

        ws = self.hop_length * window_frames
        while start < length:
            end = min(start + ws, length)
            yield int(start), int(end)
            start += ws * (1 - overlap)

    def load_windows(self, sample, loader, overlap, window_frames, start=0, seconds_limit=None):
        windows = []
        length = sample.mel_count * self.hop_length
        if seconds_limit is not None:
            length = min(length, self.SAMPLE_RATE * seconds_limit)

        for win in self.sample_windows(length, window_frames, overlap=overlap, start=start):
            d = loader(sample.fullpath, start_time=win[0] / self.SAMPLE_RATE)
            windows.append(d)

        return windows


def featurize_dataframe(args):
    df, settings = args
    do_augment = settings['augmentations'] > 0
    force = settings['force']
    featurizer = AudioFeature(settings)
    augmentizer = AudioAugmentation(settings)

    for in_fpath, out_fpath in tqdm(df[['npy', 'npz']].values):
        try:
            x = np.load(in_fpath)
            feats = featurizer.compute_mels(x)
            np.savez(out_fpath, feats)
        except Exception as e:
            print(in_fpath, e)
            continue

        if do_augment:
            if not force and os.path.exists(out_fpath.replace('.npz', '.aug_ts1.npz')):
                # if at least one augmentation exists we will pass
                continue

            aug_d = augmentizer.augmentations(x)

            for aug_name, augdata in aug_d.items():
                aug_fpath = out_fpath.replace('.npz', f'.aug_{aug_name}.npz')
                aug_features = featurizer.compute_mels(augdata)
                np.savez(aug_fpath, aug_features)


def prepare_meta_df(settings):
    df = pd.read_csv(settings['preprocessed_valid_path'])
    data_fld = os.path.join(settings['store'], settings['lang'])
    features_path = os.path.join(data_fld, settings['feature_out'])
    common.ensure_dir(features_path)

    df = df[df.label == settings['lang']]
    df['npy'] = df.path.apply(lambda p: os.path.join(data_fld, settings['preprocess_out'], p))
    df['npz'] = df.path.apply(lambda p: os.path.join(features_path, p.replace('npy', 'npz')))
    print(f"DataFrame created ({df.shape}): {df.columns}")
    if not settings['force']:
        npz_exists = {_ for _ in os.listdir(features_path) if _.endswith('.npz')}
        df = df[~(df.npz.apply(os.path.basename)).isin(npz_exists)]
        df.reset_index(inplace=True, drop=True)
        print(f"df cleared: {df.shape}")
    df = df.sample(n=settings['sample_count'], random_state=1)  # ToDo remove
    return df.copy()


def parse():
    parser = argparse.ArgumentParser(description='Feature generation')

    common.add_arguments(parser)
    a = parser.add_argument

    a('--jobs', type=int, default=1, help='Number of parallel jobs')
    a('--force', type=bool, default=False, help='Always recompute features')
    a('--lang', type=str, help="Language")

    return parser.parse_args()


def main():
    """
    parse -> load settings -> prepare DataFrame with meta info -> run featurizer
    """
    args = parse()

    settings = common.load_settings(args.settings_path, default_conf_name='feature.yml')
    settings['store'] = args.store
    settings['lang'] = args.lang
    settings['force'] = args.force

    df = prepare_meta_df(settings)
    common.parallelize_dataframe(df, featurize_dataframe, settings, n_cores=args.jobs)


if __name__ == '__main__':
    main()
