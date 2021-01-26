import os
import warnings

import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import pandas as pd
import webrtcvad
import argparse
from tqdm import tqdm

from lid import common

warnings.filterwarnings('ignore')


class AudioPreprocessor:

    def __init__(self, settings):
        self.FRAME_VAD_DURATION = settings['frame_vad_duration']
        self._do_noise_reduction = settings['noise_reduction']
        self.SAMPLE_RATE = settings['sample_rate']
        self._vad = webrtcvad.Vad(settings['vad_level'])
        self.VAD_CHUNK = self.SAMPLE_RATE // 1000 * self.FRAME_VAD_DURATION
        self.MINIMUM_DIFF = settings['minimum_diff']

    def float_to_pcm16(self, audio):
        ints = (audio * 32767).astype(np.int16)
        little_endian = ints.astype('<u2')
        return little_endian.tostring()

    def audio_strip(self, orig_x, orig_sr) -> np.array:
        """
        Trim silence from both side
        """
        # Noise reduction
        if self._do_noise_reduction:
            orig_x = nr.reduce_noise(audio_clip=orig_x, noise_clip=orig_x)

        # Resampling
        x = librosa.resample(orig_x, orig_sr=orig_sr, target_sr=self.SAMPLE_RATE)

        # Voice activity detection
        start_sample = 0
        stop_sample = x.shape[0]
        # begin
        for i in range(x.shape[0] // self.VAD_CHUNK):
            frame_buf = self.float_to_pcm16(x[self.VAD_CHUNK * i: self.VAD_CHUNK * (i + 1)])
            if self._vad.is_speech(frame_buf, self.SAMPLE_RATE):
                start_sample = (i + 1) * self.VAD_CHUNK
                break
        # end
        for i in range(x.shape[0] // self.VAD_CHUNK - 1, start_sample // self.VAD_CHUNK, -1):
            frame_buf = self.float_to_pcm16(x[self.VAD_CHUNK * i: self.VAD_CHUNK * (i + 1)])
            if self._vad.is_speech(frame_buf, self.SAMPLE_RATE):
                stop_sample = i * self.VAD_CHUNK
                break
        # less than 120 ms
        if stop_sample - start_sample <= self.MINIMUM_DIFF * (self.SAMPLE_RATE // 1000):
            return None

        return x[start_sample:stop_sample]


def preprocess_dataframe(args):
    df, settings = args
    vad = AudioPreprocessor(settings)

    for in_fpath, out_fpath in tqdm(df[['mp3', 'npy']].values):
        try:
            x, sr = librosa.load(in_fpath)
            res = vad.audio_strip(x, sr)
            if res is not None:
                np.save(out_fpath, res)
        except Exception as e:
            print(in_fpath, e)


def prepare_meta_df(settings):
    data_fld = os.path.join(settings['store'], settings['lang'])
    mp3_files = [_ for _ in os.listdir(os.path.join(data_fld, 'clips')) if _.endswith('mp3')]
    mp3_files.sort()
    npy_fld_path = os.path.join(data_fld, settings['out_fld'])
    common.ensure_dir(npy_fld_path)

    df = pd.DataFrame({'mp3': [os.path.join(data_fld, "clips", p) for p in mp3_files],
                       'npy': [os.path.join(npy_fld_path, p[:-4] + '.npy') for p in mp3_files]})
    print(f"DataFrame created ({df.shape}): {df.columns}")
    if not settings['force']:
        npy_exists = {_ for _ in os.listdir(npy_fld_path) if _.endswith('.npy')}
        df = df[~(df.npy.apply(os.path.basename)).isin(npy_exists)].copy()
        print(f"df cleared: {df.shape}")
    return df


def parse():
    parser = argparse.ArgumentParser(description='Preprocess audio files')
    common.add_arguments(parser)
    a = parser.add_argument

    a('--jobs', type=int, default=1, help='Number of parallel jobs')
    a('--force', type=bool, default=False, help='Always recompute features')
    a('--lang', type=str, help="Language")

    return parser.parse_args()


def main():
    """
    parse -> load settings -> prepare DataFrame with meta info -> run preprocess
    """
    args = parse()
    settings = common.load_settings(args.settings_path, default_conf_name='preprocess.yml')
    settings['store'] = args.store
    settings['lang'] = args.lang
    settings['force'] = args.force
    df = prepare_meta_df(settings)
    common.parallelize_dataframe(df, preprocess_dataframe, settings, n_cores=args.jobs)


if __name__ == '__main__':
    main()
