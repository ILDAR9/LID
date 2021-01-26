from collections import OrderedDict

import librosa


class AudioAugmentation:
    """
    Propose 12 modifications
    """

    def __init__(self, settings):
        self.SAMPLE_RATE = settings['sample_rate']
        aug_count = settings['augmentations']
        assert aug_count <= 12 and aug_count > 0
        ts_count = aug_count // 2
        ps_count = aug_count // 2 + aug_count % 2
        self.ts = [0.93, 1.07, 0.81, 1.23][: ts_count]  # sorted in priority
        self.ps = [-1, 1, 2, -2, 2.5, -2.5, 3.5, -3.5][:ps_count]  # sorted in priority

    def augmentations(self, audio):
        out = OrderedDict()
        for i, stretch in enumerate(self.ts, 1):
            name = f'ts{i}'
            out[name] = librosa.effects.time_stretch(audio, stretch)

        for i, shift in enumerate(self.ps, 1):
            name = f'ps{i}'
            out[name] = librosa.effects.pitch_shift(audio, self.SAMPLE_RATE, shift)

        return out
