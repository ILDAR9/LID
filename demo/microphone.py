'''
brew install portaudio
'''
import collections
import sys
from array import array
from time import time

import numpy as np
import pyaudio
import pygame
import webrtcvad


class Microphone:
    SAMPLE_RATE = 16000
    CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
    PADDING_DURATION_MS = 1500  # 1 sec jugement

    def __init__(self, vad_level=1):
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION_MS / 1000)  # chunk to read
        self.CHUNK_BYTES = self.CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM

        self.vad = webrtcvad.Vad(vad_level)

        self.init_mixer()

    @classmethod
    def init_mixer(cls):
        """
        One time Pygame init
        setup mixer to avoid sound lag
        """
        pygame.mixer.pre_init(cls.SAMPLE_RATE, -16, channels=1, buffer=2048)
        pygame.mixer.init()
        pygame.init()

    def prepare_microphone(self):
        pa = pyaudio.PyAudio()
        self.stream = pa.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=self.SAMPLE_RATE,
                              input=True,
                              start=False,
                              # input_device_index=2,
                              frames_per_buffer=self.CHUNK_SIZE)

    @staticmethod
    def normalize(snd_data):
        "Average the volume out"
        MAXIMUM = 32767  # 16384
        times = float(MAXIMUM) / max(map(abs, snd_data))
        r = array('h')
        for i in snd_data:
            r.append(int(i * times))
        return r

    def run(self, max_duration=5):
        """
        @max_duration: maximum duration in seconds for recording
        """
        # --- Steve Cox
        NUM_WINDOW_CHUNKS = int(240 / self.CHUNK_DURATION_MS)
        # NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms/ 30ms  ge

        NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2
        # START_OFFSET = int(NUM_WINDOW_CHUNKS * self.CHUNK_DURATION_MS * 0.5 * self.SAMPLE_RATE)

        NUM_PADDING_CHUNKS = int(self.PADDING_DURATION_MS / self.CHUNK_DURATION_MS)

        got_a_sentence = False
        self.prepare_microphone()
        while True:
            ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
            triggered = False
            voiced_frames = []
            ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
            ring_buffer_index = 0

            ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
            ring_buffer_index_end = 0
            buffer_in = ''
            # WangS
            raw_data = array('h')
            index = 0
            start_point = 0
            start_time = time()
            print("* recording: ")
            self.stream.start_stream()

            while not got_a_sentence:
                chunk = self.stream.read(self.CHUNK_SIZE)
                # add WangS
                raw_data.extend(array('h', chunk))
                index += self.CHUNK_SIZE
                time_use = time() - start_time

                active = self.vad.is_speech(chunk, self.SAMPLE_RATE)

                sys.stdout.write('1' if active else '_')
                ring_buffer_flags[ring_buffer_index] = 1 if active else 0
                ring_buffer_index += 1
                ring_buffer_index %= NUM_WINDOW_CHUNKS

                ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
                ring_buffer_index_end += 1
                ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

                # start point detection
                if not triggered:
                    ring_buffer.append(chunk)
                    num_voiced = sum(ring_buffer_flags)
                    if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                        sys.stdout.write(' Open ')
                        triggered = True
                        start_point = index - self.CHUNK_SIZE * 20  # start point
                        ring_buffer.clear()
                # end point detection
                else:
                    ring_buffer.append(chunk)
                    num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)

                    if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or time_use > max_duration:
                        sys.stdout.write(' Close ')
                        triggered = False
                        got_a_sentence = True

                sys.stdout.flush()

            sys.stdout.write('\n')

            self.stream.stop_stream()
            # print("* done recording")
            got_a_sentence = False

            # write to file
            raw_data.reverse()
            for index in range(start_point):
                raw_data.pop()

            raw_data.reverse()
            raw_data = self.normalize(raw_data)

            # --- Steve Cox
            # --- the wav has a header, we need to strip it off before playing
            wav_data = raw_data[44:len(raw_data)]

            sound = pygame.mixer.Sound(buffer=wav_data)
            sound.play()
            # --- Wait for the sound to finish playing or we get an echo
            while pygame.mixer.get_busy():
                pass

            yield np.array(wav_data) / 32767

        self.stream.close()


def float_to_pcm16(audio):
    ints = (audio * 32767).astype(np.int16)
    little_endian = ints.astype('<u2')
    return little_endian.tostring()


if __name__ == '__main__':
    mic = Microphone()
    mic.run()
