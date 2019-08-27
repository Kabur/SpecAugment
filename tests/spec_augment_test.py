# Copyright 2019 RnD at Spoon Radio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SpecAugment test"""

import argparse
import scipy.signal
import librosa
from SpecAugment.SpecAugment import spec_augment_tensorflow
from SpecAugment.SpecAugment import spec_augment_pytorch
import numpy as np
import torch
import torchaudio
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

parser = argparse.ArgumentParser(description='Spec Augment')
parser.add_argument('--audio-path', default='../data/61-70968-0002.wav',
                    help='The audio file.')
parser.add_argument('--time-warp-para', default=80,
                    help='time warp parameter W')
parser.add_argument('--frequency-mask-para', default=100,
                    help='frequency mask parameter F')
parser.add_argument('--time-mask-para', default=27,
                    help='time mask parameter T')
parser.add_argument('--masking-line-number', default=1,
                    help='masking line number')

args = parser.parse_args()
audio_path = args.audio_path
time_warping_para = args.time_warp_para
time_masking_para = args.frequency_mask_para
frequency_masking_para = args.time_mask_para
masking_line_number = args.masking_line_number


def load_ds2style(audio_path, normalize):
    windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
               'bartlett': scipy.signal.bartlett}
    sample_rate = 8000
    window_size = 0.02
    window_stride = 0.01
    window = windows.get("hamming", windows['hamming'])

    y, _ = torchaudio.load(audio_path, normalization=True)
    y = y.numpy().T
    y = y.squeeze()

    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)

    spect = spect.numpy()

    return spect


if __name__ == "__main__":

    # Step 0 : load audio file, extract mel spectrogram
    audio_path = "../data/sw2051B-ms98-a-0156.wav"
    # audio, sampling_rate = librosa.load(audio_path, sr=None)
    audio, sampling_rate = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=160,
                                                     hop_length=128,
                                                     fmax=8000)

    spectogram1 = load_ds2style(audio_path, normalize=True)
    # Show Raw mel-spectrogram
    spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=mel_spectrogram, title="Raw Mel Spectrogram")
    spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=spectogram1, title="Raw Mel Spectrogram")

    # # Calculate SpecAugment ver.tensorflow
    # warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram)
    # # print(warped_masked_spectrogram)
    #
    # # Show time warped & masked spectrogram
    # spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=warped_masked_spectrogram,
    #                                                   title="tensorflow Warped & Masked Mel Spectrogram")

    # Calculate SpecAugment ver.pytorch
    warped_masked_melspectrogram = spec_augment_pytorch.spec_augment(spectogram=mel_spectrogram)
    warped_masked_spectrogram1 = spec_augment_pytorch.spec_augment(spectogram=spectogram1)

    # Show time warped & masked spectrogram
    spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=warped_masked_melspectrogram, title="pytorch Warped & Masked Mel Spectrogram")
    spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=warped_masked_spectrogram1, title="pytorch Warped & Masked Mel Spectrogram")


