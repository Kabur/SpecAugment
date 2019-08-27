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
"""SpecAugment Implementation for Tensorflow.
Related paper : https://arxiv.org/pdf/1904.08779.pdf

In this paper, show summarized parameters by each open datasets in Tabel 1.
-----------------------------------------
Policy | W  | F  | m_F |  T  |  p  | m_T
-----------------------------------------
None   |  0 |  0 |  -  |  0  |  -  |  -
-----------------------------------------
LB     | 80 | 27 |  1  | 100 | 1.0 | 1
-----------------------------------------
LD     | 80 | 27 |  2  | 100 | 1.0 | 2
-----------------------------------------
SM     | 40 | 15 |  2  |  70 | 0.2 | 2
-----------------------------------------
SS     | 40 | 27 |  2  |  70 | 0.2 | 2
-----------------------------------------
LB : LibriSpeech basic
LD : LibriSpeech double
SM : Switchboard mild
SS : Switchboard strong
"""

import librosa
import librosa.display
import math
import numpy as np
import random
random.seed(12345)
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def spec_augment(spectogram, setting="swb_mild"):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
# Policy | W  | F  | m_F |  T  |  p  | m_T
# SM     | 40 | 15 |  2  |  70 | 0.2 | 2
# -----------------------------------------
# SS     | 40 | 27 |  2  |  70 | 0.2 | 2
    if setting == "swb_mild":
        time_warping_para=40
        frequency_masking_para=15
        frequency_mask_num=2
        time_masking_para=70
        time_mask_bound = 0.2
        time_mask_num=2
    elif setting == "swb_strong":
        time_warping_para=40
        frequency_masking_para=27
        frequency_mask_num=2
        time_masking_para=70
        time_mask_bound = 0.2
        time_mask_num=2
    elif setting == "libri_mild":
        time_warping_para=80
        frequency_masking_para=27
        time_masking_para=100
        frequency_mask_num=1
        time_mask_bound = 1
        time_mask_num=1
    elif setting == "libri_strong":
        time_warping_para=80
        frequency_masking_para=27
        time_masking_para=100
        frequency_mask_num=2
        time_mask_bound = 1
        time_mask_num=2
    else:
        print("Unrecognized specaugment setting, quitting...")
        exit()


    v = spectogram.shape[0]
    tau = spectogram.shape[1]
    spect_mean = np.mean(spectogram)

    # Step 1 : Time warping (TO DO...)
    # augmented_spectrogram = np.zeros(spectogram.shape, dtype=spectogram.dtype)
    #
    # for i in range(v):
    #     for j in range(tau):
    #         offset_x = 0
    #         offset_y = 0
    #         if i + offset_y < v:
    #             augmented_spectrogram[i, j] = spectogram[(i + offset_y) % v, j]
    #         else:
    #             augmented_spectrogram[i, j] = spectogram[i, j]

    augmented_spectrogram = spectogram

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v - f)
        # augmented_spectrogram[f0:f0 + f, :] = 0
        augmented_spectrogram[f0:f0 + f, :] = spect_mean

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=min(time_masking_para, time_mask_bound * tau))
        # t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau - t)
        # augmented_spectrogram[:, t0:t0 + t] = 0
        augmented_spectrogram[:, t0:t0 + t] = spect_mean

    # print("augmenting spectogram eyy")
    return augmented_spectrogram


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment

    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

