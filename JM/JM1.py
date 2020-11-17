''''TS LAB-PROJECT
HOW USEFUL IS MACHINE LEARNING ? : A SPEAKER CLASSIFICATION PROJECT
BY DUBOQUET FLORENT AND OMEZ JEAN_MAXIME'''

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.signal as sig
from scipy.signal import lfilter as fil
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
from math import ceil
from scipy.io import wavfile
import random
import os
import cmath
from xcorr import xcorr

'''I. SIGNAL PRE-PROCESSING'''

#A. Normalization

def normalization(signal):
        return signal/max(np.abs(signal))

#B. Framing

def framing(signal,sample_frequence,frame_width,shift_width):
        sample_per_frame=int(frame_width*sample_frequence)
        sample_per_shift=int(shift_width*sample_frequence)
        signal_size=len(signal)
        frame_in_signal=ceil(signal_size/sample_per_shift)

        frames=np.zeros((frame_in_signal,sample_per_frame))

        for i in range(frame_in_signal):
                for j in range(sample_per_frame):
                        position=j+i*sample_per_shift
                        if position>=signal_size:
                                break
                        else:
                                frames[i,j]=signal[position]
        return frames

'''II. FEATURES EXTRACTION ALGORITHMS'''

#A. Signal Energy

def energy(signal):
        return sum(np.abs(signal)**2)

#B. Pitch
        #1. Voiced and unvoiced sounds

def random_select_of_5_utterances_per_speacker():
        utterances=[]

        return utterances

def plot_signal_and_energy_per_frame(utterance,frame_width, shift_width):
        sample_frequence,signal=utterance

        signal = normalization(signal)

        frames=framing(signal,sample_frequence,frame_width,shift_width)

        frames_energy = []
        for frame in frames:
                frames_energy.append(energy(frame))

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.title("Signal")
        plt.plot(signal)

        plt.subplot(2,1,2)
        plt.title("Energy per frame")
        plt.plot(frames_energy)

        plt.show()

utterances=random_select_of_5_utterances_per_speacker()
for utterance in utterances:
        plot_signal_and_energy_per_frame(utterance,90 / 1000, 90 / 1000)

        #2. Autocorrelation-Based Pitch Estimation System

def pitch_autocorrelation(signal,sample_frequence,frame_width,shift_width,threshold):
        signal=normalization(signal)

        frames=framing(signal,sample_frequence,frame_width,shift_width)

        frames_energy=[]
        for frame in frames:
                frames_energy.append(energy(frame))

        f0=[]
        for i in range(len(frames)):
                if frames_energy[i]>threshold:
                        print('voiced')
                        lags, corr=xcorr(frames[i], maxlag=int(sample_frequence/50))
                        print(lags,corr)
                        f0.append(10)
                else:
                        print('unvoiced')
                        f0.append(0)

        return f0
#pitch_autocorrelation([0, 3, 3, -4, -4, -3, 1, 3, 4, 3], 1000, 3 / 1000, 2 / 1000, 2)

        #3. Cepstrum-Based Pitch Estimation System

def pitch_cepstrum(signal,sample_frequence,frame_width,shift_width,threshold):
        signal=normalization(signal)

        frames=framing(signal,sample_frequence,frame_width,shift_width)

        frames_energy=[]
        for frame in frames:
                frames_energy.append(energy(frame))

        f0=[]
        for i in range(len(frames)):
                if frames_energy[i]>threshold:
                        print('voiced')

                        f0.append(10)
                else:
                        print('unvoiced')
                        f0.append(0)

        return f0
#pitch_cepstrum([0, 3, 3, -4, -4, -3, 1, 3, 4, 3], 1000, 3 / 1000, 2 / 1000, 2)
"""
def pitch(frames, Fs, threshold=10, maxlags=800000):
        f0 = []
        for i in range(0, len(frames)):

                if sig_energy(frames[i]) > threshold:

                        # calcul l autocorrélation (2eme element)
                        x, y, *_ = plt.acorr(frames[i], maxlags=maxlags)
                        # recherche du max local de l autocorrelogramme

                        liste_temp = argrelextrema(y, np.greater)
                        loc_max_temp = np.array(liste_temp[0])
                        loc_max = []
                        maxt = 0
                        for h in range(0, len(loc_max_temp)):
                                temp = loc_max_temp[h]
                                if b[temp] > maxt:
                                        loc_max.append(loc_max_temp[h] - maxlags)
                                        maxt = y[temp]

                        loc_max = np.array(loc_max)
                        if len(loc_max) > 1:
                                dist = 0
                                for j in range(0, len(loc_max) - 1):
                                        dist += loc_max[j + 1] - loc_max[j]
                                dist = dist / (len(loc_max) - 1)
                                tps = dist / Fs
                                f0.append(1 / tps)
                        else:
                                f0.append(0)
                else:
                        f0.append(0)
        f0 = np.array(f0)
        return f0
"""
#C. Formants

def formants(signal,sample_frequence,frame_width,shift_width)

#D. MFCC



'''IV. BUILDING A RULE-BASED SYSTEM'''



'''V. BUILDING A MACHINE LEARNINGBASED SYSTEM'''


