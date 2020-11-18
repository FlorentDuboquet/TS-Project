import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.signal as sig
import scipy.fftpack as fft
from scipy.io import wavfile

from math import ceil
from statistics import mean
from os import listdir
from random import randint

from xcorr import xcorr
from scikit_talkbox_lpc import lpc_ref
from filterbanks import filter_banks

import warnings
warnings.filterwarnings("ignore")

def normalization(signal):
        return signal/max(np.abs(signal))

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

def energy(signal):
        return sum(np.abs(signal)**2)

def random_select_utterances(folder_addresses,number_of_utterances):
        utterances=[]
        for folder_adresse in folder_addresses:
                file_adresses = listdir(folder_adresse)
                number_of_selection=0
                while number_of_selection<number_of_utterances:
                        file_adresse=folder_adresse+'/'+file_adresses[randint(0,len(file_adresses)-1)]
                        if file_adresse not in utterances:
                                utterances.append(file_adresse)
                                number_of_selection+=1
        return utterances

def plot_signal_and_energy_per_frame(file_adresse,frame_width, shift_width):
        sample_frequence,signal=wavfile.read(file_adresse)

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

def pitch_autocorrelation(signal,sample_frequence,frame_width,shift_width,threshold):
        signal=normalization(signal)

        frames=framing(signal,sample_frequence,frame_width,shift_width)

        frames_energy=[]
        for frame in frames:
                frames_energy.append(energy(frame))

        fundamental_frequency_per_frame=[]
        for i in range(len(frames)):
                if frames_energy[i]>threshold: #Voiced
                        try:
                                lags, corr = xcorr(frames[i], maxlag=int(sample_frequence/50))

                                peaks,properties=sig.find_peaks(corr)

                                peaks_prominences=sig.peak_prominences(signal,peaks)[0]

                                peaks_prominences=list(peaks_prominences)
                                index_max_1=peaks_prominences.index(max(peaks_prominences))

                                peaks_prominences_copy=peaks_prominences
                                del peaks_prominences_copy[index_max_1]

                                index_max_2 = peaks_prominences_copy.index(max(peaks_prominences_copy))

                                if index_max_2 >= index_max_1:
                                        index_max_2+=1

                                postion_max_1=peaks[index_max_1]
                                postion_max_2 = peaks[index_max_2]

                                distance=abs(postion_max_1-postion_max_2)

                                fundamental_period=distance/sample_frequence

                                fundamental_frequency=1/fundamental_period
                        except:
                                fundamental_frequency = 0

                else:# Unvoiced
                        fundamental_frequency=0

                fundamental_frequency_per_frame.append(fundamental_frequency)

        return fundamental_frequency_per_frame

def pitch_cepstrum(signal,sample_frequence,frame_width,shift_width,threshold):
    signal = normalization(signal)

    frames = framing(signal, sample_frequence, frame_width, shift_width)

    frames_energy = []
    for frame in frames:
        frames_energy.append(energy(frame))

    fundamental_frequency_per_frame = []
    for i in range(len(frames)):
        if frames_energy[i] > threshold:  # Voiced
            fundamental_frequency = 'to implement'

        else:  # Unvoiced
            fundamental_frequency = 0

        fundamental_frequency_per_frame.append(fundamental_frequency)

    return fundamental_frequency_per_frame

def high_Pass(signal, a=0.67):  # a est compris dans [0.62,0.67]
    filtred_signal = []
    for i in range(0, len(signal) - 1):
        if i > 0:
            filtred_signal.append(signal[i] - a * signal[i - 1])
        else:
            filtred_signal.append(signal[i])
    filtred_signal = np.array(filtred_signal) # on change le typ de la liste avant le return
    return filtred_signal

def formant(signal,sample_frequence,frame_width,shift_width):
    #pre set
    signal = normalization(signal)
    frames = framing(signal,sample_frequence,frame_width,shift_width)

    frequences=[]

    # here we will have to use the fct lpc_ref provided in
    # scikit_talkbox_lpc.py which returns LPC coefficient predictions

    # we apply the treatment to all the frames :
    for frame in frames:

        # the high-pass filter (previously defined)
        filtred_frame = high_Pass(frame)

        # calculation of the LPC thanks to the provided fct
        temp = lpc_ref(filtred_frame, order= 10) # order peut prendre des valeurs entre 8 et 13

        # compute the roots  :
        lpc = np.roots(temp)

        # only one of the two conjugated complexes is retained
        lpc = lpc[np.imag(lpc) >= 0]

        temp = []
        for j in range(0,len(lpc)) :
            # the angle is calculated and the frequency is deduced from it
            freq = np.arctan2(np.imag(lpc[j]),np.real(lpc[j])) * (sample_frequence/8*np.pi)

            # the frequency must be between the thresholds
            if (freq<20000 and freq>500):
                temp.append(freq)
                temp.sort()
        frequences.append(temp)

    # sorting
    frequences.sort()

    return frequences

def MFCC (signal, sample_frequence,frame_width,shift_width) :
    # pre set
    signal = normalization(signal)

    # preanalyse
    signal = high_Pass(signal, a=0.97)

    # framing
    signal = framing(signal,sample_frequence,frame_width,shift_width)

    # hamming
    for i in range(len(signal)-1) :
        ham = np.hamming(len(signal[i]))
        signal[i] = signal[i] * ham

    #compute the power spectrum of the signal periodogram
    powerSpectrum = []
    ntfd = 512
    for elem in signal :
        powerSpectrum.append((np.linalg.norm(np.fft.fft(elem,ntfd)**2))/ntfd)

    # passage through the filter bank
    result = filter_banks(powerSpectrum,sample_frequence)

    # Discrete cosine transformation as foreseen in the protocol
    result = fft.dct(filter_banks, type=2, axis=1, norm='ortho')

    # we keep only the first 13
    result = result[:13]

    return result

def feature_extraction (files_adresse,frame_width,shift_width,threshold):
    list_sex = []
    list_fundamental_frequency = []
    list_energy = []
    list_formant = []
    list_MFCC = []
    for file_adresse in files_adresse:
        if 'woman' in file_adresse:
            list_sex.append(1)
        else:
            list_sex.append(0)

        sample_frequence, signal = wavfile.read(file_adresse)

        list_energy.append(energy(signal))

        fundamental_frequency_voiced = []
        for fundamental_frequency in pitch_autocorrelation(signal, sample_frequence, frame_width, shift_width, threshold):
            if fundamental_frequency != 0:
                fundamental_frequency_voiced.append(fundamental_frequency)
        list_fundamental_frequency.append(mean(fundamental_frequency_voiced))
        '''
        list_temp=formant(signal,sample_frequence,frame_width,shift_width)
        values = []
        for i in range(len(list_temp)):
            elem = min(list_temp[i])
            values.append(elem)
        value = min(values)
        list_formant.append(value)

        h = MFCC(signal, sample_frequence, frame_width, shift_width)
        list_MFCC.append(h[0])
        '''
    data_frame = pd.DataFrame()
    data_frame['Sex'] = list_sex
    data_frame['Energy'] = list_energy
    data_frame['Fundamental frequency'] = list_fundamental_frequency
    #data_frame['Formant']=list_formant
    #data_frame['MFCC']=list_MFCC

    return data_frame

def rule_based_system_on_energy_accurancy (data_frame,threshold_on_energy):
    data_frame_size=len(data_frame)
    number_of_correct_answer=0
    for i in data_frame.index.values:
        if data_frame.loc[i, 'Energy'] <= threshold_on_energy and data_frame.loc[i, 'Sex'] == 0:
            number_of_correct_answer+=1
        if data_frame.loc[i, 'Energy'] > threshold_on_energy and data_frame.loc[i, 'Sex'] == 1 :
            number_of_correct_answer+=1
    accurancy=number_of_correct_answer/data_frame_size

    return accurancy

def rule_based_system_on_fundamental_frequency_accurancy (data_frame,threshold_on_fundamental_frequency):
    data_frame_size=len(data_frame)
    number_of_correct_answer=0
    for i in data_frame.index.values:
        if data_frame.loc[i, 'Fundamental frequency'] <= threshold_on_fundamental_frequency and data_frame.loc[i, 'Sex'] == 1:
            number_of_correct_answer+=1
        if data_frame.loc[i, 'Fundamental frequency'] > threshold_on_fundamental_frequency and data_frame.loc[i, 'Sex'] == 0:
            number_of_correct_answer+=1
    accurancy=number_of_correct_answer/data_frame_size

    return accurancy

