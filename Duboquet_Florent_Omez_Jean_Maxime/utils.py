import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from math import ceil
from scipy.io import wavfile
from random import randint
from os import listdir
from xcorr import xcorr
import warnings
from scikit_talkbox_lpc import lpc_ref
from filterbanks import filter_banks
import scipy.fftpack as fft
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
                for i in range(number_of_utterances):
                        utterances.append(folder_adresse+'/'+file_adresses[randint(0,len(file_adresses)-1)])
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

                else:# Unvoiced
                        fundamental_frequency=0

                fundamental_frequency_per_frame.append(fundamental_frequency)

        return fundamental_frequency_per_frame

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

def high_Pass(signal, a=0.67):  # a est compris dans [0.62,0.67]
    filtred_signal = []
    for i in range(0, len(signal) - 1):
        if i > 0:
            filtred_signal.append(signal[i] - a * signal[i - 1])
        else:
            filtred_signal.append(signal[i])
    filtred_signal = np.array(filtred_signal) # on change le typ de la liste avant le return
    return filtred_signal


def formant (frames,fs):

    frequences = []

    # ici on va devoir utiliser la fct lpc_ref fournie dans
    # scikit_talkbox_lpc.py qui retourne les prédiction des coefficient LPC

    # on applique le traitement a tout les frames :
    for i in range(0, len(frames)):

        # le filtre passe haut (définit précédement)
        filtred_frame = high_Pass(frames[i])

        # calcul du LPC grace a la fct fournie
        temp = lpc_ref(filtred_frame, order= 10) # order peut prendre des valeurs entre 8 et 13

        # on calcule les racines du LPC :
        lpc = np.roots(temp)

        # on ne conserve que l'un des deux complexes conjugués
        lpc = lpc[np.imag(lpc) >= 0]

        temp = []
        for j in range (0,len(lpc)) :

            # on calcul l'angle et on en déduit la fréquence
            freq = np.arctan2(np.imag(lpc[j]),np.real(lpc[j])) * ( fs/8*np.pi )
            """ !!!!!!!!!! attention ici fs devra etre précisé dans main !!!!!!! """


            # la frequence doit etre comprise entre les seuils
            if (freq<20000 and freq>500):
                temp.append(freq)
                temp.sort()
        frequences.append(temp)
    # on change le type de la liste
    frequences = np.array(frequences)

    # on trie pour les assossié plus facilement au formant
    frequences = np.sort(frequences)

    return frequences

def MFCC (signal, sample_frequence,frame_width,shift_width) :
    # preanalyse
    signal = high_Pass(signal, a=0.97)

    # division en frames
    signal = framing(signal,sample_frequence,frame_width,shift_width)

    # hamming
    for i in range(len(signal)-1) :
        ham = np.hamming(len(signal[i]))
        signal[i] = signal[i] * ham

    #compute the power spectrum of the signal periodogram
    powerSpectrum = []
    ntfd = 512
    for elem in signal :
        powerSpectrum.append((np.power(np.linalg.norm(np.asarray(np.fft.fft(elem,ntfd)))),2)/ntfd)

    # passage dans le filter bank
    result = filter_banks(powerSpectrum,sample_frequence)

    #Discrete Cosine Transform as given in the protocole
    result = fft.dct(filter_banks, type=2, axis=1, norm='ortho')

    # on garde que les 13 premiers
    result = result[:13]

    return result

