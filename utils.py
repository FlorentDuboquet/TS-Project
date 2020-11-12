import matplotlib.pyplot as plt
import numpy as np

def norming(signal):
    '''
    entrée :
        signal: le signal qu on va normaliser
    sortie:
        signal_normed : signal normalisé
    '''

    sig_size = len(signal)
    max_sig = 0
    for i in range(0, sig_size): #on trouve le maximum global du signal
        if abs(signal[i]) > max_sig:
            max_sig = abs(signal[i])
    signal_normed = signal / max_sig #on divise par le maximum global

    return signal_normed


def framing(signal, shifting_step=2500, frames_size=2500):
    '''
    entrée :
        signal: le signal qu'on veut fragmenter
        shifting_step: la pas entre deux devuts de frames
        frames_size: la taille d'un fragment en nbr d echantillons
    Sortie :
         frames : array avec tout les frames
    '''

    signal_size = len(signal)
    frames = []
    i = 0
    while True:
        if (i + frames_size <= signal_size):
            fr_act_size = i + frames_size
        else:
            fr_act_size = signal_size
        frames.append(signal[i:fr_act_size])
        i += shifting_step
        if (i >= signal_size):
            break
    frames = np.array(frames)


def sig_energy(signal):
    '''
    entrée :
        signal: le signal dont on veut calculer l'énergie
    sortie :
        Energy : l'énergie du signal
    '''

    energy = 0
    for i in range(0, len(signal)):
        energy += np.power(abs(signal[i]), 2)
    return energy

#ce qui est en dessous est en cours de travail, pas sur que ca soit bon
"""
def pitch(frames,Fs, threshold=10, maxlags=800000, printing=False):
    f0 = []
    for i in range(0, len(frames)):

        if sig_energy(frames[i]) > threshold:

            a, b, *_ = plt.acorr(frames[i], maxlags=maxlags)  # we only need b, aka the autocorrelation vector

            e = argrelextrema(b, np.greater)  # Local maximum of b, the autocorrelation vector
            loc_max_temp = np.array(e[0])  # temp list
            loc_max = []
            maxt = 0
            for h in range(0, len(loc_max_temp)):
                temp = loc_max_temp[h]
                if b[temp] > maxt:
                    loc_max.append(loc_max_temp[h] - maxlags)
                    maxt = b[temp]

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