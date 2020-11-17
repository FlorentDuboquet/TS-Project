import matplotlib.pyplot as plt
import numpy as np

def norming(signal):
    '''
    entrée :
        signal: le signal qu on va normaliser
    sortie:
        signal_normed : le signal normalisé
    '''

    signal_size = len(signal)
    max_sig = 0
    for i in range(0, signal_size): #on cherche le max du signal
        if abs(signal[i]) > max_sig:
            max_sig = abs(signal[i])
    signal_normed = signal / max_sig #on divise par le maximum global pour normaliser

    return signal_normed


def framing(signal, shifting_step=2500, frames_size=2500):
    '''
    entrée :
        signal: le signal qu'on veut fragmenter
        shifting_step: la pas entre deux débuts de frames
        frames_size: la taille d'un fragment en nbr d echantillons
    Sortie :
         frames : array avec tout les frames
    '''

    signal_size = len(signal)
    frames = []
    i = 0
    while True:
        if (i + frames_size <= signal_size):
            end_frame = i + frames_size
        else:
            end_frame = signal_size # si le frame "dépasse" du signal a la fin alors on en créer un plus petit qui s arrette fatalement au dernier sample du signal
        frames.append(signal[i:end_frame])
        i += shifting_step
        if (i >= signal_size):
            break
    frames = np.array(frames)

    return frames


def sig_energy(signal):
    '''
    entrée :
        signal: le signal dont on veut calculer l'énergie
    sortie :
        Energy : l'énergie du signal
    '''

    energy = 0
    for i in range(len(signal)):
        energy += np.power(abs(signal[i]), 2)
    return energy

#ce qui est en dessous est en cours de travail, pas sur que ca soit bon
"""
def pitch(frames,Fs, threshold=10, maxlags=800000):
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