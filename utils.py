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

