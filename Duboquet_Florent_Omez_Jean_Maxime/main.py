from scipy.io import wavfile

from statistics import mean

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import random_select_utterances,plot_signal_and_energy_per_frame,energy,pitch_autocorrelation

print('Energy of the voiced and unvoiced sounds')
'''
utterances=random_select_utterances(['samples/man','samples/woman'],5)
for utterance in utterances:
        plot_signal_and_energy_per_frame(utterance,90 / 1000, 90 / 1000)
'''
print('Building a rule-based system')

utterances=random_select_utterances(['samples/man','samples/woman'],15)

frame_width=90/1000
shift_width=90/1000
threshold=10

list_sexe=[]
list_fundamental_frequency=[]
list_energy=[]
list_formant=[]
list_MFCC=[]
for i in range(len(utterances)):
    if 'woman' in utterances[i]:
        list_sexe.append(0)
    else:
        list_sexe.append(1)

    sample_frequence,signal=wavfile.read(utterances[i])

    list_energy.append(energy(signal))

    f0_voiced=[]
    for f0 in pitch_autocorrelation(signal, sample_frequence, frame_width, shift_width, threshold):
        if f0 !=0:
            f0_voiced.append(f0)
    list_fundamental_frequency.append(mean(f0_voiced))

    #list_formant.append(formant(signal,sample_frequence,frame_width,shift_width))

    #list_MFCC.append(MFCC(signal, sample_frequence,frame_width,shift_width))

data_frame = pd.DataFrame()
data_frame['Sexe']=list_sexe
data_frame['Energy']=list_energy
data_frame['Fundamental frequency']=list_fundamental_frequency
#data_frame['Formant']=list_formant
#data_frame['MFCC']=list_MFCC

print(data_frame)

sns.pairplot(data_frame, hue='Sexe')
plt.show()

sns.heatmap(data_frame.corr())
plt.show()
