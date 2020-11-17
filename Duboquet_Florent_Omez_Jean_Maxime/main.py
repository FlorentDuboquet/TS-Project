from scipy.io import wavfile
from statistics import mean
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import random_select_utterances,plot_signal_and_energy_per_frame,pitch_autocorrelation,energy,MFCC


#Energy of the voiced and unvoiced sounds
'''
utterances=random_select_utterances(['samples/man','samples/woman'],5)
for utterance in utterances:
        plot_signal_and_energy_per_frame(utterance,90 / 1000, 90 / 1000)
'''

#Building a rule-based system

utterances=random_select_utterances(['samples/man','samples/woman'],15)

frame_width=90/1000
shift_width=90/1000
threshold=50

list_sexes=[]
list_f0=[]
list_energy=[]
list_MFCC=[]
for i in range(len(utterances)):
    if 'woman' in utterances[i]:
        list_sexes.append('woman')
    else:
        list_sexes.append('man')

    sample_frequence,signal=wavfile.read(utterances[i])

    list_energy.append(energy(signal))

    f0_voiced=[]
    for f0 in pitch_autocorrelation(signal, sample_frequence, frame_width, shift_width, threshold):
        if f0 !=0:
            f0_voiced.append(f0)
    list_f0.append(mean(f0_voiced))

    list_MFCC.append(MFCC(signal, sample_frequence,frame_width,shift_width))

data_frame = pd.DataFrame()
data_frame['Sexe']=list_sexes
data_frame['Energy']=list_energy
data_frame['f0']=list_f0
data_frame['MFCC']=list_MFCC

print(data_frame)

sns.pairplot(data_frame, hue='Sexe')
plt.show()


