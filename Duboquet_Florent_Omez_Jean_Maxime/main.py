import seaborn as sns
import matplotlib.pyplot as plt

from utils import random_select_utterances,plot_signal_and_energy_per_frame,energy,pitch_autocorrelation,feature_extraction

print('-> Energy of the voiced and unvoiced sounds')
'''
utterances=random_select_utterances(['samples/man','samples/woman'],5)
for utterance in utterances:
        plot_signal_and_energy_per_frame(utterance,90 / 1000, 90 / 1000)
'''
print('-> Building a rule-based system')

files_adresse=random_select_utterances(['samples/man','samples/woman'],15)

frame_width=90/1000
shift_width=90/1000
threshold=10

data_frame=feature_extraction(files_adresse,frame_width,shift_width,threshold)

print('Data frame :\n',data_frame)

sns.pairplot(data_frame, hue='Sexe')
plt.show()

sns.heatmap(data_frame.corr())
plt.show()
