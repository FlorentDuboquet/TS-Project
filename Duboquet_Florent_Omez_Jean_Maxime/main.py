from utils import random_select_utterances,plot_signal_and_energy_per_frame


#Energy of the voiced and unvoiced sounds
'''
utterances=random_select_utterances(['samples/man','samples/woman'],5)
for utterance in utterances:
        plot_signal_and_energy_per_frame(utterance,90 / 1000, 90 / 1000)
'''

#Building a rule-based system

utterances=random_select_utterances(['samples/man','samples/woman'],15)

sexes=[]
for i in range(len(utterances)):
    if 'woman' in utterances[i]:
        sexes.append('woman')
    else:
        sexes.append('man')



