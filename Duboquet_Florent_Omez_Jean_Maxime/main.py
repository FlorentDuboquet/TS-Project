import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir

from utils import random_select_utterances,plot_signal_and_energy_per_frame,energy,pitch_autocorrelation,feature_extraction,rule_based_system_on_energy_accurancy,rule_based_system_on_fundamental_frequency_accurancy

print('\n--- Energy of the voiced and unvoiced sounds')

utterances=random_select_utterances(['samples/man','samples/woman'],5)
for utterance in utterances:
        plot_signal_and_energy_per_frame(utterance,90 / 1000, 90 / 1000)

print('\n--- Building a rule-based system')

frame_width=90/1000
shift_width=90/1000
threshold=10

print('\n-- TRAIN :')

folder_addresses=['samples/man','samples/woman']

train_files_adresse=random_select_utterances(folder_addresses,50)

train_data_frame=feature_extraction(train_files_adresse,frame_width,shift_width,threshold)

print('\n- Train data frame :\n',train_data_frame)

sns.pairplot(train_data_frame, hue='Sex')
plt.show()

sns.heatmap(train_data_frame.corr())
plt.show()

best_threshold_on_energy=0
best_accurancy_on_energy=0
for threshold_on_energy in range(1000000,100000000,10000):
    accurancy=rule_based_system_on_energy_accurancy(train_data_frame,threshold_on_energy)
    if accurancy>best_accurancy_on_energy:
        best_accurancy_on_energy= accurancy
        best_threshold_on_energy = threshold_on_energy
print('\n- Energy :','\nBest threshold :',best_threshold_on_energy,'\nBest accurancy :',best_accurancy_on_energy)

best_threshold_on_fundamental_frequency=0
best_accurancy_on_fundamental_frequency=0
for threshold_on_fundamental_frequency in range(0,1000,1):
    accurancy=rule_based_system_on_fundamental_frequency_accurancy(train_data_frame,threshold_on_fundamental_frequency)
    if accurancy>best_accurancy_on_fundamental_frequency:
        best_accurancy_on_fundamental_frequency=accurancy
        best_threshold_on_fundamental_frequency = threshold_on_fundamental_frequency
print('\n- Fundamental frequency :','\nBest threshold :',best_threshold_on_fundamental_frequency,'\nBest accurancy :',best_accurancy_on_fundamental_frequency)

print('\n-- TEST :')

test_files_adresse=[]
for folder_adresse in folder_addresses:
        file_adresses = listdir(folder_adresse)
        for file_adresse in file_adresses:
            if folder_adresse+'/'+file_adresse not in train_files_adresse:
                test_files_adresse.append(folder_adresse+'/'+file_adresse)

test_data_frame=feature_extraction(test_files_adresse,frame_width,shift_width,threshold)

print('\n- Test data frame :\n',test_data_frame)

accurancy_on_energy=rule_based_system_on_energy_accurancy(test_data_frame,best_threshold_on_energy)
print('\n- Energy :','\nAccurancy :',accurancy_on_energy)

accurancy_on_fundamental_frequency=rule_based_system_on_fundamental_frequency_accurancy(test_data_frame,best_threshold_on_fundamental_frequency)
print('\n- Fundamental frequency :','\nAccurancy :',accurancy_on_fundamental_frequency)
