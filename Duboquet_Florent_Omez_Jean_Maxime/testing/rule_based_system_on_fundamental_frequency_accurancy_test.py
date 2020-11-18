from utils import rule_based_system_on_fundamental_frequency_accurancy
import pandas as pd

data_frame=pd.DataFrame()
data_frame['Sexe']=[1,0,1,0]
data_frame['Fundamental frequency']=[0,2,2,0]
threshold_on_fundamental_frequency=1
print('Accurancy :',rule_based_system_on_fundamental_frequency_accurancy(data_frame,threshold_on_fundamental_frequency))