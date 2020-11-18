from utils import feature_extraction

files_adresse=['../samples/man/arctic_a0001.wav','../samples/woman/arctic_a0001.wav']
frame_width=35/1000
shift_width=35/1000
threshold=40
print('Data set :\n',feature_extraction(files_adresse,frame_width,shift_width,threshold))