from utils import framing
from scipy.io import wavfile

signal=[1,2,3,4,5,6,7,8,9]
sample_frequence=1
frame_width=3
shift_width=3
print('Signal :',signal)
print('Frames :\n',framing(signal,sample_frequence,frame_width,shift_width))

print('\n')

sample_frequence,signal=wavfile.read('../samples/man/arctic_a0001.wav')
frame_width=35/1000
shift_width=35/1000
print('Signal :',signal)
print('Frames :\n',framing(signal,sample_frequence,frame_width,shift_width))