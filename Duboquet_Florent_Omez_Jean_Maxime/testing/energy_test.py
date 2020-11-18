from utils import energy
from scipy.io import wavfile

signal=[1,2,3]
print('Signal :',signal)
print('Energy :',energy(signal))

print('\n')


sample_frequence,signal=wavfile.read('../samples/man/arctic_a0001.wav')
print('Signal :',signal)
print('Energy :',energy(signal))