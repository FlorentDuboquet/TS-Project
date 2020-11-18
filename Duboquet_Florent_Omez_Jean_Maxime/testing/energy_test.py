from utils import energy
from scipy.io import wavfile

signal=[1,2,3]
print(energy(signal))

sample_frequence,signal=wavfile.read('../samples/man/arctic_a0001.wav')
print(energy(signal))