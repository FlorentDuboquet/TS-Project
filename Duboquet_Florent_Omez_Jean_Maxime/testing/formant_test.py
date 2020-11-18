from utils import formant
from scipy.io import wavfile

sample_frequence,signal=wavfile.read('../samples/man/arctic_a0001.wav')
frame_width=35/1000
shift_width=35/1000
print(formant(signal,sample_frequence,frame_width,shift_width))