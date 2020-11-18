from utils import normalization
from scipy.io import wavfile
import matplotlib.pyplot as plt

sample_frequence,signal=wavfile.read('../samples/man/arctic_a0001.wav')
normalized_signal=normalization(signal)

plt.subplot(2, 1, 1)
plt.title("Signal")
plt.plot(signal)

plt.subplot(2,1,2)
plt.title("Normalized Signal")
plt.plot(normalized_signal)

plt.show()