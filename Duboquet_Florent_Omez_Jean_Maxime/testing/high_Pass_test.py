from utils import high_Pass
from scipy.io import wavfile
import matplotlib.pyplot as plt

sample_frequence,signal=wavfile.read('../samples/man/arctic_a0001.wav')
filtred_sig=high_Pass(signal)

plt.subplot(2, 1, 1)
plt.title("Signal")
plt.plot(signal)

plt.subplot(2,1,2)
plt.title("filtred Signal")
plt.plot(filtred_sig)

plt.show()