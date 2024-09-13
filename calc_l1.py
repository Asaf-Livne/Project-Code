import numpy as np
import soundfile as sf
import librosa
from signal_processing import *
import os

fx = librosa.load('./Code/fx.wav')[0]
predictions = librosa.load('Code/preditctions.wav')[0]
print(f'len fx: {len(fx)}')
print(f'len predictions: {len(predictions)}')

predictions = librosa.feature.melspectrogram(y=predictions, sr=44100, win_length = 1024, hop_length=256, n_mels=128)
fx = librosa.feature.melspectrogram(y=fx, sr=44100, win_length = 1024, hop_length=256, n_mels=128)

# Log the spectrograms
predictions = librosa.power_to_db(predictions, ref=np.max)
fx = librosa.power_to_db(fx, ref=np.max)
diff = fx-predictions


# Plot two spectrograms
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
plt.imshow(fx, aspect='auto', origin='lower')
plt.title('FX') 
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(predictions, aspect='auto', origin='lower')
plt.title('Predictions')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(diff, aspect='auto', origin='lower')
plt.title('Difference')
plt.colorbar()
plt.show()


def calc_l1(fx,predictions):
    return np.mean(np.abs(fx-predictions))

def calc_ESR(fx,predictions):
    return np.sum((fx-predictions)**2)/np.sum(fx**2)
print(calc_l1(fx,predictions))
print(calc_ESR(fx,predictions))
