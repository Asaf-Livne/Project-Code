import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import lfilter
from scipy.signal import butter




# Load an audio file
audio_file_path = 'Clean/Neck/1-0.wav'
y, sr = librosa.load(audio_file_path)

y_3 = np.array([3*val for val in y])

# Calculate the spectrogram using librosa's stft function
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

y_stft = librosa.stft(y)

print(np.max(y_stft))

magnitude = np.abs(y_stft)
angle = np.angle(y_stft)
maxi = np.max(magnitude)
lim_mag = np.where(magnitude < 0.0007 * maxi, magnitude, 0.5 * maxi)

real_part = lim_mag * np.cos(angle)
imag_part = lim_mag * np.sin(angle)

y_dist = real_part + 1j * imag_part

# Define the cutoff frequency for the low-pass filter relative to Nyquist frequency
cutoff_frequency = 2000  # Cutoff frequency in Hz
nyquist = 0.5 * sr
normalized_cutoff = cutoff_frequency / nyquist
# Design a low-pass Butterworth filter
order = 4  # Filter order
btype = 'high'  # Filter type: 'low', 'high', 'band', or 'bandstop'
b, a = butter(order, normalized_cutoff, btype=btype, analog=False, output='ba')

filtered_signal = lfilter(b, a, y_stft)



inv_y = librosa.istft(filtered_signal)
sf.write('output.wav', inv_y , sr)

# Display the spectrogram
plt.figure(figsize=(12, 8))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
#plt.show()

