import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from wavenet import *

# Read raw audio files (replace with your own file paths)
iter = 8
clean_file = f'./model_results/clean_batch.wav'
fx_file = f'./model_results/fx_batch.wav'
gen_file = f'./model_results/predictions_epoch_{iter}.wav'
gen_init = f'./model_results/predictions_epoch_{iter-1}.wav'

# Load data



# Read audio data
clean_audio = librosa.load(clean_file, sr=44100)[0] * -1
fx_audio = librosa.load(fx_file, sr=44100)[0]
gen_audio = librosa.load(gen_file, sr=44100)[0]
gen_init = librosa.load(gen_init, sr=44100)[0]

# Ensure all audio data have the same length (trim or pad if necessary)
min_length = min(len(clean_audio), len(fx_audio), len(gen_audio), len(gen_init))
clean_batch_0 = clean_audio[:min_length]
fx_batch_0 = fx_audio[:min_length]
gen_audio = gen_audio[:min_length]
gen_init = gen_init[:min_length]
print (clean_batch_0)
print (gen_audio)
# Create time array (assuming sample rate of 44100 Hz)
time = np.arange(0, 2 * 44100) / 441000 # 20 seconds

# Plot audio data
plt.figure(figsize=(10, 6))
plt.plot(time, clean_batch_0, color='blue', label='Clean')
plt.plot(time, fx_batch_0, color='red', label='Fx')
plt.plot(time, gen_audio, color='green', label='Gen')
plt.plot(time, gen_init, color='orange', label='Gen_4')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveforms')
plt.legend()
plt.grid(True)
plt.show()
