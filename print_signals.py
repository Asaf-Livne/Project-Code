import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from wavenet import *

# Read raw audio files (replace with your own file paths)
iter = 13
clean_file = f'../Project-Code/model_results/clean_output.wav'
fx_file = f'../Project-Code/model_results/fx_output.wav'
gen_file = f'../Project-Code/model_results/gen_output_epoch_{iter}.wav'
gen_init = f'../Project-Code/model_results/gen_output_epoch_11.wav'

# Load data



# Read audio data
clean_audio = librosa.load(clean_file, sr=44100)[0]
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
time = np.arange(0, 20 * 44100) / 441000 # 20 seconds

# Plot audio data
plt.figure(figsize=(10, 6))
plt.plot(time, clean_batch_0, color='blue', label='Clean')
plt.plot(time, fx_batch_0, color='red', label='Fx')
plt.plot(time, gen_audio, color='green', label='Gen')
plt.plot(time, gen_init, color='orange', label='Gen_5')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveforms')
plt.legend()
plt.grid(True)
plt.show()
