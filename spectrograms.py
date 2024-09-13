from imports import *
from audio_data_loading import AudioDataSet as adl
import torchaudio
import matplotlib.pyplot as plt
import torch
from train import ESR_loss
import signal_processing as sp


epoch = 10
# Load the audio file
data_path_clean = './model_results/validation_clean_batch.wav'
data_path_fx = './model_results/validation_fx_batch.wav'
validation_prediction = f'./model_results/validation_predictions_epoch_{epoch}.wav'
prev_prediction = f'./model_results/validation_predictions_epoch_{epoch-1}.wav'
#validation_prediction = f'../filtered_signal.wav'

#validation_prediction = '../filtered_signal.wav'

clean_audio, sr_clean = torchaudio.load(data_path_clean)
fx_audio, sr_fx = torchaudio.load(data_path_fx)
validation_audio, sr_val = torchaudio.load(validation_prediction)
if epoch > 0:
    validation_audio_prev, sr_val_prev = torchaudio.load(prev_prediction)

# Convert audio to batch format
clean_audio = clean_audio.unsqueeze(0)
fx_audio = fx_audio.unsqueeze(0)
validation_audio = validation_audio.unsqueeze(0)
if epoch > 0:
    validation_audio_prev = validation_audio_prev.unsqueeze(0)

# Ensure all audio data have the same length (trim from the beginning if necessary)
min_length = min(clean_audio.size(-1), fx_audio.size(-1), validation_audio.size(-1))
clean_audio = clean_audio[:, :, -min_length:]
fx_audio = fx_audio[:, :, -min_length:]
validation_audio = validation_audio[:, :, -min_length:]
if epoch > 0:
    validation_audio_prev = validation_audio_prev[:, :, -min_length:]

# Convert the audio files to spectrograms
clean_spec = sp.convert_batch_to_log_mel_stft(clean_audio, sr_clean, window_size=2048)
fx_spec = sp.convert_batch_to_log_mel_stft(fx_audio, sr_fx, window_size=2048)
validation_spec = sp.convert_batch_to_log_mel_stft(validation_audio, sr_val, window_size=2048)
if epoch > 0:
    validation_spec_prev = sp.convert_batch_to_log_mel_stft(validation_audio_prev, sr_val_prev, window_size=2048)


# Create an ESR function for the spectrograms
def ESR2_loss(spec1, spec2):
    err = ((spec1 - spec2) ** 2).sum()
    rmss = (spec2 ** 2).sum()
    return err / rmss

def L1_loss(spec1, spec2):
    return np.mean(np.abs(spec1 - spec2))

# Calculate the ESR
err = ESR2_loss(fx_spec, validation_spec)
print(f'ESR is {err}')
print(f'L1 is {L1_loss(fx_spec, validation_spec)}')

# Plot the spectrograms
plt.figure(figsize=(20, 6))
plt.subplot(1, 4, 1)
plt.imshow(clean_spec[0].squeeze().numpy(), aspect='auto', origin='lower')
plt.title('Clean Audio Spectrogram')
plt.colorbar()
plt.subplot(1, 4, 2)
plt.imshow(fx_spec[0].squeeze().numpy(), aspect='auto', origin='lower')
plt.title('FX Audio Spectrogram')
plt.colorbar()
plt.subplot(1, 4, 3)
plt.imshow(validation_spec[0].squeeze().numpy(), aspect='auto', origin='lower')
plt.title('Generated Audio Spectrogram')
plt.colorbar()
if epoch > 0:
    plt.subplot(1, 4, 4)
    plt.imshow(validation_spec_prev[0].squeeze().numpy(), aspect='auto', origin='lower')
    plt.title('Generated Audio Spectrogram (Previous Epoch)')
    plt.colorbar()
plt.show()
