from audio_data_loading import AudioDataSet as adl
from imports import *
import train


fx_validation = 'model_results/validation_fx_batch.wav'
valid_prediction = 'model_results/validation_predictions_epoch_7.wav'

fx_audio, sr_fx = torchaudio.load(fx_validation)
validation_audio, sr_val = torchaudio.load(valid_prediction)
fx_audio = fx_audio.squeeze(0)
validation_audio = validation_audio.squeeze(0)
fx_audio = fx_audio[-len(validation_audio):]
print(fx_audio.shape)

# Calculate the ESR
fx_audio = fx_audio.unsqueeze(0).unsqueeze(0)
validation_audio = validation_audio.unsqueeze(0).unsqueeze(0)
err = train.ESR_loss(fx_audio, validation_audio)
print(f'ESR is {err}')

# Calculate bias error
bias_err = train.gen_zero_bias_loss(validation_audio)
print(f'Bias Error is {bias_err}')

# Calculate overflow error
overflow_err = train.gen_overflow_loss(validation_audio)
print(f'Overflow Error is {overflow_err}')