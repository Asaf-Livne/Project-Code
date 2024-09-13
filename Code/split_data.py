import torch
import torchaudio
import train
# load DB01 and split it to train and validation


clean, sr = torchaudio.load('../Data/DB01/DB01 Clean.wav')
distortion, sr = torchaudio.load('../Data/DB01/DB01 EQ.wav')

# Split 80% of the data for training and 20% for validation
print(clean.shape)
print(distortion.shape)
# convert distortion to mono
distortion = distortion.mean(dim=0, keepdim=True)
print(distortion.shape)
# Match data lengths
min_length = min(clean.size(1), distortion.size(1))
clean = clean[:, :min_length]
distortion = distortion[:, :min_length]
# Split
val_size = int(0.2 * clean.size(1))
clean_val = clean[:, :val_size]
distortion_val = distortion[:, :val_size]
clean_train = clean[:, val_size:]
distortion_train = distortion[:, val_size:]

# Save the training and validation data
torchaudio.save('../Data/clean_train.wav', clean_train, sr)
torchaudio.save('../Data/equalizer_train.wav', distortion_train, sr)
torchaudio.save('../Data/clean_val.wav', clean_val, sr)
torchaudio.save('../Data/equalizer_val.wav', distortion_val, sr)