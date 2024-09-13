from imports import *
from audio_data_loading import AudioDataSet as adl
from file_operations import write_audio


train_data = adl.data_loader('../Data/clean_train.wav', '../Data/clean_train.wav', sec_sample_size=5, sr=44100, batch_size=4, shuffle=True, scramble=True)

for i, (clean, fx) in enumerate(train_data):
    print(clean[0][0][:20])
    print(fx[0][0][:20])
    
    if i == 0:
        break