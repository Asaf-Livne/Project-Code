from audio_data_loading import AudioDataSet as adl
from imports import *


clean_file = './Data/ts9_test1_in_FP32.wav'
dist_file = './Data/ts9_test1_out_FP32.wav'

dataset = adl(clean_file, dist_file, sr = 44100, sec_sample_size = 300)

print (len(dataset.clean_file))

eighty = int(len(dataset.clean_file) * 0.8)


clean_train = dataset.clean_file[:eighty]
clean_val = dataset.clean_file[eighty:]
fx_train = dataset.fx_file[:eighty]
fx_val = dataset.fx_file[eighty:]

sf.write(f'./Data/clean_train.wav', clean_train, 44100)
sf.write(f'./Data/clean_val.wav', clean_val, 44100)
sf.write(f'./Data/fx_train.wav', fx_train, 44100)
sf.write(f'./Data/fx_val.wav', fx_val, 44100)