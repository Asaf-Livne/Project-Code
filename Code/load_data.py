from imports import *

def normalize_data(data):
    data = data / np.max(np.abs(data))
    return data

def load_data(clean_path, fx_path, sr=48000, batch_size = 5, bit_size=5):
    clean_file = os.path.join(clean_path)
    fx_file = os.path.join(fx_path)
    clean_data, sr = librosa.load(clean_file, sr=sr)
    fx_data, sr = librosa.load(fx_file, sr=sr)
    clean_data = normalize_data(clean_data)
    fx_data = normalize_data(fx_data)
    data_len = len(clean_data)
    number_of_splits = int(data_len // sr // bit_size)
    bit_size = int(sr * bit_size)
    split_clean = [[clean_data[i*bit_size:(i+1)*bit_size]] for i in range(number_of_splits)]
    split_fx = [[fx_data[i*bit_size:(i+1)*bit_size]] for i in range(number_of_splits)]
    num_of_batches = len(split_clean) // batch_size
    split_clean = split_clean[:num_of_batches*batch_size]
    split_fx = split_fx[:num_of_batches*batch_size]
    data_array = []
    for i in range(num_of_batches):
        #if (i == 1 or i == 3):
         #   continue
        data_array.append((torch.tensor(split_clean[i*batch_size:(i+1)*batch_size]), torch.tensor(split_fx[i*batch_size:(i+1)*batch_size])))

    return data_array


#print (np.shape(load_data('./Project-Code/Data/LW Clean.wav', './Project-Code/Data/LW Dist.wav', sr=44100, batch_size=10, bit_size=2)))
