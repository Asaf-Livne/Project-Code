from imports import *
import random

class AudioDataSet(Dataset):
    def __init__(self, clean_path, fx_path, sec_sample_size, sr=44100, scramble=False):
        self.clean_path = clean_path
        self.fx_path = fx_path
        self.sampling_rate= sr  # sampling rate
        self.sample_size = int(sec_sample_size * sr)
        self.clean_file, _ = librosa.load(clean_path, sr=self.sampling_rate)
        if self.clean_file.dtype == 'int16':
            self.clean_file = self.clean_file.astype('float32')
        self.fx_file, _ = librosa.load(fx_path, sr=self.sampling_rate)
        if self.fx_file.dtype == 'int16':
            self.fx_file = self.fx_file.astype('float32')
        self.scramble = scramble
        if self.scramble:
            # Shuffle the indices of the fx_file for scrambling
            self.fx_indices = np.arange(len(self.clean_file) // self.sample_size)
            random.shuffle(self.fx_indices)
        else:
            self.fx_indices = np.arange(len(self.clean_file) // self.sample_size)

    def __getitem__(self, idx):
        clean_data = self.clean_file[idx * self.sample_size: (idx + 1) * self.sample_size]
        fx_data = self.fx_file[self.fx_indices[idx] * self.sample_size: (self.fx_indices[idx] + 1) * self.sample_size]
        clean_data = self.normalize_signal(clean_data)
        fx_data = self.normalize_signal(fx_data)
        return (torch.tensor([clean_data]), torch.tensor([fx_data]))

    def __len__(self):
        return len(self.clean_file) // (self.sample_size)
    
    def normalize_signal(self, data):
        return data / (np.max(np.abs(data) + 0.0001))
        

    def data_loader(clean_path, fx_path, sec_sample_size = 1, sr = 44100, batch_size=10, shuffle=False, scramble=False):
        dataset = AudioDataSet(clean_path, fx_path, sec_sample_size, sr, scramble)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
                

