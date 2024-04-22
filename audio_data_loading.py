from imports import *

class AudioDataSet(Dataset):
    def __init__(self, clean_path, fx_path, sec_sample_size, sr=44100):
        self.clean_path = clean_path
        self.fx_path = fx_path
        self.sampling_rate= sr  # sampling rate
        self.sample_size = int(sec_sample_size * sr)
        self.clean_file, _ = librosa.load(clean_path, sr=self.sampling_rate)
        self.clean_file = self.clean_file[sr:sr*31]
        self.fx_file, _ = librosa.load(fx_path, sr=self.sampling_rate)
        self.fx_file = self.fx_file[sr:sr*31]

    def __getitem__(self, idx):
        clean_data = self.clean_file[idx * self.sample_size: (idx + 1) * self.sample_size]
        fx_data = self.fx_file[idx * self.sample_size: (idx + 1) * self.sample_size]
        clean_data = self.normalize_signal(clean_data)
        fx_data = self.normalize_signal(fx_data)
        return (torch.tensor([clean_data]), torch.tensor([fx_data]))

    def __len__(self):
        return len(self.clean_file) // (self.sample_size)
    
    def normalize_signal(self, data):
        return data / (np.max(np.abs(data) + 0.0001))
        

    def data_loader(clean_path, fx_path, sec_sample_size = 1, sr = 44100, batch_size=10, shuffle=False):
        dataset = AudioDataSet(clean_path, fx_path, sec_sample_size, sr)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def infinite_batch(self, dataset):
        while True:
            for clean, fx in dataset:
                yield clean, fx
                

