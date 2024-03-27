from imports import *

class AudioDataSet(Dataset):
    def __init__(self, clean_path, fx_path, sr=48000):
        self.clean_path = clean_path
        self.fx_path = fx_path
        self.sampling_rate= sr  # sampling rate
        self.samples = os.listdir(clean_path)


    def __getitem__(self, idx):
        filename = self.samples[idx]
        clean_file = os.path.join(self.clean_path, filename)
        fx_file = os.path.join(self.fx_path, filename)
        clean_data, sr = librosa.load(clean_file, sr=self.sampling_rate)
        fx_data, sr = librosa.load(fx_file, sr=self.sampling_rate)
        return (clean_data, fx_data)

    def __len__(self):
        return len(self.samples)
    

    def data_loader(clean_path, fx_path, batch_size=32, shuffle=True, num_workers=4):
        dataset = AudioDataSet(clean_path, fx_path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
