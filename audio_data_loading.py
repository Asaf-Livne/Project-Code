from imports import *

class AudioDataSet(Dataset):
    def __init__(self, clean_path, fx_path, sr=48000):
        self.clean_path = clean_path
        self.fx_path = fx_path
        self.sampling_rate= sr  # sampling rate
        self.samples = os.listdir(clean_path) # files names are the same for clean and fx folders


    def __getitem__(self, idx):
        filename = self.samples[idx]
        clean_file = os.path.join(self.clean_path, filename)
        fx_file = os.path.join(self.fx_path, filename)
        clean_data, sr = librosa.load(clean_file, sr=self.sampling_rate)
        fx_data, sr = librosa.load(fx_file, sr=self.sampling_rate)
        # Find the minimum length among clean and fx audio samples
        min_length = min(len(clean_data), len(fx_data))
        #print(f"Original lengths: Clean: {len(clean_data)}, FX: {len(fx_data)}, Min: {min_length}")
        clean_data = clean_data[:min_length]
        fx_data = fx_data[:min_length]
        #print(f"After the Truncate: lengths: Clean: {len(clean_data)}, FX: {len(fx_data)}, Min: {min_length}")
        return (clean_data, fx_data)

    def __len__(self):
        return len(self.samples)
        

    def data_loader(clean_path, fx_path, batch_size=1, shuffle=True):
        dataset = AudioDataSet(clean_path, fx_path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

