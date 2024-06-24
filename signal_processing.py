from imports import *


def convert_batch_to_log_mel_stft(batch, sample_rate, n_fft=2048, hop_length=512, n_mels=128):
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_transform(batch))
    # Normalize the log mel spectrogram between -1 and 1
    log_mel_spectrogram = (log_mel_spectrogram + 40) / 40
    log_mel_spectrogram = log_mel_spectrogram.squeeze(1)
    return log_mel_spectrogram