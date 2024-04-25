import numpy as np
import matplotlib.pyplot as plt
from wavenet import *

def create_graph(curr,  prev = 0):
    
    clean_file = f'./model_results/validation_clean_batch.wav'
    fx_file = f'./model_results/validation_fx_batch.wav'
    gen_file = f'./model_results/validation_predictions_epoch_{curr}.wav'
    if prev != 0:
        gen_prev = f'./model_results/validation_predictions_epoch_{prev}.wav'



    # Read audio data
    clean_audio = librosa.load(clean_file, sr=44100)[0] * -1
    fx_audio = librosa.load(fx_file, sr=44100)[0]
    gen_audio = librosa.load(gen_file, sr=44100)[0]
    if prev != 0:
        gen_prev = librosa.load(gen_prev, sr=44100)[0]
    
    #Ensure all audio data have the same length (trim or pad if necessary)
    if prev != 0:
        min_length = min(len(clean_audio), len(fx_audio), len(gen_audio), len(gen_prev))
        print (f'clean audio: {len(clean_audio)}')
        print (f'fx audio: {len(fx_audio)}')
        print (f'gen audio: {len(gen_audio)}')
        print (f'gen init: {len(gen_prev)}')
    else:
        min_length = min(len(clean_audio), len(fx_audio), len(gen_audio))
    clean_audio = clean_audio[-min_length:]
    fx_audio = fx_audio[-min_length:]
    gen_audio = gen_audio[-min_length:]
    if prev != 0:
        gen_prev = gen_prev[-min_length:]
    # Create time array (assuming sample rate of 44100 Hz)
    
    time = np.arange(0, min_length) / 441000 # 20 seconds

    # Plot audio data
    plt.figure(figsize=(10, 6))
    plt.plot(time, clean_audio, color='blue', label='Clean')
    plt.plot(time, fx_audio, color='red', label='Fx')
    plt.plot(time, gen_audio, color='green', label='Generated')
    if prev:
        plt.plot(time, gen_prev, color='orange', label=f'Generated {curr-prev} Epochs Ago')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveforms')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(epochs, train_losses, valid_losses, title, xlabel, ylabel):
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.3)
    plt.savefig(f'{title}.png')
    plt.close()

def plot_graph (x, y, title, xlabel, ylabel):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.close()

#create_graph(45,43)