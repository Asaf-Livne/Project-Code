import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data
from wavenet import *

def create_graph(iter, valid = True):
    if valid:
        clean_file = f'./model_results/valid_clean_batch_{iter}.wav'
        fx_file = f'./model_results/valid_fx_batch_{iter}.wav'
        gen_file = f'./model_results/valid_predictions_epoch_{iter}.wav'
        gen_init = f'./model_results/valid_predictions_epoch_{iter-1}.wav'
    else:
        clean_file = f'./model_results/training_clean_batch_{iter}.wav'
        fx_file = f'./model_results/training_fx_batch_{iter}.wav'
        gen_file = f'./model_results/training_predictions_epoch_{iter}.wav'


    # Read audio data
    clean_audio = librosa.load(clean_file, sr=44100)[0] * -1
    fx_audio = librosa.load(fx_file, sr=44100)[0]
    gen_audio = librosa.load(gen_file, sr=44100)[0]
    if valid:
        gen_init = librosa.load(gen_init, sr=44100)[0]
    
    #Ensure all audio data have the same length (trim or pad if necessary)
    if valid:
        min_length = min(len(clean_audio), len(fx_audio), len(gen_audio), len(gen_init))
    else:
        min_length = min(len(clean_audio), len(fx_audio), len(gen_audio))
    clean_audio = clean_audio[:min_length]
    fx_audio = fx_audio[:min_length]
    gen_audio = gen_audio[:min_length]
    if valid:
        gen_init = gen_init[:min_length]
    # Create time array (assuming sample rate of 44100 Hz)
    
    time = np.arange(0, min_length) / 441000 # 20 seconds

    # Plot audio data
    plt.figure(figsize=(10, 6))
    plt.plot(time, clean_audio, color='blue', label='Clean')
    plt.plot(time, fx_audio, color='red', label='Fx')
    plt.plot(time, gen_audio, color='green', label='Gen')
    if valid:
        plt.plot(time, gen_init, color='orange', label='Gen_4')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveforms')
    plt.legend()
    plt.grid(True)
    plt.show()


create_graph(4, valid = True)