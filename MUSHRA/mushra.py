import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt

def band_pass(path, low_freq, high_freq, low_taper_width, high_taper_width):
    # Create a bandpass filter

    # Load your signal
    signal, sr = librosa.load(path, sr=44100)
    print(sr)

    # Compute the STFT
    D = librosa.stft(signal)

    # Create the frequency mask with smooth transitions
    fft_frequencies = librosa.fft_frequencies(sr=44100)
    mask = np.ones_like(fft_frequencies)  # Initialize the mask with ones

    # Low-frequency taper
    low_taper =  (1 + np.cos(np.pi * (fft_frequencies - low_freq) / low_taper_width)) / 2
    mask *= np.where((fft_frequencies >= (low_freq - low_taper_width)) & (fft_frequencies < low_freq), low_taper, 1)

    # Passband
    mask *=  np.where((fft_frequencies >= low_freq) & (fft_frequencies <= high_freq), 1, 1)

    # High-frequency taper
    high_taper = 0.3 + 0.7 * (1 + np.cos(np.pi * (fft_frequencies - high_freq) / high_taper_width)) / 2
    mask *= np.where((fft_frequencies > high_freq) & (fft_frequencies <= (high_freq + high_taper_width)), high_taper, 1)

    # Set mask to 0.5 outside the taper ranges
    mask = np.where(fft_frequencies > (high_freq + high_taper_width), 0.3, mask)
    mask = np.where(fft_frequencies < (low_freq - low_taper_width), 0.5, mask)

    # Ensure the values are in the correct range
    mask = np.clip(mask, 0, 1)

    plt.figure()
    plt.plot(fft_frequencies, mask)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Mask')
    plt.grid(True)
    plt.show()



    # Apply the frequency mask
    D_band_pass = D * mask[:, np.newaxis]

    # Inverse STFT
    filtered_signal = librosa.istft(D_band_pass)


    # Save or play the filtered signal
    sf.write('./Code/MUSHRA/midq_test_fx_batch.wav', filtered_signal, 44100)

    return filtered_signal

#band_pass('Code/MUSHRA/test_fx_batch.wav', 0, 7000, 10, 1000)

fx = 'Code/MUSHRA/test_fx_batch.wav'
gen = 'Code/MUSHRA/test_gen_batch.wav'
lowq = 'Code/MUSHRA/lowq_test_fx_batch.wav'
fx, sr = librosa.load(fx, sr=None)
gen, sr = librosa.load(gen, sr=None)
lowq, sr = librosa.load(lowq, sr=None)
midq, sr = librosa.load('Code/MUSHRA/midq_test_fx_batch.wav', sr=None)
sr = 44100
sample_length = 3 * sr

fx_scores = []
gen_scores = []
lowq_scores = []
midq_scores = []

for i in range(10):

    # Choose reference sample
    random_sample = np.random.randint(0, len(fx) - 2 * sample_length)
    print(random_sample)
    reference = fx[random_sample: random_sample + sample_length]
    # Save reference sample
    sf.write('Code/MUSHRA/reference.wav', reference, sr)

    # Create test samples
    test_fx = fx[random_sample + sample_length: random_sample + 2 * sample_length]
    test_gen = gen[random_sample + sample_length: random_sample + 2 * sample_length]
    lowq_test_fx = lowq[random_sample + sample_length: random_sample + 2 * sample_length]
    midq_test_fx = midq[random_sample + sample_length: random_sample + 2 * sample_length]

    sounds_array = np.array([test_fx, test_gen, lowq_test_fx, midq_test_fx])

    indices = np.arange(len(sounds_array))

    np.random.shuffle(indices)

    shuffled_sounds_array = sounds_array[indices]

    names = ['test_fx', 'test_gen', 'low_quality_test_fx', 'mid_quality_test_fx']




    # Write the test samples to files
    sf.write('Code/MUSHRA/A.wav', shuffled_sounds_array[0], sr)
    sf.write('Code/MUSHRA/B.wav', shuffled_sounds_array[1], sr)
    sf.write('Code/MUSHRA/C.wav', shuffled_sounds_array[2], sr)
    sf.write('Code/MUSHRA/D.wav', shuffled_sounds_array[3], sr)

    # Ask the user's input for each sample
    rating_A = int(input(f'How would you rate the sample A? '))
    rating_B = int(input(f'How would you rate the sample B? '))
    rating_C = int(input(f'How would you rate the sample C? '))
    rating_D = int(input(f'How would you rate the sample D? '))

    scores = [rating_A, rating_B, rating_C, rating_D]
    scores = np.array(scores)
    scores = scores[indices]


    print (f'A is {names[indices[0]]},\n B is {names[indices[1]]},\n C is {names[indices[2]]}\n, D is {names[indices[3]]} \n\n')
    print (f'You rated {names[indices[0]]} as {rating_A},\n {names[indices[1]]} as {rating_B},\n {names[indices[2]]} as {rating_C}, \n {names[indices[3]]} as {rating_D} \n\n')

    fx_scores.append(scores[0])
    gen_scores.append(scores[1])
    lowq_scores.append(scores[2])
    midq_scores.append(scores[3])

print(f'fx_scores - {fx_scores}')
print(f'fx_scores mean - {np.mean(fx_scores)}')
print(f'fx_scores std - {np.std(fx_scores)}\n\n')

print(f'gen_scores - {gen_scores}')
print(f'gen_scores mean - {np.mean(gen_scores)}')
print(f'gen_scores std - {np.std(gen_scores)}\n\n')

print(f'lowq_scores - {lowq_scores}')
print(f'lowq_scores mean - {np.mean(lowq_scores)}')
print(f'lowq_scores std - {np.std(lowq_scores)}\n\n')

print(f'midq_scores - {midq_scores}')
print(f'midq_scores mean - {np.mean(midq_scores)}')
print(f'midq_scores std - {np.std(midq_scores)}\n\n')


