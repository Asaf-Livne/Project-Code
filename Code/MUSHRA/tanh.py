import librosa
import numpy as np
import soundfile as sf

clean = 'Code/MUSHRA/test_clean_batch.wav'
fx = 'Code/MUSHRA/test_fx_batch.wav'
gen = 'Code/MUSHRA/test_gen_batch.wav'

clean, sr = librosa.load(clean, sr=None)
fx, sr = librosa.load(fx, sr=None)
gen, sr = librosa.load(gen, sr=None)

clean = clean[-len(gen)+44100:]
fx = fx[-len(gen)+44100:]
gen = gen[44100:]

print(f"clean - {clean.shape}, fx - {fx.shape}, gen - {gen.shape}")

sf.write('Code/MUSHRA/test_clean_batch.wav', clean, sr)
sf.write('Code/MUSHRA/test_fx_batch.wav', fx, sr)
sf.write('Code/MUSHRA/test_gen_batch.wav', gen, sr)