from imports import *


valid = pickle.load(open('./valid_losses.pkl', 'rb'))
gen = pickle.load(open('./train_gen_losses.pkl', 'rb'))
disc = pickle.load(open('./train_disc_losses.pkl', 'rb'))

disc = 100 - np.array(gen)

plt.figure(figsize=(10, 6))
plt.plot(valid, label='Validation Loss (ESR)')
plt.plot(gen, label='Generator Loss (100-Success Rate)')
plt.plot(disc, label='Discriminators Average Loss (100-Success Rate)')

plt.xlabel('Epoch')
plt.ylabel('Loss [%]')
plt.title('Validation and Training Losses')
plt.legend()
plt.grid()
plt.show()