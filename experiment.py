from imports import *
from wavenet import WaveNetModel
from file_operations import write_audio
from audio_data_loading import AudioDataSet as adl
import test
import train

exp_params = ['dilation_repeats', 'dilation_depth', 'num_channels', 'kernel_size']


def experiment(exp_param, exp_range, exp_name, train_data, valid_data, params_dict, device):
    training_losses = []
    valid_losses = []
    training_durations = []
    valid_best_losses = []
    for param in exp_range:
        start_time = time.time()
        params_dict[exp_param] = param
        train_losses, valid_losses, valid_best_loss = train.gen_train(train_data, valid_data, device, params_dict)
        end_time = time.time()
        training_losses.append(train_losses)
        valid_losses.append(valid_losses)
        training_durations.append(end_time - start_time)
        valid_best_losses.append(valid_best_loss)
    arrays_dict = {'exp_param': exp_param, 'exp_range': exp_range, 'training_losses': training_losses, 'valid_losses': valid_losses, 'training_durations': training_durations, 'valid_best_losses': valid_best_losses}
    with open(f'experiment_results/{exp_name}.pkl', 'wb') as f:
        pickle.dump(arrays_dict, f)








        
