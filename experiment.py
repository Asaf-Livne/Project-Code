from imports import *
from wavenet import WaveNetModel
from file_operations import write_audio
from audio_data_loading import AudioDataSet as adl
import test
import train

exp_params = ['dilation_repeats', 'dilation_depth', 'num_channels', 'kernel_size']


def experiment(exp_param, exp_range, train_data, valid_data, params_dict, device):
    training_losses = []
    valid_losses = []
    training_durations = []
    for param in exp_range:
        start_time = time.time()
        params_dict[exp_param] = param
        train_losses, valid_losses = train.gen_train(train_data, valid_data, device, params_dict)
        end_time = time.time()
        training_losses.append(train_losses)
        valid_losses.append(valid_losses)
        training_durations.append(end_time - start_time)
    np.savetxt(f'experiment results for {exp_param} in the range {exp_range}.csv', np.array([training_losses, valid_losses, training_durations]), delimiter=',', fmt='%1.4e')







        
