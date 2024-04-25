from imports import *
from audio_data_loading import AudioDataSet as adl
import train
import print_signals
import test
from wavenet import WaveNetModel
import experiment


def main(action, exp_param = 0, exp_range = 0, cuda=False):
    # Init params
    train_data = adl.data_loader('./Data/clean_train.wav', './Data/fx_train.wav', sec_sample_size=0.2, sr=44100, batch_size=8, shuffle=True)
    valid_data = adl.data_loader('./Data/clean_val.wav', './Data/fx_val.wav', sec_sample_size=10, sr=44100, batch_size=1, shuffle=False)
    dilation_repeats, dilation_depth, num_channels, kernel_size, num_epochs, lr = 2, 9, 16, 2, 100, 0.0003
    params_dict = {'dilation_repeats': dilation_repeats, 'dilation_depth': dilation_depth, 'num_channels': num_channels, 'kernel_size': kernel_size, 'epochs_num': num_epochs, 'lr': lr}
    if torch.cuda.is_available and cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    if action == 'train':
        train_losses, valid_losses = train.gen_train(train_data, valid_data, device, params_dict)
        print_signals.plot_loss(range(num_epochs), train_losses, valid_losses, 'Loss vs Epochs', 'Epochs', 'Loss [%]')
    elif action == 'test':
        best_gen_path = input("Enter the path to the best generator model: ")
        test.test_gen(dilation_repeats, dilation_depth, num_channels, kernel_size, best_gen_path, './Data/LW Clean.wav', 44100, device)
    elif action == 'exp':
        experiment.experiment(exp_param, exp_range, train_data, valid_data, params_dict, device)


        

    


if __name__ == '__main__':
    action = input("What do you want to do? (train/exp/test): ").lower()
    cuda = input("Do you want to use a GPU? (y/n): ").lower() == 'y'
    exp_param= 0
    exp_range = 0
    if action == 'exp':
        exp_param = input("Enter the parameter to experiment with: ").lower()
        exp_range = input("Enter the range of the parameter: ").split(',')
        exp_range = list(map(int, exp_range))
    main(action, exp_param, exp_range, cuda)
