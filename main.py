from imports import *
from audio_data_loading import AudioDataSet as adl
import train
import test
from wavenet import WaveNetModel
import experiment



def main(action, paired = 1, exp_param = 0, exp_range = 0, cuda=False, load_models=False):
    # Init params
    # The last run with wavenet - some of the cnn layers's kernel size were changed
    train_data = adl.data_loader('Data/clean_old_train.wav', 'Data/fx_old_train.wav', sec_sample_size=1, sr=44100, batch_size=5, shuffle=True, scramble=True)
    valid_data = adl.data_loader('Data/clean_old_val.wav', 'Data/fx_old_val.wav', sec_sample_size=10, sr=44100, batch_size=1, shuffle=False, scramble=False)
    dilation_repeats, dilation_depth, num_channels, kernel_size, num_epochs, lr = 2, 9, 16, 5, 100, 0.0001
    print(f'dilation_repeats={dilation_repeats}, dilation_depth={dilation_depth}, num_channels={num_channels}, kernel_size={kernel_size}, lr={lr} with tanh as the last layer')
    params_dict = {'dilation_repeats': dilation_repeats, 'dilation_depth': dilation_depth, 'num_channels': num_channels, 'kernel_size': kernel_size, 'epochs_num': num_epochs, 'lr': lr}
    if torch.cuda.is_available and cuda: 
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    if action == 'train':
        if paired:
            train_losses, valid_losses, valid_best_loss = train.gen_train(train_data, valid_data, device, params_dict)
            #plot_loss(range(num_epochs), train_losses, valid_losses, 'Loss vs Epochs', 'Epochs', 'Loss [%]')
        else:
            train_gen_losses, train_disc_losses, valid_losses, valid_best_loss = train.gen_and_disc_train(train_data, valid_data, device, params_dict, load_models)
            #plot_loss(range(num_epochs), train_losses, valid_losses, 'Loss vs Epochs', 'Epochs', 'Loss [%]')
            # save the results
            # save the results
            pickle.dump(train_gen_losses, open('train_gen_losses.pkl', 'wb'))
            pickle.dump(train_disc_losses, open('train_disc_losses.pkl', 'wb'))
            pickle.dump(valid_losses, open('valid_losses.pkl', 'wb'))
    elif action == 'test':
        best_gen_path = './best_models/EQ.pt'
        test.test_gen(dilation_repeats, dilation_depth, num_channels, kernel_size, best_gen_path, 'Data/DB01/clean_sample.wav', './Data/fx_old_val.wav', 44100, device)
    elif action == 'exp':
        experiment.experiment(exp_param, exp_range, exp_name, train_data, valid_data, params_dict, device)


        

    


if __name__ == '__main__':
    #action = input("What do you want to do? (train/exp/test): ").lower()
    #cuda = input("Do you want to use a GPU? (y/n): ").lower() == 'y'
    action = 'test'
    cuda = False
    exp_param= 0
    exp_range = 0
    paired = 0
    load_models = False
    if action == 'exp':
        exp_param = input("Enter the parameter to experiment with: ").lower()
        exp_range = input("Enter the range of the parameter: ").split(',')
        exp_range = list(map(float, exp_range))
        exp_name = input("Enter the name of the experiment: ")
    main(action,paired, exp_param, exp_range, cuda, load_models)
