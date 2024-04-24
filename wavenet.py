from imports import *
from audio_data_loading import AudioDataSet as adl
from load_data import load_data, normalize_data

# Parameter set #
LR = 0.0005
EPOCHS_NUM = 2000



class WaveNetModel(nn.Module):
    def __init__(self,dilation_repeats, dilation_depth, num_channels=8, skip_channels=8, end_channels = 8, kernel_size=3):

        self.dilation_repeats = dilation_repeats
        self.dilation_depth = dilation_depth
        self.num_blocks = dilation_repeats * dilation_depth

        super(WaveNetModel, self).__init__()
        self.casual_conv = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1)
        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels, out_channels=end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(in_channels=end_channels, out_channels=1, kernel_size=1)
        self.linear_mixer = nn.Conv1d(in_channels = self.num_blocks*num_channels, out_channels=1, kernel_size=1)
        # dilated conv
        self.gated_convs = nn.ModuleList()
        self.filtered_convs = nn.ModuleList()
        # 1*1 conv list for residual connection
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        
        for repeat_num in range(dilation_repeats):
            layer_gated_convs = nn.ModuleList()
            layer_filtered_convs = nn.ModuleList()
            for depth_num in range(dilation_depth):
                layer_gated_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, dilation=2**depth_num, padding = 0))
                layer_filtered_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, dilation=2**depth_num, padding = 0))
                self.residual_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=1))
                self.skip_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=skip_channels, kernel_size=1))
            self.gated_convs.append(layer_gated_convs)
            self.filtered_convs.append(layer_filtered_convs)
    
    def wavenet_block(self, x, repeat_num, dilation_depth, block_num):
        residual = x.clone() 
        gate_x = x.clone()  
        filter_x = x.clone()  
        gate_x = self.gated_convs[repeat_num][dilation_depth](gate_x)
        filter_x = self.filtered_convs[repeat_num][dilation_depth](filter_x)
        # Apply activation functions
        filter = torch.tanh(filter_x)
        gate = torch.sigmoid(gate_x)
        new_x = filter * gate
        skip = new_x
        res_conv = self.residual_convs[block_num](new_x)
        residual = residual [:, :, -res_conv.size(2):]
        residual += res_conv
        return skip, residual

    
    def wavenet_end(self, skip_sum, output_size):
        skip_sum = torch.cat([s[:, :, -output_size :] for s in skip_sum], dim=1)
        return self.linear_mixer(skip_sum)


    def forward(self, input):
        x = self.casual_conv(input)
        skip_sum = []
        for block_num in range(self.num_blocks):
            repeat_num = block_num // self.dilation_depth
            dilation_depth = block_num % self.dilation_depth
            skip, residual = self.wavenet_block(x, repeat_num, dilation_depth, block_num)
            x = residual
            skip_sum.append(skip)
        output_size = x.size(2)
        output = self.wavenet_end(skip_sum, output_size)
        #print (f'output is {output}')
        return output
    



    
def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x, x - coeff * x), dim=1)

def ESR_loss(preds, labels): # FIXME: check the Log MEL loss as described in the article
    # Calculate error as the difference between predictions and true labels
    preds, labels = pre_emphasis_filter(preds), pre_emphasis_filter(labels)
    labels = labels[:, :, -preds.size(2):]
    error = preds - labels
    # Calculate Root Square Error (RMSE)
    rmse = torch.sum(torch.square(error))
    # Calculate Root Square Signal (RMSS)
    rmss = torch.sum(torch.square(labels))
    # Calculate Error-to-Signal Ratio (ESR)
    esr = torch.abs(rmse / rmss)
    return esr


def mse_loss(preds, labels):
    return F.mse_loss(preds, labels)



def write_audio(clean, fx, predictions, epoch, training='training'):
    export_clean = clean.view(-1)
    export_fx = fx.view(-1)
    export_predictions = predictions.view(-1).detach().numpy()
    sf.write(f'model_results/{training}_clean_batch_{epoch+1}.wav', export_clean, 44100)
    sf.write(f'model_results/{training}_fx_batch_{epoch+1}.wav', export_fx, 44100)
    sf.write(f'model_results/{training}_predictions_epoch_{epoch+1}.wav', export_predictions, 44100)


def train_epoch(model, train_data, optimizer, device, epoch):
    model.train()
    train_losses = []
    i = 0
    for clean_batch, fx_batch in tqdm.tqdm(train_data):
        optimizer.zero_grad()
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = model(clean_batch)
        loss = ESR_loss(predictions, fx_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if i == 5:
            write_audio(clean_batch, fx_batch, predictions, epoch, 'training')
        i += 1
    avg_train_loss = np.mean(train_losses)
    return avg_train_loss



def valid_epoch(model, valid_data, device, epoch):
    model.eval()
    valid_losses = []
    i = 0
    for clean_batch, fx_batch in tqdm.tqdm(valid_data):
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = model(clean_batch)
        loss = ESR_loss(predictions, fx_batch)
        valid_losses.append(loss.item())
        if i == 5:
            write_audio(clean_batch, fx_batch, predictions, epoch, 'valid')
        i += 1
    avg_valid_loss = np.mean(valid_losses)
    return avg_valid_loss



def gen_train(gen, optimizer, train_data, valid_data, device):
    train_losses = []
    valid_losses = []
    for epoch in range(EPOCHS_NUM):
        avg_train_loss = train_epoch(gen, train_data, optimizer, device, epoch)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{EPOCHS_NUM}: training loss = {avg_train_loss}")
        avg_valid_loss = valid_epoch(gen, valid_data, device, epoch)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch {epoch+1}/{EPOCHS_NUM}: validation loss = {avg_valid_loss}")
    return train_losses, valid_losses

# FIXME: Check if the generator and discriminatr are trained togerther
def model_run(gen, optimizer, train_data):
    gen_best_loss = float('inf')
    for epoch in range(EPOCHS_NUM):
        loss = gen_train(gen, optimizer, train_data, epoch)
        print(f"Epoch {epoch+1}/{EPOCHS_NUM}: gen_loss = {loss} | discriminator_loss = {0}")
        if loss < gen_best_loss:
            print(f"Loss improvment: old loss {gen_best_loss} -> new_loss {loss}")
            gen_best_loss = loss
            torch.save(gen.state_dict(), f'gen_best_model.pt')


def model_load(generator):
    generator.load_state_dict(torch.load(f'gen_best_model.pt'))
    return generator


def test_model(gen, data_path, sr):
    libro_data, sr = librosa.load(data_path, sr=sr)
    test_data = torch.tensor(libro_data)
    #test_data = DataLoader(libro_data, batch_size=1)
    print(f"data len - {len(test_data)}")
    gen_audio = gen(test_data)
    print(gen_audio)
    gen_audio_np = gen_audio[0].detach().numpy()
    print(gen_audio_np)
    sf.write(f'trained_out.wav', gen_audio_np, sr) 
    sf.write(f'test_data.wav', libro_data, sr) 


def main(train, cuda=False):
    dilation_repeats, dilation_depth = 2, 9
    gen = WaveNetModel(dilation_repeats = dilation_repeats, dilation_depth=dilation_depth)
    # Google Colab credit check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(device == 'cuda'): print('Running on GPU')
    else: print('Running on CPU')
    gen = gen.to(device)
    if train:
        train_data = adl.data_loader('./Data/clean_train.wav', './Data/fx_train.wav', sec_sample_size = 0.1, sr=44100, batch_size=20, shuffle=True)
        valid_data = adl.data_loader('./Data/clean_val.wav', './Data/fx_val.wav', sec_sample_size= 2, sr=44100, batch_size=1, shuffle=False)
        optimizer = optim.Adam(gen.parameters(), lr=LR)
        train_losses, valid_losses = gen_train(gen, optimizer, train_data, valid_data, device)
        print("*************************")
        print(f'End of model training with learning rate - {LR}, repeats num - {gen.dilation_repeats} and depth - {dilation_depth}')
        print("*************************")
    else:
        gen = model_load(gen)
        test_model(gen, data_path='LW short.wav', sr=44100)


if __name__ == '__main__':
    #train = input("Do you want to train the model? (y/n): ").lower() == 'y'
    main(1)
