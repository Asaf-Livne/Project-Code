from imports import *
from audio_data_loading import AudioDataSet as adl
from load_data import load_data, normalize_data

# Parameter set #
LR = 0.0002
EPOCHS_NUM = 2000



class WaveNetModel(nn.Module):
    def __init__(self,dilation_repeats, dilation_depth, num_channels=8, skip_channels=8, end_channels = 8, kernel_size=1):

        self.dilation_repeats = dilation_repeats
        self.dilation_depth = dilation_depth
        self.num_blocks = dilation_repeats * dilation_depth

        super(WaveNetModel, self).__init__()
        self.casual_conv = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=1)
        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels, out_channels=end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(in_channels=end_channels, out_channels=1, kernel_size=1)
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
                layer_gated_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, dilation=2**depth_num))
                layer_filtered_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, dilation=2**depth_num))
                self.residual_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size))
                self.skip_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=skip_channels, kernel_size=kernel_size))
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
        skip = self.skip_convs[block_num](new_x)
        res_conv = self.residual_convs[block_num](new_x)
        residual += res_conv
        return skip, residual

    
    def wavenet_end(self, skip_sum):
        x = F.tanh(skip_sum)
        x = self.end_conv_1(x)
        x = F.tanh(x)
        x = self.end_conv_2(x)
        return x


    def forward(self, input):
        x = self.casual_conv(input)
        skip_sum = 0
        for block_num in range(self.num_blocks):
            repeat_num = block_num // self.dilation_depth
            dilation_depth = block_num % self.dilation_depth
            skip, residual = self.wavenet_block(x, repeat_num, dilation_depth, block_num)
            x = residual
            skip_sum += skip
        output = self.wavenet_end(skip_sum)
        #print (f'output is {output}')
        return output
    
def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x, x - coeff * x), dim=1)

def ESR_loss(preds, labels): # FIXME: check the Log MEL loss as described in the article
    # Calculate error as the difference between predictions and true labels
    preds, labels = pre_emphasis_filter(preds), pre_emphasis_filter(labels)
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

def gen_train(gen, optimizer, train_data, valid_data):
    train_losses = []
    valid_losses = []
    i = 0
    for epoch in range(EPOCHS_NUM):
        gen.train()
        train_loss = 0.0
        i = 0
        for clean_batch, fx_batch in tqdm.tqdm(train_data):
            optimizer.zero_grad()
            before_gen = time.time()
            predictions = gen(clean_batch)
            after_gen = time.time()
            #print(f"Time for gen - {after_gen - before_gen}")
            #predictions = predictions.view(-1)
            #fx_batch = fx_batch.view(-1)
            loss = ESR_loss(predictions, fx_batch)
            after_loss = time.time()
            #print(f'time for loss - {after_loss - after_gen}')
            train_loss += loss
            loss.backward()
            after_backward = time.time()
            #print(f'time for backward - {after_backward - after_loss}')
            optimizer.step()
            after_step = time.time()
            #print(f'time for step - {after_step - after_backward}')
            #print(f"Loss - {loss}")
            #print('\n')
            if i == 2:
                export_clean = clean_batch.view(-1)
                export_fx = fx_batch.view(-1)
                export_predictions = predictions.view(-1).detach().numpy()
            i += 1
        avg_train_loss = train_loss / i
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{EPOCHS_NUM}: training loss = {avg_train_loss}")
        sf.write(f'model_results/training_clean_batch_{epoch+1}.wav', export_clean, 44100)
        sf.write(f'model_results/training_fx_batch_{epoch+1}.wav', export_fx, 44100)
        sf.write(f'model_results/training_predictions_epoch_{epoch+1}.wav', export_predictions, 44100)
        gen.eval()
        valid_loss = 0.0
        i = 0
        for clean_batch, fx_batch in tqdm.tqdm(valid_data):
            predictions = gen(clean_batch)
            loss = ESR_loss(predictions, fx_batch)
            valid_loss += loss
            if i == 2:
                export_clean = clean_batch.view(-1)
                export_fx = fx_batch.view(-1)
                export_predictions = predictions.view(-1).detach().numpy()
            i += 1
        avg_valid_loss = valid_loss / i
        print(f"Epoch {epoch+1}/{EPOCHS_NUM}: validation loss = {avg_valid_loss}")
        sf.write(f'model_results/valid_clean_batch_{epoch+1}.wav', export_clean, 44100)
        sf.write(f'model_results/valid_fx_batch_{epoch+1}.wav', export_fx, 44100)
        sf.write(f'model_results/valid_predictions_epoch_{epoch+1}.wav', export_predictions, 44100)
        valid_losses.append(avg_valid_loss)
    return train_losses, valid_losses
'''
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
'''

def main(train):
    dilation_repeats, dilation_depth = 2, 9
    gen = WaveNetModel(dilation_repeats = dilation_repeats, dilation_depth=dilation_depth)
    print (gen)
    if train:
        train_data = adl.data_loader('./Data/clean_train.wav', './Data/fx_train.wav', sec_sample_size = 0.1, sr=44100, batch_size=20, shuffle=True)
        valid_data = adl.data_loader('./Data/clean_val.wav', './Data/fx_val.wav', sec_sample_size= 2, sr=44100, batch_size=1, shuffle=False)
        gen = WaveNetModel(dilation_repeats=dilation_repeats, dilation_depth=dilation_depth)
        optimizer = optim.Adam(gen.parameters(), lr=LR)
        train_losses, valid_losses = gen_train(gen, optimizer, train_data, valid_data)
        print("*************************")
        print(f'End of model training with learning rate - {LR}, repeats num - {gen.dilation_repeats} and depth - {dilation_depth}')
        print("*************************")
'''    else:
        gen = model_load(gen)
        test_model(gen, data_path='LW short.wav', sr=44100)

'''


if __name__ == '__main__':
    #train = input("Do you want to train the model? (y/n): ").lower() == 'y'
    main(1)
