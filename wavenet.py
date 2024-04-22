from imports import *
from audio_data_loading import AudioDataSet as adl
from load_data import load_data, normalize_data

# Parameter set #
LR = 0.001
EPOCHS_NUM = 1000



class WaveNetModel(nn.Module):
    def __init__(self, num_blocks, dilation_layers, dilation_depth, num_channels=512, skip_channels=512, end_channels = 512, kernel_size=1):

        self.num_blocks = num_blocks
        self.num_layers = dilation_layers
        self.num_depth = dilation_depth

        super(WaveNetModel, self).__init__()
        self.casual_conv = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=1)
        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels, out_channels=end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(in_channels=end_channels, out_channels=1, kernel_size=1)
        # dilated conv
        self.dilated_convs = nn.ModuleList()
        # 1*1 conv list for skip connection
        self.skip_convs = nn.ModuleList()
        # 1*1 conv list for residual connection
        self.residual_convs = nn.ModuleList()

        for block_num in range(num_blocks):
            dilated_convs_block = nn.ModuleList()
            for layer_num in range(dilation_layers):
                for depth_num in range(dilation_depth):
                    dilated_convs_block.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, dilation=2**depth_num))
            self.skip_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=skip_channels, kernel_size=kernel_size))
            self.residual_convs.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size))
            self.dilated_convs.append(dilated_convs_block)
    
    def wavenet_block(self, x, block_num):
        residual = x.clone()  # Clone x to avoid in-place operations
        new_x = x.clone()  # Clone x to avoid in-place operations
        for dilated_conv in self.dilated_convs[block_num]:
            new_x = dilated_conv(new_x)
        # Apply activation functions
        filter = torch.tanh(new_x)
        gate = torch.sigmoid(new_x)
        new_x = filter * gate
        # Compute skip and residual connections
        skip = self.skip_convs[block_num](new_x)  # Assuming self.skip_conv is a single convolutional layer
        res_conv = self.residual_convs[block_num](new_x)  # Assuming self.residual_conv is a single convolutional layer
        residual += res_conv
        return skip, residual

    
    def wavenet_end(self, skip_sum):
        x = F.relu(skip_sum)
        x = self.end_conv_1(x)
        x = F.relu(x)
        x = self.end_conv_2(x)
        return x


    def forward(self, input):
        x = self.casual_conv(input)
        skip_sum = 0
        for block_num in range(self.num_blocks):
            skip, residual = self.wavenet_block(x, block_num)
            x = residual
        skip_sum += skip
        output = self.wavenet_end(x)
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

def batch_loss(gen, clean_batch, fx_batch):
    total_loss = 0.0
    batch_size = len(clean_batch)
    for i in range(batch_size):
        gen_audio = gen(clean_batch[i])
        loss = mse_loss(gen_audio, fx_batch[i])
        total_loss += loss
    return total_loss / batch_size

def concat_batch(batch):
    arr = []
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            for k in range(len(batch[i][j])):
                    arr.append(batch[i][j][k])
    return np.array(arr)

def loss_fn(preds, labels):
    return nn.MSELoss(preds, labels)

def gen_train(gen, optimizer, train_data):
    train_losses = []
    for epoch in range(EPOCHS_NUM):
        gen.train()
        train_loss = 0.0
        i = 0
        for clean_batch, fx_batch in tqdm.tqdm(train_data):
            if i == 0:
                i += 1
                continue
            optimizer.zero_grad()
            before_gen = time.time()
            predictions = gen(clean_batch)
            after_gen = time.time()
            #print(f"Time for gen - {after_gen - before_gen}")
            predictions = predictions.view(-1)
            fx_batch = fx_batch.view(-1)
            loss = mse_loss(predictions, fx_batch)
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
           # print('\n')
            if i == 2:
                export_clean = clean_batch.view(-1)
                export_fx = fx_batch
                export_predictions = predictions.detach().numpy()
            i += 1
        train_losses.append(train_loss / len(train_data.dataset))
        print(f"Epoch {epoch+1}/{EPOCHS_NUM}: gen_loss = {train_loss / len(train_data.dataset)}")
        if epoch == 0:
            sf.write(f'model_results/clean_batch.wav', export_clean, 44100)
            sf.write(f'model_results/fx_batch.wav', export_fx, 44100)
        sf.write(f'model_results/predictions_epoch_{epoch}.wav', export_predictions, 44100)
    return train_losses
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
    blk_nums, dilation_layers, dilation_depth = 2, 1, 8
    gen = WaveNetModel(num_blocks=blk_nums, dilation_layers = dilation_layers, dilation_depth=dilation_depth)
    if train:
        train_data = adl.data_loader('./Data/ts9_test1_in_FP32.wav', './Data/ts9_test1_out_FP32.wav', sec_sample_size= 0.5, sr=44100, batch_size=4, shuffle=False)
        gen = WaveNetModel(num_blocks=blk_nums, dilation_layers=dilation_layers, dilation_depth=dilation_depth)
        optimizer = optim.Adam(gen.parameters(), lr=LR)
        gen_losses = gen_train(gen, optimizer, train_data)
        print("*************************")
        print(f'End of model training with learning rate - {LR}, block num - {blk_nums} and layers num - {dilation_depth}')
        print("*************************")
'''    else:
        gen = model_load(gen)
        test_model(gen, data_path='LW short.wav', sr=44100)

'''


if __name__ == '__main__':
    #train = input("Do you want to train the model? (y/n): ").lower() == 'y'
    main(1)
