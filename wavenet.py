from imports import *
from audio_data_loading import AudioDataSet as audio_ds
from load_data import load_data, normalize_data

# Parameter set #
LR = 0.001
EPOCHS_NUM = 150



class WaveNetModel(nn.Module):
    def __init__(self, num_blocks, dilation_depth, num_channels=16, skip_channels=16, kernel_size=1):

        self.num_blocks = num_blocks
        self.num_layers = dilation_depth

        super(WaveNetModel, self).__init__()
        self.casual_conv = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=1)
        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels, out_channels=8, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1)
        # dilated conv
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # 1*1 conv list for skip connection
        self.skip_convs = nn.ModuleList()
        # 1*1 conv list for residual connection
        self.residual_conv = nn.ModuleList()

        for block_num in range(num_blocks):
            filter_convs_block = nn.ModuleList()
            gate_convs_block = nn.ModuleList()
            skip_convs_block = nn.ModuleList()
            residual_conv_block = nn.ModuleList()
            for layer_num in range(dilation_depth):
                filter_convs_block.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=kernel_size, dilation=2**layer_num))
                gate_convs_block.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=kernel_size, dilation=2**layer_num))
                skip_convs_block.append(nn.Conv1d(in_channels=num_channels*2, out_channels=skip_channels, kernel_size=kernel_size))
                residual_conv_block.append(nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels, kernel_size=kernel_size))
            self.filter_convs.append(filter_convs_block)
            self.gate_convs.append(gate_convs_block)
            self.skip_convs.append(skip_convs_block)
            self.residual_conv.append(residual_conv_block)
    
    def wavenet_layer(self, x, block_num, layer_num):
        residual = x 
        filter = self.filter_convs[block_num][layer_num](x)
        filter = F.tanh(filter)
        gate = self.gate_convs[block_num][layer_num](x)
        gate = F.sigmoid(gate)
        x = filter * gate
        return self.skip_convs[block_num][layer_num](x), self.residual_conv[block_num][layer_num](x), residual
    
    def wavenet_end(self, skip_sum):
        x = F.relu(skip_sum)
        #print(f"x after the first RELU is {x}")
        x = self.end_conv_1(x)
        x = F.relu(x)
        #print(f"x is {x}")
        return self.end_conv_2(x)


    def forward(self, input):

        x = self.casual_conv(input)
        skip_sum = 0
        for block_num in range(self.num_blocks):
            for layer_num in range(self.num_layers):
             #   print(f"layer_num is {layer_num}")
                skip, x, residual = self.wavenet_layer(x, block_num, layer_num)
             #   print(f"after the wavenet layer -> x is {x}")
                x += residual
                skip_sum += skip
       # print(f"x after the block and the layer num loop is {x}")
        return self.wavenet_end(skip_sum)
    
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


def gen_train(gen, optimizer, train_data, epoch):
    gen.train()
    train_loss = 0.0
    i = 0
    gen_audio_np = []
    clean_audio_np = []
    fx_audio_np = []
    for clean_batch, fx_batch in tqdm.tqdm(train_data):
        if i == 0:
            gen_audio_np = [gen(clean_batch[i]).detach().numpy() for i in range(len(clean_batch))]
            gen_audio_np = concat_batch(gen_audio_np)
            clean_audio_np = concat_batch(clean_batch)
            fx_audio_np = concat_batch(fx_batch)
        optimizer.zero_grad()
        loss = batch_loss(gen, clean_batch, fx_batch)
        train_loss += loss
        loss.backward()
        optimizer.step()
        i += 1
        print(f'train_loss = {(loss.item())}')
    sf.write(f'model_results/gen_output_epoch_{epoch}.wav', gen_audio_np, 44100) 
    if epoch == 0:
        sf.write(f'model_results/clean_output.wav', clean_audio_np, 44100)
        sf.write(f'model_results/fx_output.wav', fx_audio_np, 44100)
  
    return train_loss / len(train_data)

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


def main(train):
    blk_nums, dilation_depth = 2, 11
    gen = WaveNetModel(num_blocks=blk_nums, dilation_depth=dilation_depth)
    if train:
        train_data = load_data('./Data/LW Clean.wav', './Data/LW Dist.wav', sr=44100, batch_size=10, bit_size=2)
        gen = WaveNetModel(num_blocks=blk_nums, dilation_depth=dilation_depth)
        optimizer = optim.Adam(gen.parameters(), lr=LR, weight_decay=1e-4)
        model_run(gen, optimizer, train_data)
        print("*************************")
        print(f'End of model training with learning rate - {LR}, block num - {blk_nums} and layers num - {dilation_depth}')
        print("*************************")
    else:
        gen = model_load(gen)
        test_model(gen, data_path='LW short.wav', sr=44100)




if __name__ == '__main__':
    #train = input("Do you want to train the model? (y/n): ").lower() == 'y'
    main(1)
