from imports import *
from audio_data_loading import AudioDataSet as audio_ds

# Parameter set #
LR = 0.001
EPOCHS_NUM = 1 



class WaveNetModel(nn.Module):
    def __init__(self, num_blocks, dilation_depth, num_channels=32, skip_channels=256, kernel_size=1):

        self.num_blocks = num_blocks
        self.num_layers = dilation_depth
        kernel_size

        super(WaveNetModel, self).__init__()
        self.casual_conv = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=1)
        self.end_conv_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
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
                filter_convs_block[block_num].append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=kernel_size, dilation=2**layer_num))
                gate_convs_block[block_num].append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=kernel_size, dilation=2**layer_num))
                skip_convs_block[block_num].append(nn.Conv1d(in_channels=num_channels*2, out_channels=skip_channels))
                residual_conv_block[block_num].append(nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels))
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
        return self.skip_convs[block_num][layer_num](x), self.residual_conv[block_num][layer_num](x)
    
    def wavenet_end(self, skip_sum):
        x = F.relu(skip_sum)
        x = self.end_conv_1(x)
        x = F.relu(x)
        return self.end_conv_2(x)


    def forward(self, input):
        x = self.casual_conv(input)
        skip_sum = 0
        for block_num in range(self.num_blocks):
            for layer_num in range(self.num_layers):
                skip, residual = self.wavenet_layer(x, block_num, layer_num)
                x += residual
                skip_sum += skip
        return self.wavenet_end(skip_sum)
    
    #def genertor: FIXME: use ChatGpt to build a function that convert wave files to generate data

def data_loader():
    return 0

def ESR_loss(preds, labels): # FIXME: check the Log MEL loss as described in the article
    # Calculate error as the difference between predictions and true labels
    error = preds - labels
    # Calculate Root Mean Square Error (RMSE)
    rmse = np.mean(np.square(error))
    # Calculate Root Mean Square Signal (RMSS)
    rmss = np.mean(np.square(labels))
    # Calculate Error-to-Signal Ratio (ESR)
    esr = rmse / rmss
    return esr

def gen_train(gen, optimizer, train_data):
    gen.train()
    train_loss = 0.0
    for batch_idx, (data, _) in tqdm(train_data):
        optimizer.zero_grad()
        fake = gen(data)
        loss = ESR_loss()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10:
            print(f'Batch_idx ({batch_idx}/{len(train_data.dataset)} - train_loss = {(loss.item())*100}%)')
    return train_loss / len(train_data.dataset)

# FIXME: Check if the generator and discriminatr are trained togerther
def model_run(gen, optimizer, train_data, epochs_num):
    gen_best_loss = float('inf')
    for epoch in range(epochs_num):
        loss = gen_train(gen, optimizer, train_data)
        print(f"Epoch {epoch+1}: gen_loss = {loss} | discriminator_loss = {0}")

        if loss < gen_best_loss:
            print(f"Loss improvment: old loss {gen_best_loss} -> new_loss {loss}")
            gen_best_loss = loss
            torch.save(gen.state_dict(), f'gen_best_model_epcoh_{epoch}')


def main():
    train_data = audio_ds.data_loader(clean_path='./Clean/Neck', fx_path='./RAT/Neck')
    print(train_data)
    for data, label in train_data:
        print(data, label)
'''        for clean_audio, fx_audio in zip(clean_batch, fx_batch):
            clean_audio_np = clean_audio.numpy()
            fx_audio_np = fx_audio.numpy()
             # Play clean audio
            print("Playing clean audio...")
            sf.write('clean_output.wav', clean_audio_np, 48000)
            
            # Play FX audio
            print("Playing FX audio...")
            sf.write('fx_output.wav', fx_audio_np, 48000)            
            # Break the loop after playing the first audio clip from each batch
            break

    #gen = WaveNetModel(num_blocks=4, dilation_depth=7)
    gen = WaveNetModel(num_blocks=1, dilation_depth=1)
    optimizer = optim.Adam(gen.parameters(), lr=LR)

'''
if __name__ == '__main__':
    main()


                



