from imports import *
from wavenet import WaveNetModel
from file_operations import write_audio


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



def train_epoch(model, train_data, optimizer, device, epoch):
    model.train()
    train_losses = []
    for clean_batch, fx_batch in tqdm.tqdm(train_data):
        optimizer.zero_grad()
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = model(clean_batch)
        loss = ESR_loss(predictions, fx_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)
    return avg_train_loss * 100



def valid_epoch(model, valid_data, device, epoch, valid_best_loss, rand_idx):
    model.eval()
    valid_losses = []
    i = 0
    for clean_batch, fx_batch in tqdm.tqdm(valid_data):
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = model(clean_batch)
        loss = ESR_loss(predictions, fx_batch)
        valid_losses.append(loss.item())
        if i == rand_idx:
            sample_clean, sample_fx, sample_pred = clean_batch, fx_batch, predictions
        i += 1
    avg_valid_loss = np.mean(valid_losses)
    if avg_valid_loss < valid_best_loss:
        write_audio(sample_clean, sample_fx, sample_pred, epoch, 'validation')
    return avg_valid_loss * 100



def gen_train(train_data, valid_data, device, params_dict):
    train_losses = []
    valid_losses = []
    num_repeats = params_dict['dilation_repeats']
    dilation_depth = params_dict['dilation_depth']
    num_channels = params_dict['num_channels']
    kernel_size = params_dict['kernel_size']
    epochs_num = params_dict['epochs_num']
    lr = params_dict['lr']
    gen = WaveNetModel(num_repeats, dilation_depth, num_channels, kernel_size)
    optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    valid_best_loss = float('inf')
    rand_idx = np.random.randint(0, len(valid_data))
    for epoch in range(1,epochs_num+1):
        avg_train_loss = train_epoch(gen, train_data, optimizer, device, epoch)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}/{epochs_num}: training loss = {format(avg_train_loss, '.2f')}%")
        avg_valid_loss = valid_epoch(gen, valid_data, device, epoch, valid_best_loss, rand_idx)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch {epoch}/{epochs_num}: validation loss = {format(avg_valid_loss, '.2f')}%")
        if avg_valid_loss < valid_best_loss:
            print(f"Loss improvment: old loss {valid_best_loss} -> new_loss {avg_valid_loss}")
            valid_best_loss = avg_valid_loss
            torch.save(gen.state_dict(), f'gen_best_model_num_repeats_{num_repeats}_dilation_depth_{dilation_depth}_num_channels_{num_channels}_kernel_size_{kernel_size}.pt')
    return train_losses, valid_losses
