from imports import *
from wavenet import WaveNetModel
from discriminator import Discriminator
from time_disc import TimeDiscriminator
from alexnet import AlexNet
import auraloss
from signal_processing import convert_batch_to_log_mel_stft
from file_operations import write_audio

lambda_gp = 0.1


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



def gen_train_epoch(model, train_data, optimizer, device, epoch):
    model.train()
    train_losses = []
    stft = auraloss.freq.STFTLoss().to(device)
    i = 0
    smooth = torch.nn.SmoothL1Loss().to(device)
    for clean_batch, fx_batch in tqdm.tqdm(train_data):
        optimizer.zero_grad()
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        #write_audio(clean_batch, fx_batch, fx_batch, epoch, 'training')
        predictions = model(clean_batch)
        # Trim the longer tensor to match the size of the shorter one
        min_len = min(predictions.size(2), fx_batch.size(2))
        predictions = predictions[:, :, :min_len]
        fx_batch = fx_batch[:, :, :min_len]

        loss = (stft(predictions, fx_batch) + smooth(predictions, fx_batch) + ESR_loss(predictions, fx_batch))/3
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        i +=1

    avg_train_loss = np.mean(train_losses)
    return avg_train_loss * 100



def gen_valid_epoch(model, valid_data, device, epoch, valid_best_loss, rand_idx):
    model.eval()
    valid_losses = []
    i = 0
 
    for clean_batch, fx_batch in tqdm.tqdm(valid_data):
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = model(clean_batch)
        loss = min(ESR_loss(predictions, fx_batch), ESR_loss(predictions, -1*fx_batch))
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
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=20, verbose=True)
    #gen.load_state_dict(torch.load(f'../trained_generators/gen_best_model_num_repeats_{num_repeats}_dilation_depth_{dilation_depth}_num_channels_{num_channels}_kernel_size_{kernel_size}.pt'))
    valid_best_loss = float('inf')
    rand_idx = np.random.randint(0, len(valid_data))
    for epoch in range(1,epochs_num+1):
        avg_train_loss = gen_train_epoch(gen, train_data, optimizer, device, epoch)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}/{epochs_num}: training loss = {format(avg_train_loss, '.2f')}%")
        avg_valid_loss = gen_valid_epoch(gen, valid_data, device, epoch, valid_best_loss, rand_idx)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch {epoch}/{epochs_num}: validation loss = {format(avg_valid_loss, '.2f')}%")
        if avg_valid_loss < valid_best_loss:
            print(f"Loss improvment: old loss {format(valid_best_loss, '.2f')}%, new loss {format(avg_valid_loss, '.2f')}%")
            valid_best_loss = avg_valid_loss
            torch.save(gen.state_dict(), f'trained_generators/gen_best_model_num_repeats_{num_repeats}_dilation_depth_{dilation_depth}_num_channels_{num_channels}_kernel_size_{kernel_size}.pt')
        #scheduler.step(avg_valid_loss)
    return train_losses, valid_losses, valid_best_loss



def discriminator_loss(real_preds, fake_preds):
    real_loss = torch.mean(torch.relu(1.0 - real_preds))
    fake_loss = torch.mean(torch.relu(1.0 + fake_preds))
    loss = (real_loss + fake_loss) / 2
    return loss


def gen_overflow_loss(fake_sound):
    total_loss = 0
    for i in range(len(fake_sound)):
        tensor = fake_sound[i][0]
        overflow = tensor[tensor > 0.99]
        underflow = tensor[tensor < -0.99]
        loss = len(overflow) + len(underflow)
        loss = loss/float(len(tensor))
        total_loss += loss
    return total_loss/len(fake_sound) 

def gen_zero_bias_loss(fake_sound):
    total_loss = 0
    for i in range(fake_sound.size(0)):
        tensor = fake_sound[i][0]
        pos_channel = tensor[tensor > 0]
        neg_channel = tensor[tensor < 0]
        loss_n1 = np.abs(len(pos_channel) - len(neg_channel))
        loss_n2 = torch.abs(torch.sum(tensor))
        if len(pos_channel) < 10 or len(neg_channel) <= 10:
            loss_var = 1
        else:
            loss_var = torch.abs(torch.var(pos_channel) - torch.var(neg_channel)) * len(tensor) * 2
        loss = loss_n1 + loss_n2 + loss_var
        loss /= tensor.size(0)
        total_loss += loss
    total_loss = total_loss/fake_sound.size(0)
    total_loss = torch.relu(total_loss-0.01)
    return total_loss

def gradient_penalty(disc, fake_img, real_img):
    BATCH_SIZE, _, H, W = real_img.shape # Batch, Channels, Heights, Width
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, 1, H, W)
    # Create interpolated images
    interpolated_images = real_img * alpha + fake_img * (1 - alpha)

    # Calculate critic 
    score = disc(interpolated_images)

    # Take the gradient of the scores with respect to the images
    grad = torch.autograd.grad(inputs=interpolated_images, outputs=score, grad_outputs=torch.ones_like(score), create_graph=True, retain_graph=True,)[0]
    grad_norm = (grad.view(grad.shape[0], -1)).norm(2, dim=1)
    return torch.mean((grad_norm - 1) ** 2)

# --- defines the discical loss function --- #
def disc_loss_calc(disc, disc_fake, fake_img, disc_real, real_img):
    gp = gradient_penalty(disc, fake_img, real_img)
    return lambda_gp * gp - (torch.mean(disc_real) - torch.mean(disc_fake))  



def gen_and_disc_train_example(gen, disc, clean_batch, fx_batch, optimizer_gen, optimizer_disc, device, cnt, window, n_mels):
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)        
        fake_sound = gen(clean_batch)
        fx_batch = fx_batch[:, :, :fake_sound.size(2)]
        if type(disc) == TimeDiscriminator:
            real_preds = disc(fx_batch)
            fake_preds = disc(fake_sound)
            loss_disc = discriminator_loss(real_preds, fake_preds)
        else:
            log_mel_real = convert_batch_to_log_mel_stft(fx_batch, 44100, window_size=window, n_mels=n_mels)
            log_mel_fake = convert_batch_to_log_mel_stft(fake_sound, 44100, window_size=window, n_mels=n_mels)
            real_preds = disc(log_mel_real)
            fake_preds = disc(log_mel_fake)
            loss_disc = discriminator_loss(real_preds, fake_preds)


        loss_fake = torch.mean(real_preds - fake_preds) / 2
        loss_gen = loss_fake

        #loss_disc = discriminator_loss(real_preds, fake_preds)
        

        if cnt % 2 == 1:
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()
        else:
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()
        loss_gen = loss_gen.item()
        loss_disc = loss_disc.item()
        real_preds_avg = torch.mean(real_preds).item()
        fake_preds_avg = torch.mean(fake_preds).item()
        return loss_gen, loss_disc, real_preds_avg, fake_preds_avg


def gen_and_discs_train_epoch(gen, discs, train_data, optimizer_gen, optimizer_discs, device, epoch, windows, n_mels):
    gen.train()
    for disc in discs:
        disc.train()
        
    train_losses_gen, train_losses_disc, real_avg, fake_avg = ([[] for _ in range(len(discs))] for _ in range(4))
    cnt = 0
    for clean_batch, fx_batch in tqdm.tqdm(train_data):
        for i, disc in enumerate(discs):
            loss_gen, loss_disc, real_preds, fake_preds = gen_and_disc_train_example(gen, disc, clean_batch, fx_batch, optimizer_gen, optimizer_discs[i], device, cnt, windows[i], n_mels[i])
            train_losses_gen[i].append(loss_gen)
            train_losses_disc[i].append(loss_disc)
            real_avg[i].append(real_preds)
            fake_avg[i].append(fake_preds)
            if cnt % 10 == 0:
                print(f"Epoch {epoch}: batch{cnt}: disc{i} - loss_gen = {format(loss_gen*100, '.2f')}%, loss_disc = {format(loss_disc*100, '.2f')}%, real_preds = {format(real_preds, '.2f')}, fake_preds = {format(fake_preds, '.2f')}")
        cnt += 1
    avg_train_loss_gen = [np.mean(losses) * 100 for losses in train_losses_gen]        
    avg_train_loss_disc = [np.mean(losses) * 100 for losses in train_losses_disc]
    real_avg_avg = [np.mean(avg) for avg in real_avg]
    fake_avg_avg = [np.mean(avg) for avg in fake_avg]
    
    return avg_train_loss_gen, avg_train_loss_disc, real_avg_avg, fake_avg_avg


def spectral_loss(fx_batch, fake_audio, window_size, n_mels):
    log_mel_real = convert_batch_to_log_mel_stft(fx_batch, 44100, window_size=window_size, n_mels=n_mels)
    log_mel_fake = convert_batch_to_log_mel_stft(fake_audio, 44100, window_size=window_size, n_mels=n_mels)
    err = (log_mel_real - log_mel_fake) ** 2
    rmss = (log_mel_real ** 2)
    return torch.sum(err) / torch.sum(rmss) / len(fx_batch)

def gen_spectral_validation(gen, valid_data, device, epoch, valid_best_loss, rand_idx, window, n_mels):
    gen.eval()
    valid_losses = []
    i = 0
    for clean_batch, fx_batch in tqdm.tqdm(valid_data):
        clean_batch, fx_batch = clean_batch.to(device), fx_batch.to(device)
        predictions = gen(clean_batch)
        # Len matching
        min_len = min(fx_batch.size(2), predictions.size(2))
        fx_batch = fx_batch[:, :, -min_len:]
        predictions = predictions[:, :, -min_len:]
        loss = spectral_loss(fx_batch, predictions, window, n_mels)
        valid_losses.append(loss.item())
        if i == rand_idx:
            sample_clean, sample_fx, sample_pred = clean_batch, fx_batch, predictions
        i += 1
    avg_valid_loss = np.mean(valid_losses)
    if avg_valid_loss < valid_best_loss:
        write_audio(sample_clean, sample_fx, sample_pred, epoch, 'validation')
    return avg_valid_loss * 100    

def gen_and_disc_train(train_data, valid_data, device, params_dict, load_models=False):
    train_avg_gen_losses = []
    train_avg_disc_losses = []
    valid_losses = []
    num_repeats, dilation_depth, num_channels, kernel_size, epochs_num, lr = params_dict['dilation_repeats'], params_dict['dilation_depth'], params_dict['num_channels'], params_dict['kernel_size'], params_dict['epochs_num'], params_dict['lr']

    gen = WaveNetModel(num_repeats, dilation_depth, num_channels, kernel_size)
    discs = [Discriminator((3,10), (1,2), 9), Discriminator(5, 1, 9), Discriminator((10,2), 1, 5)]
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=lr)
    optimizers_discs = [torch.optim.Adam(disc.parameters(), lr=lr) for disc in discs]

    

    if load_models:
        gen.load_state_dict(torch.load(f'Code/trained_generators/gen_best_model_R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}.pt'))
        for i, disc in enumerate(discs):
            disc.load_state_dict(torch.load(f'Code/trained_discriminators/disc_best_model_{2**(i+9)}.pt'))

    valid_best_loss = float('inf')
    for epoch in range(1,epochs_num+1):
        gen_losses, disc_losses, real_preds_avgs, fake_preds_avgs = gen_and_discs_train_epoch(gen, discs, train_data, optimizer_gen, optimizers_discs, device, epoch, windows=[512,1024,2048], n_mels=[64,128,256])
        print(f'Epoch {epoch}/{epochs_num}:')
        print(f'Generator loss = {format(np.mean(gen_losses), ".2f")}%')
        for disc_idx, disc in enumerate(discs):
            print(f'Discriminator loss for disc[{disc_idx}] = {format(disc_losses[disc_idx], ".2f")}%')
            print(f'Real and fake preds for disc[{disc_idx}]: {format(real_preds_avgs[disc_idx], ".2f")}, {format(fake_preds_avgs[disc_idx], ".2f")}')
        gen_loss = np.mean(gen_losses)
        disc_loss = np.mean(disc_losses)
        train_avg_gen_losses.append(gen_loss)
        train_avg_disc_losses.append(disc_loss)
        avg_valid_loss = 0
        if epoch  != 1 or epoch == 1:
            avg_valid_loss = gen_spectral_validation(gen, valid_data, device, epoch, valid_best_loss, 0, 1024, 128)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch {epoch}/{epochs_num}: validation loss = {format(avg_valid_loss, '.2f')}%")
        if avg_valid_loss < valid_best_loss:
            print(f"Loss improvment: old loss {format(valid_best_loss, '.2f')}%, new loss {format(avg_valid_loss, '.2f')}%")
            valid_best_loss = avg_valid_loss
            torch.save(gen.state_dict(), f'Code/trained_generators/gen_best_model_R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}.pt')
            for i, disc in enumerate(discs):
                torch.save(disc.state_dict(), f'Code/trained_discriminators/disc_best_model_{2**(i+9)}.pt')
        torch.save(gen.state_dict(), f'Code/trained_generators/gen_model_R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}_epoch{epoch}.pt')
        for i, disc in enumerate(discs):
            torch.save(disc.state_dict(), f'Code/trained_discriminators/disc_model_{2**(i+9)}_epoch{epoch}.pt')
        torch.save(gen.state_dict(), f'Code/trained_generators/gen_last_model_R{num_repeats}_D{dilation_depth}_C{num_channels}_K{kernel_size}.pt')
        for i, disc in enumerate(discs):
            torch.save(disc.state_dict(), f'Code/trained_discriminators/disc_last_model_{2**(i+9)}.pt')
    return train_avg_gen_losses, train_avg_disc_losses, valid_losses, valid_best_loss