from imports import *
from wavenet import WaveNetModel
from file_operations import write_audio
from audio_data_loading_test import AudioDataSet as adl
from train import ESR_loss





def create_effects(gen, data_path_clean, data_path_fx, sr, device):
    gen.eval()
    gen.to(device)
    test_data = adl.data_loader(data_path_clean, data_path_fx, sec_sample_size=5, sr=sr, batch_size=1, shuffle=False)
    #test_data = DataLoader(libro_data, batch_size=1)
    print(f"data len - {len(test_data)}")
    i = 1
    loss = 0
    for clean_batch, fx_batch in tqdm.tqdm(test_data):
        clean_batch = clean_batch.to(device)
        print(f"clean batch shape - {clean_batch.shape}")
        gen_audio = gen(clean_batch)
        write_audio(clean_batch, fx_batch, gen_audio, i, 'test')
        loss += ESR_loss(gen_audio, fx_batch)
        i += 1
    print(f"Total loss - {loss/(i-1)}")


def test_gen(dilation_repeats, dilation_depth, num_channels, kernel_size, best_gen_path, data_path_clean, data_path_fx, sr, device):
    gen = WaveNetModel(dilation_repeats=dilation_repeats, dilation_depth=dilation_depth, num_channels=num_channels, kernel_size=kernel_size)
    gen.load_state_dict(torch.load(best_gen_path))
    create_effects(gen, data_path_clean, data_path_fx, sr, device)
    print("Done!")