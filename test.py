from imports import *
from wavenet import WaveNetModel
from file_operations import write_audio
from audio_data_loading import AudioDataSet as adl





def create_effects(gen, data_path, sr, device):
    gen.eval()
    gen.to(device)
    test_data = adl.data_loader(data_path, data_path, sec_sample_size=35, sr=sr, batch_size=1, shuffle=False)
    #test_data = DataLoader(libro_data, batch_size=1)
    print(f"data len - {len(test_data)}")
    i = 0
    for clean_batch, fx_batch in tqdm.tqdm(test_data):
        clean_batch = clean_batch.to(device)
        print(f"clean batch shape - {clean_batch.shape}")
        gen_audio = gen(clean_batch)
        write_audio(clean_batch, clean_batch, gen_audio, i, 'test')
        i += 1


def test_gen(dilation_repeats, dilation_depth, num_channels, kernel_size, best_gen_path, data_path, sr, device):
    gen = WaveNetModel(dilation_repeats=dilation_repeats, dilation_depth=dilation_depth, num_channels=num_channels, kernel_size=kernel_size)
    gen.load_state_dict(torch.load(best_gen_path))
    create_effects(gen, data_path, sr, device)
    print("Done!")