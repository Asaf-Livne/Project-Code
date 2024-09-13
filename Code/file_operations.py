from imports import *



def write_audio(clean, fx, predictions, epoch, training='training'):
    export_clean = clean.view(-1)
    export_fx = fx.view(-1)
    export_predictions = predictions.view(-1).detach().numpy()
    if training == 'training':
        sf.write(f'model_results/{training}_clean_batch_{epoch}.wav', export_clean, 44100)
        sf.write(f'model_results/{training}_fx_batch_{epoch}.wav', export_fx, 44100)
        sf.write(f'model_results/{training}_predictions_epoch_{epoch}.wav', export_predictions, 44100)
    else:
        sf.write(f'model_results/{training}_predictions_epoch_{epoch}.wav', export_predictions, 44100)
        if epoch == 1:
            sf.write(f'model_results/{training}_clean_batch.wav', export_clean, 44100)
            sf.write(f'model_results/{training}_fx_batch.wav', export_fx, 44100)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

