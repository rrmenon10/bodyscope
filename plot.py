import argparse
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def time_domain(audiopath):

    # Load the audio file (we get the wave form and the sampling rate as output)
    x , _ = librosa.load(audiopath)
    
    # Plot waveform
    plt.figure()
    librosa.display.waveshow(x, x_axis='s',)
    plt.show()

def spectrogram(audiopath):

    # Load the audio file (we get the wave form and the sampling rate as output)
    x , _ = librosa.load(audiopath)

    # Compute the fourier transform and convert amplitude spectrogram to dB spectrogram 
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X), ref=np.max)

    # Plot spectrogram
    plt.figure()
    librosa.display.specshow(Xdb, y_axis='linear', x_axis='s')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--audiopath", \
                        default="usis_presentation_audio.wav", help='audio file path')
    args = parser.parse_args()
    time_domain(args.audiopath)
    spectrogram(args.audiopath)