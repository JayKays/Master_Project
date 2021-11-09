

import numpy as np
from scipy.signal import butter, lfilter
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift


def butter_lowpass(data, fs = 100, cutoff = 6, idx = None):

    if idx is None:
        idx = [i for i in range(1,data.shape[1])]

    fs = 100#1E9 # 1 ns -> 1 GHz
    cutoff = 6 # 10 MHz
    B, A = butter(2, cutoff / (fs / 2), btype='low') # 1st order Butterworth low-pass
    data[:, idx] = lfilter(B, A, data[:,idx], axis=0)

    return data


def lowpass_fft(data, fs = 100, cutoff = 6, idx = None):

    print(idx)
    if idx is None:
        idx = [i for i in range(1,data.shape[1])]

    data_fft = fft(data[:,idx], axis = 0)
    freqs = fftshift(fftfreq(data_fft.shape[0], d = 1/fs))
    fftVar = fftshift(data_fft)

    #Remove high frequencies
    fft_filtered = fftVar.copy()
    fft_filtered[np.abs(freqs) > cutoff] = 0

    filtered_data = ifft(ifftshift(fft_filtered), axis = 0)

    data[:,idx] = filtered_data
    
    return data 
