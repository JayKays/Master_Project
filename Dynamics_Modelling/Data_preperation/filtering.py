

import numpy as np
from scipy.signal import butter, lfilter, medfilt
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift


def lowpass_butter(data, fs = 100, cutoff = 6, idx = None):

    '''
    Implements a digital lowpass butterworth filter
    inputs:
        data    -> Numpy array of data to filter
        fs      -> sampling frequency of the given data
        cutoff  -> Cutoff frequency for the filter
        idx     -> array of indeces in data to be filtered, if None all but first column (time) is filtered
    
    returns:
        filtered version of data array
    '''
    filtered_data = data.copy()
    if idx is None:
        # idx = [i for i in range(1,data.shape[1])]
        idx = np.arange(data.shape[1])

    B, A = butter(2, cutoff / (fs / 2), btype='low') # 1st order Butterworth low-pass
    filtered_data[:, idx] = lfilter(B, A, data[:,idx], axis=0)

    return filtered_data


def lowpass_fft(data, fs = 100, cutoff = 6, idx = None):
    '''
    Implements a digital lowpass filter using the fast fourier transform
    inputs:
        data    -> Numpy array of data to filter
        fs      -> sampling frequency of the given data
        cutoff  -> Cutoff frequency for the filter
        idx     -> array of indeces in data to be filtered, if None all but first column (time) is filtered
    
    returns:
        filtered version of data array
    '''

    if idx is None:
        idx = [i for i in range(data.shape[1])]

    #Compute fft of data
    data_fft = fft(data[:,idx], axis = 0)
    freqs = fftshift(fftfreq(data_fft.shape[0], d = 1/fs))
    fftVar = fftshift(data_fft)

    #Remove frequencies above cutoff
    fft_filtered = fftVar.copy()
    fft_filtered[np.abs(freqs) > cutoff] = 0

    #Undo the fft with filtered frequency data
    filtered_data = ifft(ifftshift(fft_filtered), axis = 0)

    #Update data array with filtered version
    data[:,idx] = filtered_data
    
    return data 

def median_filter(data, window_size = 3):
    '''
    Performs a running median filtering of the input data array along each time axis
    '''

    filtered_data = medfilt(data.copy(), kernel_size=(window_size,1))

    return filtered_data

def filter_med_lowpass(data, cutoff = 4, fs = 100, window = 5):

    med_data = median_filter(data,window_size=window)
    lp_data = lowpass_butter(med_data, fs = fs, cutoff=cutoff)

    return lp_data