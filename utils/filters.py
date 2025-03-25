#!/usr/bin/env python3

## Taken from https://github.com/MultiScale-BCI/IV-2a


'''	Functions used for bandpass filtering and freuquency band generation'''

import numpy as np
from scipy import signal 
from scipy.signal import butter, sosfilt, sosfreqz


__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

# def bandpass_filter(signal_in,f_band_nom):
# 	'''	Filter N channels with fir filter of order 101

# 	Keyword arguments:
# 	signal_in -- numpy array of size [NO_channels, NO_samples]
# 	f_band_nom -- normalized frequency band [freq_start, freq_end]

# 	Return:	filtered signal 
# 	'''
# 	numtabs = 101
# 	h = signal.firwin(numtabs,f_band_nom,pass_zero=False)
# 	NO_channels ,NO_samples = signal_in.shape 
# 	sig_filt = np.zeros((NO_channels ,NO_samples))

# 	for channel in range(0,NO_channels):
# 		sig_filt[channel] = signal.convolve(signal_in[channel,:],h,mode='same') # signal has same size as signal_in (centered)

# 	return sig_filt

def bandpass_filter(signal_in,f_band_nom): 


	'''	Filter N channels with fir filter of order 101

	Keyword arguments:
	signal_in -- numpy array of size [NO_channels, NO_samples]
	f_band_nom -- normalized frequency band [freq_start, freq_end]

	Return:	filtered signal 
	'''
	order = 4
	sos = butter(order, f_band_nom, analog=False, btype='band', output='sos')
	sig_filt = sosfilt(sos, signal_in)

	return sig_filt

# def load_bands(fb,f_s):
#     '''	Filter N channels with fir filter of order 101

#     Keyword arguments:
#     bandwith -- numpy array containing bandwiths ex. [2,4,8,16,32]
#     f_s -- sampling frequency

#     Return:	numpy array of normalized frequency bands
#     '''
#     f_bands = np.zeros((99,2)).astype(float)

#     band_counter = 0
#     for i in range(len(fb) - 1):
#         f_bands[band_counter] = [fb[i], fb[i+1]]
#         band_counter += 1 
            
#     # convert array to normalized frequency 
#     f_bands_nom = 2*f_bands[:band_counter]/f_s
#     return f_bands_nom

def load_bands(fb,f_s):
    '''	Filter N channels with fir filter of order 101

    Keyword arguments:
    bandwith -- numpy array containing bandwiths ex. [2,4,8,16,32]
    f_s -- sampling frequency

    Return:	numpy array of normalized frequency bands
    '''
    f_bands = np.zeros((99,2)).astype(float)

    band_counter = 0
    for i in range(len(fb) - 1):
        f_bands[band_counter] = [fb[i], fb[i+1]]
        band_counter += 1 
    # if len(fb) >2:        
    #     f_bands[band_counter] = [fb[0], fb[-1]]
    #     band_counter += 1
    # convert array to normalized frequency 
    f_bands_nom = 2*f_bands[:band_counter]/f_s
    return f_bands_nom


def load_filterbank(fb, fs, order = 4, ftype = 'butter'): 
    '''	Calculate Filters bank with Butterworth filter  

    Keyword arguments:
    bandwith -- numpy array containing bandwiths ex. [2,4,8,16,32]
    f_s -- sampling frequency

    Return:	numpy array containing filters coefficients dimesnions 'butter': [N_bands,order,6] 'fir': [N_bands,order]
    '''
    # if not multifreq:
    #     f_band_nom = load_bands(bandwidth,fs,max_freq) # get normalized bands 
    # else:
    #     f_band_nom = load_bands_v2(bandwidth, fs, max_freq)

    f_band_nom = load_bands(fb,fs)
    n_bands = f_band_nom.shape[0]

    if ftype == 'butter': 
        filter_bank = np.zeros((n_bands,order,6))
    elif ftype == 'fir':
        filter_bank = np.zeros((n_bands,order))


    for band_idx in range(n_bands):
        if ftype == 'butter': 
            filter_bank[band_idx] = butter(order, f_band_nom[band_idx], analog=False, btype='band', output='sos')
        elif ftype == 'fir':


            filter_bank[band_idx] = signal.firwin(order,f_band_nom[band_idx],pass_zero=False)
    return filter_bank

def butter_fir_filter(signal_in,filter_coeff):

	if filter_coeff.ndim == 2: # butter worth 
		return sosfilt(filter_coeff, signal_in)
	elif filter_coeff.ndim ==1: # fir filter 
		
		NO_channels ,NO_samples = signal_in.shape 
		sig_filt = np.zeros((NO_channels ,NO_samples))

		for channel in range(0,NO_channels):
			sig_filt[channel] = signal.convolve(signal_in[channel,:],filter_coeff,mode='same') # signal has same size as signal_in (centered)
		
		return sig_filt
		

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


def chebyshevFilter(data, bandFiltCutF,  fs, filtAllowance=2, axis=1, filtType='filter'):
    """
        Filter a signal using cheby2 iir filtering.

    Parameters
    ----------
    data: 2d/ 3d np array
        trial x channels x time
    bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
        if any value is specified as None then only one sided filtering will be performed
    fs: sampling frequency
    filtAllowance: transition bandwidth in hertz
    filtType: string, available options are 'filtfilt' and 'filter'

    Returns
    -------
    dataOut: 2d/ 3d np array after filtering
        Data after applying bandpass filter.
    """
    aStop = 30 # stopband attenuation
    aPass = 3 # passband attenuation
    nFreq= fs/2 # Nyquist frequency
    
    if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
        # no filter
        print("Not doing any filtering. Invalid cut-off specifications")
        return data
    
    elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
        # low-pass filter
        print("Using lowpass filter since low cut hz is 0 or None")
        fPass =  bandFiltCutF[1]/ nFreq
        fStop =  (bandFiltCutF[1]+filtAllowance)/ nFreq
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        b, a = signal.cheby2(N, aStop, fStop, 'lowpass')
    
    elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
        # high-pass filter
        print("Using highpass filter since high cut hz is None or nyquist freq")
        fPass =  bandFiltCutF[0]/ nFreq
        fStop =  (bandFiltCutF[0]-filtAllowance)/ nFreq
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        b, a = signal.cheby2(N, aStop, fStop, 'highpass')
    
    else:
        # band-pass filter
        # print("Using bandpass filter")
        fPass =  (np.array(bandFiltCutF)/ nFreq).tolist()
        fStop =  [(bandFiltCutF[0]-filtAllowance)/ nFreq, (bandFiltCutF[1]+filtAllowance)/ nFreq]
        # find the order
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

    if filtType == 'filtfilt':
        dataOut = signal.filtfilt(b, a, data, axis=axis)
    else:
        dataOut = signal.lfilter(b, a, data, axis=axis)
    return dataOut