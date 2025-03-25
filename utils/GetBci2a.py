import numpy as np
import scipy.io as sio
from scipy.linalg import sqrtm
from .filters import load_filterbank, butter_fir_filter, chebyshevFilter

def get_fb(data, fb, is_ea, ftype="butter"):
    """
        One frequency
    """
    fs = 250 # Sampling frequency
    forder = 8

    if len(fb) > 2:
        filter_bank = []
        for i in range(len(fb) - 1):
            filter_bank.append([fb[i], fb[i+1]])
        # filter_bank.append([fb[0], fb[-1]])
        n_freq = len(filter_bank)
    else:
        filter_bank = load_filterbank(fb = fb, fs = fs, order = forder, ftype = ftype)
        n_freq = filter_bank.shape[0]

    # time_windows = (np.array([2, 6]) * fs).astype(int)
    time_windows = (np.array([0, 4]) * fs).astype(int)
    t_start, t_end = time_windows[0], time_windows[1]
    n_samples = t_end-t_start

    n_tr_trial, n_channel, _ = data.shape
    
    # rho = 0.1

    cov_mat = np.zeros((n_channel, n_channel))
    filtered_data = np.zeros((n_freq, n_tr_trial, n_channel, n_samples))
    ea_data = np.zeros((n_freq, n_tr_trial, n_channel, n_samples))

    # calculate training covariance matrices  
    # for freq_idx in range(n_freq)
    for freq_idx in range(n_freq):
        for trial_idx in range(n_tr_trial):
            # filter signal
            if len(fb) > 2:
                data_filter = chebyshevFilter(data[trial_idx,:,t_start:t_end], filter_bank[freq_idx], fs)
            else:
                data_filter = butter_fir_filter(data[trial_idx,:,t_start:t_end], filter_bank[freq_idx])
            # regularized covariance matrix
            filtered_data[freq_idx, trial_idx] = data_filter
            cov_mat += 1/(n_samples-1)*np.dot(data_filter,np.transpose(data_filter))            
        mean_cov_mat = cov_mat / n_tr_trial
        ea_data[freq_idx] = np.dot(np.linalg.inv(sqrtm(mean_cov_mat)), filtered_data[freq_idx]).transpose(1, 0, 2)
    # ea_data = np.dot(np.linalg.inv(sqrtm(mean_cov_mat)), filtered_data).transpose(1, 2, 0, 3)
    if is_ea:
        needed_data = ea_data
    else: 
        needed_data = filtered_data
    return needed_data.transpose(1, 0, 2, 3)

def get_raw_data(subject, dataset_id, PATH, ftype="butter"):
    if dataset_id==0:
        a = sio.loadmat(PATH+'s00'+str(subject)+'.mat')
    else:
        a = sio.loadmat(PATH+'se00'+str(subject)+'.mat')
    data_return = np.transpose(a['x'], (2, 0, 1))
    class_return = np.squeeze(a['y'])
    return data_return, class_return

def get_data_bcic4_2a(data_path, subject, dataset_id, is_ea, freq_band):
    X, y = get_raw_data(subject, dataset_id, data_path)
    X_processed = get_fb(X, freq_band, is_ea)

    return X_processed, y