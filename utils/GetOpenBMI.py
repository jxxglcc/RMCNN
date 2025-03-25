import scipy.io as sio
import numpy as np
import os
from scipy.linalg import sqrtm
from .filters import load_filterbank, butter_fir_filter

def get_fb(data, fs, fb, is_ea, ftype="butter"):
    """
        One frequency
    """
    forder = 8
    time_windows = (np.array([0, 4]) * fs).astype(int)
    t_start, t_end = time_windows[0], time_windows[1]
    n_samples = t_end-t_start


    filter_bank = load_filterbank(fb = fb, fs = fs, order = forder, ftype = ftype)

    n_tr_trial, n_channel, _ = data.shape
    n_freq = filter_bank.shape[0]
    # rho = 0.1

    cov_mat = np.zeros((n_channel, n_channel))
    filtered_data = np.zeros((n_tr_trial, n_freq, n_channel, n_samples))

    # calculate training covariance matrices  
    for trial_idx in range(n_tr_trial):	

        for freq_idx in range(n_freq):
            # filter signal
            data_filter = butter_fir_filter(data[trial_idx,:,t_start:t_end], filter_bank[freq_idx])
            # regularized covariance matrix
            filtered_data[trial_idx, freq_idx] = data_filter
            cov_mat += 1/(n_samples-1)*np.dot(data_filter,np.transpose(data_filter))
            
    mean_cov_mat = cov_mat / n_tr_trial
    ea_data = np.dot(np.linalg.inv(sqrtm(mean_cov_mat)), filtered_data).transpose(1, 2, 0, 3)

    needed_data = ea_data if is_ea else filtered_data
    return needed_data


def get_data(PATH, chans=None):
    data = sio.loadmat(PATH)

    x = np.concatenate((data['EEG_MI_train'][0, 0]['smt'], data['EEG_MI_test'][0, 0]['smt']), axis=1).astype(
        np.float32)
    y = np.concatenate(
        (data['EEG_MI_train'][0, 0]['y_dec'].squeeze(), data['EEG_MI_test'][0, 0]['y_dec'].squeeze()),
        axis=0).astype(int) - 1
    c = np.array([m.item() for m in data['EEG_MI_train'][0, 0]['chan'].squeeze().tolist()])
    s = data['EEG_MI_train'][0, 0]['fs'].squeeze().item()
    del data

    # extract the requested channels:
    if chans is not None:
        x = x[:, :, np.array(chans)]
        c = c[np.array(chans)]

    # change the data dimensions to be in a format: Chan x time x trials
    x = np.transpose(x, axes=(1, 2, 0))

    return {'x': x, 'y': y, 'c': c, 's':s}

# split dataset
def get_data_open_bmi(data_path, subject, dataset_id, is_ea, freq_band):
    chans = [4,32,8,9,33,34,12,35,13,36,14,37,38,18,39,19,40,41,24,42,43]
    data_train = get_data(os.path.join(data_path, 'session' + str(dataset_id + 1),'s'+str(subject), 'EEG_MI.mat'), chans)
    X_processed = get_fb(data_train['x'], data_train['s'], freq_band, is_ea)
    return X_processed, data_train['y']