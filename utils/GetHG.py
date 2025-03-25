import scipy.io as sio
import numpy as np
import os
from scipy.linalg import sqrtm
from .filters import load_filterbank, butter_fir_filter
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from collections import OrderedDict
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import exponential_running_standardize

def get_fb(data, fb, is_ea, ftype="butter"):
    """
        One frequency
    """
    fs = 500
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


def get_data(subject, dataset_id, data_path, chans=None):
    if dataset_id == 0:
        filename =  os.path.join(data_path, 'train/{:d}.mat'.format(subject))
    else:
        filename =  os.path.join(data_path, 'test/{:d}.mat'.format(subject))

    loader = BBCIDataset(filename)
    cnt = loader.load()

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800
    C_sensors = chans

    cnt = cnt.pick_channels(C_sensors)
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, 0, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)
    ival = [0, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset

# split dataset
def get_data_hg(data_path, subject, dataset_id, is_ea, freq_band):
    chans = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    data = get_data(subject, dataset_id, data_path, chans)
    X_processed = get_fb(data.X, freq_band, is_ea)
    return X_processed, data.y

