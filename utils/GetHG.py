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

    # time_windows_flt = np.array([[2.5,4.5], [4,6], [2.5,6],
    #                             [2.5,3.5], [3,4], [4,5]])*fs
    
    # time_windows = time_windows_flt.astype(int)
    # restrict time windows and frequency bands 
    # time_windows = time_windows[2:3]
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

# def get_ea(data):
#     n_tr_trial, n_channel, n_samples = data.shape
#     # rho = 0.1

#     cov_mat = np.zeros((n_channel, n_channel))

#     # calculate training covariance matrices  
#     for trial_idx in range(n_tr_trial):	
#         # filter signal
#         cov_mat += 1/(n_samples-1)*np.dot(data[trial_idx],np.transpose(data[trial_idx]))
            
#     mean_con_mat = cov_mat / n_tr_trial
#     ea_data = np.dot(np.linalg.inv(sqrtm(mean_con_mat)), data).transpose(1, 0, 2)
#     return ea_data

def get_data(subject, dataset_id, data_path, chans=None):
    if dataset_id == 0:
        filename =  os.path.join(data_path, 'train/{:d}.mat'.format(subject))
    else:
        filename =  os.path.join(data_path, 'test/{:d}.mat'.format(subject))
    # load_sensor_names = None
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename)

    # log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    # log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    # log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
    #     np.sum(clean_trial_mask),
    #     len(set_for_cleaning.X),
    #     np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    # C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
    #              'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
    #              'C6',
    #              'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
    #              'FCC5h',
    #              'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
    #              'CPP5h',
    #              'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
    #              'CCP1h',
    #              'CCP2h', 'CPP1h', 'CPP2h']
    C_sensors = chans
    # if debug:
    #     C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    # log.info("Resampling...")
    # cnt = resample_cnt(cnt, 250.0)
    # log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, 0, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    # log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [0, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset

def add_noisy(eeg_data, ratio_noisy):
    # 定义噪声标准差（可以根据需求调整）
    # noise_std = 0.1  # 噪声的标准差
    signal_std = np.std(eeg_data)  # EEG 数据的标准差
    noise_std = ratio_noisy * signal_std  # 设置噪声标准差为信号标准差的 5%

    # 生成与原数据同维度的噪声
    noise = np.random.normal(0, noise_std, eeg_data.shape)

    # 向原始 EEG 数据添加噪声
    noisy_eeg_data = eeg_data + noise

    return  noisy_eeg_data


# split dataset
def get_data_hg(data_path, subject, dataset_id, is_ea, freq_band, ratio_noisy = 0):
    chans = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    data = get_data(subject, dataset_id, data_path, chans)

    X_noisy = add_noisy(data.X, ratio_noisy)

    X_processed = get_fb(X_noisy, freq_band, is_ea)
    return X_processed, data.y

