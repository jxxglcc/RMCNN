'''
############################################################################################################################
Discription: 

The two data loader classes are for Korea University dataset and the BCIC-IV-2a dataset. Each loader will
serve Tensor-CSPNet and Graph-CSPNet on two scenairos, i.e., the cross-validation scenario and the holdout scenario. 

Keep in mind that the segmentation plan in this study is a simple example without a deep reason in neurophysiology, 
but achiving a relative good result. More reasonable segmentation plans may yield better performance. 

The class of FilterBank is from https://github.com/ravikiran-mane/FBCNet

#############################################################################################################################
'''

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.signal import cheb2ord
from scipy.linalg import eigvalsh
import torch as th
from pyriemann.estimation import Covariances
import os
from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from collections import OrderedDict
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import exponential_running_standardize
from scipy.signal import decimate


class FilterBank:
    def __init__(self, fs, pass_width=4, f_width=4):
        self.fs           = fs
        self.f_trans      = 2
        self.f_pass       = np.arange(4, 40, pass_width)
        self.f_width      = f_width
        self.gpass        = 3
        self.gstop        = 30
        self.filter_coeff = {}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass    = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop    = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp        = f_pass/Nyquist_freq
            ws        = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a      = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape

        if window_details:
            n_samples = int(self.fs * (window_details.get('tmax') - window_details.get('tmin')))
            #+1

        filtered_data = np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))

        for i, fb in self.filter_coeff.items():
          
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])

            if window_details:
                eeg_data_filtered  = eeg_data_filtered[:,:,int((window_details.get('tmin'))*self.fs):int((window_details.get('tmax'))*self.fs)]
            filtered_data[i,:,:,:] = eeg_data_filtered

        return filtered_data

def get_data_openbmi(PATH, chans=None):
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
    # import resampy
    # # down-sample if requested .
    # if downsampleFactor is not None:
    #     xNew = np.zeros((int(x.shape[0] / downsampleFactor), x.shape[1], x.shape[2]), np.float32)
    #     for i in range(x.shape[2]):  # resampy.resample cant handle the 3D data.
    #         xNew[:, :, i] = resampy.resample(x[:, :, i], s, s / downsampleFactor, axis=0)
    #     x = xNew
    #     s = s / downsampleFactor

    # change the data dimensions to be in a format: Chan x time x trials
    x = np.transpose(x, axes=(1, 2, 0))

    return {'x': x, 'y': y, 'c': c, 's':s}

def cross_validate_sequential_split(kfold, y_labels):
    from sklearn.model_selection import StratifiedKFold
    train_indices = {}
    test_indices = {}
    skf_model = StratifiedKFold(n_splits=kfold, shuffle=False)
    i = 0
    for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
        train_indices.update({i: train_idx})
        test_indices.update({i: test_idx})
        i += 1
    return train_indices, test_indices

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class load_KU:
    def __init__(self, subject, dataset_id=0, alg_name ='Graph_CSPNet', scenario = 'CV', data_path = '/dataset/'):

        # self.channel_index = [7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        self.channel_index = [4,32,8,9,33,34,12,35,13,36,14,37,38,18,39,19,40,41,24,42,43]
        self.alg_name = alg_name
        self.scenario = scenario


        if self.alg_name  == 'Tensor_CSPNet':
            #For Tensor-CSPNet
            self.freq_seg = 4
            self.time_seg =[[0, 1500], [500, 2000], [1000, 2500]]

        elif self.alg_name == 'Graph_CSPNet':
            #For Graph-CSPNet
            self.freq_seg  = 4
            self.time_freq_graph = {
                '1':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '2':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '3':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '4':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '5':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '6':[[0, 500], [500, 1000], [1000, 1500], [1500, 2000], [2000, 2500]],
                '7':[[0, 250], [250, 500], [500, 750], [750, 1000],[1000, 1250], [1250,1500],
                     [1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]],
                '8':[[0, 250], [250, 500], [500, 750], [750, 1000],[1000, 1250], [1250,1500],
                     [1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]],
                '9':[[0, 250], [250, 500], [500, 750], [750, 1000],[1000, 1250], [1250,1500],
                     [1500, 1750], [1750, 2000], [2000, 2250], [2250, 2500]]
            }
            self.block_dims = [
                          len(self.time_freq_graph['1']), 
                          len(self.time_freq_graph['2']), 
                          len(self.time_freq_graph['3']) + len(self.time_freq_graph['4']) + len(self.time_freq_graph['5']) + len(self.time_freq_graph['6']), 
                          len(self.time_freq_graph['7']) + len(self.time_freq_graph['8']) + len(self.time_freq_graph['9'])
                          ]
            self.time_windows = [5, 5, 5, 10]


        if scenario == 'CV':
            # self.x        = np.load(data_folder[0] + '_x.npy')[:, self.channel_index, :]
            # self.y_labels = np.load(data_folder[0] + '_y.npy')
            data = get_data_openbmi(os.path.join(data_path, 'session' + str(dataset_id + 1),'s'+str(subject), 'EEG_MI.mat'), self.channel_index)
            self.x = data['x']
            self.y_labels = data['y']

            fbank     = FilterBank(fs = 1000, pass_width = self.freq_seg)
            _         = fbank.get_filter_coeff()

            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            x_fb = fbank.filter_data(self.x, window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
            self.x_stack = self._tensor_stack(x_fb)
            
            self.train_indices, self.test_indices = cross_validate_sequential_split(10, data['y'])

        elif scenario == 'Holdout':
            data = get_data_openbmi(os.path.join(data_path, 'session1','s'+str(subject), 'EEG_MI.mat'), self.channel_index)
            self.x_1        = data['x']
            self.y_labels_1 = data['y']

            data = get_data_openbmi(os.path.join(data_path, 'session2','s'+str(subject), 'EEG_MI.mat'), self.channel_index)
            self.x_2        = data['x']
            self.y_labels_2 = data['y']


            fbank    = FilterBank(fs = 1000, pass_width = self.freq_seg)
            _        = fbank.get_filter_coeff()

            x_train_fb = fbank.filter_data(self.x_1,       window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
            x_valid_fb = fbank.filter_data(self.x_2[:100], window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)
            x_test_fb  = fbank.filter_data(self.x_2[100:], window_details={'tmin':0.0, 'tmax':2.5}).transpose(1, 0, 2, 3)

            self.x_train_stack = self._tensor_stack(x_train_fb)
            self.x_valid_stack = self._tensor_stack(x_valid_fb)
            self.x_test_stack  = self._tensor_stack(x_test_fb)


    def _tensor_stack(self, x_fb):

        if self.alg_name == 'Tensor_CSPNet':
            '''

            For Tensor-CSPNet:

            Step 1: Segment the signal on each given time intervals.
                    e.g., (trials, frequency bands, channels, timestamps) ---> 
                    (trials, temporal segments, frequency bands, channels, timestamps);
            Step 2: Take covariance.
                    e.g., (trials, temporal segments, frequency bands, channels, timestamps) --->
                    (trials, temporal segments, frequency bands, channels, channels).

            '''
            temporal_seg = []
            for [a, b] in self.time_seg:
                temporal_seg.append(np.expand_dims(x_fb[:, :, :, a:b], axis = 1))
            temporal_seg = np.concatenate(temporal_seg, axis = 1)


            stack_tensor  = []
            for i in range(temporal_seg.shape[0]):
                cov_stack = []
                for j in range(temporal_seg.shape[1]):
                    cov_stack.append(Covariances().transform(temporal_seg[i, j]))
                stack_tensor.append(np.stack(cov_stack, axis = 0))
            stack_tensor = np.stack(stack_tensor, axis = 0)

        elif self.alg_name == 'Graph_CSPNet':
            '''

            For Graph-CSPNet:
            Take covariance on each temporal intervals given in the time-frequency graph. 


            '''
            stack_tensor  = []
            for i in range(1, x_fb.shape[1]+1):
              for [a, b] in self.time_freq_graph[str(i)]:
                cov_record = []
                for j in range(x_fb.shape[0]):
                  cov_record.append(Covariances().transform(x_fb[j, i-1:i, :, a:b]))
                stack_tensor.append(np.expand_dims(np.concatenate(cov_record, axis = 0), axis = 1))
            stack_tensor = np.concatenate(stack_tensor, axis = 1)

        return stack_tensor

    def _riemann_distance(self, A, B):
        #AIRM 
        return np.sqrt((np.log(eigvalsh(A, B))**2).sum())

    def LGT_graph_matrix_fn(self, gamma = 50, time_step = [2, 2, 2, 5], freq_step = [1, 1, 4, 3]):
          '''

          time_step: a list, step of diffusion to right direction.
          freq_step: a list, step of diffusion to down direction.
          gamma: Gaussian coefficent.
          
          '''
          A = np.zeros((sum(self.block_dims), sum(self.block_dims))) + np.eye(sum(self.block_dims))
          start_point = 0
          for m in range(len(self.block_dims)):
            for i in range(self.block_dims[m]):
              max_time_step = min(self.time_windows[m] - 1 - (i % self.time_windows[m]), time_step[m])
              for j in range(i+1, i + max_time_step + 1):
                  A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                  A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
              for freq_mul in range(1, freq_step[m]+1):
                for j in range(i+ freq_mul*self.time_windows[m], i + freq_mul*self.time_windows[m] + max_time_step + 1):
                    if j < self.block_dims[m]: 
                        A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                        A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
            start_point += self.block_dims[m]

          D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

          return np.matmul(D, A), A

    def GGT_graph_matrix_fn(self, k = 12, gamma = 50):

          '''
          time_step: a list, step of diffusion to right direction.
          freq_step: a list, step of diffusion to down direction.
          gamma: Gaussian coefficent.
          '''

          A = np.zeros((sum(self.block_dims), sum(self.block_dims))) 

          for m in range(sum(self.block_dims)):
              row_record = []
              for n in range(sum(self.block_dims)):
                row_record.append(np.exp(-self._riemann_distance(self.lattice[m], self.lattice[n])**2/gamma))
              k_index = sorted(range(len(row_record)), key=lambda i: row_record[i])[-k:]
              for index in k_index:
                A[m, index] = row_record[index]

          A = (np.abs(A.T - A) + A.T + A)/2

          D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

          return np.matmul(D, A), A


    def generate_training_test_set_CV(self, kf_iter):

        train_idx = self.train_indices[kf_iter]
        test_idx  = self.test_indices[kf_iter]

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_stack[train_idx], axis = 0)

        return self.x_stack[train_idx], self.x_stack[test_idx], self.y_labels[train_idx], self.y_labels[test_idx]


    def generate_training_valid_test_set_Holdout(self):

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_train_stack, axis = 0)

        return self.x_train_stack, self.x_valid_stack, self.x_test_stack, self.y_labels_1, self.y_labels_2[:100], self.y_labels_2[100:]


class dataloader_in_main(th.utils.data.Dataset):

    def __init__(self, data_root, data_label):
        self.data  = data_root
        self.label = data_label
 
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
 
    def __len__(self):
        return len(self.data)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def get_data(subject,dataset_id,PATH):

    NO_channels   = 22
    NO_tests      = 6*48     
    Window_Length = 7*250 

    class_return = np.zeros(NO_tests)
    data_return  = np.zeros((NO_tests,NO_channels,Window_Length))

    NO_valid_trial = 0
    
    if dataset_id == 0:
      a = sio.loadmat(PATH+'A0'+str(subject)+'T.mat')
    else:
      a = sio.loadmat(PATH+'A0'+str(subject)+'E.mat')
    
    a_data = a['data']

    for ii in range(0,a_data.size):
        a_data1     = a_data[0,ii]
        a_data2     = [a_data1[0,0]]
        a_data3     = a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_fs        = a_data3[3]
        a_classes   = a_data3[4]
        a_artifacts = a_data3[5]
        a_gender    = a_data3[6]
        a_age       = a_data3[7]
        for trial in range(0,a_trial.size):
            data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
            class_return[NO_valid_trial]    = int(a_y[trial])
            NO_valid_trial                 += 1

    return data_return, class_return-1 

class load_BCIC:
    def __init__(self, sub, dataset_id = 0, alg_name ='Tensor_CSPNet', session_no = 1, scenario = 'CV', path='dataset/'):

        self.alg_name = alg_name
        self.scenario = scenario

        if self.alg_name == 'Tensor_CSPNet':
            self.freq_seg = 4
            self.time_seg =[[0, 625], [375, 1000]]

        elif self.alg_name == 'Graph_CSPNet':
            self.freq_seg  = 4
            # self.time_freq_graph = {
            # '1':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '2':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '3':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '4':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '5':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '6':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '7':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]],
            # '8':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]],
            # '9':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]]
            # }
            self.time_freq_graph = {
            '1':[[0, 500], [500, 1000]],
            '2':[[0, 500], [500, 1000]],
            '3':[[0, 500], [500, 1000]],
            '4':[[0, 500], [500, 1000]],
            '5':[[0, 500], [500, 1000]],
            '6':[[0, 500], [500, 1000]],
            '7':[[0, 250], [250, 500],[500,750],[750, 1000]],
            '8':[[0, 250], [250, 500],[500,750],[750, 1000]],
            '9':[[0, 250], [250, 500],[500,750],[750, 1000]]
            }
            self.block_dims = [
                          len(self.time_freq_graph['1']), 
                          len(self.time_freq_graph['2']), 
                          len(self.time_freq_graph['3']) + len(self.time_freq_graph['4']) + len(self.time_freq_graph['5']) + len(self.time_freq_graph['6']), 
                          len(self.time_freq_graph['7']) + len(self.time_freq_graph['8']) + len(self.time_freq_graph['9'])
                          ]
            # self.time_windows = [4, 4, 4, 8]
            self.time_windows = [2, 2, 2, 4]

        if scenario == 'CV':
            self.x, self.y_labels = get_data(sub, dataset_id, path)
            fbank     = FilterBank(fs = 250, pass_width = self.freq_seg)
            _         = fbank.get_filter_coeff()

            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            x_fb         = fbank.filter_data(self.x, window_details={'tmin':3.0, 'tmax':7.0}).transpose(1, 0, 2, 3)
            self.x_stack = self._tensor_stack(x_fb)
            
            self.train_indices, self.test_indices = cross_validate_sequential_split(10, self.y_labels)

        elif scenario == 'Holdout':
            ratio_noisy = 0.05
            self.x_1, self.y_labels_1 = get_data(sub, 0, path)
            self.x_2, self.y_labels_2 = get_data(sub, 1, path)

            fbank    = FilterBank(fs = 250, pass_width = self.freq_seg)
            _        = fbank.get_filter_coeff()

            x_train_fb = fbank.filter_data(self.x_1, window_details={'tmin':3.0, 'tmax':7.0}).transpose(1, 0, 2, 3)
            x_train_fb, y_train, x_valid_fb, y_valid = self._split_train_valid_set(x_train_fb, ratio=4)
            x_test_fb  = fbank.filter_data(self.x_2, window_details={'tmin':3.0, 'tmax':7.0}).transpose(1, 0, 2, 3)

            self.x_train_stack = self._tensor_stack(x_train_fb)
            self.y_train = y_train
            self.x_valid_stack = self._tensor_stack(x_valid_fb)
            self.y_valid = y_valid
            self.x_test_stack  = self._tensor_stack(x_test_fb)


    def _split_train_valid_set(self, x_train, ratio):
        s = self.y_labels_1.argsort()
        x_train = x_train[s]
        y_train = self.y_labels_1[s]

        cL = int(len(x_train) / 4)

        class1_x = x_train[ 0 * cL : 1 * cL ]
        class2_x = x_train[ 1 * cL : 2 * cL ]
        class3_x = x_train[ 2 * cL : 3 * cL ]
        class4_x = x_train[ 3 * cL : 4 * cL ]

        class1_y = y_train[ 0 * cL : 1 * cL ]
        class2_y = y_train[ 1 * cL : 2 * cL ]
        class3_y = y_train[ 2 * cL : 3 * cL ]
        class4_y = y_train[ 3 * cL : 4 * cL ]

        vL = int(len(class1_x) / ratio)

        x_train = np.concatenate((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]), axis=0)
        y_train = np.concatenate((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]), axis=0)

        x_valid = np.concatenate((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]), axis=0)
        y_valid = np.concatenate((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]), axis=0)

        return x_train, y_train, x_valid, y_valid

    def _tensor_stack(self, x_fb):

        if self.alg_name == 'Tensor_CSPNet':
            '''
            For Tensor-CSPNet:

            Step 1: Segment the signal on each given time intervals.
                    e.g., (trials, frequency bands, channels, timestamps) ---> 
                    (trials, temporal segments, frequency bands, channels, timestamps);
            Step 2: Take covariance.
                    e.g., (trials, temporal segments, frequency bands, channels, timestamps) --->
                    (trials, temporal segments, frequency bands, channels, channels).
            '''
            temporal_seg   = []
            for [a, b] in self.time_seg:
                temporal_seg.append(np.expand_dims(x_fb[:, :, :, a:b], axis = 1))
            temporal_seg   = np.concatenate(temporal_seg, axis = 1)

            stack_tensor   = []
            for i in range(temporal_seg.shape[0]):
                cov_stack  = []
                for j in range(temporal_seg.shape[1]):
                    cov_stack.append(Covariances().transform(temporal_seg[i, j]))
                stack_tensor.append(np.stack(cov_stack, axis = 0))
            stack_tensor   = np.stack(stack_tensor, axis = 0)

        elif self.alg_name == 'Graph_CSPNet':
            '''
            For Graph-CSPNet:
            Take covariance on each temporal intervals given in the time-frequency graph. 

            '''
            stack_tensor   = []
            for i in range(1, x_fb.shape[1]+1):
              for [a, b] in self.time_freq_graph[str(i)]:
                cov_record = []
                for j in range(x_fb.shape[0]):
                  cov_record.append(Covariances().transform(x_fb[j, i-1:i, :, a:b]))
                stack_tensor.append(np.expand_dims(np.concatenate(cov_record, axis = 0), axis = 1))
            stack_tensor   = np.concatenate(stack_tensor, axis = 1)

        return stack_tensor

    def _riemann_distance(self, A, B):
        #geodesic distance under metric AIRM. 
        return np.sqrt((np.log(eigvalsh(A, B))**2).sum())

    def LGT_graph_matrix_fn(self, gamma = 50, time_step = [2, 2, 2, 4], freq_step = [1, 1, 4, 3]):
          '''
          time_step: a list, step of diffusion to right direction.
          freq_step: a list, step of diffusion to down direction.
          gamma: Gaussian coefficent.
          '''
        
          A = np.zeros((sum(self.block_dims), sum(self.block_dims))) + np.eye(sum(self.block_dims))
          start_point = 0
          for m in range(len(self.block_dims)):
            for i in range(self.block_dims[m]):
              max_time_step = min(self.time_windows[m] - 1 - (i % self.time_windows[m]), time_step[m])
              for j in range(i + 1, i + max_time_step + 1):
                  A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                  A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
              for freq_mul in range(1, freq_step[m]+1):
                for j in range(i+ freq_mul*self.time_windows[m], i + freq_mul*self.time_windows[m] + max_time_step + 1):
                    if j < self.block_dims[m]: 
                        A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                        A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
            start_point += self.block_dims[m]

          D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

          return np.matmul(D, A), A

    def GGT_graph_matrix_fn(self, k = 12, gamma =50):

          '''
          time_step: a list, step of diffusion to right direction.
          freq_step: a list, step of diffusion to down direction.
          gamma: Gaussian coefficent.
          '''

          A = np.zeros((sum(self.block_dims), sum(self.block_dims)))
          
          for m in range(sum(self.block_dims)):
              row_record = []
              for n in range(sum(self.block_dims)):
                row_record.append(np.exp(-self._riemann_distance(self.lattice[m], self.lattice[n])**2/gamma))
              k_index = sorted(range(len(row_record)), key=lambda i: row_record[i])[-k:]
              for index in k_index:
                A[m, index] = row_record[index]

          A = (np.abs(A.T - A) + A.T + A)/2
          D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

          return np.matmul(D, A), A


    def generate_training_test_set_CV(self, kf_iter):

        train_idx = self.train_indices[kf_iter]
        test_idx  = self.test_indices[kf_iter]

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_stack[train_idx], axis = 0)

        return self.x_stack[train_idx], self.x_stack[test_idx], self.y_labels[train_idx], self.y_labels[test_idx]


    def generate_training_valid_test_set_Holdout(self):

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_train_stack, axis = 0)

        return self.x_train_stack, self.x_valid_stack,self.x_test_stack, self.y_train, self.y_valid, self.y_labels_2

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_data_HG(subject, dataset_id, data_path, chans=None):
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

    downsample_factor = 2  # 降采样倍数
    dataset.X = decimate(dataset.X, downsample_factor, axis=2, ftype='fir', zero_phase=True)
    return dataset.X, dataset.y


class load_HG:
    def __init__(self, sub, dataset_id = 0, alg_name ='Tensor_CSPNet', session_no = 1, scenario = 'CV', path='dataset/'):

        self.alg_name = alg_name
        self.scenario = scenario

        if self.alg_name == 'Tensor_CSPNet':
            self.freq_seg = 4
            self.time_seg =[[0, 625], [375, 1000]]

        elif self.alg_name == 'Graph_CSPNet':
            self.freq_seg  = 4
            # self.time_freq_graph = {
            # '1':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '2':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '3':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '4':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '5':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '6':[[0, 250], [250, 500], [500, 750], [750, 1000]],
            # '7':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]],
            # '8':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]],
            # '9':[[0, 125], [125, 250], [250, 375], [375, 500],[500, 625], [625,750],[750, 875], [875, 1000]]
            # }
            self.time_freq_graph = {
            '1':[[0, 500], [500, 1000]],
            '2':[[0, 500], [500, 1000]],
            '3':[[0, 500], [500, 1000]],
            '4':[[0, 500], [500, 1000]],
            '5':[[0, 500], [500, 1000]],
            '6':[[0, 500], [500, 1000]],
            '7':[[0, 250], [250, 500],[500,750],[750, 1000]],
            '8':[[0, 250], [250, 500],[500,750],[750, 1000]],
            '9':[[0, 250], [250, 500],[500,750],[750, 1000]]
            }
            self.block_dims = [
                          len(self.time_freq_graph['1']), 
                          len(self.time_freq_graph['2']), 
                          len(self.time_freq_graph['3']) + len(self.time_freq_graph['4']) + len(self.time_freq_graph['5']) + len(self.time_freq_graph['6']), 
                          len(self.time_freq_graph['7']) + len(self.time_freq_graph['8']) + len(self.time_freq_graph['9'])
                          ]
            # self.time_windows = [4, 4, 4, 8]
            self.time_windows = [2, 2, 2, 4]

        if scenario == 'CV':
            self.x, self.y_labels = get_data(sub, dataset_id, path)
            fbank     = FilterBank(fs = 250, pass_width = self.freq_seg)
            _         = fbank.get_filter_coeff()

            '''The shape of x_fb is No. of (trials, frequency bands, channels, timestamps)'''
            x_fb         = fbank.filter_data(self.x, window_details={'tmin':3.0, 'tmax':7.0}).transpose(1, 0, 2, 3)
            self.x_stack = self._tensor_stack(x_fb)
            
            self.train_indices, self.test_indices = cross_validate_sequential_split(10, self.y_labels)

        elif scenario == 'Holdout':
            chans = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

            self.x_1, self.y_labels_1 = get_data_HG(sub, 0, path, chans)
            self.x_2, self.y_labels_2 = get_data_HG(sub, 1, path, chans)

            fbank    = FilterBank(fs = 250, pass_width = self.freq_seg)
            _        = fbank.get_filter_coeff()

            x_train_fb = fbank.filter_data(self.x_1, window_details={'tmin':0, 'tmax':4}).transpose(1, 0, 2, 3)
            x_train_fb, y_train, x_valid_fb, y_valid = self._split_train_valid_set(x_train_fb, ratio=4)
            x_test_fb  = fbank.filter_data(self.x_2, window_details={'tmin':0, 'tmax':4}).transpose(1, 0, 2, 3)

            self.x_train_stack = self._tensor_stack(x_train_fb)
            self.y_train = y_train
            self.x_valid_stack = self._tensor_stack(x_valid_fb)
            self.y_valid = y_valid
            self.x_test_stack  = self._tensor_stack(x_test_fb)


    def _split_train_valid_set(self, x_train, ratio):
        s = self.y_labels_1.argsort()
        x_train = x_train[s]
        y_train = self.y_labels_1[s]

        cL = int(len(x_train) / 4)

        class1_x = x_train[ 0 * cL : 1 * cL ]
        class2_x = x_train[ 1 * cL : 2 * cL ]
        class3_x = x_train[ 2 * cL : 3 * cL ]
        class4_x = x_train[ 3 * cL : 4 * cL ]

        class1_y = y_train[ 0 * cL : 1 * cL ]
        class2_y = y_train[ 1 * cL : 2 * cL ]
        class3_y = y_train[ 2 * cL : 3 * cL ]
        class4_y = y_train[ 3 * cL : 4 * cL ]

        vL = int(len(class1_x) / ratio)

        x_train = np.concatenate((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]), axis=0)
        y_train = np.concatenate((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]), axis=0)

        x_valid = np.concatenate((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]), axis=0)
        y_valid = np.concatenate((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]), axis=0)

        return x_train, y_train, x_valid, y_valid

    def _tensor_stack(self, x_fb):

        if self.alg_name == 'Tensor_CSPNet':
            '''
            For Tensor-CSPNet:

            Step 1: Segment the signal on each given time intervals.
                    e.g., (trials, frequency bands, channels, timestamps) ---> 
                    (trials, temporal segments, frequency bands, channels, timestamps);
            Step 2: Take covariance.
                    e.g., (trials, temporal segments, frequency bands, channels, timestamps) --->
                    (trials, temporal segments, frequency bands, channels, channels).
            '''
            temporal_seg   = []
            for [a, b] in self.time_seg:
                temporal_seg.append(np.expand_dims(x_fb[:, :, :, a:b], axis = 1))
            temporal_seg   = np.concatenate(temporal_seg, axis = 1)

            stack_tensor   = []
            for i in range(temporal_seg.shape[0]):
                cov_stack  = []
                for j in range(temporal_seg.shape[1]):
                    cov_stack.append(Covariances().transform(temporal_seg[i, j]))
                stack_tensor.append(np.stack(cov_stack, axis = 0))
            stack_tensor   = np.stack(stack_tensor, axis = 0)

        elif self.alg_name == 'Graph_CSPNet':
            '''
            For Graph-CSPNet:
            Take covariance on each temporal intervals given in the time-frequency graph. 

            '''
            stack_tensor   = []
            for i in range(1, x_fb.shape[1]+1):
              for [a, b] in self.time_freq_graph[str(i)]:
                cov_record = []
                for j in range(x_fb.shape[0]):
                  cov_record.append(Covariances().transform(x_fb[j, i-1:i, :, a:b]))
                stack_tensor.append(np.expand_dims(np.concatenate(cov_record, axis = 0), axis = 1))
            stack_tensor   = np.concatenate(stack_tensor, axis = 1)

        return stack_tensor

    def _riemann_distance(self, A, B):
        #geodesic distance under metric AIRM. 
        return np.sqrt((np.log(eigvalsh(A, B))**2).sum())

    def LGT_graph_matrix_fn(self, gamma = 50, time_step = [2, 2, 2, 4], freq_step = [1, 1, 4, 3]):
          '''
          time_step: a list, step of diffusion to right direction.
          freq_step: a list, step of diffusion to down direction.
          gamma: Gaussian coefficent.
          '''
        
          A = np.zeros((sum(self.block_dims), sum(self.block_dims))) + np.eye(sum(self.block_dims))
          start_point = 0
          for m in range(len(self.block_dims)):
            for i in range(self.block_dims[m]):
              max_time_step = min(self.time_windows[m] - 1 - (i % self.time_windows[m]), time_step[m])
              for j in range(i + 1, i + max_time_step + 1):
                  A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                  A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
              for freq_mul in range(1, freq_step[m]+1):
                for j in range(i+ freq_mul*self.time_windows[m], i + freq_mul*self.time_windows[m] + max_time_step + 1):
                    if j < self.block_dims[m]: 
                        A[start_point + i, start_point + j] = np.exp(-self._riemann_distance(self.lattice[start_point + i], self.lattice[start_point + j])**2/gamma)
                        A[start_point + j, start_point + i] = A[start_point + i, start_point + j]
            start_point += self.block_dims[m]

          D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

          return np.matmul(D, A), A

    def GGT_graph_matrix_fn(self, k = 12, gamma =50):

          '''
          time_step: a list, step of diffusion to right direction.
          freq_step: a list, step of diffusion to down direction.
          gamma: Gaussian coefficent.
          '''

          A = np.zeros((sum(self.block_dims), sum(self.block_dims)))
          
          for m in range(sum(self.block_dims)):
              row_record = []
              for n in range(sum(self.block_dims)):
                row_record.append(np.exp(-self._riemann_distance(self.lattice[m], self.lattice[n])**2/gamma))
              k_index = sorted(range(len(row_record)), key=lambda i: row_record[i])[-k:]
              for index in k_index:
                A[m, index] = row_record[index]

          A = (np.abs(A.T - A) + A.T + A)/2
          D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis = 0))))

          return np.matmul(D, A), A


    def generate_training_test_set_CV(self, kf_iter):

        train_idx = self.train_indices[kf_iter]
        test_idx  = self.test_indices[kf_iter]

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_stack[train_idx], axis = 0)

        return self.x_stack[train_idx], self.x_stack[test_idx], self.y_labels[train_idx], self.y_labels[test_idx]


    def generate_training_valid_test_set_Holdout(self):

        if self.alg_name == 'Graph_CSPNet':
            self.lattice = np.mean(self.x_train_stack, axis = 0)

        return self.x_train_stack, self.x_valid_stack,self.x_test_stack, self.y_train, self.y_valid, self.y_labels_2
