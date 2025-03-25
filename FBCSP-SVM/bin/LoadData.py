import mne
import os
import glob
import numpy as np
import scipy.io as sio

from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from collections import OrderedDict
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import exponential_running_standardize

class LoadData:
    def __init__(self,eeg_file_path: str):
        self.eeg_file_path = eeg_file_path

    def load_raw_data_gdf(self,file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        import scipy.io as sio
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self,file_path_extension: str =''):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)

class LoadBCIC(LoadData):
    '''Subclass of LoadData for loading BCI Competition IV Dataset 2a'''
    def __init__(self, file_to_load,*args):
        self.stimcodes=('769','770','771','772')
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC,self).__init__(*args)

    def get_epochs(self, tmin=-4.5,tmax=5.0,baseline=None):
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data,
                  'y_labels':self.y_labels,
                  'fs':self.fs}
        return eeg_data

class LoadKU(LoadData):
    '''Subclass of LoadData for loading KU Dataset'''
    def __init__(self,subject_id,*args):
        self.subject_id=subject_id
        self.fs=1000
        super(LoadKU,self).__init__(*args)

    def get_epochs(self,sessions=[1, 2]):
        for i in sessions:
            file_to_load=f'session{str(i)}/s{str(self.subject_id)}/EEG_MI.mat'
            self.load_raw_data_mat(file_to_load)
            x_data = self.raw_eeg_subject['EEG_MI_train']['smt'][0, 0]
            x_data = np.transpose(x_data,axes=[1, 2, 0])
            labels = self.raw_eeg_subject['EEG_MI_train']['y_dec'][0, 0][0]
            y_labels = labels - np.min(labels)
            if hasattr(self,'x_data'):
                self.x_data=np.append(self.x_data,x_data,axis=0)
                self.y_labels=np.append(self.y_labels,y_labels)
            else:
                self.x_data = x_data
                self.y_labels = y_labels
        ch_names = self.raw_eeg_subject['EEG_MI_train']['chan'][0, 0][0]
        ch_names_list = [str(x[0]) for x in ch_names]
        eeg_data = {'x_data': self.x_data,
                    'y_labels': self.y_labels,
                    'fs': self.fs,
                    'ch_names':ch_names_list}

        return eeg_data

def Loadbcic(subject, dataset_id, PATH):
    '''	Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets

    Keyword arguments:
    subject -- number of subject in [1, .. ,9]
    training -- if True, load training data
                if False, load testing data
    
    Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
            class_return 	numpy matrix 	size = NO_valid_trial
    '''
    if 'horizon' in PATH:
        NO_channels = 22
        NO_tests = 6*48 	
        Window_Length = 7*250 

        class_return = np.zeros(NO_tests)
        data_return = np.zeros((NO_tests,NO_channels,Window_Length))

        NO_valid_trial = 0
        if dataset_id==0:
            a = sio.loadmat(PATH+'A0'+str(subject)+'T.mat')
        else:
            a = sio.loadmat(PATH+'A0'+str(subject)+'E.mat')
        a_data = a['data']
        for ii in range(0,a_data.size):
            a_data1 = a_data[0,ii]
            a_data2=[a_data1[0,0]]
            a_data3=a_data2[0]
            a_X 		= a_data3[0]
            a_trial 	= a_data3[1]
            a_y 		= a_data3[2]
            # a_fs 		= a_data3[3]
            # a_classes 	= a_data3[4]
            a_artifacts = a_data3[5]
            # a_gender 	= a_data3[6]
            # a_age 		= a_data3[7]
            for trial in range(0,a_trial.size):
                if(a_artifacts[trial]==0):
                    data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
                    class_return[NO_valid_trial] = int(a_y[trial] - 1)
                    NO_valid_trial +=1
            # for trial in range(0,a_trial.size):
            #     data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
            #     class_return[NO_valid_trial] = int(a_y[trial])
            #     NO_valid_trial +=1
        return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]
    else:
        if dataset_id==0:
            a = sio.loadmat(PATH+'s00'+str(subject)+'.mat')
        else:
            a = sio.loadmat(PATH+'se00'+str(subject)+'.mat')
        data_return = np.transpose(a['x'], (2, 0, 1))
        class_return = np.squeeze(a['y'])

    eeg_data={'x_data':data_return,
            'y_labels':class_return,
            'fs':250}
    return eeg_data

def Loadopenbmi(subject, session, data_path):
    PATH = os.path.join(data_path, 'session' + str(session + 1),'s'+str(subject), 'EEG_MI.mat')
    data = sio.loadmat(PATH)
    chans = [4,32,8,9,33,34,12,35,13,36,14,37,38,18,39,19,40,41,24,42,43]

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
    eeg_data = {'x_data': x,
                'y_labels': y,
                'fs': s,
                'ch_names':c}

    return eeg_data
    

def Loadseed(subject, dataset_id, PATH):
    folder_path = PATH + '/Preprocessed_EEG/session' + str(dataset_id % 3 + 1)
    label_path = PATH + 'label.mat'
    labels = sio.loadmat(label_path)["label"][0] + 1
    try:
        all_mat_file = os.walk(folder_path)
        skip_set = {'label.mat', 'readme.txt'}
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                if file_name not in skip_set and file_name.split('_')[0] == str(subject):
                    all_trials_dict = sio.loadmat(os.path.join(folder_path, file_name),
                                                   verify_compressed_data_integrity=False)
                    # experiment_name = file_name.split('.')[0]
                    # feature_vector_trial_dict = {}
                    # label_trial_dict = {}
                    feature_vector_list = []
                    label_list = []
                    for key in all_trials_dict.keys():
                        if 'eeg' not in key:
                            continue
                        # feature_vector_list = []
                        # label_list = []
                        cur_trial = all_trials_dict[key]  # 维度为 62 * N，每200个采样点截取一个样本，不足200时舍弃
                        length = len(cur_trial[0])
                        pos = 0
                        while pos + 800 <= length:
                            feature_vector_list.append(np.asarray(cur_trial[:, pos:pos + 800]))
                            raw_label = labels[int(key.split('_')[-1][3:]) - 1]  # 截取片段对应的 label，-1, 0, 1
                            label_list.append(raw_label)
                            pos += 800
                    #     trial = key.split('_')[1][3:]
                    #     feature_vector_trial_dict[trial] = np.asarray(feature_vector_list)
                    #     label_trial_dict[trial] = np.asarray(label_2_onehot(label_list))

                    # feature_vector_dict[experiment_name] = feature_vector_trial_dict
                    # label_dict[experiment_name] = label_trial_dict
                else:
                    continue

    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    feature = np.array(feature_vector_list)
    label = np.array(label_list)
    eeg_data = {'x_data': feature,
            'y_labels': label,
            'fs': 200}

    return eeg_data

def Loadhg(subject, dataset_id, data_path):
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
    C_sensors = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
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

    eeg_data = {'x_data': dataset.X ,
        'y_labels': dataset.y,
        'fs': 500}

    return eeg_data