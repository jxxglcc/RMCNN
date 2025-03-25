import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord
from bin.FBCSP import FBCSP
from bin.Classifier import Classifier
import bin.LoadData as LoadData
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix

class MLEngine:
    def __init__(self,data_path='',file_to_load='',subject_id=1, test_id=1, m_filters=2,window_details={}):
        self.data_path = data_path
        self.subject_id=subject_id
        self.file_to_load = file_to_load
        self.window_details = window_details
        self.m_filters = m_filters
        self.test_id = test_id 

    def experiment(self):

        
        '''for BCIC Dataset'''
        if 'BCI-Ⅳ-2A' in self.data_path:
            eeg_data_train =LoadData.Loadbcic(self.subject_id, 0, self.data_path)
            eeg_data_test =LoadData.Loadbcic(self.subject_id, 1, self.data_path)
        elif 'OpenBMI' in self.data_path:
            '''for KU dataset'''
            eeg_data_train =LoadData.Loadopenbmi(self.subject_id, 0, self.data_path)
            eeg_data_test =LoadData.Loadopenbmi(self.subject_id, 1, self.data_path)
        elif 'SEED' in self.data_path:
            eeg_data_train0 =LoadData.Loadseed(self.subject_id, self.test_id + 1, self.data_path)
            eeg_data_train1 =LoadData.Loadseed(self.subject_id, self.test_id + 2, self.data_path)
            train_data = np.concatenate((eeg_data_train0['x_data'], eeg_data_train1['x_data']),axis=0)
            train_label = np.concatenate((eeg_data_train0['y_labels'], eeg_data_train1['y_labels']),axis=0)
            eeg_data_train = {'x_data': train_data ,
                        'y_labels': train_label,
                        'fs': 200}

            eeg_data_test =LoadData.Loadseed(self.subject_id, self.test_id, self.data_path)
        elif 'High-Gamma' in self.data_path:
            eeg_data_train =LoadData.Loadhg(self.subject_id, 0, self.data_path)
            eeg_data_test =LoadData.Loadhg(self.subject_id, 1, self.data_path)



        fbank = FilterBank(eeg_data_train.get('fs'))
        fbank_coeff = fbank.get_filter_coeff()
        filtered_data_train = fbank.filter_data(eeg_data_train.get('x_data'),self.window_details)
        y_train_labels = eeg_data_train.get('y_labels')
        filtered_data_test = fbank.filter_data(eeg_data_test.get('x_data'),self.window_details)
        y_test_labels = eeg_data_test.get('y_labels')

        y_train, y_test = y_train_labels, y_test_labels
        x_train_fb, x_test_fb = filtered_data_train, filtered_data_test

        y_classes_unique = np.unique(y_train)
        n_classes = len(np.unique(y_train))

        fbcsp = FBCSP(self.m_filters)
        fbcsp.fit(x_train_fb,y_train)
        y_train_predicted = np.zeros((y_train.shape[0], n_classes), dtype=np.float)
        y_test_predicted = np.zeros((y_test.shape[0], n_classes), dtype=np.float)

        for j in range(n_classes):
            cls_of_interest = y_classes_unique[j]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            y_test_cls = np.asarray(select_class_labels(cls_of_interest, y_test))

            x_features_train = fbcsp.transform(x_train_fb,class_idx=cls_of_interest)
            x_features_test = fbcsp.transform(x_test_fb,class_idx=cls_of_interest)

            classifier_type = SVR(gamma='auto')
            classifier = Classifier(classifier_type)
            y_train_predicted[:,j] = classifier.fit(x_features_train,np.asarray(y_train_cls,dtype=np.float))
            y_test_predicted[:,j] = classifier.predict(x_features_test)


        y_train_predicted_multi = self.get_multi_class_regressed(y_train_predicted)
        y_test_predicted_multi = self.get_multi_class_regressed(y_test_predicted)

        tr_acc =np.sum(y_train_predicted_multi == y_train, dtype=np.float) / len(y_train)
        te_acc =np.sum(y_test_predicted_multi == y_test, dtype=np.float) / len(y_test)
        te_cm = confusion_matrix(y_test, y_test_predicted_multi)

        print('*'*10,'\n')
        print(f'Training Accuracy = {str(tr_acc)}\n')
        print(f'Testing Accuracy = {str(te_acc)}\n \n')
        print('*' * 10, '\n')
        return te_acc, te_cm

    def cross_validate_Ntimes_Kfold(self, y_labels, ifold=0):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        random_seed = ifold
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=random_seed)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_sequential_split(self, y_labels):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=False)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_half_split(self, y_labels):
        import math
        unique_classes = np.unique(y_labels)
        all_labels = np.arange(len(y_labels))
        train_idx =np.array([])
        test_idx = np.array([])
        for cls in unique_classes:
            cls_indx = all_labels[np.where(y_labels==cls)]
            if len(train_idx)==0:
                train_idx = cls_indx[:math.ceil(len(cls_indx)/2)]
                test_idx = cls_indx[math.ceil(len(cls_indx)/2):]
            else:
                train_idx=np.append(train_idx,cls_indx[:math.ceil(len(cls_indx)/2)])
                test_idx=np.append(test_idx,cls_indx[math.ceil(len(cls_indx)/2):])

        train_indices = {0:train_idx}
        test_indices = {0:test_idx}

        return train_indices, test_indices

    def split_xdata(self,eeg_data, train_idx, test_idx):
        x_train_fb=np.copy(eeg_data[:,train_idx,:,:])
        x_test_fb=np.copy(eeg_data[:,test_idx,:,:])
        return x_train_fb, x_test_fb

    def split_ydata(self,y_true, train_idx, test_idx):
        y_train = np.copy(y_true[train_idx])
        y_test = np.copy(y_true[test_idx])

        return y_train, y_test

    def get_multi_class_label(self,y_predicted, cls_interest=0):
        y_predict_multi = np.zeros((y_predicted.shape[0]))
        for i in range(y_predicted.shape[0]):
            y_lab = y_predicted[i, :]
            lab_pos = np.where(y_lab == cls_interest)[0]
            if len(lab_pos) == 1:
                y_predict_multi[i] = lab_pos
            elif len(lab_pos > 1):
                y_predict_multi[i] = lab_pos[0]
        return y_predict_multi

    def get_multi_class_regressed(self, y_predicted):
        y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
        return y_predict_multi


class FilterBank:
    def __init__(self,fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4,40,4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:,:,int((window_details.get('tmin'))*self.fs):int((window_details.get('tmax'))*self.fs)]
            filtered_data[i,:,:,:]=eeg_data_filtered

        return filtered_data

