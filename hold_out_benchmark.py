import torch
import numpy as np
from utils.train_model_benchmark import *
from utils.utils import ensure_path, get_mean_cov_mat
from utils.GetBci2a import get_data_bcic4_2a
from utils.GetOpenBMI import get_data_open_bmi
from utils.GetHG import get_data_hg
from sklearn.model_selection import train_test_split


class HoldOut:
    def __init__(self, args):
        self.args = args
        ensure_path(self.args.save_path)
        ensure_path(self.args.save_path_model)
        self.txt_file = args.save_path_acc
        file = open(self.txt_file, 'a')
        file.write(self.args.save_path_random + '\n'
                    + 'dataset:' + self.args.dataset + ', chans: ' + str(self.args.in_chan)  + ', is_aug: ' + str(self.args.is_aug)  + '\n'
                    + 'network: ' + self.args.model  + '\n'
                    + 'downsample_rate:' + str(self.args.downsample_rate) + ', learning_rate: ' + str(self.args.learning_rate) + '\n'
                    + 'max_epoch: ' + str(self.args.max_epoch) + ', min_epoch:' + str(self.args.min_epoch) + '\n'
                    + 'max_epoch_cmb: ' + str(self.args.max_epoch_cmb) + ', min_epoch_cmb:' + str(self.args.min_epoch_cmb) + '\n'
                    + 'split_ratio: ' + str(self.args.split_ratio) + ', patience: ' +  '\n'
                    + 'loss:CrossEntrophy: ' + str(self.args.coefficient_clf) + '\n'
                    + 'loss:CenterLoss: ' + str(self.args.coefficient_intra) + ', Enable: ' + str(self.args.enable_intra) +  '\n'
                    + 'notes:' +  '\n'
        )
        file.close()
        if args.dataset=='BCIC4_2a':
            self.get_data = get_data_bcic4_2a
        elif args.dataset=='Open_BMI':
            self.get_data = get_data_open_bmi
        elif args.dataset=='HG':
            self.get_data = get_data_hg


    def prepare_data(self, data, label, is_ea=0):
        """
        numpy.array-->torch.tensor
        :param data: (trials, 1, channel, data)
        :param label: (trials,)
        :return: data and label
        """
        if is_ea:
            if torch.is_tensor(data):
                data = data.numpy()
            if torch.is_tensor(label):
                label = label.numpy()
            mean_cov_mat = get_mean_cov_mat(data)
            data = np.dot(np.linalg.inv(sqrtm(mean_cov_mat)), data).transpose(1, 2, 0, 3)
        if not torch.is_tensor(data):
            data = torch.from_numpy(data).float()
        if not torch.is_tensor(label):
            label = torch.from_numpy(label).long()
        return data, label
    
    def run(self, subject=[0]):
        tta = []  # total test accuracy
        ttcm = []  # total validation accuracy

        for sub in subject:
            self.log2txt('sub{}:'.format(sub))
            data_train_raw, label_train_raw = self.get_data(self.args.data_path, sub, dataset_id=0, is_ea=0, freq_band=self.args.freq_band)
            data_test, label_test = self.get_data(self.args.data_path, sub, dataset_id=1, is_ea=0, freq_band=self.args.freq_band)
            data_train, data_val, label_train, label_val = train_test_split(data_train_raw, label_train_raw, 
                                                                            test_size=self.args.split_ratio, 
                                                                            random_state=2023, 
                                                                            shuffle=True,
                                                                            stratify=label_train_raw
                                                                            ) 
            data_train_raw, label_train_raw = self.prepare_data(data_train_raw, label_train_raw, is_ea=0)
            data_train, label_train = self.prepare_data(data_train, label_train, is_ea=0)
            data_val, label_val = self.prepare_data(data_val, label_val, is_ea=0)
            data_test, label_test = self.prepare_data(data_test, label_test, is_ea=0)

            if not self.args.reproduce:
                if self.args.model == 'conformer':
                    acc_val, target_loss = train(args=self.args,
                                                data_train=data_train_raw, label_train=label_train_raw, 
                                                data_val=data_val, label_val=label_val, 
                                                subject=sub
                                                )
                else:
                    acc_val, target_loss = train(args=self.args,
                                                data_train=data_train,label_train=label_train,
                                                data_val=data_val,label_val=label_val,
                                                subject=sub
                                                )
                    
                    combine_train(args=self.args, 
                                data_train=data_train_raw, label_train=label_train_raw, 
                                data_val=data_val, label_val=label_val, 
                                subject=sub, target_loss=target_loss)
            

            pred, act = test(args=self.args, data=data_test, label=label_test, subject=sub)
            acc, cm = get_metrics(y_pred=pred, y_true=act)

            # tva.append(acc_val.item())
            tta.append(acc)
            ttcm.append(cm)
            result = '  acc:'.ljust(100) + str(acc)
            self.log2txt(result)

        # prepare final report
        tta = np.array(tta)
        # tva = np.array(tva)
        mACC = np.mean(tta)
        std = np.std(tta)
        # mACC_val = np.mean(tva)
        # std_val = np.std(tva)
        

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        # print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        results = 'test array={}\n     mAcc={:.4f} + {:.4f}'.format(tta, mACC, std)
        self.log2txt(results)
        ttcm = np.concatenate(ttcm, axis=1)
        return tta, mACC, ttcm
    
    def log2txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        file = open(self.txt_file, 'a')
        file.write(str(content) + '\n')
        file.close()