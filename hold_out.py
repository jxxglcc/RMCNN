import torch
import numpy as np
from utils.train_model import *
from utils.utils import ensure_path, get_mean_cov_mat
from utils.GetBci2a import get_data_bcic4_2a
# from utils.GetOpenBMI import get_data_open_bmi
# from utils.GetHG import get_data_hg
from sklearn.model_selection import train_test_split
import argparse
import itertools

class HoldOut:
    def __init__(self, args):
        self.args = args
        self.params = {}
        ensure_path(self.args.save_path)
        ensure_path(self.args.save_path_model)
        self.txt_file = args.save_path_acc
        file = open(self.txt_file, 'a')
        file.write(self.args.save_path_random + '\n'
                    + 'dataset:' + self.args.dataset + ', chans: ' + str(self.args.in_chan)  + ', is_aug: ' + str(self.args.is_aug) + '\n'
                    + 'network: ' + self.args.model  + '(' + str(self.args.st_chan) + ', ' + str(self.args.r_chan) + ')'  + '\n'
                    + 'downsample_rate:' + str(self.args.downsample_rate) + ', blocks : ' + str(self.args.blocks) + ', learning_rate: ' + str(self.args.learning_rate) + '\n'
                    + 'max_epoch: ' + str(self.args.max_epoch) + ', min_epoch:' + str(self.args.min_epoch) + '\n'
                    + 'max_epoch_cmb: ' + str(self.args.max_epoch_cmb) + ', min_epoch_cmb:' + str(self.args.min_epoch_cmb) + '\n'
                    + 'split_ratio: ' + str(self.args.split_ratio) + ', patience: ' + str(self.args.patience)+ ', weight_decay: ' + str(self.args.weight_decay) + '\n' # + ', patience_cmb: ' + str(self.args.patience_cmb) + '\n'
                    + 'scheduler_patience : ' + str(self.args.patience_scheduler) + ', scheduler_eps : ' + str(self.args.scheduler_eps) + '\n'
                    + 'loss:CrossEntrophy: ' + str(self.args.coefficient_clf) + '\n'
                    + 'loss:TripletMarginLoss(margin=' + str(self.args.tripletloss_margin) +'): ' + str(self.args.coefficient_inter) + ', Enable: ' + str(self.args.enable_inter) + '\n'
                    + 'notes:' + self.args.notes + '\n'
        )
        file.close()
        if args.dataset=='BCIC4_2a':
            self.get_data = get_data_bcic4_2a
        # elif args.dataset=='Open_BMI':
        #     self.get_data = get_data_open_bmi
        # elif args.dataset=='HG':
        #     self.get_data = get_data_hg


    def update_hyparameters(self, params, is_best):
        self.args = vars(self.args)
        self.args.update(params)    
        self.args = argparse.Namespace(**self.args)
        if is_best:
            self.args.save_path_model_param = os.path.join(self.args.save_path_model, 'best_model')
            self.args.save_path_log_param = os.path.join(self.args.save_path_log, 'best_model')
            self.args.save_path_iter_param = os.path.join(self.args.save_path_iter, 'best_model')
        else:           
            save_path = 'blocks' + str(self.args.blocks)
            self.save_param = save_path
            self.args.save_path_model_param = os.path.join(self.args.save_path_model, save_path)
            self.args.save_path_log_param = os.path.join(self.args.save_path_log, save_path)
            self.args.save_path_iter_param = os.path.join(self.args.save_path_iter, save_path)

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
            data_train, label_train = self.get_data(self.args.data_path, sub, dataset_id=0, is_ea=1, freq_band=self.args.freq_band)
            data_test, label_test = self.get_data(self.args.data_path, sub, dataset_id=1, is_ea=1, freq_band=self.args.freq_band)
            # from scipy.io import savemat
            # savemat('E:/Datasets/BCI-Ⅳ-2A/data/braindecode_data/A01E_EA.mat', {'X':np.squeeze(data_test)})
            data_train_val_sim, data_test_sim, label_train_val_sim, label_test_sim = train_test_split(data_train, label_train, 
                                                                                            test_size=self.args.split_ratio, 
                                                                                            random_state=2023, 
                                                                                            shuffle=True,
                                                                                            stratify=label_train
                                                                                            )
            data_train_sim, data_val_sim, label_train_sim, label_val_sim = train_test_split(data_train_val_sim, label_train_val_sim, 
                                                                                            test_size=self.args.split_ratio, 
                                                                                            random_state=2023, 
                                                                                            shuffle=True,
                                                                                            stratify=label_train_val_sim
                                                                                            )   
            data_train, label_train = self.prepare_data(data_train, label_train)
            data_train_val_sim, label_train_val_sim = self.prepare_data(data_train_val_sim, label_train_val_sim, is_ea=1)
            data_train_sim, label_train_sim = self.prepare_data(data_train_sim, label_train_sim, is_ea=1)
            data_val_sim, label_val_sim = self.prepare_data(data_val_sim, label_val_sim, is_ea=1)
            data_test_sim, label_test_sim = self.prepare_data(data_test_sim, label_test_sim, is_ea=1)
            data_test, label_test = self.prepare_data(data_test, label_test)
            
            ## 记得改前面的save_path
            hyperparams = {
                "blocks": [8, 6, 4],
            }
            keys, values = zip(*hyperparams.items())
            permuts_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            if len(permuts_params) > 1 :
                max_test_sim_acc = 0.0

                for params in permuts_params:
                    self.update_hyparameters(params, False)
                    if not self.args.reproduce:        
                    # to train new models
                        acc_val, target_loss = train(args=self.args,
                                            data_train=data_train_sim,
                                            label_train=label_train_sim,
                                            data_val=data_val_sim,
                                            label_val=label_val_sim,
                                            subject=sub
                                            )

                        combine_train(args=self.args, 
                                    data_train=data_train_val_sim, label_train=label_train_val_sim, 
                                    data_val=data_val_sim, label_val=label_val_sim, 
                                    subject=sub, target_loss=target_loss)

                    pred, act = test(args=self.args, data=data_test_sim, label=label_test_sim, subject=sub)
                    acc, _ = get_metrics(y_pred=pred, y_true=act)

                    result = ('    ' + self.save_param).ljust(100) + str(acc)
                    self.log2txt(result)

                    if acc >= max_test_sim_acc:
                        best_params = params
                        max_test_sim_acc = acc
                        self.save_best_param = self.save_param
                self.params[sub] = best_params
                self.update_hyparameters(best_params, True)
            else:
                self.update_hyparameters(permuts_params[0], False)

            if not self.args.reproduce:
                acc_val, target_loss = train(args=self.args,
                                    data_train=data_train_val_sim,
                                    label_train=label_train_val_sim,
                                    data_val=data_test_sim,
                                    label_val=label_test_sim,
                                    subject=sub
                                    )

                combine_train(args=self.args, 
                            data_train=data_train, label_train=label_train, 
                            data_val=data_test_sim, label_val=label_test_sim, 
                            subject=sub, target_loss=target_loss)

            pred, act = test(args=self.args, data=data_test, label=label_test, subject=sub)
            acc, cm = get_metrics(y_pred=pred, y_true=act)

            # tva.append(acc_val.item())
            tta.append(acc)
            ttcm.append(cm)
            if len(permuts_params) > 1 :
                result = '  best_acc:'.ljust(100) + str(acc)
                self.log2txt(result)
                params_choosed = '  best_params:' + self.save_best_param
                self.log2txt(params_choosed)
            else:
                result = '  acc:'.ljust(100) + str(acc)
                self.log2txt(result)
                params_choosed = '  params:' + self.save_param
                self.log2txt(params_choosed)

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
        self.log2txt(self.params)
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