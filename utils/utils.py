import os
import time
import numpy as np
import pprint
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from model.RMCNN import *
from model.mAtt import mAtt
from model.conformer import Conformer
from model.FBMSNet import FBMSNet
from model.FBCNet import FBCNet
from model.TSFCNet import TSFCNet4a
from scipy.linalg import sqrtm


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def get_model(args):
    if args.model == 'RMCNN':
        model = RMCNN(
            in_chan=args.in_chan,
            st_chan=args.st_chan,
            r_chan=args.r_chan,
            fb=args.freq_band,
            downsample_rate=args.downsample_rate,
            blocks=args.blocks,
            num_class=args.num_class
            )
    elif args.model == 'MAtt':
        model = mAtt(
            in_chan=args.in_chan,
            downsample_rate=args.downsample_rate,
            blocks=args.blocks,
            num_class=args.num_class
            )
    elif args.model == 'conformer':
        model = Conformer(
            in_chan=args.in_chan,
            downsample_rate=args.downsample_rate,
            depth=args.num_depth,
            num_heads=args.num_head,
            n_classes=args.num_class
            )
    elif args.model == 'FBMSNet':
        model = FBMSNet(
            nChan=args.in_chan,
            nTime=1000,
            dropoutP=0.5,
            nBands=len(args.freq_band) - 1,
            m=32,
            temporalLayer='LogVarLayer',
            nClass=args.num_class,
            doWeightNorm=True
            )
    elif args.model == 'FBCNet':
        model = FBCNet(
            nChan=args.in_chan,
            nTime=1000,
            dropoutP=0.5,
            nBands=len(args.freq_band) - 1,
            m=32,
            temporalLayer='LogVarLayer',
            nClass=args.num_class,
            doWeightNorm=True
            )
    elif args.model == 'TSFCNet4a':
        model = TSFCNet4a(
            nChan=args.in_chan,
            nTime=1000,
            dropoutP=0.5,
            nBands=len(args.freq_band) - 1,
            m=32,
            temporalLayer='LogVarLayer',
            nClass=args.num_class,
            doWeightNorm=True
            )
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # for name, parameter in model.named_parameters():
    #     print(name, ':', parameter.size())
    return model

class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)

def get_dataloader(data, label, batch_size):
    # load the data
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred, average='micro')
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err

def get_mean_cov_mat(data):
    n_tr_trial, _, n_channel, n_samples = data.shape
    cov_mat = np.zeros((n_channel, n_channel))
    for trial_idx in range(n_tr_trial):	
        data_tr = data[trial_idx, 0]
        cov_mat += 1/(n_samples-1)*np.dot(data_tr,np.transpose(data_tr))
            
    mean_cov_mat = cov_mat / n_tr_trial

    # ea_data = np.dot(np.linalg.inv(sqrtm(mean_cov_mat)), data).transpose(1, 2, 0, 3)
    return mean_cov_mat
