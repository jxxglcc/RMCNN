from utils.utils import *
import argparse
from cross_validation_baseline import CrossValidation
from hold_out_baseline_seed import HoldOut
# from visualization import visualization
from scipy.io import savemat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    #####bcic4-2a
    # parser.add_argument('--dataset', type=str, default='BCIC4_2a')
    # # parser.add_argument('--data_path', type=str, default='E:/Datasets/BCI-Ⅳ-2A/data/horizon/')
    # # parser.add_argument('--data_path', type=str, default='/home/xglcc/BCICIV_2a/data/horizon/')
    # parser.add_argument('--data_path', type=str, default='E:/Datasets/BCI-Ⅳ-2A/code/FBCNet-master/data/bci42a/rawMat/')
    # # parser.add_argument('--data_path', type=str, default='/home/xglcc/BCICIV_2a/data/rawMat/')
    # parser.add_argument('--subjects', type=int, default=9)
    # parser.add_argument('--num_class', type=int, default=4)
    # parser.add_argument('--sampling_rate', type=int, default=250)
    # parser.add_argument('--signal_length', type=int, default=1000)
    # parser.add_argument('--downsample_rate', type=int, default=2)
    # parser.add_argument('--in_chan', type=int, default=22)
    # parser.add_argument('--blocks', type=int, default=3)
    # parser.add_argument('--max_epoch', type=int, default=500)      # 500
    # parser.add_argument('--max_epoch_cmb', type=int, default=500)  # 500
    # parser.add_argument('--min_epoch', type=int, default=300)        # 300
    # parser.add_argument('--min_epoch_cmb', type=int, default=300)    # 300
    ####open-bmi
    # parser.add_argument('--dataset', type=str, default='Open_BMI')
    # parser.add_argument('--data_path', type=str, default='E:/Datasets/OpenBMI/data/raw')
    # # parser.add_argument('--data_path', type=str, default='/home/xglcc/OpenBMI/data/originalData/')
    # parser.add_argument('--subjects', type=int, default=54)
    # parser.add_argument('--num_class', type=int, default=2)
    # parser.add_argument('--sampling_rate', type=int, default=1000)
    # parser.add_argument('--signal_length', type=int, default=4000)
    # parser.add_argument('--downsample_rate', type=int, default=8)
    # parser.add_argument('--in_chan', type=int, default=21)
    # # parser.add_argument('--in_chan', type=int, default=62)
    # parser.add_argument('--blocks', type=int, default=3)
    # parser.add_argument('--max_epoch', type=int, default=300)
    # parser.add_argument('--max_epoch_cmb', type=int, default=300)
    # parser.add_argument('--min_epoch', type=int, default=100)
    # parser.add_argument('--min_epoch_cmb', type=int, default=100)
    #### HG
    # parser.add_argument('--dataset', type=str, default='HG')
    # parser.add_argument('--data_path', type=str, default='E:/Datasets/High-Gamma/data')
    # # parser.add_argument('--data_path', type=str, default='/home/xglcc/High-Gamma/data')
    # parser.add_argument('--subjects', type=int, default=14)
    # parser.add_argument('--num_class', type=int, default=4)
    # parser.add_argument('--sampling_rate', type=int, default=500)
    # parser.add_argument('--signal_length', type=int, default=2000)
    # parser.add_argument('--downsample_rate', type=int, default=2)
    # parser.add_argument('--in_chan', type=int, default=22)
    # parser.add_argument('--blocks', type=int, default=3)
    # parser.add_argument('--max_epoch', type=int, default=500)      # 500
    # parser.add_argument('--max_epoch_cmb', type=int, default=500)  # 500
    # parser.add_argument('--min_epoch', type=int, default=300)        # 300
    # parser.add_argument('--min_epoch_cmb', type=int, default=300)    # 300
    #####stroke eeg
    # parser.add_argument('--dataset', type=str, default='Stroke EEG')
    # # parser.add_argument('--data_path', type=str, default='E:/Datasets/stroke_eeg/data/sourcedata/')
    # parser.add_argument('--data_path', type=str, default='/home/xglcc/stroke_eeg/sourcedata/')
    # parser.add_argument('--subjects', type=int, default=50)
    # parser.add_argument('--num_class', type=int, default=2)
    # parser.add_argument('--sampling_rate', type=int, default=500)
    # parser.add_argument('--downsample_rate', type=int, default=4)
    # parser.add_argument('--in_chan', type=int, default=29)
    # parser.add_argument('--blocks', type=int, default=4)
    #####SEED
    parser.add_argument('--dataset', type=str, default='SEED')
    # parser.add_argument('--data_path', type=str, default='E:/Datasets/SEED/data/SEED_PR_PL/')
    parser.add_argument('--data_path', type=str, default='/home/xglcc/SEED/data/SEED_PR_PL/')
    parser.add_argument('--subjects', type=int, default=15)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--sampling_rate', type=int, default=200)
    parser.add_argument('--signal_length', type=int, default=800)
    parser.add_argument('--downsample_rate', type=int, default=1)
    # parser.add_argument('--in_chan', type=int, default=22)
    parser.add_argument('--in_chan', type=int, default=62)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--max_epoch_cmb', type=int, default=500)
    parser.add_argument('--min_epoch', type=int, default=200)
    parser.add_argument('--min_epoch_cmb', type=int, default=200)
    #####cross session
    # parser.add_argument('--dataset', type=str, default='CSV')
    # # parser.add_argument('--data_path', type=str, default='E:/Datasets/cross-session variability/data/')
    # parser.add_argument('--data_path', type=str, default='/home/xglcc/cross-session variability/data/')
    # parser.add_argument('--subjects', type=int, default=25)
    # parser.add_argument('--num_class', type=int, default=2)
    # parser.add_argument('--sampling_rate', type=int, default=250)
    # parser.add_argument('--downsample_rate', type=int, default=1)
    # parser.add_argument('--in_chan', type=int, default=32)
    # parser.add_argument('--blocks', type=int, default=8)

    ######## Training Process ########
    parser.add_argument('--test_pattern', type=str, default='holdout')
    # parser.add_argument('--test_pattern', type=str, default='cross_validation')
    parser.add_argument('--cv_dataset_id', type=int, default=1)
    parser.add_argument('--reproduce', type=bool, default=False)
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=58)
    # parser.add_argument('--weight_decay', type=float, default=1e-5)   #1e-2
    # parser.add_argument('--freq_band', type=list, default=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
    # parser.add_argument('--freq_band', type=list, default=[4, 8, 12, 40])
    parser.add_argument('--freq_band', type=list, default=[4, 40])
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--CUDA', type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--is_aug', type=bool, default=False)

    parser.add_argument('--patience', type=int, default=50)
    # parser.add_argument('--patience_cmb', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    # parser.add_argument('--learning_rate_cmb', type=float, default=1e-5)
    parser.add_argument('--patience_scheduler', type=int, default=30)
    parser.add_argument('--scheduler_eps', type=float, default=1e-7)
    ## inter-tripletloss, intra-centerloss
    parser.add_argument('--enable_inter', type=bool, default=False)
    parser.add_argument('--enable_intra', type=bool, default=False)
    parser.add_argument('--coefficient_inter', type=float, default=1)
    parser.add_argument('--coefficient_intra', type=float, default=5e-4)
    parser.add_argument('--coefficient_clf', type=float, default=1)
    parser.add_argument('--tripletloss_margin', type=float, default=0.001)
    ######## Model Parameters ########
    parser.add_argument('--notes', type=str, default='hold_out_basline')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--st_chan', type=int, default=30)
    parser.add_argument('--r_chan', type=int, default=30)
    ######## Model Save ########
    parser.add_argument('--save_path', default='./test/')
    parser.add_argument('--save_param', default='')
    parser.add_argument('--save_path_random', default='./save/output')
    parser.add_argument('--save_path_model', default='./save/output/model')
    parser.add_argument('--save_path_model_param', default='./save/output/model/param')
    parser.add_argument('--save_path_log', default='./save/output/log')
    parser.add_argument('--save_path_iter', default='./save/output/iter')
    parser.add_argument('--save_path_tsne', default='./save/output/tsne')
    parser.add_argument('--save_path_cat', default='./save/output/cat')
    parser.add_argument('--save_path_acc', default='./save/output/result.txt')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_all(args.random_seed)

    # ### conformer
    # args.model='conformer'
    # args.is_aug = True
    # args.learning_rate = 1e-3
    # args.num_depth = 6
    # args.num_head = 10
    # args.max_epoch=500
    # args.max_epoch_cmb=500
    # args.min_epoch=300
    # args.min_epoch_cmb=200
    # args.patience=50
    # # args.enable_inter = True
    # args.scheduler_eps = 1e-3


    #### mAtt
    # args.model='MAtt'
    # args.CUDA = False
    # args.stchan = 22
    # args.rchan = 18
    # args.scheduler_eps = 1e-3
    
    ### FBMSNet
    # args.model='FBMSNet'
    # args.freq_band=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    # args.split_ratio = 0.2
    # args.max_epoch=1500
    # args.max_epoch_cmb=600
    # args.min_epoch=0
    # args.min_epoch_cmb=0
    # args.patience=200
    # args.scheduler_eps = 1e-3
    # enable_intra = True

    # #### FBCNet
    # args.model='FBCNet'
    # args.freq_band=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    # args.split_ratio = 0.2
    # args.max_epoch=1500
    # args.max_epoch_cmb=600
    # args.min_epoch=0
    # args.min_epoch_cmb=0
    # args.patience=200
    # args.scheduler_eps = 1e-3

    ### TSFNet4a
    args.model='TSFCNet4a'
    args.learning_rate = 0.001
    args.split_ratio = 0.1
    args.batch_size = 32
    args.freq_band=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    args.max_epoch=1500
    args.max_epoch_cmb=600
    args.min_epoch=0
    args.min_epoch_cmb=0
    args.patience=200
    args.scheduler_eps = 0.001
    args.enable_intra = True
    args.coefficient_intra = 0.001

    # args.max_epoch=50







    if args.test_pattern == 'cross_validation':
        args.save_path = os.path.join(args.save_path, args.dataset, args.model, args.test_pattern + '_dataset_id_' + str(args.cv_dataset_id))
    else:
        args.save_path = os.path.join(args.save_path, args.dataset, args.model, args.test_pattern)

    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime()))
    # randomFolder = '2024-09-30--16-58-59' # conformer
    # randomFolder = '2024-09-30--22-22-29' # FBMSNet
    # randomFolder = '2024-10-01--01-11-29' # TSFCNet4a 
    # randomFolder = '2024-10-01--08-16-11' # MAtt
    # randomFolder = '2024-10-13--23-32-11' # EA-MSCNN

    args.save_path_random = os.path.join(args.save_path, 'output-' + randomFolder + str(args.cv_dataset_id))
    args.save_path_acc = os.path.join(args.save_path_random, 'result.txt')
    args.save_path_model = os.path.join(args.save_path_random, 'model')
    args.save_path_log = os.path.join(args.save_path_random, 'log')
    args.save_path_iter = os.path.join(args.save_path_random, 'iter')
    args.save_path_tsne = os.path.join(args.save_path_random, 'tsne')
    args.save_path_cat = os.path.join(args.save_path_random, 'cat')

    # args.max_epoch = 2
    # args.max_epoch_cmb = 2
    # args.min_epoch = 1
    # args.min_epoch_cmb = 1

    sub_to_run = np.arange(args.subjects) + 1
    # sub_to_run = [2, 4, 6]
    if args.test_pattern == 'cross_validation':
        cv = CrossValidation(args)
        cv.run(subject=sub_to_run)
    else:
        ho = HoldOut(args)
        _, _, ttcm = ho.run(subject=sub_to_run)
        savemat(args.save_path_random + '/kappa_ho_' + args.model + '_' + args.dataset + '.mat', {'data':ttcm})

        # params_ho = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        # # params_ho = [0.001, 0.01, 0.05, 0.1, 0.2]
        # # params_ho = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        # # params_ho = [0.01, 0.05]
        # ttas = np.zeros((len(params_ho), args.subjects+2))
        # for i, param in enumerate(params_ho):

        #     randomFolder = str(time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime()))
        #     args.save_path_random = os.path.join(args.save_path, 'output-' + randomFolder)
        #     args.save_path_acc = os.path.join(args.save_path_random, 'result.txt')
        #     args.save_path_model = os.path.join(args.save_path_random, 'model')
        #     args.save_path_log = os.path.join(args.save_path_random, 'log')
        #     args.save_path_iter = os.path.join(args.save_path_random, 'iter')
        #     args.save_path_tsne = os.path.join(args.save_path_random, 'tsne')
        #     args.save_path_cat = os.path.join(args.save_path_random, 'cat')

        #     args.split_ratio = param
        #     ho = HoldOut(args)
        #     tta, mACC, ttcm = ho.run(subject=sub_to_run)
        #     ttas[i, 0] = param
        #     ttas[i, 1:-1] = tta
        #     ttas[i, -1] = mACC
        #     savemat(args.save_path_random + '/kappa_ho_' + args.model + '_' + args.dataset + '.mat', {'data':ttcm})
        # # savemat('E:/李长春/博士/成果/SPDL/data_ho_lamada1.mat', {'data_param':ttas}) 
        # savemat(args.save_path_random + '/data_ho_split_ratio.mat', {'data_param':ttas}) 
