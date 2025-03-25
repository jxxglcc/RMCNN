from utils.utils import *
import argparse
from hold_out import HoldOut
from scipy.io import savemat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    #####bcic4-2a
    # parser.add_argument('--dataset', type=str, default='BCIC4_2a')
    # parser.add_argument('--data_path', type=str, default='./BCIC4-2a/data')
    # parser.add_argument('--subjects', type=int, default=9)
    # parser.add_argument('--num_class', type=int, default=4)
    # parser.add_argument('--sampling_rate', type=int, default=250)
    # parser.add_argument('--signal_length', type=int, default=1000)
    # parser.add_argument('--downsample_rate', type=int, default=2)
    # parser.add_argument('--in_chan', type=int, default=22)
    # parser.add_argument('--blocks', type=int, default=None)
    # parser.add_argument('--max_epoch', type=int, default=500)      # 500
    # parser.add_argument('--max_epoch_cmb', type=int, default=500)  # 500
    # parser.add_argument('--min_epoch', type=int, default=300)        # 300
    # parser.add_argument('--min_epoch_cmb', type=int, default=300)    # 300
    ####open-bmi
    # parser.add_argument('--dataset', type=str, default='Open_BMI')
    # parser.add_argument('--data_path', type=str, default='./OpenBMI/data')
    # parser.add_argument('--subjects', type=int, default=54)
    # parser.add_argument('--num_class', type=int, default=2)
    # parser.add_argument('--sampling_rate', type=int, default=1000)
    # parser.add_argument('--signal_length', type=int, default=4000)
    # parser.add_argument('--downsample_rate', type=int, default=8)
    # parser.add_argument('--in_chan', type=int, default=21)
    # # parser.add_argument('--in_chan', type=int, default=62)
    # parser.add_argument('--blocks', type=int, default=None)
    # parser.add_argument('--max_epoch', type=int, default=300)
    # parser.add_argument('--max_epoch_cmb', type=int, default=300)
    # parser.add_argument('--min_epoch', type=int, default=100)
    # parser.add_argument('--min_epoch_cmb', type=int, default=100)
    ### HG
    parser.add_argument('--dataset', type=str, default='HG')
    parser.add_argument('--data_path', type=str, default='./High-Gamma/data')
    parser.add_argument('--subjects', type=int, default=14)
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--sampling_rate', type=int, default=500)
    parser.add_argument('--signal_length', type=int, default=2000)
    parser.add_argument('--downsample_rate', type=int, default=2)
    parser.add_argument('--in_chan', type=int, default=22)
    parser.add_argument('--blocks', type=int, default=None)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--max_epoch_cmb', type=int, default=500)
    parser.add_argument('--min_epoch', type=int, default=300)
    parser.add_argument('--min_epoch_cmb', type=int, default=300)


    ######## Training Process ########
    parser.add_argument('--test_pattern', type=str, default='holdout')
    parser.add_argument('--reproduce', type=bool, default=False)
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=58)
    parser.add_argument('--weight_decay', type=float, default=None)   #1e-2
    parser.add_argument('--freq_band', type=list, default=[4, 40])
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--CUDA', type=bool, default=False)
    parser.add_argument('--is_aug', type=bool, default=False)

    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--patience_scheduler', type=int, default=30)
    parser.add_argument('--scheduler_eps', type=float, default=1e-7)
    parser.add_argument('--enable_inter', type=bool, default=True, help='tripletloss')
    parser.add_argument('--coefficient_inter', type=float, default=1)
    parser.add_argument('--coefficient_clf', type=float, default=1)
    parser.add_argument('--tripletloss_margin', type=float, default=0.001)
    ######## Model Parameters ########
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--model', type=str, default='RMCNN')
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
    seed_all(args.random_seed)

    args.save_path = os.path.join(args.save_path, args.dataset, args.model, args.test_pattern)
    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime()))

    args.save_path_random = os.path.join(args.save_path, 'output-' + randomFolder)
    args.save_path_acc = os.path.join(args.save_path_random, 'result.txt')
    args.save_path_model = os.path.join(args.save_path_random, 'model')
    args.save_path_log = os.path.join(args.save_path_random, 'log')


    sub_to_run = np.arange(args.subjects) + 1
    ho = HoldOut(args)
    _, _, ttcm = ho.run(subject=sub_to_run)
    savemat(args.save_path_random + '/kappa_ho_' + args.model+ '_' + args.dataset + '.mat', {'data':ttcm})

    




