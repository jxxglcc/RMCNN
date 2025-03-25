from bin.MLEngine_ho import MLEngine
import numpy as np
import os
from scipy.io import savemat

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def log2txt(file, content):
    """
    this function log the content to results.txt
    :param content: string, the content to log
    """
    file = open(file, 'a')
    file.write(str(content) + '\n')
    file.close()

if __name__ == "__main__":
    # dataset = 'BCIC4_2a'
    # num_sub = 9
    # dataset = 'Open_BMI'
    # num_sub = 54
    dataset = 'HG'
    num_sub = 14
    save_path = './save/'
    save_path = os.path.join(save_path, dataset)
    ensure_path(save_path)
    txt_file = os.path.join(save_path, 'result.txt')
    log2txt(txt_file, 'Holdout:')
    sub_to_run = np.arange(num_sub) + 1

    sub_total_acc = []
    ttcm = []
    for sub in sub_to_run:
        '''Example for loading Korea University Dataset'''
        # dataset_details = {
        #     'data_path': "./OpenBMI/data",
        #     'subject_id': sub,
        #     'm_filters': 2,
        #     'window_details':{'tmin':0.5,'tmax':4}
        # }

        '''Example for loading BCI Competition IV Dataset 2a'''
        # dataset_details={
        #     'data_path' : "./BCIC4-2a/data",
        #     # 'file_to_load': 'A01T.gdf',
        #     'subject_id':sub,
        #     'm_filters':2,
        #     'window_details':{'tmin':0,'tmax':4}
        # }


        '''Example for loading high gamma'''
        dataset_details={
            'data_path' : './High-Gamma/data',
            'subject_id':sub,
            'm_filters':2,
            'window_details':{'tmin':0,'tmax':4}
        }


        ML_experiment = MLEngine(**dataset_details)
        accuracy, cm = ML_experiment.experiment()
        
        log2txt(txt_file, 'sub{}: {}'.format(sub, accuracy))
        sub_total_acc.append(accuracy)
        ttcm.append(cm)
    sub_mean_acc = np.mean(sub_total_acc)
    ttcm = np.concatenate(ttcm, axis=1)
    savemat('./kappa_ho_FBCSP_'+ dataset +'.mat', {'data':ttcm})
    log2txt(txt_file, 'sub accuracy:         {}'.format(sub_total_acc))
    log2txt(txt_file, 'sub mean accuracy:    {}\n'.format(sub_mean_acc))
