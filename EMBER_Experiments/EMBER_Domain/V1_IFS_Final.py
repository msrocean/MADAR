import numpy as np
import random
import datetime
import time
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
import pandas as pd
import seaborn as sns

import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader

from ember_utils import *
from ember_model import *
from ember_pjr_utils import *




def get_IFBased_samples_V3(family_name, family_data,\
                        contamination,\
                        num_samples_per_malware_family):
    data_X = np.array(family_data)
    
    if len(data_X) > num_samples_per_malware_family:
        
        # fit the model
        clf = IsolationForest(max_samples=len(data_X), contamination=contamination)
        clf.fit(data_X)
        #scores_prediction = clf.decision_function(data_X)
        y_pred = clf.predict(data_X)


        anomalous_idx = np.where(y_pred == -1.0)
        similar_idx = np.where(y_pred == 1.0)

        #print(f'{family_name}: all-{len(y_pred)} anomalous-{len(anomalous_idx[0])} similar-{len(similar_idx[0])}')
        assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)

        anomalous_samples = data_X[anomalous_idx]
        
#         anomaly_proportion = (len(similar_idx) / len(y_pred))
#         similar_proportion = (len(anomalous_idx) / len(y_pred))
#         anomalous_samples_pool = list(anomalous_samples)
#         anomalous_samples = random.sample(anomalous_samples_pool, int(num_samples_per_malware_family * anomaly_proportion))
#         similar_samples_pool = list(data_X[similar_idx])
#         similar_samples = random.sample(similar_samples_pool, int(num_samples_per_malware_family * similar_proportion))
                                          
        if len(anomalous_samples) >= num_samples_per_malware_family:
            anomalous_samples_pool = list(anomalous_samples)
            remaining_samples_to_pick = int(num_samples_per_malware_family/2)
            anomalous_samples = random.sample(anomalous_samples_pool, remaining_samples_to_pick)

        else:
            remaining_samples_to_pick = num_samples_per_malware_family - len(anomalous_samples)
            
        similar_samples_pool = list(data_X[similar_idx])
        
        #print(f'similar_samples_pool {len(similar_samples_pool)} len(similar_idx) {len(similar_idx)}')
        if remaining_samples_to_pick > len(similar_samples_pool):
            similar_samples = similar_samples_pool #data_X[similar_idx]
        else:
            #print(f'similar_samples_pool {len(similar_samples_pool)} remaining_samples_to_pick*3 {remaining_samples_to_pick*3}')
            similar_samples = random.sample(similar_samples_pool, remaining_samples_to_pick)

        replay_samples = np.concatenate((anomalous_samples, similar_samples))
    else:
        replay_samples = data_X
    
    return replay_samples



def get_IFBased_samples_V2(family_name, family_data,\
                        contamination,\
                        num_samples_per_malware_family):
    data_X = np.array(family_data)
    
    if len(data_X) > num_samples_per_malware_family:
        
        # fit the model
        clf = IsolationForest(max_samples=len(data_X), contamination=contamination)
        clf.fit(data_X)
        #scores_prediction = clf.decision_function(data_X)
        y_pred = clf.predict(data_X)


        anomalous_idx = np.where(y_pred == -1.0)
        similar_idx = np.where(y_pred == 1.0)

        #print(f'{family_name}: all-{len(y_pred)} anomalous-{len(anomalous_idx[0])} similar-{len(similar_idx[0])}')
        assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)

        anomalous_samples = data_X[anomalous_idx]
        remaining_samples_to_pick = num_samples_per_malware_family - len(anomalous_samples)
        if abs(remaining_samples_to_pick)/remaining_samples_to_pick == -1:
            remaining_samples_to_pick = min(int(len(anomalous_samples) * 0.50), abs(remaining_samples_to_pick))
            print(f'remaining_samples_to_pick {remaining_samples_to_pick}')
        
        if remaining_samples_to_pick >= len(similar_idx):
            similar_samples = data_X[similar_idx]
        else:
            similar_samples_pool = list(data_X[similar_idx])
            similar_samples = random.sample(similar_samples_pool, remaining_samples_to_pick)

        replay_samples = np.concatenate((anomalous_samples, similar_samples))
    else:
        replay_samples = data_X
    
    return replay_samples


def get_IFBased_samples_V1(family_name, family_data, num_samples_per_malware_family):
    data_X = np.array(family_data)
    
    if len(data_X) > 1:
        
        # fit the model
        clf = IsolationForest(max_samples=len(data_X))
        clf.fit(data_X)
        #scores_prediction = clf.decision_function(data_X)
        y_pred = clf.predict(data_X)


        anomalous_idx = np.where(y_pred == -1.0)
        similar_idx = np.where(y_pred == 1.0)

        #print(f'{family_name}: all-{len(y_pred)} anomalous-{len(anomalous_idx[0])} similar-{len(similar_idx[0])}')
        assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)

        anomalous_samples = data_X[anomalous_idx]
        remaining_samples_to_pick = num_samples_per_malware_family - len(anomalous_samples)

        if remaining_samples_to_pick >= len(similar_idx):
            similar_samples = data_X[similar_idx]
        else:
            similar_samples_pool = list(data_X[similar_idx])
            similar_samples = random.sample(similar_samples_pool, remaining_samples_to_pick)

        replay_samples = np.concatenate((anomalous_samples, similar_samples))
    else:
        replay_samples = data_X
    
    return replay_samples


def get_IFBased_samples_V4(family_name, family_data,\
                        contamination,\
                        num_samples_per_malware_family):
    data_X = np.array(family_data)
    
    if len(data_X) > num_samples_per_malware_family:
        
        # fit the model
        clf = IsolationForest(max_samples=len(data_X), contamination=contamination)
        clf.fit(data_X)
        #scores_prediction = clf.decision_function(data_X)
        y_pred = clf.predict(data_X)


        anomalous_idx = np.where(y_pred == -1.0)
        similar_idx = np.where(y_pred == 1.0)

        #print(f'{family_name}: all-{len(y_pred)} anomalous-{len(anomalous_idx[0])} similar-{len(similar_idx[0])}')
        assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)

        anomalous_samples_pool = list(data_X[anomalous_idx])
        similar_samples_pool = list(data_X[similar_idx])
        
        num_samples_to_pick = int(num_samples_per_malware_family/2)
        
        if len(anomalous_samples_pool) >= num_samples_to_pick:
            anomalous_samples = random.sample(anomalous_samples_pool, num_samples_to_pick)

        else:
            anomalous_samples = anomalous_samples_pool
            

        if len(anomalous_samples) == num_samples_to_pick:
            similar_samples = random.sample(similar_samples_pool, num_samples_to_pick)
        else:
            similar_samples = random.sample(similar_samples_pool, len(anomalous_samples))

        replay_samples = np.concatenate((anomalous_samples, similar_samples))
    else:
        replay_samples = data_X
    
    return replay_samples


def get_IFBased_samples(family_name, family_data,\
                        contamination,\
                        num_samples_per_malware_family):
    
    
    data_X = np.array(family_data)
    
    if len(data_X) >= num_samples_per_malware_family:
        
        # fit the model
        clf = IsolationForest(max_samples=len(data_X), contamination=contamination)
        clf.fit(data_X)
        #scores_prediction = clf.decision_function(data_X)
        y_pred = clf.predict(data_X)


        anomalous_idx = np.where(y_pred == -1.0)
        similar_idx = np.where(y_pred == 1.0)

        #print(f'{family_name}: all-{len(y_pred)} anomalous-{len(anomalous_idx[0])} similar-{len(similar_idx[0])}')
        assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)

        anomalous_samples_pool = list(data_X[anomalous_idx])
        similar_samples_pool = list(data_X[similar_idx])
        
        num_samples_to_pick = int(num_samples_per_malware_family/2)
        
        if len(anomalous_samples_pool) >= num_samples_to_pick:
            anomalous_samples = random.sample(anomalous_samples_pool, num_samples_to_pick)

        else:
            anomalous_samples = anomalous_samples_pool
            

        if len(anomalous_samples) == num_samples_to_pick:
            similar_samples = random.sample(similar_samples_pool, num_samples_to_pick)
        else:
            similar_samples = random.sample(similar_samples_pool, len(anomalous_samples))

        replay_samples = np.concatenate((anomalous_samples, similar_samples))
        
        #print(f'family w/ MORE than budget replay_samples {replay_samples.shape}')
        
    elif len(data_X) < 50:
        replay_samples = data_X
    else:
        
        # fit the model
        clf = IsolationForest(max_samples=len(data_X), contamination=contamination)
        clf.fit(data_X)
        
        
        #scores_prediction = clf.decision_function(data_X)
        y_pred = clf.predict(data_X)


        anomalous_idx = np.where(y_pred == -1.0)
        similar_idx = np.where(y_pred == 1.0)

        #print(f'{family_name}: all-{len(y_pred)} anomalous-{len(anomalous_idx[0])} similar-{len(similar_idx[0])}')
        assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)

        anomalous_samples_pool = list(data_X[anomalous_idx])
        similar_samples_pool = list(data_X[similar_idx])
        
        anomalous_samples = anomalous_samples_pool
        similar_samples = random.sample(similar_samples_pool, len(anomalous_samples))

        replay_samples = np.concatenate((anomalous_samples, similar_samples))
        
        #print(f'family LESS than budget replay_samples {replay_samples.shape}')
        
    return replay_samples

def get_replay_samples_IFBased(global_family_dict, contamination, num_samples_per_malware_family):
    
    
    tmp_family_dict = defaultdict(list)
    
    pre_malware_samples = []

    cnt = 0
    for k, v in global_family_dict.items():
        if k != 'goodware':
            cnt += 1
            selected_family_samples = get_IFBased_samples(k, v, contamination, num_samples_per_malware_family)
            tmp_family_dict[k] = list(selected_family_samples)
            
            #print(selected_family_samples)
            for sample in selected_family_samples:
                pre_malware_samples.append(sample)
                
    if len(global_family_dict['goodware']) < len(pre_malware_samples):
        pre_goodware_samples = random.sample(global_family_dict['goodware'], len(global_family_dict['goodware']))
    else:
        pre_goodware_samples = random.sample(global_family_dict['goodware'], len(pre_malware_samples))
    
    tmp_family_dict['goodware'] = list(pre_goodware_samples)
    
    samples_to_replay = np.concatenate((np.array(pre_goodware_samples), np.array(pre_malware_samples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_goodware_samples)), np.ones(len(pre_malware_samples))))


    #print(f'X_replay {samples_to_replay.shape} Y_replay {labels_to_replay.shape}')
    #print(f'Replay {len(pre_malware_samples)} malware samples of {len(global_family_dict.keys()) -1} families')
    #print(f'and Replay {len(pre_goodware_samples)} goodware samples')
    
    
    return samples_to_replay, labels_to_replay, tmp_family_dict


parser = argparse.ArgumentParser()
parser.add_argument('--num_exps', type=int, default=1, required=False, help='Number of Experiments to Run.')
parser.add_argument('--contamination', type=float, default=0.1, required=False)
parser.add_argument('--num_epoch', type=int, default=500, required=False)
parser.add_argument('--batch_size', type=int, default=2000, required=False)
parser.add_argument('--num_replay_sample', type=int, required=True)
args = parser.parse_args()


all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

data_dir = '../../month_based_processing_with_family_labels/'



patience = 5
replay_type = 'ifs'



num_exps = args.num_exps
num_epoch = args.num_epoch
batch_size = args.batch_size
num_samples_per_malware_family = args.num_replay_sample

exp_type = 'ifs'

contamination = args.contamination #0.1 #[0.2, 0.3, 0.4, 0.5]

exp_seeds = [random.randint(1, 99999) for i in range(num_exps)]


expSaveDir = '../IFS_Final_' + str(contamination) + '_'
resSaveDir = './IFS_Results'
expSaveFile = '/IFS_'  + str(contamination) + '_'

cnt =  1    
for exp in exp_seeds:
    start_time = time.time()
    use_cuda = True
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(exp)

    model = Ember_MLP_Net()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
       
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print(f'Model has {count_parameters(model)/1000000}m parameters')    
    criterion = nn.BCELoss()    

    
    
    stored_global_family_dict = defaultdict(list)
    
    standardization = StandardScaler()
    standard_scaler = None
    for task_month in range(len(all_task_months)):
                
        print(f'\n{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Round {cnt} ...')
        task_start = time.time()
        
        #task_month = task_month
        current_task = all_task_months[task_month]
        task_months = all_task_months[:task_month+1]
        print(f'Current Task {current_task} w/ {num_samples_per_malware_family} samples to Replay per Malware family.')


        model_save_dir = str(expSaveDir) + 'model_' +\
                    str(exp_type) + str(expSaveFile) +\
                    str(num_samples_per_malware_family) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)
        
        opt_save_path = str(expSaveDir) + 'optimizer_' +\
                    str(exp_type) + str(expSaveFile) +\
                    str(num_samples_per_malware_family) + '/' + str(current_task) + '/'
        create_parent_folder(opt_save_path)
        
        
        results_save_dir = str(resSaveDir) + str(expSaveFile) +\
                    str(num_samples_per_malware_family) + '/' 
        create_parent_folder(results_save_dir)

        
        X_train, Y_train, Y_train_family = get_family_labeled_month_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)
        
        
        # to debug
        #X_train, Y_train, Y_train_family = X_train[:500], Y_train [:500], Y_train_family[:500]
        #X_test, Y_test, Y_test_family = X_test[:50], Y_test[:50], Y_test_family[:50]
        

        if current_task == all_task_months[0]:
            num_replay_samples = 0
            stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,\
                                                                   current_task, stored_global_family_dict)
        else:
#             num_replay_samples = sum([len(v) for k, v in stored_global_family_dict.items()])
#             print(f'replay samples of task {num_replay_samples}')
            
            stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,\
                                                                   current_task, stored_global_family_dict)
            X_train, Y_train, stored_global_family_dict =\
                get_replay_samples_IFBased(stored_global_family_dict, contamination, num_samples_per_malware_family)
            
            num_replay_samples = len(Y_train)
            
        print()
        print(f'X_train {X_train.shape} Y_train {Y_train.shape}')
        print()
        
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)

        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)        
               
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Training ...')
        task_training_time, epoch_ran, training_loss, validation_loss  =\
                                training_early_stopping(model, model_save_dir, opt_save_path,\
                                X_train, Y_train, X_test, Y_test, patience,\
                                batch_size, device, optimizer, num_epoch,\
                                 criterion, replay_type, current_task, exp, earlystopping=True)
        
        
        
        model = Ember_MLP_Net()
        model = model.to(device)
        #load the best model for this task
        best_model_path = model_save_dir + os.listdir(model_save_dir)[0]
        print(f'loading best model {best_model_path}')
        model.load_state_dict(torch.load(best_model_path))
        
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
        best_optimizer = opt_save_path + os.listdir(opt_save_path)[0]
        print(f'loading best optimizer {best_optimizer}')
        optimizer.load_state_dict(torch.load(best_optimizer))
        
        
        
        acc, precision, recall, f1score = testing_aucscore(model, X_test, Y_test, batch_size, device)
        
        
        end_time = time.time()

        print(f'Elapsed time {(end_time - start_time)/60} mins.')    
        

        task_end = time.time()
        task_run_time = (task_end - task_start)/60
        
        num_training = Y_train.shape[0]
        results_f = open(os.path.join('./ICDM_Domain/' + 'ifs_V2_' + str(args.num_replay_sample) + '_results.txt'), 'a')
        result_string = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(current_task, acc, precision, recall, f1score, num_training, task_training_time)
        
        results_f.write(result_string)
        results_f.flush()
        results_f.close()
        
    
    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')
   
