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




def get_IFBased_samples(family_name, family_data, num_samples_per_malware_family):
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

        if remaining_samples_to_pick > len(similar_idx):
            similar_samples = data_X[similar_idx]
        else:
            similar_samples = random.sample(data_X[similar_idx], remaining_samples_to_pick)

        replay_samples = np.concatenate((anomalous_samples, similar_samples))
    else:
        replay_samples = data_X
    
    return replay_samples



def get_replay_samples_IFBased(global_family_dict, num_samples_per_malware_family):
    pre_malware_samples = []

    cnt = 0
    for k in global_family_dict.keys():
        if k != 'goodware':
            cnt += 1
            selected_family_samples = get_IFBased_samples(k, global_family_dict[k],                                                  num_samples_per_malware_family)

            #print(selected_family_samples)
            for sample in selected_family_samples:
                pre_malware_samples.append(sample)
                
    if len(global_family_dict['goodware']) < len(pre_malware_samples):
        pre_goodware_samples = random.sample(global_family_dict['goodware'], len(global_family_dict['goodware']))
    else:
        pre_goodware_samples = random.sample(global_family_dict['goodware'], len(pre_malware_samples))

    samples_to_replay = np.concatenate((np.array(pre_goodware_samples), np.array(pre_malware_samples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_goodware_samples)), np.ones(len(pre_malware_samples))))


    #print(f'X_replay {samples_to_replay.shape} Y_replay {labels_to_replay.shape}')
    #print(f'Replay {len(pre_malware_samples)} malware samples of {len(global_family_dict.keys()) -1} families')
    #print(f'and Replay {len(pre_goodware_samples)} goodware samples')
    
    
    return samples_to_replay, labels_to_replay




parser = argparse.ArgumentParser()
parser.add_argument('--num_exps', type=int, default=4, required=False, help='Number of Experiments to Run.')
parser.add_argument('--num_epoch', type=int, default=2000, required=False)
parser.add_argument('--batch_size', type=int, default=6000, required=False)
parser.add_argument('--num_replay_samples', type=float, default=500, required=False)

args = parser.parse_args()

all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

data_dir = '../../month_based_processing_with_family_labels/'



patience = 5
replay_type = 'partial_joint_replay'



num_exps = args.num_exps
#task_month = args.task_month
num_epoch = args.num_epoch
batch_size = args.batch_size
#replay_portion = args.replay_portion
num_samples_per_malware_family = args.num_replay_samples

exp_type = 'ifbased' #{'', 'last', 'random', 'ifbased'}

exp_seeds = [random.randint(1, 99999) for i in range(num_exps)]


allexps_acc = {}
allexps_rocauc = {}
allexps_training_time = {}
all_exps_best_epoch = {}

mistaken_stats = {}

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
        
        current_task = all_task_months[task_month]
        task_months = all_task_months[:task_month+1]
        print(f'Current Task {current_task} w/ {num_samples_per_malware_family} samples to Replay per Malware family.')


        model_save_dir = '../IFBased_pjr_saved_model_' +                    str(exp_type) + '/IFBased_PJR_replay_' +                    str(num_samples_per_malware_family) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)
        
        results_save_dir = './IFBased_saved_results_' +                    str(exp_type) + '/IFBased_PJR_replay_' +                    str(num_samples_per_malware_family) + '/' 
        create_parent_folder(results_save_dir)

        
        X_train, Y_train, Y_train_family = get_family_labeled_month_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)
        
        
        # to debug
        #X_train, Y_train, Y_train_family = X_train[:500], Y_train [:500], Y_train_family[:500]
        #X_test, Y_test, Y_test_family = X_test[:50], Y_test[:50], Y_test_family[:50]
        

        if current_task == all_task_months[0]:
            stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,                                                               current_task, stored_global_family_dict)
        else:
            if str(exp_type) == 'first':
                X_replay, Y_replay = get_replay_samples_first(stored_global_family_dict, num_samples_per_malware_family)
            elif str(exp_type) == 'last':
                X_replay, Y_replay = get_replay_samples_last(stored_global_family_dict, num_samples_per_malware_family)
            elif str(exp_type) == 'random':
                X_replay, Y_replay = get_replay_samples(stored_global_family_dict, num_samples_per_malware_family)
            else:
                X_replay, Y_replay = get_replay_samples_IFBased(stored_global_family_dict, num_samples_per_malware_family)
            
            
            stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,                                                               current_task, stored_global_family_dict)
        
        
        if current_task == all_task_months[0]:
            print(f'Initial Task {current_task} X_train {X_train.shape} Y_train {Y_train.shape}')
            print(f'************** ************** **************')
            print()
        else:
            print(f'W/O replay samples \n X_train {X_train.shape} Y_train {Y_train.shape}')
            X_train, Y_train = np.concatenate((X_train, X_replay)), np.concatenate((Y_train, Y_replay))
            print(f'With replay samples \n X_train {X_train.shape} Y_train {Y_train.shape}')
            print(f'************** ************** **************')
            print()
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)

        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)        
                
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Training ...')
        task_training_time, epoch_ran, training_loss, validation_loss  = training_early_stopping(model, model_save_dir,                                X_train, Y_train, X_test, Y_test, patience,                                batch_size, device, optimizer, num_epoch,                                 criterion, replay_type, current_task, exp, earlystopping=True)

        top_k = 10
        acc, rocauc, wrong_good, wrong_mal, top_k_mistaken_families =                                    testing_aucscore_with_mistaken_stats(model, X_test, Y_test,                                    Y_test_family, batch_size, top_k, device)
        
        mistaken_stats[str(current_task)] = (wrong_good, wrong_mal, top_k_mistaken_families)
        
        end_time = time.time()

        print(f'Elapsed time {(end_time - start_time)/60} mins.')    
        

        task_end = time.time()
        task_run_time = (task_end - task_start)/60
        
        
        try:
            allexps_acc[str(current_task)].append(acc)
            allexps_rocauc[str(current_task)].append(rocauc)
            allexps_training_time[str(current_task)].append(task_run_time)
            all_exps_best_epoch[str(current_task)].append(epoch_ran)
        except:
            allexps_acc[str(current_task)] = [acc]
            allexps_rocauc[str(current_task)] = [rocauc]
            allexps_training_time[str(current_task)] = [task_run_time]
            all_exps_best_epoch[str(current_task)] = [epoch_ran]
        
        

        results_f = open(os.path.join(results_save_dir + 'results_accumulated_replay_' + str(num_samples_per_malware_family) + '_results.txt'), 'a')
        result_string = '{}\t{}\t{}\t{}\t{}\t\n'.format(current_task,epoch_ran, task_training_time, acc, rocauc)
        results_f.write(result_string)
        results_f.flush()
        results_f.close()
        

        
        wf = open(os.path.join(results_save_dir + 'Results_' + str(current_task) + '_' + str(num_epoch) + '_replay_' + str(num_samples_per_malware_family) + '_results.txt'), 'a')
        task_exp_string = '\n\nSeed\t{}\t\tRun time\t{}\tAcc:\t{}\t\tROC_AUC:\t{}\n\tepoch_ran\t{}\t\n\ntraining_loss\t{}\n\nValid_loss\t{}\n\n'.format(exp,task_training_time, acc, rocauc, epoch_ran, training_loss, validation_loss)
        
        wf.write('\n ########################### ########################### ###########################\n')
        wf.write(str(model))
        wf.write(task_exp_string)
        
        wf.flush()
        wf.close()
    
    mistakes_f = open(os.path.join(results_save_dir + 'mistaken_stats_replay_' + str(num_samples_per_malware_family) + '.txt'), 'a')
    mis_string = '{}\n'.format(mistaken_stats)
    mistakes_f.write(mis_string)
    mistakes_f.flush()
    mistakes_f.close()
        
    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')

#print(f'all results saved')



