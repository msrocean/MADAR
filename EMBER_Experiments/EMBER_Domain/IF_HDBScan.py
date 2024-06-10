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
import hdbscan

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



def get_HDBScanBased_similar_samples(ifbased_similar_samples,
                                     min_cluster_size, min_samples,
                                     samples_remaining):
    
    
    
    clf_hdbscan = hdbscan.HDBSCAN(min_cluster_size, min_samples)
    clf_fit_hdbscan = clf_hdbscan.fit(ifbased_similar_samples)
    unique_hdbscan_labels = np.unique(clf_fit_hdbscan.labels_)

    if len(unique_hdbscan_labels) == 1:
        
        if len(ifbased_similar_samples) <= samples_remaining:
            similar_malware_samples = ifbased_similar_samples
        else:
            similar_malware_samples = random.sample(list(ifbased_similar_samples), samples_remaining)
    else:
        
        similar_malware_samples = []
        exemplars = clf_fit_hdbscan.exemplars_
        #print(selected_family_samples)
        for cluster in exemplars:
            for sample in cluster:
                similar_malware_samples.append(sample)
        if len(similar_malware_samples) > samples_remaining:
            similar_malware_samples = random.sample(similar_malware_samples, samples_remaining)
        
    return similar_malware_samples

        

def get_IFBased_samples(family_name, family_data,
                        min_cluster_size, min_samples,
                        num_samples_per_malware_family):
    data_X = np.array(family_data)
    
    if len(data_X) > num_samples_per_malware_family:
        
        # fit the model
        clf = IsolationForest(max_samples=len(data_X))
        clf.fit(data_X)
        #scores_prediction = clf.decision_function(data_X)
        y_pred = clf.predict(data_X)


        anomalous_idx = np.where(y_pred == -1.0)
        similar_idx = np.where(y_pred == 1.0)

        #print(f'{family_name}: all-{len(y_pred)} 
        # anomalous-{len(anomalous_idx[0])} similar-{len(similar_idx[0])}')
        assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)
        
        split_num_sample = int(num_samples_per_malware_family/2)
        
        if len(anomalous_idx[0]) <= split_num_sample:
            anomalous_samples = data_X[anomalous_idx]
        else:
            anomalous_samples = random.sample(list(data_X[anomalous_idx]), split_num_sample)
            
        
        samples_remaining = num_samples_per_malware_family - len(anomalous_samples)
        
        similar_samples = get_HDBScanBased_similar_samples(data_X[similar_idx],\
                               min_cluster_size, min_samples, split_num_sample)
        
        replay_samples = np.concatenate((anomalous_samples, similar_samples))
    else:
        replay_samples = data_X
    
    return replay_samples



def make_family_based_dict(X_train, Y_train, Y_train_family, task_month, global_family_dict):
    count = 0
    for x_ind, x_sample in enumerate(X_train):
        count += 1
        #print(x_ind, Y_train[x_ind])

        if Y_train[x_ind] == 0:
            global_family_dict["goodware"].append(x_sample)
        if Y_train[x_ind] == 1:
            if Y_train_family[x_ind] == '':
                global_family_dict["others_family"].append(x_sample)
            else:
                global_family_dict[Y_train_family[x_ind]].append(x_sample)

    print(f'Task {task_month} and #-of new samples stored {count}')
    
    return global_family_dict


def get_sample_count(dataDict):
    mal_count_ = []
    good_count = 0
    for k, v in dataDict.items():
        if k != 'goodware':
            mal_count_ += [len(v)]
        else:
            good_count += len(v)
            
    mal_count = np.sum(np.array(mal_count_))
    print(f'good {good_count} mal {mal_count}')
    
    return mal_count

def get_replay_samples_IF_HDBScan_Based(global_family_dict,
                                        min_cluster_size, min_samples,
                                        num_samples_per_malware_family):
    
    tmp_family_dict = defaultdict(list)
    pre_malware_samples = []
    num_exemplars = 0
    cnt = 0
    for k, v in global_family_dict.items():
        if k != 'goodware':
            cnt += 1
            selected_family_samples = get_IFBased_samples(k, v,
                                                          min_cluster_size, min_samples,
                                                          num_samples_per_malware_family)
            tmp_family_dict[k] = list(selected_family_samples)
            num_exemplars += len(selected_family_samples)
            #print(selected_family_samples)
            for sample in selected_family_samples:
                pre_malware_samples.append(sample)
                
    if len(global_family_dict['goodware']) < len(pre_malware_samples):
        pre_goodware_samples = random.sample(global_family_dict['goodware'], len(global_family_dict['goodware']))
    else:
        pre_goodware_samples = random.sample(global_family_dict['goodware'], len(pre_malware_samples))
    
    
    tmp_family_dict['goodware'] = list(pre_goodware_samples)
    print(f'pre_goodware_samples {len(pre_goodware_samples)}')
    
    samples_to_replay = np.concatenate((np.array(pre_goodware_samples), np.array(pre_malware_samples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_goodware_samples)), np.ones(len(pre_malware_samples))))


    validate_mal_count = get_sample_count(tmp_family_dict)
    
    print(validate_mal_count == len(pre_malware_samples))
    
    
    return samples_to_replay, labels_to_replay, tmp_family_dict, num_exemplars




all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

data_dir = '../../month_based_processing_with_family_labels/'



patience = 5
replay_type = 'IFHDBSCAN_tuned'

min_cluster_size, min_samples = 10, 5

num_exps = 2 #args.num_exps
#task_month = args.task_month
num_epoch = 500 #args.num_epoch
batch_size = 6000 #args.batch_size
#replay_portion = args.replay_portion
num_samples_per_malware_family = 500

exp_type = 'if_hdbscan_' #{'', 'last', 'random', 'ifbased'}

exp_seeds = [random.randint(1, 99999) for i in range(num_exps)]


allexps_acc = {}
allexps_rocauc = {}
allexps_training_time = {}
all_exps_best_epoch = {}

mistaken_stats = {}

cnt =  1    
for exp in exp_seeds:
    #try:
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


        model_save_dir = '../IFBased_pjr_saved_model_' +\
                    str(exp_type) + '/IFBased_PJR_replay_' +\
                    str(num_samples_per_malware_family) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)

        opt_save_path = '../IFBased_pjr_saved_optimizer_' +\
                    str(exp_type) + '/IFBased_PJR_replay_' +\
                    str(num_samples_per_malware_family) + '/' + str(current_task) + '/'
        create_parent_folder(opt_save_path)


        results_save_dir = './IFBased_saved_results_' +\
                    str(exp_type) + '/IFBased_PJR_replay_' +\
                    str(num_samples_per_malware_family) + '/' 
        create_parent_folder(results_save_dir)


        X_train, Y_train, Y_train_family = get_family_labeled_month_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)


        # to debug
        #X_train, Y_train, Y_train_family = X_train[:500], Y_train [:500], Y_train_family[:500]
        #X_test, Y_test, Y_test_family = X_test[:50], Y_test[:50], Y_test_family[:50]


        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)

        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)   



        if current_task == all_task_months[0]:
            num_replay_samples = 0
            stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,\
                                                                   current_task, stored_global_family_dict)

            _, _, stored_global_family_dict, num_exemplar_samples =\
                get_replay_samples_IF_HDBScan_Based(stored_global_family_dict,
                                                min_cluster_size, min_samples,
                                                num_samples_per_malware_family)

        else:

            stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,\
                                                                   current_task, stored_global_family_dict)
            X_train, Y_train, stored_global_family_dict, num_exemplar_samples =\
                get_replay_samples_IF_HDBScan_Based(stored_global_family_dict,
                                                    min_cluster_size, min_samples,
                                                    num_samples_per_malware_family)
            print(f'task {current_task} exemplars used {num_exemplar_samples}')
            
        print()
        print(f'X_train {X_train.shape} Y_train {Y_train.shape}')
        print()      


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

        acc, rocauc = testing_aucscore(model, X_test, Y_test, batch_size, device)


        task_end = time.time()

        print(f'Task Elapsed time {(task_end - task_start)/60} mins.')    


        results_f = open(os.path.join(results_save_dir + 'ifhd_10_5_exemplarsStrict_' + str(num_samples_per_malware_family) + '_results.txt'), 'a')
        result_string = '{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(current_task,epoch_ran, task_training_time, acc, rocauc, num_replay_samples)
        results_f.write(result_string)
        results_f.flush()
        results_f.close()


    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')
    #except: 
    #    exp_seeds += [random.randint(99999, 999999)]
    #    pass