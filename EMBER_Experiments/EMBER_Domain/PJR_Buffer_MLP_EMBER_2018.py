import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time, random
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import datetime
from collections import defaultdict
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


parser = argparse.ArgumentParser()
parser.add_argument('--num_exps', type=int, default=5, required=False, help='Number of Experiments to Run.')
parser.add_argument('--num_epoch', type=int, default=500, required=False)
parser.add_argument('--batch_size', type=int, default=6000, required=False)
parser.add_argument('--num_replay_sample', type=int, default=1, required=True)

args = parser.parse_args()


all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

data_dir = '../../month_based_processing_with_family_labels/'


num_exps = args.num_exps
num_epoch = args.num_epoch
batch_size = args.batch_size

patience = 5
replay_type = 'partial_joint_replay'

num_samples_per_malware_family = args.num_replay_sample

exp_type = 'first' #'last'


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


        model_save_dir = '../pjr_saved_model_' + str(exp_type) + '/NEW_PJR_replay_' + str(num_samples_per_malware_family) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)
        
        results_save_dir = './saved_results_' + str(exp_type) + '/NEW_PJR_replay_' + str(num_samples_per_malware_family) + '/' 
        create_parent_folder(results_save_dir)

        X_train, Y_train, Y_train_family = get_family_labeled_month_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)
        
        #X_train, Y_train, Y_train_family = get_only_av_class_labeled_samples(X_train, Y_train, Y_train_family)
        #X_test, Y_test, Y_test_family = get_only_av_class_labeled_samples(X_test, Y_test, Y_test_family)
        
        # to debug
        #X_train, Y_train, Y_train_family = X_train[:500], Y_train [:500], Y_train_family[:500]
        #X_test, Y_test, Y_test_family = X_test[:50], Y_test[:50], Y_test_family[:50]
        
        '''
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        
        
        standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)

        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)        
        '''
      

        if current_task == all_task_months[0]:
            stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,\
                                                               current_task, stored_global_family_dict)
        else:
            
            if str(exp_type) == 'first':
                X_replay, Y_replay = get_replay_samples_first(stored_global_family_dict, num_samples_per_malware_family)
            elif str(exp_type) == 'last':
                X_replay, Y_replay = get_replay_samples_last(stored_global_family_dict, num_samples_per_malware_family)
            else:
                X_replay, Y_replay = get_replay_samples(stored_global_family_dict, num_samples_per_malware_family)
            
            
            stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,\
                                                               current_task, stored_global_family_dict)
        
        
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
        task_training_time, epoch_ran, training_loss, validation_loss  = training_early_stopping(model, model_save_dir,\
                                X_train, Y_train, X_test, Y_test, patience,\
                                batch_size, device, optimizer, num_epoch,\
                                 criterion, replay_type, current_task, exp, earlystopping=True)

        top_k = 10
        acc, rocauc, wrong_good, wrong_mal, top_k_mistaken_families =\
                                    testing_aucscore_with_mistaken_stats(model, X_test, Y_test,\
                                    Y_test_family, batch_size, top_k, device)
        
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
    
results_save_dir = './saved_results/NEW_PJR_replay_' + str(num_samples_per_malware_family) + '/' 
create_parent_folder(results_save_dir)

all_results_save_file = results_save_dir + 'PJR_acc_rocauc_tr_time_num_exps_' + str(args.num_exps) + '.npz'

np.savez_compressed(all_results_save_file,
                        accuracy = allexps_acc, rocauc = allexps_rocauc, tr_time = allexps_training_time, best_epochs = all_exps_best_epoch)
print(f'all results saved')
