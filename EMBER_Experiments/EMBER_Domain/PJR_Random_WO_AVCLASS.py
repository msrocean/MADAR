import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time, random
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import datetime
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


def get_only_av_class_labeled_samples(all_X, all_Y, Y_Family):
    
    new_X = []
    new_Y = []
    new_Y_family = []
    
    for ind, y in enumerate(all_Y):
        if y == 1:
            if Y_Family[ind] != '':
                new_Y.append(y)
                new_X.append(all_X[ind])
                new_Y_family.append(Y_Family[ind])
        else:
            new_Y.append(y)
            new_X.append(all_X[ind])
            new_Y_family.append(Y_Family[ind])
            
            
    print(len(all_Y==1) == (len(new_Y) + (len(all_Y) - len(new_Y))))  
    print(len(new_X) == len(new_Y))
    return new_X, new_Y, new_Y_family

def get_family_labeled_month_data(data_dir, month, train=True):
    
    if train:
        data_dir = data_dir + str(month) + '/'
        XY_train = np.load(data_dir + 'XY_train.npz')
        X_tr, Y_tr, Y_tr_family = XY_train['X_train'], XY_train['Y_train'], XY_train['Y_family_train']

        print(f'X_train {X_tr.shape} Y_train {Y_tr.shape} Y_tr_family {Y_tr_family.shape}')
        
        return X_tr, Y_tr, Y_tr_family
    else:
        data_dir = data_dir + str(month) + '/'
        XY_test = np.load(data_dir + 'XY_test.npz')
        X_test, Y_test, Y_test_family = XY_test['X_test'], XY_test['Y_test'], XY_test['Y_family_test']

        return X_test, Y_test, Y_test_family

    
    
def get_family_labeled_task_test_data(data_dir, task_months, mlp_net=False):
    
    X_te, Y_te, Y_te_family = get_family_labeled_month_data(data_dir, task_months[-1], train=False)
    
    for month in task_months[:-1]:
        pre_X_te, pre_Y_te, pre_Y_te_family = get_family_labeled_month_data(data_dir, month, train=False)
        X_te, Y_te, Y_te_family = np.concatenate((X_te, pre_X_te)), np.concatenate((Y_te, pre_Y_te)),\
                                np.concatenate((Y_te_family, pre_Y_te_family))
        

    X_test, Y_test, Y_test_family = X_te, Y_te, Y_te_family
    print(f'X_test {X_test.shape} Y_test {Y_test.shape} Y_te_family {Y_te_family.shape}')
    
    return X_test, Y_test, Y_test_family




def getNOAVCLASS_partial_data(X, Y, Y_fam, replay_portion):
    indx = [i for i in range(len(Y))]
    random.shuffle(indx)

    replay_data_size = int(len(indx)*replay_portion)
    replay_index = indx[:replay_data_size]

    X_train = X[replay_index]
    Y_train = Y[replay_index]
    Y_family = Y_fam[replay_index]
    
    return X_train, Y_train, Y_family        
        
def getNOAVCLASS_PJR_random_training_data(data_dir, task_months, replay_portion):
    
    X_tr, Y_tr, Y_tr_fam = get_family_labeled_month_data(data_dir, task_months[-1])
    X_tr, Y_tr, Y_tr_fam = get_only_av_class_labeled_samples(X_tr, Y_tr, Y_tr_fam)
    
    X_tr, Y_tr, Y_tr_fam = np.array(X_tr), np.array(Y_tr), np.array(Y_tr_fam)
    
    print(f'Current Task month {task_months[-1]} data X {np.array(X_tr).shape} Y {np.array(Y_tr).shape} Y_family {np.array(Y_tr_fam).shape}')
    
    if replay_portion != 0.0:
        for month in task_months[:-1]:
            pre_X_tr, pre_Y_tr, pre_Y_tr_fam = get_family_labeled_month_data(data_dir, month)
            pre_X_tr, pre_Y_tr, pre_Y_tr_fam = get_only_av_class_labeled_samples(pre_X_tr, pre_Y_tr, pre_Y_tr_fam)

            pre_X_tr, pre_Y_tr, pre_Y_tr_fam = np.array(pre_X_tr), np.array(pre_Y_tr), np.array(pre_Y_tr_fam)

            pre_X_tr, pre_Y_tr, pre_Y_tr_fam = getNOAVCLASS_partial_data(pre_X_tr, pre_Y_tr, pre_Y_tr_fam, replay_portion)

            print(f'previous month {month} data X {pre_X_tr.shape} Y {pre_Y_tr.shape} Y_family {pre_Y_tr_fam.shape}')

            X_tr, Y_tr, Y_tr_fam  = np.concatenate((X_tr, pre_X_tr)),\
                        np.concatenate((Y_tr, pre_Y_tr)), np.concatenate((Y_tr_fam, pre_Y_tr_fam))

    
    
    X_tr, Y_tr, Y_tr_fam = np.array(X_tr), np.array(Y_tr), np.array(Y_tr_fam)
    print()
    print(f'X_train {X_tr.shape} Y_train {Y_tr.shape} Y_family {Y_tr_fam.shape}\n')
    
    return X_tr, Y_tr, Y_tr_fam 


parser = argparse.ArgumentParser()
parser.add_argument('--num_exps', type=int, default=1, required=False, help='Number of Experiments to Run.')
parser.add_argument('--num_run', type=int, required=True)
parser.add_argument('--num_epoch', type=int, default=2000, required=False)
parser.add_argument('--batch_size', type=int, default=6000, required=False)
parser.add_argument('--replay_portion', type=float, default=1.0, required=True)

args = parser.parse_args()


all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

#data_dir = '../../ember2018/month_based_processing/'
data_dir = '../../month_based_processing_with_family_labels/'

num_exps = args.num_exps
#task_month = args.task_month
num_epoch = args.num_epoch
batch_size = args.batch_size
replay_portion = args.replay_portion
patience = 5


replay_type = 'joint_partial'
exp_type = 'RANDOM_BUFFER'

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
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
       
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print(f'Model has {count_parameters(model)/1000000}m parameters')    
    criterion = nn.BCELoss()    
    
    
    standardization = StandardScaler()
    standard_scaler = None
    for task_month in range(len(all_task_months)):
                
        print(f'\n{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Round {cnt} ...')
        task_start = time.time()
        #task_month = 11
        current_task = all_task_months[task_month]
        task_months = all_task_months[:task_month+1]
        print(f'Current Task {current_task} with Replay {replay_portion*100}%')


        model_save_dir = '../pjr_saved_model_' + str(exp_type) + '/NOAVCLASS_VALID_PJR_replay_' +\
                            str(replay_portion) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)
        
        results_save_dir =  './pjr_saved_results_' + str(exp_type) + '/NOAVCLASS_VALID_PJR_replay_' + str(replay_portion) + '/' 
        create_parent_folder(results_save_dir)
        
        X_train, Y_train, Y_train_family = getNOAVCLASS_PJR_random_training_data(data_dir, task_months, replay_portion)

        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)
    

        # to debug
        #X_train = X_train[:500]
        #Y_train = Y_train [:500]
        #X_valid, Y_valid = X_valid[:500], Y_valid[:500]
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        
        if replay_portion == 1.0:
                standardization = StandardScaler()
                standard_scaler = None
                standard_scaler = standardization.fit(X_train)
        else:        
                standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)


        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)        
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Training ...')
        task_training_time, epoch_ran, training_loss, validation_loss  =\
                            training_early_stopping(model, model_save_dir, X_train, Y_train,\
                            X_test, Y_test, patience, batch_size, device, optimizer, num_epoch,\
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
        
        
        #now = datetime.datetime.now()
        #print ("Current date and time : ")
        #current_time = now.strftime("%Y-%m-%d %H:%M:%S")   
        
       
        results_f = open(os.path.join(results_save_dir + 'results_accumulated_replay_' + str(replay_portion) + '_results.txt'), 'a')
        result_string = '{}\t{}\t{}\t{}\t{}\t\n'.format(current_task,epoch_ran, task_training_time, acc, rocauc)
        results_f.write(result_string)
        results_f.flush()
        results_f.close()

        
        wf = open(os.path.join(results_save_dir + 'Results_' + str(current_task) + '_' + str(num_epoch) + '_replay_' + str(replay_portion) + '_results.txt'), 'a')
        task_exp_string = '\n\nSeed\t{}\t\tRun time\t{}\tAcc:\t{}\t\tROC_AUC:\t{}\n\tepoch_ran\t{}\t\n\ntraining_loss\t{}\n\nValid_loss\t{}\n\n'.format(exp,task_training_time, acc, rocauc, epoch_ran, training_loss, validation_loss)
        
        wf.write('\n ########################### ########################### ###########################\n')
        wf.write(str(model))
        wf.write(task_exp_string)
        
        wf.flush()
        wf.close()
   
    
    mistakes_f = results_save_dir + 'mistaken_stats.npz'
    np.savez_compressed(mistakes_f, mistakes = mistaken_stats)
    
    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')



#results_save_dir = './pjr_with_random_buffer_results/PJR_replay_' + str(replay_portion) + '/' 
#create_parent_folder(results_save_dir)

#all_results_save_file = results_save_dir + 'PJR_acc_rocauc_tr_time_best_epoch_' + str(args.num_run) + '.npz'
#np.savez_compressed(all_results_save_file,
#                        accuracy = allexps_acc, rocauc = allexps_rocauc, tr_time = allexps_training_time, best_epochs = all_exps_best_epoch)
#print(f'all results saved')


