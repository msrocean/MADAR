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

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
plt.rcParams['font.size'] = 18
#plt.rcParams['font.family'] = "serif"
tdir = 'in'
major = 5.0
minor = 3.0
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor


from ember_utils import *
from ember_model import *
from ember_pjr_utils import *



def get_weights(model, data, layer, device):
    
    #print(f'i am in get_weights')
    model.eval()
    
    weights_ = []
    
    #sample_cnt = 0
    for sample in data:
        #print()
        #sample_cnt += 1
        X_ = torch.from_numpy(sample).type(torch.FloatTensor)
        X_ = X_.reshape(1, -1).to(device)
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        
        if layer == 'fc2':
            model.fc2.register_forward_hook(get_activation(layer))
        elif layer == 'fc3':
            model.fc3.register_forward_hook(get_activation(layer))
        elif layer == 'fc4':
            model.fc4.register_forward_hook(get_activation(layer))
            
        output = model(X_)
        weights_.append(activation[layer].cpu().numpy()[0])
        
        #print(f'sample_cnt {sample_cnt}')
    #print(f'weights_ {len(weights_)}')
    return weights_


def get_anomalyScoresSamples(fam, weight_data,\
                             raw_data, samples_in_family,\
                             chooseSample = True):
    clf = IsolationForest(max_samples=len(weight_data))
    clf.fit(weight_data)
    scores_prediction = clf.decision_function(weight_data)
    y_pred = clf.predict(weight_data)


    anomalous_idx = np.where(y_pred == -1.0)
    similar_idx = np.where(y_pred == 1.0)

    print(f'family {fam} anomaly-similar: {len(anomalous_idx[0])} - {len(similar_idx[0])}')
    
    if chooseSample:
        raw_data = np.array(raw_data)
        anomaly_samples = raw_data[anomalous_idx]
        
        
        remaining_samples_to_pick = samples_in_family - len(anomaly_samples)

        if remaining_samples_to_pick >= len(similar_idx):
            similar_samples = raw_data[similar_idx]
        else:
            similar_samples_pool = list(raw_data[similar_idx])
            similar_samples = random.sample(similar_samples_pool, remaining_samples_to_pick)
        

        mal_replay_samples = np.concatenate((np.array(anomaly_samples), np.array(similar_samples)))

        return np.array(mal_replay_samples)
    else:
        return scores_prediction


def get_anomalySamples(X_train, Y_train, Y_train_family,\
                       family_dict,\
                       samples_in_family,\
                       model, layer, device):
    
    
    #layer = 'fc4'
    good_weight_samples = X_train[np.where(Y_train == 0)]
    #print(f'good_weight_samples {len(good_weight_samples)}')
    
    pre_malware_samples = []
    for k, v in family_dict.items():
        #print(f'family {k} ....')
        if k != 'goodware':
            if len(v) > num_samples_per_malware_family:
                #print(f'family {k} to IF.')
                
#                 if k == 'others_family':
#                     maly = np.where(Y_train == 1)
#                     Ytrainmaly = Y_train_family[maly]
#                     k_weights_ind = np.where(Ytrainmaly == '')
#                 else:
#                     k_weights_ind = np.where(Y_train_family == k)
                    
#                 k_samples = X_train[k_weights_ind]
                #print(f'k_samples {len(k_samples)}')
    
                k_samples = v
                k_weights = get_weights(model, k_samples, layer, device) 
                #print(f'k_samples {len(k_samples)}  k_weights {len(k_weights)}')
                
                
                k_selected_samples = get_anomalyScoresSamples(k, k_weights,\
                                                     k_samples, samples_in_family,\
                                                     chooseSample = True)
                for sample in k_selected_samples:
                    pre_malware_samples.append(sample)
                
                family_dict[k] = list(k_selected_samples)
            else:
                for sample in v:
                    pre_malware_samples.append(sample)
    

    if len(good_weight_samples) <= len(pre_malware_samples):
        pre_goodware_samples = good_weight_samples
    else:
        pre_goodware_samples = random.sample(list(good_weight_samples), abs(len(pre_malware_samples)))

    family_dict['goodware'] = list(pre_goodware_samples)
    
    replay_samples = np.concatenate((np.array(pre_goodware_samples),\
                                     np.array(pre_malware_samples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_goodware_samples)),\
                                       np.ones(len(pre_malware_samples))))
    
    return replay_samples, labels_to_replay, family_dict



all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

data_dir = '../../month_based_processing_with_family_labels/'



patience = 5
replay_type = 'pjr'



num_exps = 1 #args.num_exps
#task_month = args.task_month
num_epoch = 500 #args.num_epoch
batch_size = 6000 #args.batch_size
num_samples_per_malware_family = 500

layer = 'fc4'

exp_type = 'weights'

exp_seeds = [random.randint(1, 99999) for i in range(num_exps)]


expSaveDir = '../Weights_'
resSaveDir = './Weights_'
expSaveFile = '/Weights_replay_'


raw_anomalyScores_Dict = {}
weight_anomalyScores_Dict = {}


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
        
        
        results_save_dir = str(resSaveDir) + 'results_' +\
                    str(exp_type) + str(expSaveFile) +\
                    str(num_samples_per_malware_family) + '/' 
        create_parent_folder(results_save_dir)

        
        X_train, Y_train, Y_train_family = get_family_labeled_month_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)
        
        stored_global_family_dict = make_family_based_dict(\
                                   X_train, Y_train, Y_train_family,\
                                   current_task, stored_global_family_dict)
        
        
        if current_task != all_task_months[0]:
            X_train = np.concatenate((np.array(X_train), np.array(X_replay)))
            Y_train = np.concatenate((np.array(Y_train), np.array(Y_replay)))
        
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)

        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)        
        
        
        # to debug
        #X_train, Y_train, Y_train_family = X_train[:500], Y_train [:500], Y_train_family[:500]
        #X_test, Y_test, Y_test_family = X_test[:50], Y_test[:50], Y_test_family[:50]
        
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
        
        
        end_time = time.time()

        print(f'Elapsed time {(end_time - start_time)/60} mins.')    
        

        task_end = time.time()
        task_run_time = (task_end - task_start)/60
        
        
        #print()
        X_replay, Y_replay, stored_global_family_dict = get_anomalySamples(X_train, Y_train,\
                                                     Y_train_family, stored_global_family_dict,\
                                                     num_samples_per_malware_family,\
                                                     model, layer, device)
        
        num_replay_samples = len(X_replay)
        
        print()
        
        
        results_f = open(os.path.join(results_save_dir + 'v2_weight_' + str(num_samples_per_malware_family) + '_results.txt'), 'a')
        result_string = '{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(current_task,epoch_ran, task_training_time, acc, rocauc, num_replay_samples)
        results_f.write(result_string)
        results_f.flush()
        results_f.close()
        
    
    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')
   
