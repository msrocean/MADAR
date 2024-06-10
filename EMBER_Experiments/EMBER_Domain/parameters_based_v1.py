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
    
    
    model.eval()
    
    weights_ = []
    
    for sample in data:
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
    
    return weights_


def get_anomalyScoresSamples(weight_data, raw_data, num_samples_per_category, chooseSample = True):
    clf = IsolationForest(max_samples=len(weight_data))
    clf.fit(weight_data)
    scores_prediction = clf.decision_function(weight_data)
    y_pred = clf.predict(weight_data)


    anomalous_idx = np.where(y_pred == -1.0)
    similar_idx = np.where(y_pred == 1.0)

    print(f'anomaly-similar samples {len(anomalous_idx[0])} - {len(similar_idx[0])}')
    
    if chooseSample:
        raw_data = np.array(raw_data)
        anomaly_samples = raw_data[anomalous_idx]
        similar_samples = random.sample(list(raw_data[similar_idx]),
                            (num_samples_per_category - len(anomaly_samples)))

        mal_replay_samples = np.concatenate((np.array(anomaly_samples), np.array(similar_samples)))

        return mal_replay_samples, scores_prediction
    else:
        return scores_prediction


def get_anomalySamples(X_train, model, device):
    
    num_samples_per_category = int((len(X_train) * 0.20)/2)
    layer = 'fc4'
    wegt = get_weights(model, X_train, layer, device)
    mal_raw = X_train[np.where(Y_train == 1)]
    mal_weight = np.array(wegt)[np.where(Y_train == 1)]
    
    
    good_weight_samples = X_train[np.where(Y_train == 0)]
    mal_weight_samples = X_train[np.where(Y_train == 1)]
    
    replay_mal_samples, anomalyscores = get_anomalyScoresSamples(mal_weight,\
                             mal_weight_samples, num_samples_per_category, chooseSample = True)
    
    
    if len(good_weight_samples) < num_samples_per_category: #len(replay_mal_samples):
        replay_good_samples = good_weight_samples
    else:
        replay_good_samples = random.sample(list(good_weight_samples), num_samples_per_category) #len(replay_mal_samples))
    
    
    replay_samples = np.concatenate((np.array(replay_good_samples), np.array(replay_mal_samples)))
    labels_to_replay = np.concatenate((np.zeros(len(replay_good_samples)), np.ones(len(replay_mal_samples))))
    
    return replay_samples, labels_to_replay, anomalyscores



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
        
        

#         if current_task == all_task_months[0]:
#             num_replay_samples = 0
#             stored_global_family_dict = make_family_based_dict(\
#                                        X_train, Y_train, Y_train_family,\
#                                        current_task, stored_global_family_dict)
#         else:
#             stored_global_family_dict = make_family_based_dict(X_train, Y_train, Y_train_family,\
#                                                                    current_task, stored_global_family_dict)
            
#             X_train, Y_train, stored_global_family_dict =\
#                 get_replay_samples_IFBased(stored_global_family_dict, num_samples_per_malware_family)
            
        print()
        print(f'X_train {X_train.shape} Y_train {Y_train.shape}')
        print()
        
        '''
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)

        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)        
        '''        
        
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
        
        
        print()
        X_replay, Y_replay, mal_weights_anomaly = get_anomalySamples(X_train, model, device)
        num_replay_samples = len(Y_replay)
        
#         mal_raw = X_train[np.where(Y_train == 1)]
#         mal_raw_anomaly = get_anomalyScoresSamples(mal_raw,\
#                                          mal_raw, chooseSample = False)
    
        #raw_anomalyScores_Dict[str(current_task)] = mal_raw_anomaly
        weight_anomalyScores_Dict[str(current_task)] = mal_weights_anomaly
        print()
        
        
        results_f = open(os.path.join(results_save_dir + 'v1_20_weight_' + str(num_samples_per_malware_family) + '_results.txt'), 'a')
        result_string = '{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(current_task,epoch_ran, task_training_time, acc, rocauc, num_replay_samples)
        results_f.write(result_string)
        results_f.flush()
        results_f.close()
        
    
    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')
    
    
all_results_save_file = results_save_dir + 'weightAnomalyScores.npz'
np.savez_compressed(all_results_save_file, anomaly = raw_anomalyScores_Dict)

    
