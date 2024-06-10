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



class Ember_MLP_Net_V2(nn.Module):
    def __init__(self):
        super(Ember_MLP_Net_V2, self).__init__()
        
        input_features = 2381
        self.fc1 = nn.Linear(input_features, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        self.fc1_drop = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.act2 = nn.ReLU()
        self.fc2_drop = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.act3 = nn.ReLU()
        self.fc3_drop = nn.Dropout(p=0.5)        
        
        self.fc4 = nn.Linear(256, 128)
        self.fc4_bn = nn.BatchNorm1d(128)
        self.act4 = nn.ReLU()
        self.fc4_drop = nn.Dropout(p=0.5)  
        
        self.fc_last = nn.Linear(128, 1) 
        self.out = nn.Sigmoid()
        
        #self.activate = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        #x = self.fc1_bn(x)
        x = self.act1(x) 
        x = self.fc1_drop(x)

        x = self.fc2(x)
        #x = self.fc2_bn(x)
        x = self.act2(x) 
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        #x = self.fc3_bn(x)
        x = self.act3(x) 
        x = self.fc3_drop(x)
        
        x = self.fc4(x)
        #x = self.fc4_bn(x)
        x = self.act4(x)
        x = self.fc4_drop(x)
        
        x = self.fc_last(x)
        x = self.out(x)
        return x


def get_dataloader_weights(X, y, batch_size):
    
    X_ = torch.from_numpy(np.array(X)).type(torch.FloatTensor)
    y_ = torch.from_numpy(y).type(torch.FloatTensor)
    
    data_tensored = torch.utils.data.TensorDataset(X_,y_)    
    
    data_loader = torch.utils.data.DataLoader(data_tensored, batch_size = batch_size,
                                              num_workers=1, drop_last=False)
    return data_loader



def get_weights(model, layer, X_, Y_, batch_size, device):
    
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
    elif layer == 'fc2_bn':
        model.fc2_bn.register_forward_hook(get_activation(layer))
    elif layer == 'fc3_bn':
        model.fc3_bn.register_forward_hook(get_activation(layer))
    elif layer == 'fc4_bn':
        model.fc4_bn.register_forward_hook(get_activation(layer))
    elif layer == 'act2':
        model.act2.register_forward_hook(get_activation(layer))
    elif layer == 'act3':
        model.act3.register_forward_hook(get_activation(layer))
    elif layer == 'act4':
        model.act4.register_forward_hook(get_activation(layer)) 
    
    dataloader = get_dataloader_weights(X_, Y_, batch_size)   
    
    model.eval()
    
    features = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            feats_batch = activation[layer].cpu().numpy()
            
            for f in feats_batch:
                features.append(f)
 
            
    assert len(features) == len(X_)      
    return np.array(features)



def get_anomalyScoresSamples_V1(fam, weight_data,\
                             raw_data, samples_in_family,\
                             chooseSample = True):
    clf = IsolationForest(max_samples=len(weight_data))
    clf.fit(weight_data)
    #scores_prediction = clf.decision_function(weight_data)
    y_pred = clf.predict(weight_data)


    anomalous_idx = np.where(y_pred == -1.0)
    similar_idx = np.where(y_pred == 1.0)

    #print(f'family {fam} anomaly-similar: {len(anomalous_idx[0])} - {len(similar_idx[0])}')
    
    if chooseSample:
        raw_data = np.array(raw_data)
        anomaly_samples = raw_data[anomalous_idx]
        
        remaining_samples_to_pick = samples_in_family - len(anomaly_samples)
        
        if remaining_samples_to_pick == 0 or int(abs(remaining_samples_to_pick)/remaining_samples_to_pick) == -1:
            remaining_samples_to_pick = min(int(len(anomaly_samples) * 0.50), abs(remaining_samples_to_pick))
        
        
        if remaining_samples_to_pick >= len(similar_idx):
            similar_samples = raw_data[similar_idx]
        else:
            similar_samples_pool = list(raw_data[similar_idx])
            similar_samples = random.sample(similar_samples_pool, remaining_samples_to_pick)
        
        try:
            mal_replay_samples = np.concatenate((np.array(anomaly_samples), np.array(similar_samples)))
        except:
            print(np.array(anomaly_samples).shape, np.array(similar_samples).shape)

        return mal_replay_samples
    else:
        return scores_prediction

    
    
def get_anomalyScoresSamples(fam, weight_data,\
                             raw_data, samples_in_family,\
                             chooseSample = True):
    clf = IsolationForest(max_samples=len(weight_data))
    clf.fit(weight_data)
    #scores_prediction = clf.decision_function(weight_data)
    y_pred = clf.predict(weight_data)


    anomalous_idx = np.where(y_pred == -1.0)
    similar_idx = np.where(y_pred == 1.0)

    #print(f'family {fam} anomaly-similar: {len(anomalous_idx[0])} - {len(similar_idx[0])}')
    assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)
   
    
    if chooseSample:
        raw_data = np.array(raw_data)
        anomalous_samples = raw_data[anomalous_idx]
        
        if len(anomalous_samples) >= num_samples_per_malware_family:
            anomalous_samples_pool = list(anomalous_samples)
            remaining_samples_to_pick = int(num_samples_per_malware_family/2)
            anomalous_samples = random.sample(anomalous_samples_pool, remaining_samples_to_pick)

        else:
            remaining_samples_to_pick = num_samples_per_malware_family - len(anomalous_samples)        
        
        if remaining_samples_to_pick >= len(similar_idx):
            similar_samples = raw_data[similar_idx]
        else:
            similar_samples_pool = list(raw_data[similar_idx])
            similar_samples = random.sample(similar_samples_pool, remaining_samples_to_pick)
                
        try:
            mal_replay_samples = np.concatenate((np.array(anomalous_samples), np.array(similar_samples)))
        except:
            print(np.array(anomalous_samples).shape, np.array(similar_samples).shape)

        return mal_replay_samples
    else:
        return scores_prediction    
  
    

def get_anomalySamples(family_dict,\
                       samples_in_family,\
                       model, layer, device):
    
    
    
    pre_malware_samples = []
    for k, v in family_dict.items():
        #print(f'family {k} ....')
        if k != 'goodware':
            if len(v) > samples_in_family:
                #print(f'family {k} to IF.')
                
                k_samples = v
                k_Y = np.ones(len(k_samples))
                k_weights = get_weights(model, layer, k_samples, k_Y, batch_size, device)
                
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
    

    if len(family_dict['goodware']) < len(pre_malware_samples):
        pre_goodware_samples = random.sample(family_dict['goodware'], len(family_dict['goodware']))
    else:
        pre_goodware_samples = random.sample(family_dict['goodware'], len(pre_malware_samples))
    
    family_dict['goodware'] = list(pre_goodware_samples)
    
    replay_samples = np.concatenate((list(pre_goodware_samples),\
                                     list(pre_malware_samples)))
    labels_to_replay = np.concatenate((list(np.zeros(len(pre_goodware_samples))),\
                                       (np.ones(len(pre_malware_samples)))))
    
    #random.shuffle(replay_samples, labels_to_replay)
    from sklearn.utils import shuffle
    X_, Y_ = shuffle(replay_samples, labels_to_replay)
    
    return X_, Y_, family_dict


parser = argparse.ArgumentParser()
# parser.add_argument('--num_exps', type=int, default=5, required=False, help='Number of Experiments to Run.')
#parser.add_argument('--contamination', type=float, default=0.1, required=True)
parser.add_argument('--batch_norm', required=True, choices=["yes", "no"],\
                    help = 'Get weights from the model with or withour batch normalization layer.')
parser.add_argument('--layer', choices=["act4", "fc4", "fc4_bn"], required=True)
args = parser.parse_args()

all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

data_dir = '../../month_based_processing_with_family_labels/'



patience = 5
replay_type = 'AHRS'



num_exps = 5 #args.num_exps
#task_month = args.task_month
num_epoch = 500 #args.num_epoch
batch_size = 6000 #args.batch_size
num_samples_per_malware_family = 500

layer = args.layer #'act4'
batch_norm = args.batch_norm


exp_type = 'weights'

exp_seeds = [random.randint(1, 99999) for i in range(num_exps)]


expSaveDir = '../WeightsFinal_'
resSaveDir = './AHRS_Results/'
expSaveFile = 'Weights_replay_'


raw_anomalyScores_Dict = {}
weight_anomalyScores_Dict = {}

restartMonth = 100
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
        
        #task_month = restartMonth - 1
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
        
        
        if task_month == restartMonth -1:
            tmpcurrent_task = all_task_months[task_month - 1]
            tmp_model_save_dir = str(expSaveDir) + 'model_' +\
                    str(exp_type) + str(expSaveFile) +\
                    str(num_samples_per_malware_family) + '/' + str(tmpcurrent_task) + '/'
            
            dict_save_file = tmp_model_save_dir + 'global_family_dict_' + str(task_month) + '.npz'
            dictFile = np.load(dict_save_file)
            stored_global_family_dict = dictFile['dictfile']
        else:
            dict_save_file = model_save_dir + 'global_family_dict_' + str(task_month) + '.npz'
            np.savez_compressed(dict_save_file, dictfile = stored_global_family_dict)
        
        X_train, Y_train, Y_train_family = get_family_labeled_month_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)
        
        
        # msr --> get X_train in standardized space and then update the global dict
        # does not work
#         if current_task != all_task_months[0]:
#             X_train = standard_scaler.transform(X_train)
#             X_train = np.array(X_train, np.float32)
        
        stored_global_family_dict = make_family_based_dict(\
                                   X_train, Y_train, Y_train_family,\
                                   current_task, stored_global_family_dict)
        
        
        
        if batch_norm == 'yes':
            if current_task != all_task_months[0]:
                X_train, Y_train, stored_global_family_dict = get_anomalySamples(
                                                             stored_global_family_dict,\
                                                             num_samples_per_malware_family,\
                                                             model, layer, device)
            results_f = open(os.path.join(results_save_dir + layer + '_ahrs_bn_' + str(num_samples_per_malware_family) + '_results.txt'), 'a')
        
        else:
            if current_task != all_task_months[0]:
                model_new = Ember_MLP_Net_V2()
                model_new = model_new.to(device)
                #load the best model for this task
                #best_model_path = model_save_dir + os.listdir(model_save_dir)[0]
                print(f'loading best model {best_model_path}')
                model_new.load_state_dict(torch.load(best_model_path))

                X_train, Y_train, stored_global_family_dict = get_anomalySamples(
                                                             stored_global_family_dict,\
                                                             num_samples_per_malware_family,\
                                                             model_new, layer, device)
                
            results_f = open(os.path.join(results_save_dir + layer + '_ahrs_wobn_' + str(num_samples_per_malware_family) + '_results.txt'), 'a')
        
        
        
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
        
        
        num_replay_samples = len(X_train)
        
        print()
        
        
     
        result_string = '{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(current_task,epoch_ran, task_training_time, acc, rocauc, num_replay_samples)
        results_f.write(result_string)
        results_f.flush()
        results_f.close()
        
    
    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')
   
