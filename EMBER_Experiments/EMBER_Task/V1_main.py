#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np
import time
import torch
from torch import optim
import utils
import pandas as pd
from param_stamp import get_param_stamp, get_param_stamp_from_args
import evaluate
from data import get_malware_multitask_experiment
from encoder import Classifier
import callbacks as cb
from train_diversity import train_cl
from continual_learner import ContinualLearner
from exemplars import ExemplarHandler
from replayer import Replayer
from param_values import set_default_values


parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str,\
                    default='../../../ember2018/top_class_bases/top_classes_100',\
                    dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results_dir', type=str, default='./results', dest='r_dir', help="default: %(default)s")

# expirimental task parameters
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, required=False,\
                         default='EMBER_Class', choices=['EMBER_Class', 'EMBER_Task'])
task_params.add_argument('--dataset', type=str, default='EMBER', choices=['EMBER'])

task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')
task_params.add_argument('--target_classes', type=int, default=100, required=False, help='number of classes')
task_params.add_argument('--init_classes', type=int, default=50, required=False, help='number of classes for the first task')
task_params.add_argument('--replay_portion', type=float, default=1.0,\
                         required=False, help='percentage of number of stored samples to replay')
task_params.add_argument('--cnt_rate', type=float, default=0.1,\
                         required=False, help='contamination rate for isolation forest')
task_params.add_argument('--num_replay_sample', type=int, default=200,\
                         required=False, help='number of stored samples per class to replay')
task_params.add_argument('--replay_config', type=str,\
                         choices=['grs', 'frs', 'hs', 'ifs', 'aws'],\
                         required=True, help='diversity aware replay configurations')

parser.add_argument('--layer', choices=["fcLayer4", "fc4", "fc4_bn"],  default='act4', required=False)


# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")
loss_params.add_argument('--bce-distill', action='store_true', help='distilled loss on previous classes for new'
                                                                    ' examples (only if --bce & --scenario="class")')

# model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-layers', type=int, default=5, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="yes", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                   " (instead of a 'multi-headed' one)")

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--exp_name', type=str, help="experiment name")
train_params.add_argument('--logger_file', type=str, help="logger file name")
train_params.add_argument('--iters', type=int, default=10000, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, default=0.001, help="learning rate") #default=0.01, 
train_params.add_argument('--batch', type=int, default=256, help="batch-size")
train_params.add_argument('--epoch', type=int, default=30, help="number of training epochs")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='sgd')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--feedback', action="store_true", help="equip model with feedback connections")
replay_params.add_argument('--z-dim', type=int, default=100, help='size of latent representation (default: 100)')
replay_choices = ['offline', 'none']
replay_params.add_argument('--replay', type=str, default='offline', choices=replay_choices)
replay_params.add_argument('--distill', action='store_true', help="use distillation for replay?")
replay_params.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
replay_params.add_argument('--agem', action='store_true', help="use gradient of replay as inequality constraint")
# -generative model parameters (if separate model)
genmodel_params = parser.add_argument_group('Generative Model Parameters')
genmodel_params.add_argument('--g-z-dim', type=int, default=100, help='size of latent representation (default: 100)')
genmodel_params.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
genmodel_params.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
# - hyper-parameters for generative model (if separate model)
gen_params = parser.add_argument_group('Generator Hyper Parameters')
gen_params.add_argument('--g-iters', type=int, help="# batches to train generator (default: as classifier)")
gen_params.add_argument('--lr-gen', type=float, help="learning rate generator (default: lr)")

# "memory allocation" parameters
cl_params = parser.add_argument_group('Memory Allocation Parameters')
cl_params.add_argument('--ewc', action='store_true', help="use 'EWC' (Kirkpatrick et al, 2017)")
cl_params.add_argument('--lambda', type=float, dest="ewc_lambda", help="--> EWC: regularisation strength")
cl_params.add_argument('--fisher-n', type=int, help="--> EWC: sample size estimating Fisher Information")
cl_params.add_argument('--online', action='store_true', help="--> EWC: perform 'online EWC'")
cl_params.add_argument('--gamma', type=float, help="--> EWC: forgetting coefficient (for 'online EWC')")
cl_params.add_argument('--emp-fi', action='store_true', help="--> EWC: estimate FI with provided labels")
cl_params.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
cl_params.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")
cl_params.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="--> SI: dampening parameter")
cl_params.add_argument('--xdg', action='store_true', help="Use 'Context-dependent Gating' (Masse et al, 2018)")
cl_params.add_argument('--gating-prop', type=float, metavar="PROP", help="--> XdG: prop neurons per layer to gate")

# data storage ('exemplars') parameters
store_params = parser.add_argument_group('Data Storage Parameters')
store_params.add_argument('--icarl', action='store_true', help="bce-distill, use-exemplars & add-exemplars")
store_params.add_argument('--use-exemplars', action='store_true', help="use exemplars for classification")
store_params.add_argument('--add-exemplars', action='store_true', help="add exemplars to current task's training set")
store_params.add_argument('--budget', type=int, default=1000, dest="budget", help="how many samples can be stored?")
store_params.add_argument('--herding', action='store_true', help="use herding to select stored data (instead of random)")
store_params.add_argument('--norm-exemplars', action='store_true', help="normalize features/averages of exemplars")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
eval_params.add_argument('--metrics', action='store_true', help="calculate additional metrics (e.g., BWT, forgetting)")
eval_params.add_argument('--pdf', action='store_true', help="generate pdf with results")
eval_params.add_argument('--visdom', action='store_true', help="use visdom for on-the-fly plots")
eval_params.add_argument('--log-per-task', action='store_true', help="set all visdom-logs to [iters]")
eval_params.add_argument('--loss-log', type=int, default=200, metavar="N", help="# iters after which to plot loss")
eval_params.add_argument('--prec-log', type=int, default=200, metavar="N", help="# iters after which to plot precision")
eval_params.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating solver's precision")
eval_params.add_argument('--sample-log', type=int, default=500, metavar="N", help="# iters after which to plot samples")
eval_params.add_argument('--sample-n', type=int, default=64, help="# images to show")

def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def run(args, verbose=True):
    #print(f'batch iteration {args.iters}')
    visdom = None

    
    # -if [log_per_task], reset all logs
    if args.log_per_task:
        args.prec_log = args.iters
        args.loss_log = args.iters
        args.sample_log = args.iters
        
    # -if 'BCEdistill' is selected for other than scenario=="class", give error
    if args.bce_distill and not args.scenario=="class":
        raise ValueError("BCE-distill can only be used for class-incremental learning.")
    
    # -create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
        
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    scenario = args.scenario

    # If only want param-stamp, get it printed to screen and exit
    if hasattr(args, "get_stamp") and args.get_stamp:
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")
    #print("CUDA is {}used".format("" if cuda else "NOT(!!) "))
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))
    
    
    args.seed = random.randint(1, 99999)
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    #----------------#
    #----- DATA -----#
    #----------------#
    #print(args)
    #args.tasks = 3
    
    target_classes = args.target_classes
    orig_feats_length, target_feats_length = 2381, 2381
    dataset_name = args.dataset #'EMBER'
    init_classes = args.init_classes
    
    #print(args)
    # Prepare data for chosen experiment
    if verbose:
        print("\nPreparing the data...")
        
    num_training_samples, ember_train, ember_test, train_datasets, test_datasets, config, classes_per_task = get_malware_multitask_experiment(
        dataset_name=dataset_name, target_classes=target_classes,\
        init_classes = init_classes, scenario=scenario, orig_feats_length=orig_feats_length,\
        target_feats_length=target_feats_length, tasks=args.tasks, data_dir=args.d_dir,
        verbose=verbose)
    
    
    #num_training_samples = 303331
    #args.iters = 2000

    #------------------------------#
    #----- MODEL (CLASSIFIER) -----#
    #------------------------------#
    #print(config)
    
    
    model = Classifier(
        image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
        fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
        fc_bn=True if args.fc_bn=="yes" else False, excit_buffer=True if args.xdg and args.gating_prop>0 else False,
        binaryCE=args.bce, binaryCE_distill=args.bce_distill, AGEM=args.agem).to(device)
    
#     print(args)
    # Define optimizer (only include parameters that "requires_grad")
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type in ("adam", "adam_reset"):
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type=="sgd":
        model.optimizer = optim.SGD(model.optim_list, momentum=0.9, weight_decay=0.000001)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))
    
    
    
    
    #print(model)
    
    
    #-------------------------------------------------------------------------------------------------#

    #----------------------------------#
    #----- CL-STRATEGY: EXEMPLARS -----#
    #----------------------------------#
   
    
    # Store in model whether, how many and in what way to store exemplars
    if isinstance(model, ExemplarHandler) and (args.use_exemplars or args.add_exemplars or args.replay=="exemplars"):
        model.memory_budget = args.budget
        model.norm_exemplars = args.norm_exemplars
        model.herding = args.herding



    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    generator = None


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#
    #print(model)
    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp(
        args, 'ember_MLP', verbose=verbose, replay=True if (not args.replay=="none") else False,
        replay_model_name=generator.name if (args.replay=="generative" and not args.feedback) else None,
    )

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        utils.print_model_info(model, title="MAIN MODEL")


    # Prepare for keeping track of statistics required for metrics (also used for plotting in pdf)
    if args.pdf or args.metrics:
        # -define [metrics_dict] to keep track of performance during training for storing & for later plotting in pdf
        metrics_dict = evaluate.initiate_metrics_dict(args, n_tasks=args.tasks, scenario=args.scenario)
        # -evaluate randomly initiated model on all tasks & store accuracies in [metrics_dict] (for calculating metrics)
        if not args.use_exemplars:
            metrics_dict = evaluate.intial_accuracy(args, model, test_datasets, metrics_dict,
                                                    classes_per_task=classes_per_task, scenario=scenario,
                                                    test_size=None, no_task_mask=False)
    else:
        metrics_dict = None
    
    #print(metrics_dict)
    
    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Callbacks for reporting on and visualizing loss
    
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, tasks=args.tasks,
                           iters_per_task=args.iters, replay=False if args.replay=="none" else True)
    ] if (not args.feedback) else [None]



    # Callbacks for reporting and visualizing accuracy
    # -visdom (i.e., after each [prec_log]
    eval_cbs = [
        cb._eval_cb(log=args.prec_log, test_datasets=test_datasets, visdom=visdom,
                    iters_per_task=args.iters, test_size=args.prec_n, classes_per_task=classes_per_task,
                    scenario=scenario, with_exemplars=False)
    ] if (not args.use_exemplars) else [None]
    #--> during training on a task, evaluation cannot be with exemplars as those are only selected after training
    #    (instead, evaluation for visdom is only done after each task, by including callback-function into [metric_cbs])

    # Callbacks for calculating statists required for metrics
    # -pdf / reporting: summary plots (i.e, only after each task) (when using exemplars, also for visdom)
    metric_cbs = [
        cb._metric_cb(args=args, log=args.iters, test_datasets=test_datasets,
                      classes_per_task=classes_per_task, metrics_dict=metrics_dict, scenario=scenario,
                      iters_per_task=args.iters, with_exemplars=args.use_exemplars),
        cb._eval_cb(log=args.iters, test_datasets=test_datasets, visdom=visdom,
                    iters_per_task=args.iters, test_size=args.prec_n, classes_per_task=classes_per_task,
                    scenario=scenario, with_exemplars=True) if args.use_exemplars else None
    ]


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if verbose:
        print("\nTraining...")
    model_save_path = './ember_saved_models/'
    # Keep track of training-time
    start = time.time()
    
    #print(args) 
    results_save_dir = 'ember_results_cl/ICDM_Submission/' + str(args.scenario) + '/' + str(target_classes) + '/'
    create_parent_folder(results_save_dir)
    
    if args.replay_config == 'grs':
        results_file = str(args.replay_config)  + '_MC_' + str(args.replay_portion) + '_results.txt'
    
    if args.replay_config == 'frs':
        results_file = str(args.replay_config)  + '_' + str(args.num_replay_sample) + '_results.txt'
        
            
    if args.replay_config == 'ifs':
        results_file = str(args.replay_config)  + '_MC_' + str(args.num_replay_sample) + '_' + str(args.cnt_rate) + '_results.txt'
    
    if args.replay_config == 'aws':
        results_file = str(args.replay_config) + '_MC_' + str(args.layer)  + '_' + str(args.num_replay_sample) + '_' + str(args.cnt_rate) + '_results.txt'
    
    
    
    
    args.r_dir = os.path.join(results_save_dir + results_file)
    
    #print(f'args.r_dir {args.r_dir}')
    
    # Train model
    train_cl(
        model, model_save_path, ember_train, ember_test, train_datasets,\
        test_datasets, args, replay_mode=args.replay, scenario=scenario,\
        classes_per_task=classes_per_task,\
        iters=args.iters, batch_size=args.batch,\
        eval_cbs=eval_cbs, loss_cbs=solver_loss_cbs,\
        metric_cbs=metric_cbs, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
    )
    
    # Get total training-time in seconds, and write to file
    if args.time:
        training_time = time.time() - start
        time_file = open("{}/time-{}.txt".format(args.r_dir, param_stamp), 'w')
        time_file.write('{}\n'.format(training_time))
        time_file.close()

        
    #print(f'metrics_dict {metrics_dict}')
#     metrics_average = metrics_dict['average']
    
#     print(f'metrics_dict average {metrics_average} Mean {np.mean(metrics_average)}')

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
#     accs, precisions, recalls, f1scores = [evaluate.validate(
#         model, test_datasets[i], verbose=True, test_size=None, task=i+1, with_exemplars=False,
#         allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#     ) for i in range(args.tasks)]
    
#     accs, precisions, recalls, f1scores = [], [], [], []
#     task = args.tasks
#     for i in range(task):
#         acc, prec, rec, f1 = evaluate.validate(
#             model, test_datasets[i], verbose=True, test_size=None, task=i+1, with_exemplars=False,
#             allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#         )
#         accs.append(acc)
#         precisions.append(prec)
#         recalls.append(rec)
#         f1scores.append(f1)    


#     accs = evaluate.validate(
#             model, test_datasets[i], verbose=True, test_size=None, task=i+1, with_exemplars=False,
#             allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#         )


#     precisions = evaluate.validate(
#             model, test_datasets[i], whichMetric='precision', verbose=True, test_size=None, task=i+1, with_exemplars=False,
#             allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#         )

#     recalls = evaluate.validate(
#             model, test_datasets[i], whichMetric='recall', verbose=True, test_size=None, task=i+1, with_exemplars=False,
#             allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#         )

#     f1scores = evaluate.validate(
#             model, test_datasets[i], whichMetric='f1score', verbose=True, test_size=None, task=i+1, with_exemplars=False,
#             allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#         )
    
#     print(f'\n accs {accs}')
#     print(f'\n precisions {precisions}')
#     print(f'\n recalls {recalls}')
#     print(f'\n f1scores {f1scores}')
    


#     if args.metrics:
#         # Accuracy matrix
#         if args.scenario in ('task', 'domain'):
#             R = pd.DataFrame(data=metrics_dict['acc per task'],
#                              index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#             R.loc['at start'] = metrics_dict['initial acc per task'] if (not args.use_exemplars) else [
#                 'NA' for _ in range(args.tasks)
#             ]
            
            
            
#             R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
#             BWTs = [(R.loc['after task {}'.format(args.tasks), 'task {}'.format(i + 1)] - \
#                      R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 1)]) for i in range(args.tasks - 1)]
#             FWTs = [0. if args.use_exemplars else (
#                 R.loc['after task {}'.format(i+1), 'task {}'.format(i + 2)] - R.loc['at start', 'task {}'.format(i+2)]
#             ) for i in range(args.tasks-1)]
            
            
#             forgetting = []
            
#             for i in range(args.tasks - 1):
#                 forgetting.append(max(R.iloc[1:args.tasks, i]) - R.iloc[args.tasks, i])
#             R.loc['FWT (per task)'] = ['NA'] + FWTs
#             R.loc['BWT (per task)'] = BWTs + ['NA']
#             R.loc['F (per task)'] = forgetting + ['NA']
#             BWT = sum(BWTs) / (args.tasks - 1)
#             F = sum(forgetting) / (args.tasks - 1)
#             FWT = sum(FWTs) / (args.tasks - 1)
#             metrics_dict['BWT'] = BWT
#             metrics_dict['F'] = F
#             metrics_dict['FWT'] = FWT
#             # -print on screen
#             if verbose:
#                 print("Accuracy matrix")
#                 print(R)
#                 print("\nFWT = {:.4f}".format(FWT))
#                 print("BWT = {:.4f}".format(BWT))
#                 print("  F = {:.4f}\n\n".format(F))
#         else:
#             if verbose:
#                 # Accuracy matrix based only on classes in that task (i.e., evaluation as if Task-IL scenario)
#                 R = pd.DataFrame(data=metrics_dict['acc per task (only classes in task)'],
#                                  index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 R.loc['at start'] = metrics_dict[
#                     'initial acc per task (only classes in task)'
#                 ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
#                 R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 print("Accuracy matrix, based on only classes in that task ('as if Task-IL scenario')")
#                 print(R)

#                 # Accuracy matrix, always based on all classes
#                 R = pd.DataFrame(data=metrics_dict['acc per task (all classes)'],
#                                  index=['after task {}'.format(i + 1) for i in range(args.tasks)])
                
#                 R.loc['at start'] = metrics_dict[
#                     'initial acc per task (only classes in task)'
#                 ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
#                 R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 print("\nAccuracy matrix, always based on all classes")
#                 print(R)

#                 # Accuracy matrix, based on all classes thus far
#                 R = pd.DataFrame(data=metrics_dict['acc per task (all classes up to trained task)'],
#                                  index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#                 print("\nAccuracy matrix, based on all classes up to the trained task")
#                 print(R)
                
                

#             # Accuracy matrix, based on all classes up to the task being evaluated
#             # (this is the accuracy-matrix used for calculating the metrics in the Class-IL scenario)
#             R = pd.DataFrame(data=metrics_dict['acc per task (all classes up to evaluated task)'],
#                              index=['after task {}'.format(i + 1) for i in range(args.tasks)])
#             R.loc['at start'] = metrics_dict[
#                 'initial acc per task (only classes in task)'
#             ] if not args.use_exemplars else ['NA' for _ in range(args.tasks)]
#             R = R.reindex(['at start'] + ['after task {}'.format(i + 1) for i in range(args.tasks)])
#             BWTs = [(R.loc['after task {}'.format(args.tasks), 'task {}'.format(i + 1)] - \
#                      R.loc['after task {}'.format(i + 1), 'task {}'.format(i + 1)]) for i in range(args.tasks-1)]
#             FWTs = [0. if args.use_exemplars else (
#                 R.loc['after task {}'.format(i+1), 'task {}'.format(i+2)] - R.loc['at start', 'task {}'.format(i+2)]
#             ) for i in range(args.tasks-1)]
#             forgetting = []
#             for i in range(args.tasks - 1):
#                 forgetting.append(max(R.iloc[1:args.tasks, i]) - R.iloc[args.tasks, i])
#             R.loc['FWT (per task)'] = ['NA'] + FWTs
#             R.loc['BWT (per task)'] = BWTs + ['NA']
#             R.loc['F (per task)'] = forgetting + ['NA']
#             BWT = sum(BWTs) / (args.tasks-1)
#             F = sum(forgetting) / (args.tasks-1)
#             FWT = sum(FWTs) / (args.tasks-1)
#             metrics_dict['BWT'] = BWT
#             metrics_dict['F'] = F
#             metrics_dict['FWT'] = FWT
#             # -print on screen
#             if verbose:
#                 print("\nAccuracy matrix, based on all classes up to the evaluated task")
#                 print(R)
#                 print("\n=> FWT = {:.4f}".format(FWT))
#                 print("=> BWT = {:.4f}".format(BWT))
#                 print("=>  F = {:.4f}\n".format(F))

    if verbose and args.time:
        print("=> Total training time = {:.1f} seconds\n".format(training_time))

          
if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -run experiment
    run(args, verbose=True)
