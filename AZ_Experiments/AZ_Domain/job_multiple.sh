#!/bin/sh

echo $'############ START ############'



CUDA_VISIBLE_DEVICES=2 python IFS_Final.py --contamination 0.1

CUDA_VISIBLE_DEVICES=2 python IFS_Final.py --contamination 0.3

CUDA_VISIBLE_DEVICES=2 python IFS_Final.py --contamination 0.4

CUDA_VISIBLE_DEVICES=2 python IFS_Final.py --contamination 0.5




# CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --num_exps 2 --num_epoch 500 --batch_size 6000 --replay_portion 0.00

#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 1 ###############'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 0 --num_epoch 500 --batch_size 6000


#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 2 ###############' 
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 1 --num_epoch 500 --batch_size 6000

# now="$(date)"
# printf "Current date and time %s\n" "$now"
# echo $'###########  TASK 3 ###############'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 2 --num_epoch 500 --batch_size 6000

#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 4 ###############'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 3 --num_epoch 500 --batch_size 6000

# now="$(date)"
# printf "Current date and time %s\n" "$now"
# echo $'###########  TASK 5 ###############'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 4 --num_epoch 500 --batch_size 6000

#now="$(date)"
#echo $'###########  TASK 6 ###############'
#printf "Current date and time %s\n" "$now"
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 5 --num_epoch 500 --batch_size 6000

#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 7 ###############'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 6 --num_epoch 500 --batch_size 6000

#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 8 ###############'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 7 --num_epoch 500 --batch_size 6000

#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 9 ###############'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 8 --num_epoch 500 --batch_size 6000

#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 10 ###############'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 9 --num_epoch 500 --batch_size 6000

#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 11 ###############'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 10 --num_epoch 500 --batch_size 6000

#now="$(date)"
#printf "Current date and time %s\n" "$now"
#echo $'###########  TASK 12 ###############'
#CUDA_VISIBLE_DEVICES=0,1,2,3 python continual_ember.py --task_month 11 --num_epoch 500 --batch_size 6000


