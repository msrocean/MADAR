#!/bin/sh


NUM_TASK=20
SCENARIO=task
DATASET=ember
GPU_NUMBER=3
NUM_REPLAY_SAMPLE=200
REPLAY_PORTION=0.50
REPLAY_CONFIG=ifs
CNT_RATE=0.5


now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START IFS ############'
counter=1
while [ $counter -le 1 ]
do
echo done w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --metrics --scenario=${SCENARIO} --replay=offline --replay_portion=${REPLAY_PORTION} --replay_config=${REPLAY_CONFIG} --num_replay_sample=${NUM_REPLAY_SAMPLE} --cnt_rate=${CNT_RATE}
done
echo All done
