#!/bin/sh


NUM_TASK=11
SCENARIO=class
DATASET=ember
GPU_NUMBER=1
NUM_REPLAY_SAMPLE=500
REPLAY_PORTION=1.0
REPLAY_CONFIG=grs

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START GRS ############'
counter=1
while [ $counter -le 1 ]
do
echo done w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --scenario=${SCENARIO} --replay=offline --replay_portion=${REPLAY_PORTION} --replay_config=${REPLAY_CONFIG} --num_replay_sample=${NUM_REPLAY_SAMPLE}
done
echo All done
