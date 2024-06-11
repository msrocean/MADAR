#!/bin/sh


NUM_TASK=20
SCENARIO=task
DATASET=ember
GPU_NUMBER=3
NUM_REPLAY_SAMPLE=200
REPLAY_PORTION=0.50
REPLAY_CONFIG=frs

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START FRS ############'
counter=1
while [ $counter -le 40 ]
do
echo done w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --metrics --scenario=${SCENARIO} --replay=offline --replay_portion=${REPLAY_PORTION} --replay_config=${REPLAY_CONFIG} --num_replay_sample=${NUM_REPLAY_SAMPLE}
done
echo All done
