#!/bin/sh


NUM_TASK=12
SCENARIO=class
DATASET=ember
GPU_NUMBER=2
NUM_REPLAY_SAMPLE=500

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START ER ############'
counter=1
while [ $counter -le 10 ]
do
echo done w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --metrics --scenario=${SCENARIO} --replay=offline --replay_config ifs --num_replay_sample=${NUM_REPLAY_SAMPLE}
done
echo All done
