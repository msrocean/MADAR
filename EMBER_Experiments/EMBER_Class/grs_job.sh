#!/bin/sh


#python main.py --metrics --scenario=task --replay=offline --replay_portion=1.0 --replay_config=grs --num_replay_sample=500 --iters=2


ITERS=1
NUM_TASK=20
SCENARIO=task
DATASET=ember
GPU_NUMBER=2
NUM_REPLAY_SAMPLE=500
REPLAY_PORTION=1.0
REPLAY_CONFIG=grs


now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'############ START GRS ############'
counter=1
while [ $counter -le 1 ]
do
echo Start w/ $counter time 
((counter++))
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python main.py --iters=${ITERS} --metrics --scenario=${SCENARIO} --replay=offline --replay_portion=${REPLAY_PORTION} --replay_config=${REPLAY_CONFIG} --num_replay_sample=${NUM_REPLAY_SAMPLE}
echo done w/ $counter time
((counter++))
done
echo All done
