#!/bin/sh

GPU_NUMBER=2
SCENARIO=Domain
REPLAY_CONFIG=IFS
NUM_REPLAY_SAMPLE=500

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'----------' ${SCENARIO} $'----------'
echo $'##### START' ${REPLAY_CONFIG} ${NUM_REPLAY_SAMPLE} $'#####'

counter=1
while [ $counter -le 2 ]
do
echo start w/ $counter time 
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python IFS_Final.py --num_replay_sample=${NUM_REPLAY_SAMPLE}
echo done w/ $counter time
((counter++))
done
echo All done