#!/bin/sh

GPU_NUMBER=1
SCENARIO=Domain
REPLAY_CONFIG=GRS
REPLAY_PORTION=1.0

now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'----------' ${SCENARIO} $'----------'
echo $'##### START' ${REPLAY_CONFIG} $'#####'

counter=1
while [ $counter -le 4 ]
do
echo start w/ $counter time 
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python GRS.py --replay_portion=${REPLAY_PORTION}
echo done w/ $counter time
((counter++))
done
echo All done


