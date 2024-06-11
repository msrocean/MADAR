#!/bin/bash




#SBATCH --job-name=cAWS1K              # Name for your job
#SBATCH --comment="CL Runs"         # Comment for your job

#SBATCH --account=conlearn              # Project account to run your job under
#SBATCH --partition=tier3               # Partition to run your job on

#SBATCH --output=logs/%x_%j.out              # Output file
#SBATCH --error=logs/%x_%j.err               # Error file

#SBATCH --mail-user=mr6564@rit.edu	# Email address to notify
#SBATCH --mail-type=END                 # Type of notification emails to send

#SBATCH --time=0-1:59:59               # Time limit
#SBATCH --nodes=1                       # How many nodes to run on
#SBATCH --ntasks=1                      # How many tasks per node
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem-per-cpu=20g               # Memory per CPU

#SBATCH --gres=gpu:v100:1


source ~/anaconda/etc/profile.d/conda.sh
conda activate PyTorch


ITERS=10000
ANOMALY_PERCT=0.5
NUM_TASK=11
SCENARIO=class
DATASET=ember
NUM_REPLAY_SAMPLE=1000
REPLAY_PORTION=1.0
REPLAY_CONFIG=aws
CNT_RATE=0.1


now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'----------' ${SCENARIO} $'----------'
echo $'##### START' ${REPLAY_CONFIG} $'#####'
#echo ${SCENARIO}
counter=1
while [ $counter -le 1 ]
do
echo start w/ $counter time 
python main.py --scenario=${SCENARIO} --iters=${ITERS} --replay=none --replay_portion=${REPLAY_PORTION} --replay_config=${REPLAY_CONFIG} --num_replay_sample=${NUM_REPLAY_SAMPLE} --cnt_rate=${CNT_RATE} --anomaly_perct=${ANOMALY_PERCT}
echo done w/ $counter time
((counter++))
done
echo All done


