#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=FM-"$5-$6"              # Name for your job
#SBATCH --comment="CL Runs"         # Comment for your job

#SBATCH --account=conlearn              # Project account to run your job under
#SBATCH --partition=tier3               # Partition to run your job on

#SBATCH --output=logs/%x_%j.out              # Output file
#SBATCH --error=logs/%x_%j.err               # Error file

#SBATCH --mail-user=mr6564@rit.edu	# Email address to notify
#SBATCH --mail-type=END                 # Type of notification emails to send

#SBATCH --time=0-13:59:59               # Time limit
#SBATCH --nodes=1                       # How many nodes to run on
#SBATCH --ntasks=1                      # How many tasks per node
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem-per-cpu=30g               # Memory per CPU

#SBATCH --gres=gpu:v100:1

# 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 10K, 15K, 20K, M-7K-20

# $1: scenario; $2: replay config, $3: ifs option; $4: iters; $5: memory budget; $6: min samples 

source ~/anaconda/etc/profile.d/conda.sh
conda activate PyTorch
now=$(date)
printf "Current date and time %s\n" "$now" 
echo $'##### START' "$1" "$2" "$3" "$5" $'#####'
counter=1
while [[ "$counter" -le 1 ]]
do
echo Start w/ ${counter} time 
python main.py --metrics --scenario="$1" --replay_config="$2" --ifs_option="$3" --iters=$4 --memory_budget=$5 --min_samples=$6
echo done w/ $counter time
((counter++))
done
echo All done
conda deactivate
EOT