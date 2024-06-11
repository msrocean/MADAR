#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="$4"              # Name for your job
#SBATCH --comment="CL Runs"         # Comment for your job

#SBATCH --account=conlearn              # Project account to run your job under
#SBATCH --partition=tier3               # Partition to run your job on

#SBATCH --output=logs/%x_%j.out              # Output file
#SBATCH --error=logs/%x_%j.err               # Error file

#SBATCH --mail-user=mr6564@rit.edu	# Email address to notify
#SBATCH --mail-type=END                 # Type of notification emails to send

#SBATCH --time=0-2:59:59               # Time limit
#SBATCH --nodes=1                       # How many nodes to run on
#SBATCH --ntasks=1                      # How many tasks per node
#SBATCH --cpus-per-task=1               # Number of CPUs per task
#SBATCH --mem-per-cpu=30g               # Memory per CPU

#SBATCH --gres=gpu:v100:1


source ~/anaconda/etc/profile.d/conda.sh
conda activate PyTorch

# ITERS=10000
# NUM_TASK=11
# SCENARIO=class
# DATASET=AZ
# MEMORY_BUDGET=4500
# REPLAY_CONFIG=grs
# --min_samples="$6"


now="$(date)"
printf "Current date and time %s\n" "$now"
echo $'##### START' "$1" "$2" "$3" $'#####'

for counter in {1..1}
do
    echo Start w/ $counter time 
    python main.py --metrics --replay_config="$1" --ifs_option="$2" --memory_budget="$3"
    echo done w/ $counter time
    
done
echo All done
conda deactivate
EOT