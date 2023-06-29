#!/bin/bash
#SBATCH -J script_test # name
#SBATCH --account=hallmark
#SBATCH --partition=normal_q # scheduling policy
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=5-00:00:00 # days-hours:minutes:seconds
#SBATCH --exclusive

# Bash script used to schedule jobs in Slurm.

# Test print
echo "hello world from..."
hostname
# moved the below line for a quick test
# Reset modules & load Anaconda
echo "reseting module..."
module reset
module load Anaconda3

# Source the conda setup script
echo "loading python environment..."
source "${EBROOTANACONDA3}"/etc/profile.d/conda.sh
# Activate punch environment (may not be necessary)
conda activate punch
echo "  $CONDA_DEFAULT_ENV"

# Run python script.
echo "running python script..."
# 4 parts to this line:
#   - First is srun and srun options
#   - Next is the Python interpreter. Make sure you point at the right python env manually; don't just prepend "python" to the command. Activating `punch` above is not sufficient. 
#   - Next is the Python script to use.
#   - Last is the argument to the Python script, which is the config file (.json)
PYDIR=`which python`
echo "python directory: $PYDIR"
PWDDIR=${PWD}
echo "working directory: $PWDDIR"
# NEWDIR=$PWDDIR"/.conda/envs/punch/bin/python"
# echo "new directory: $NEWDIR"

echo "config file path: $1"
# /home/dylanrpenn/.conda/envs/punch/bin/python punchclock/training_scripts/run_tune_script.py $1
srun --cpus-per-task=128 /home/dylanrpenn/.conda/envs/punch/bin/python punchclock/training_scripts/run_tune_script.py $1

# To run file:
#   1. Make sure your working directory is the repo
#   2. Enter the following into bash terminal: sbatch punchclock/training_scripts/bash_script_test.sh
# 
# You can also use `srun` instead of `sbatch`.
# There are lots of optional arguments to add to sbatch and srun.
