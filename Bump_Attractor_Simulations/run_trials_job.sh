#!/bin/bash

#  run_trials_job.sh
#  
#
#  Created by John on 03/12/2020.
#  
# Grid Engine options (lines prefixed with #$)
#$ -N run_trials_job
#$ -cwd
#$ -l h_vmem=24G
## $ -l h_rt=48:00:00
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
## $ -l h_rt=72:00:00
#  runtime limit of 72 hours: -l h_rt
#  memory limit of 128 Gbyte: -l h_vmem

vmem=16000000000 # Available RAM in Bytes
 
# Defaults for optional parameters
VERSION="eddie02"
NOISE="2.3"
TAU_MEM="-1"   # Default is an invalid number that is ignored by scripts
SKEW_W="0.05"  # results in 180deg shift per ~10s
SKEW_W="0.005" # results in 180deg shift per ~??s

# Usage string
USAGE=$(cat <<- EOM
USAGE
    `basename $0` <MODEL> <NEURONS> <DURATION> <TRIALS> [NOISE] [VERSION] [TAU_M] [SKEW_W]

     MODEL      : The model to use: 'NMDA', 'EC_LV_1', 'SIMPLE', 'NMDA-TAU', 'NMDA-SHIFT', 'full', or 'reduced' network
     NEURONS    : The number of excitatory neurons
     DURATION   : The duration of the simulation in seconds
     TRIALS     : The number of trials to run
     NOISE      : The amount of neuronal noise (optional, default 2.3)
     VERSION    : The version code to use in the filenames (optional)
     TAU_M      : The neuronal membrane time constant in ms (optional, not used by all model, default -1 an invalid value that is ignored)
     SKEW_W     : The amount of skewing of the Gaussian distribution of excitatory synaptic weights (optional, only used by the NMDA-SHIFT model, and if provided both TAU_M and SKEW_W must be provided even though the TAU_M will be ignored but needed for the positional variable assignment. Default 0.05: small skewing towards the right)
EOM
)

# Check if the required arguments were given
if [ "$#" -lt "4" ]; then
    echo "ERROR"
    echo "  Expected at least 4 arguments."
    echo
    echo "${USAGE}"
    exit $E_BADARGS
fi

MODEL=${1}               # $1 : Model to use: full or reduced network
NEURONS=${2}             # $2 : the number of excitatory neurons
DURATION=${3}            # $3 : the duration of the simulation
TRIALS=${4}              # $4 : the number of trials to run
NOISE=${5:-${NOISE}}     # $5 : the amount of neuronal noise (optional, default 2.3)
VERSION=${6:-${VERSION}} # $6 : the version code to use in the filenames (optional)
TAU_M=${7:-${TAU_MEM}}   # $7 : the neuronal membrane time constant in ms (optional, only used by the SIMPLE model)
SKEW_W=${8:-${SKEW_W}}   # $8 : the amount of skewing of the Gaussian distribution of excitatory synaptic weights that causes shift of the activity bump (optional, only used by the NMDA-SHIFT model, and if provided both TAU_M and SKEW_W must be provided even though the TAU_M will be ignored but needed for the positional variable assignment. Default 0.05: small skewing towards the right)

# Initialise the environment modules
. /etc/profile.d/modules.sh

source ~/.bashrc

# Load Python
module load anaconda/5.3.1

conda activate Brian2

# Run the program
if [ "${MODEL}" == "reduced" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_reduced_2_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_reduced_2.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_reduced_2_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "full" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "full_1" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_1.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_1_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_EC_LV_Principal_Neurons_1.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_EC_LV_1_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "NMDA" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_NMDA_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_NMDA_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

# The membrane time constant is changed using capacitance = tau * conductance
if [ "${MODEL}" == "NMDA-TAU" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} --tau_m ${TAU_M} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_NMDA-TAU_duration${DURATION}s_tau${TAU_M}ms_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" --tau_m "${TAU_M}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_NMDA-TAU_duration${DURATION}s_tau${TAU_M}ms_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi


if [ "${MODEL}" == "SIMPLE" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} --tau_m ${TAU_M} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_SIMPLE_duration${DURATION}s_tau${TAU_M}ms_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" --tau_m "${TAU_M}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_SIMPLE_duration${DURATION}s_tau${TAU_M}ms_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

# Unlike "SIMPLE" that uses conductance = capacitance / tau this uses capacitance = tau * conductance
if [ "${MODEL}" == "SIMPLE-TAU2" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} --tau_m ${TAU_M} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_SIMPLE-TAU2_duration${DURATION}s_tau${TAU_M}ms_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" --tau_m "${TAU_M}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_SIMPLE-TAU2_duration${DURATION}s_tau${TAU_M}ms_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi

if [ "${MODEL}" == "NMDA-SHIFT" ]; then
   echo    "Running: python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N ${NEURONS} -t ${TRIALS} -D ${DURATION} --neuronal_noise_Hz ${NOISE} --weights-skewness ${SKEW_W} -a ${vmem} -f /exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_NMDA-SHIFT${SKEW_W}_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
   python3 ./run_trials-simplified-neurons_NMDA_Neurons.py -N "${NEURONS}" -t "${TRIALS}" -D "${DURATION}" --neuronal_noise_Hz "${NOISE}" --weights-skewness "${SKEW_W}" -a "${vmem}" -f "/exports/eddie/scratch/s0093128/Data/collected_drift_trials_all_NMDA-SHIFT${SKEW_W}_duration${DURATION}s_noise${NOISE}Hz_v${VERSION}_${NEURONS}.npy"
fi
