# Attractor Based Memory Plaussibility Study


## Files and directories:

`README.md` This file.

`neurodynex/` Contains the the libraries from the Neuronal Dynamics book from which I am using the model of ring attractor. It also includes my modified version of the ring attractor used in the various experiments here. 

`Python_Libs/` Contains useful python libraries.

`Data/` The simulation results are stored in this directory.

`optimize_synaptic_conductances.py` This script was used to optimise the conductances of the attractor network. 

`run_trials_job.sh` The script starting the data collection job. It is made for running on the Eddie grid but can also be run on other machines. 

`merge_all_files_into_one_keep_only_thetas.py` Process the collected data files (.npy) extract the theta time series from the population activity and store only the theta and some other useful data disregarding the individual neuronal activity to save space. 

`Bump_attractor_experiments_analysis.ipynb` The plots for the paper. 



## Examples of how to collect and process the bump attractor simulation data

### General description of data collection utility usage

This will start a job on the computing cluster:

`run_trials_job.sh <MODEL> <NEURONS> <DURATION> <TRIALS> [NOISE] [VERSION] [TAU_M] [SKEW_W]`

where

```
     MODEL      : The model to use: 'NMDA', 'EC_LV_1', 'SIMPLE', 'NMDA-TAU', 'NMDA-SHIFT', 'full', or 'reduced' network. 'NMDA' was used to run the base model simulations, 'EC_LV_1' was used to run the simulations with the non-specific cation conductances, 'NMDA-TAU', was used for running simulations with different time constant values, and 'NMDA-SHIFT' was used for running simulations with asymmetric latteral excitation to produce biased drift of the attractor state. 
     
     NEURONS    : The number of excitatory neurons
     
     DURATION   : The duration of the simulation in seconds
     
     TRIALS     : The number of trials to run
     
     NOISE      : The amount of neuronal noise (optional, default 2.3)
     
     VERSION    : The version code to use in the filenames (optional)
     
     TAU_M      : The neuronal membrane time constant in ms (optional, not used by all models, default -1 an invalid value that is ignored)
     
     SKEW_W     : The amount of skewing of the Gaussian distribution of excitatory synaptic weights (optional, only used by the NMDA-SHIFT model, and if provided both TAU_M and SKEW_W must be provided even though the TAU_M will be ignored but needed for the positional variable assignment. Default 0.05: small skewing towards the right)
```




## Collect data with the base neuron model

### Collect data simulating the attractor with NMDA neuron model for 300s. Simulations are run in batches of 5 trials. 

> for size in 128 256 512 1024 2048 4096 8192; do duration=300; N=5; noise=0.005; for i in `seq 100 160`; do qsub ./run_trials_job.sh NMDA $size $duration $N $noise eddie$i 0; done; done

#### Extract the population vector theta time series and store them in one file collected-wrapped-NMDA-N.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 1 -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-wrapped-NMDA-N.npy

#### Extract the population vector theta time series, unwrap them, and store them in one file collected-unwrapped-NMDA-N.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --unwrap-angles -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-unwrapped-NMDA-N.npy



### Collect data simulating the attractor with NMDA neuron model for 300s and different noise levels. Simulations are run in batches of 5 trials. 

> for noise in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009; do  size=256; duration=300; N=5; noise=0.005; for i in `seq 100 160`; do qsub ./run_trials_job.sh NMDA $size $duration $N $noise eddie$i 0; done; done

#### Extract the population vector theta time series and store them in one file collected-wrapped-NMDA-noise.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 1 -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-wrapped-NMDA-noise.npy

#### Extract the population vector theta time series, unwrap them, and store them in one file collected-unwrapped-NMDA-noise.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --unwrap-angles -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-unwrapped-NMDA-noise.npy



### Collect data for different membrane time constant values.

#### The NMDA model with different tau values (manipulating tau by changing the capacitance instead of the conductance).
> noise=1.4; do size=256; N=5; for tau in 0.5 1 5 10 20 30 40 50 60 70 80 90 100; do for i in `seq 100 160`; do qsub ./run_trials_job.sh NMDA-TAU $size 300 $N $noise eddie$i $tau; done; done

#### Extract the population vector theta time series, and store them in one file collected-wrapped-NMDA-TAU.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 2 -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-wrapped-NMDA-TAU.npy

#### Extract the population vector theta time series, unwrap them, and store them in one file collected-unwrapped-NMDA-TAU.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 2 --unwrap-angles -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-unwrapped-NMDA-TAU.npy



### Collect data with systematic bump shifting.

#### Collect with NMDA-SHIFT-0.001 and NMDA-SHIFT-0.0005 for 300s (--weights-skewness 0.05 results in 180deg shift per ~8.6s (20.96deg/s))

> NMDA_SHIFTING=-0.0005; for noise in 0.005; do size=256; N=5; for i in `seq 100 160`; do qsub ./run_trials_job_64G.sh NMDA-SHIFT $size 300 $N $noise eddie$i 0 ${NMDA_SHIFTING}; done; done
> NMDA_SHIFTING=-0.001; for noise in 0.005; do size=256; N=5; for i in `seq 100 160`; do qsub ./run_trials_job_64G.sh NMDA-SHIFT $size 300 $N $noise eddie$i 0 ${NMDA_SHIFTING}; done; done

#### Extract the population vector theta time series, and store them in one file collected-wrapped-NMDA-SHIFT.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 3 -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-wrapped-NMDA-SHIFT.npy

#### Extract the population vector theta time series, unwrap them, and store them in one file collected-unwrapped-NMDA-SHIFT.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 3 --unwrap-angles -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-unwrapped-NMDA-SHIFT.npy



## Collect data with the persistent graded activity neuron model

### Collect data simulating the attractor with EC_LV_1 neuron model for 300s. Simulations are run in batches of 5 trials. 

> for size in 128 256 512 1024 2048 4096 8192; do duration=300; N=5; noise=0.005; for i in `seq 100 160`; do qsub ./run_trials_job.sh EC_LV_1 $size $duration $N $noise eddie$i 0; done; done

#### Extract the population vector theta time series and store them in one file collected-wrapped-EC_LV_1-N.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --filename-template 1 -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-wrapped-EC_LV_1-N.npy

#### Extract the population vector theta time series, unwrap them, and store them in one file collected-unwrapped-EC_LV_1-N.npy
> python3 merge_all_files_into_one_keep_only_thetas.py --unwrap-angles -i /PATH_TO_DATA/ -o /PATH_TO_DATA/collected-unwrapped-EC_LV_1-N.npy




## Data analysis

The script `Bump_attractor_experiments_analysis.ipynb` is used to process the data and produce the plots in Figures 2A, 2C, 2D, 3A, 3B, 4A, 4C, and 4D.

