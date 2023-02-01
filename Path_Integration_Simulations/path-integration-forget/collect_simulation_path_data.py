# collect_simulation_path_data.py
# Python script to use to run simulations and collect simulated path data.

# USAGE EXAMPLES:

# Collect memory manipulation data with noise_syn=0.1 and noise_turn=7 noise_slope=9. 
# --------------------------------------------------------------------------------------------------------------------

# Collected outbound paths using 
# iter=001; ve=Generate_outbound_route; noise_slope=9.0; noise_syn=0.1; noise_turn=7.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; python collect_simulation_path_data.py ${ve} SAVE SHOW:SAVE "${iter}" "${noise_syn}" "${noise_turn}" DEFAULT "${noise_slope}"
# outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.npz straight north.
# outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.png
# with_Pontin_Holonomic_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_002.npz east a bit down
# all_trajectories_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_002.png
# with_Pontin_Holonomic_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_003.npz south west
# all_trajectories_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_003.png


# Simulated FV agent paths with noise added to the memory
# The distance memory state of a bump attractor network follows a stochastic diffusion over time. This is what must happen while the ants wait for release. We simulate the stochastic diffusion of the distance memory as a random walk (Wiener process or 1D Brownian motion) during a wait period. Random walk with Gaussian noise mu=0 and std=0.0055 results in a diffusion rate 0.0342m/h which is the closest match to the ants' 0.034m/h. To simplify the computational complexity we substitute the full random walk simulations with the distribution of the final locations of random walks. This distribution is a function of the std=0.0055 used to generate the value random walk and the duration of the wait before release <duration>. The distribution of the final locations of the random walks is again a Gaussian with mu=0 and std_2=sqrt(duration)*std=sqrt(duration)*0.0055.
# Used wait_duration_hours = np.array([1, 24, 48, 96]) # hours waiting before release
# rand_walk_std=0.0055
# mem_noise_std = np.sqrt(wait_duration_hours * 60 * 60) * rand_walk_std
# mem_noise_std = mem_noise_std / 128 # Scaled in the context of the physical space represented in memory (max 128m) = [0.00257812, 0.01263018, 0.01786177, 0.02526036].

# The homing paths are quite noisy and cover the effect of memory noise
# Simulated FV agent paths without and with stochastic diffusion of memory before release
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=7.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem in DEFAULT n0.00257812 n0.01263018 n0.01786177 n0.02526036; do for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python collect_simulation_path_data.py ${ve}_release SAVE:LOAD:outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.npz DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done


# Trying with less memory noise
# Simulated FV agent paths without and with stochastic diffusion of memory before release
# outbound_file=outbound_route_only_NE_to_SW_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_003.npz

# outbound_file=outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.npz


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# For the paper: Can the insect path integration memory be a bump attactor? I used:
# outbound_file=outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.npz

# Collect FV agent paths with noise_syn=0.1 and noise_turn=7 noise_slope=9. 
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python3 collect_simulation_path_data.py ${ve}_release SAVE:LOAD:${outbound_file} DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" DEFAULT "${noise_slope}"; done

# Collect ZV agent paths with noise_syn=0.1 and noise_turn=7 noise_slope=9. 
# ve=ZV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python3 collect_simulation_path_data.py ${ve}_release SAVE DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" DEFAULT "${noise_slope}"; done

# n0.00257812 n0.01263018 n0.01786177 n0.02526036 n0.0309375 n0.03572355 n0.03994014 n0.04375223, n0.04725781, n0.05052073 correspond to accumulative drift due to noise after 1, 24, 48, 96, 144, 192, 240, 288, 336, 384 hours calculated as sqrt(wait_hours*60*60)*0.0055.
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem in DEFAULT n0.00257812 n0.01263018 n0.01786177 n0.02526036 n0.0309375 n0.03572355 n0.03994014 n0.04375223 n0.04725781 n0.05052073; do for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python3 collect_simulation_path_data.py ${ve}_release SAVE:LOAD:${outbound_file} DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done

# Move data files to individual directories
# noise_slope="9.0";
# noiseRot="2.0";
# noiseSyn="0.1";
# Conditions=();
# Conditions+=('ZV' 'FV');
# for i in 0.00257812 0.01263018 0.01786177 0.02526036 0.0309375 0.03572355 0.03994014 0.04375223, 0.04725781, 0.05052073; do
#     Conditions+=("FVIcen${i}");
# done;
# for cond in ${Conditions[@]}; do
#     mkdir -p data/Conditions/Memory/${cond}/ ;
#     for i in `seq 1001 1100`; do
#         mv data/with_Pontin_Holonomic_noiseSyn${noiseSyn}_noiseRot${noiseRot}_noiseSlope${noise_slope}_route_${cond}_${i}.npz data/Conditions/Memory/${cond}/ ;
#     done;
# done

# Simulated FV agent paths with memory decay following inverse logistic function fit to the ants homing distance decay over time. Did this for different waiting periods:
# 0h (l0), 1h (l1), 24h (l24), 48h (l48), 96h (l96), 144h (l144), 192h (l192), 240h (b240)
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem in l0 l1 l24 l48 l96 l144 l192 l240; do for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python collect_simulation_path_data.py ${ve}_release SAVE:LOAD:${outbound_file} DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done

# Move data files to individual directories
# noise_slope="9.0"; 
# noiseRot="2.0"; 
# noiseSyn="0.1"; 
# Conditions=();
# for i in 0 1 24 48 96 144 192 240; do
#     Conditions+=("FVWaitl${i}");
# done;
# for cond in ${Conditions[@]}; do 
#     mkdir -p data/Conditions/Memory/${cond}/ ; 
#     for i in `seq 1001 1100`; do 
#         mv data/with_Pontin_Holonomic_noiseSyn${noiseSyn}_noiseRot${noiseRot}_noiseSlope${noise_slope}_route_${cond}_${i}.npz data/Conditions/Memory/${cond}/ ; 
#     done; 
# done

# Simulated FV agent paths with memory decay following inverse logistic function fit to the ants homing distance decay over time plus additive noise accumulating over time creating diffusion dynamics. Did this for different waiting periods:
# 0h (b0), 1h (b1), 24h (b24), 48h (b48), 96h (b96), 144h (l144), 192h (b192), 240h (b240), 288h (b288), 336h (b336), 384h (b384)
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem_wait in 0 1 24 48 96 144 192 240 288 336 384; do for mem_noise in 0.000 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.010; do mem="b${mem_wait},${mem_noise}"; for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python collect_simulation_path_data.py ${ve}_release SAVE:LOAD:${outbound_file} DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done; done

# Move data files to individual directories
# noise_slope="9.0";
# noiseRot="2.0";
# noiseSyn="0.1";
# Conditions=();
# for h in 0.0 1.0 24.0 48.0 96.0 144.0 192.0 240.0 288.0 336.0 384.0; do
#     for n in 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01; do
#         Conditions+=("FVWait${h}hNoise${n}");
#     done;
# done;
# for cond in ${Conditions[@]}; do
#     mkdir -p data/Conditions/Memory/${cond}/ ;
#     for i in `seq 1001 1100`; do
#         mv data/with_Pontin_Holonomic_noiseSyn${noiseSyn}_noiseRot${noiseRot}_noiseSlope${noise_slope}_route_${cond}_${i}.npz data/Conditions/Memory/${cond}/ ;
#     done;
# done

# Simulated FV agent paths with memory decay following inverse logistic function fit to the ants homing distance decay over time plus additive noise accumulating over time creating diffusion dynamics. This version also gets two parameters (mem_Nl, mem_r) for the shape of the logistic function. Did this for different waiting periods:
# 0h (p0), 1h (p1), 24h (p24), 48h (p48), 96h (p96), 144h (p144), 192h (p192), 240h (p240), 288h (p288), 336h (p336), 384h (p384)
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem_wait in 0 1 24 48 96 144 192 240 288 336 384 432; do for mem_noise in 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.015 0.02; do for mem_Nl in 0.002 0.004 0.006  0.008  0.01 0.012 0.014 0.016 0.018 0.02; do for mem_r in -0.008 -0.010 -0.012 -0.014 -0.016 -0.018 -0.020  -0.022  -0.024 -0.026 -0.028 -0.030 -0.032; do mem="p${mem_wait},${mem_noise},${mem_Nl},${mem_r}"; for iter in $(seq 1001 1040); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python3 collect_simulation_path_data.py ${ve}_release SAVE:LOAD:${outbound_file} DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done; done; done; done

# I actually run this script instead tp spawn several longjobs in a server to run in parallel batches of the data collection:
# cat spawn_longjobs.sh
# #!/bin/bash
# echo "Spawning a set of longjobs in the computer..."
# for mem_noise in 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.015 0.02; do
# for mem_wait in 0 1 24 48 96 144 192 240 288 336 384 432; do
# (
# cat <<EOF
# #!/bin/bash
# # Starts longjob for collecting data with mem_wait=${mem_wait} and mem_noise=${mem_noise}...
# echo "Starts longjob for collecting data with mem_wait=${mem_wait} and mem_noise=${mem_noise}..."
# mem_wait=${mem_wait}
# mem_noise=${mem_noise}
# outbound_file=outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.npz
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0;
# echo "Noise synaptic   \${noise_syn}";
# echo "Noise locomotive \${noise_turn}";
# echo "Noise slope      \${noise_slope}";
# echo "=======================";
# for mem_Nl in 0.0 0.002 0.004 0.006 0.008 0.01 0.012 0.014 0.016 0.018 0.02  0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2; do
# for mem_r in -0.008 -0.010 -0.012 -0.014 -0.016 -0.018 -0.020  -0.022  -0.024 -0.026 -0.028 -0.030 -0.032; do
#     mem="p\${mem_wait},\${mem_noise},\${mem_Nl},\${mem_r}";
#     for iter in \$(seq 1001 1040); do
#         echo "Iteration           : \$iter";
#         echo "Memory              : \$mem";
#         python3 collect_simulation_path_data.py \${ve}_release SAVE:LOAD:\${outbound_file} DEFAULT "\${iter}" "\${noise_syn}" "\${noise_turn}" "\${mem}" "\${noise_slope}";
#     done;
# done;
# done
# echo "longjob for collecting data with mem_wait=${mem_wait} and mem_noise=${mem_noise} done."
# exit 0
# EOF
# ) > collect_data_longjob_${mem_wait}_${mem_noise}_$(uname -n).sh
# chmod u+x collect_data_longjob_${mem_wait}_${mem_noise}_$(uname -n).sh
# echo "Starting longjob for collecting data with mem_wait=${mem_wait} and mem_noise=${mem_noise}..."
# echo "Please, give password:"
# longjob -28day -c "./collect_data_longjob_${mem_wait}_${mem_noise}_$(uname -n).sh" >> log_file_${mem_wait}_${mem_noise}_$(uname -n).log
# done;
# done
# echo "Spawning done."
# exit 0
#
# Move data files to individual directories under directory
# cd /disk/scratch/ipisokas/data/
# OUTPUT_PATH=Conditions/Memory/
# noise_slope="9.0";
# noiseRot="2.0";
# noiseSyn="0.1";
# Conditions=();
# for h in 0.0 1.0 24.0 48.0 96.0 144.0 192.0 240.0 288.0 336.0 384.0 432.0; do
#   for n in 0.01; do
#     for mem_Nl in 0.002 0.004 0.006 0.008 0.01 0.012 0.014 0.016 0.018 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2; do
#       for mem_r in -0.008 -0.01 -0.012 -0.014 -0.016 -0.018 -0.02  -0.022  -0.024 -0.026 -0.028 -0.03 -0.032; do
#         Conditions+=("FVWait${h}hNoise${n}Nl${mem_Nl}r${mem_r}");
#       done;
#     done;
#   done;
# done;
# for cond in ${Conditions[@]}; do
#   mkdir -p "${OUTPUT_PATH}${cond}/" ;
#   for i in `seq 1001 1040`; do
#     mv with_Pontin_Holonomic_noiseSyn${noiseSyn}_noiseRot${noiseRot}_noiseSlope${noise_slope}_route_${cond}_${i}.npz "${OUTPUT_PATH}${cond}/" ;
#   done;
# done

# Then used this script to convert npz files to csv files containing only the trajectories and deleted the npz files
# python3 npz_to_csv.py -i=/disk/data/ipisokas/data/Conditions/Memory/ -o=/disk/data/ipisokas/data/Converted_to_CSV/Conditions/Memory/

# Actually in the new collections I used the scripts spawn_data_collection_longjobs.sh 
# for running on servers and spawn_data_collection_qjobs.sh for running on the eddie grid. 

# Then used this script to analyse the data
# python3 analyse_trajectories_publication_script.py /disk/data/ipisokas/data/Converted_to_CSV/Conditions/Memory/
# which produced the results file 'path-integration-forget/data/path_analysis_calculation_results_3parameters.npz'
# The data in this file will have the structure dict[wait_noise_sd_str][mem_Nl_str][mem_r_str][measure] = [list of values in order of waiting time]
# Note that I actually first transfered the csv files in the local external hard disk and then run
# python3 analyse_trajectories_publication_script.py /Volumes/WD\ Elements\ 25A3\ Media/Documents/Research/PhD/Projects/Recurrent_Net_Memory/Attractor_Based_Memory_Plaussibility_Study/Path_Integration_Simulations/path-integration-forget/data/outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot2.0_noiseSlope9.0_route_Generate_outbound_route_001/T_inbound3000steps/3_params_scanning_correct_Nl_range/Converted_to_CSV/Conditions/Memory/ path-integration-forget/data/path_analysis_calculation_results_3parameters_new_batch.npz
# python3 analyse_trajectories_publication_script.py /disk/data/ipisokas/data/eddie/Converted_to_CSV/Conditions/Memory/ path-integration-forget/data/path_analysis_calculation_results_3parameters_new_batch.npz 

# In practice since this processes all conditions in series it takes a lot of time so I run 
# the above script in batches one for each noise level. Then combined the 
# resulting files into one by using the notebook combine_dicts.ipynb which produced the file 
# 'path-integration-forget/data/path_analysis_calculation_results_3parameters.npz'

# Then used this script to calculate the Mean Squared Error (MSE) of the statistics
# and the actual ants behaviour.
# python analyse_trajectories_step2_script.py
# This script reads the file 'path-integration-forget/data/path_analysis_calculation_results_3parameters.npz'
# and stores the resulting MSE values to the file 'path-integration-forget/data/path_analysis_calculation_results_3parameters_MSE_values.npz'
# The data in this file will have the structure dict[measure][wait_noise_sd_str][mem_Nl_str][mem_r_str] = MSE value
# I actually used two different values for the variable slice_t_max that determines the number of waiting times to include in the MSE calculations, 
# the default first 8 waiting times which are the same as in the Ziegler1997 paper which produced 
# the results file path_analysis_calculation_results_3parameters_MSE_values_slice_t_max_8.npz
# and with keeping all 11 waiting times which extend beyond what Ziegler1997 reports which produced 
# the results file path_analysis_calculation_results_3parameters_MSE_values_slice_t_max_11.npz

# The files 'path_analysis_calculation_results_3parameters.npz' and 
# 'path_analysis_calculation_results_3parameters_MSE_values_slice_t_max_8.npz' and 
# 'path_analysis_calculation_results_3parameters_MSE_values_slice_t_max_11.npz' were moved to 
# 'path-integration-forget/data/3_parameters_results_correct_Nl_range_01Dec2022/'
# Then they were analysed with the scripts visualize_n_minimize.ipynb and analyse_trajectories_publication.ipynb
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


# Collect ZV agent paths with noise_syn=0.1 and noise_turn=7 noise_slope=9. 
# --------------------------------------------------------------------------------------------------------------------
# ve=ZV; noise_slope=9.0; for noise_syn in 0.1; do for noise_turn in 7.0; do echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; python collect_simulation_path_data.py ${ve}_release SAVE DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" DEFAULT "${noise_slope}"; done; done; done

# Collect FV agent paths with noise_syn=0.1 and noise_turn=7 noise_slope=9. 
# --------------------------------------------------------------------------------------------------------------------
# ve=FV; noise_slope=9.0; for noise_syn in 0.1; do for noise_turn in 7.0; do echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; python collect_simulation_path_data.py ${ve}_release SAVE:LOAD:outbound_route_only_NW_to_SE_1500steps_03.npz DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" DEFAULT "${noise_slope}"; done; done; done


# Collect memory manipulation data with noise_syn=0.1 and noise_turn=7 noise_slope=9. 
# --------------------------------------------------------------------------------------------------------------------

# Simulated FV agent paths with proportional manipulation of the memory
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=7.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem in DEFAULT \=0.5 x0.0 x0.05 x0.1 x0.15 x0.2 x0.25 x0.3 x0.35 x0.4 x0.45 x0.5 x0.55 x0.6 x0.65 x0.7 x0.75 x0.8 x0.85 x0.9 x0.95 x1.0; do for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python collect_simulation_path_data.py ${ve}_release SAVE:LOAD:outbound_route_only_NW_to_SE_1500steps_03.npz DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done

# Simulated FV agent paths with subtractive manipulation of the memory
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=7.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem in -0.0 -0.05 -0.1 -0.15 -0.2 -0.25 -0.3 -0.35 -0.4 -0.45 -0.5 -0.55 -0.6 -0.65 -0.7 -0.75 -0.8 -0.85 -0.9 -0.95 -1.0; do for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python collect_simulation_path_data.py ${ve}_release SAVE:LOAD:outbound_route_only_NW_to_SE_1500steps_03.npz DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done

# Simulated FV agent paths with noise added to the memory
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=7.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem in n0.00 n0.01 n0.02 n0.03 n0.04 n0.05 n0.06 n0.07 n0.08 n0.09 n0.1 n0.11 n0.12 n0.13 n0.14 n0.15 n0.16 n0.17 n0.18 n0.19 n0.2 n0.3 n0.4 n0.5 n0.6 n0.7 n0.8 n0.9 n1.0; do for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python collect_simulation_path_data.py ${ve}_release SAVE:LOAD:outbound_route_only_NW_to_SE_1500steps_03.npz DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done

# Move data files to individual directories
# noise_slope="9.0"; 
# noiseRot="7.0"; 
# noiseSyn="0.1"; 
# Conditions=(); 
# Conditions+=('ZV'); 
# Conditions+=('FV'); 
# Conditions+=("FVIce=0.5"); 
# for i in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do 
#     Conditions+=("FVIce-${i}"); 
# done; 
# for i in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do 
#     Conditions+=("FVIcex${i}"); 
# done;
# for i in 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do 
#     Conditions+=("FVIcen${i}"); 
# done;
# for cond in ${Conditions[@]}; do 
#     mkdir -p data/Conditions/Memory/${cond}/ ; 
#     for i in `seq 1001 1100`; do 
#         mv data/with_Pontin_Holonomic_noiseSyn${noiseSyn}_noiseRot${noiseRot}_noiseSlope${noise_slope}_route_${cond}_${i}.npz data/Conditions/Memory/${cond}/ ; 
#     done; 
# done

# Simulated FV agent paths with Gaussian noise added to the memory to emulate the effect of bump attractor error accumulation during different waiting periods:
# 0h (n0.0), 1h (n0.00257812), 24h (n0.01263018), 48h (n0.01786177), 96h (n0.02526036), 192h (n0.03572355)
# ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0; echo "Noise synaptic   ${noise_syn}"; echo "Noise locomotive ${noise_turn}"; echo "Noise slope      ${noise_slope}"; echo "======================="; for mem in DEFAULT n0.0 n0.00257812 n0.01263018 n0.01786177 n0.02526036 n0.03572355; do for iter in $(seq 1001 1100); do echo "Iteration           : $iter"; echo "Memory              : $mem"; python collect_simulation_path_data.py ${ve}_release SAVE:LOAD:${outbound_file} DEFAULT "${iter}" "${noise_syn}" "${noise_turn}" "${mem}" "${noise_slope}"; done; done

# Compatibility between Python 2 and Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import central_complex
import cx_basic
import cx_rate
import trials
import analysis
import plotter

# For saving files on the big disk on Matthias' server
# if available use the big disk on Matthias' server
if os.path.isdir("/disk/scratch/ipisokas/tmp"):
    os.environ["TMP"] = "/disk/scratch/ipisokas/tmp"
    os.environ["TMPDIR"] = "/disk/scratch/ipisokas/tmp"
    os.environ["TEMPDIR"] = "/disk/scratch/ipisokas/tmp"
# Data paths
# if available store the data on the big disk on Matthias' server
if os.path.isdir('/disk/data/ipisokas/data'):
    data_path = '/disk/data/ipisokas/data'
elif os.path.isdir('/disk/scratch/ipisokas/data'):
    data_path = '/disk/scratch/ipisokas/data'
elif os.path.isdir('/exports/eddie/scratch/s0093128/data'):
    data_path = '/exports/eddie/scratch/s0093128/data'
else: # else use the subdir data in the working directory
    data_path = 'data'

def usage():
    print('')
    print('SYNTAX:')
    print('    python collect_simulation_path_data.py <MODE> <ROUTE> <FIGURE> <FILE_NUM> <SYN_NOISE> <MOTOR_NOISE> [FORGET]')
    print('    <MODE> one of:')
    print('           Generate_outbound_route')
    print('                 Generate only a random outbound route without homing.')
    print('           ZV_release')
    print('                 Simulate release of a Zero Vector ant.')
    print('           FV_release')
    print('                 Simulate release of a Full Vector ant.')
    print('    <ROUTE> any combination of:')
    print('           SAVE, LOAD or both, or DEFAULT (: is separator).')
    print('           LOAD')
    print('                 Load route from the subdir data/ It must be followed by the filename to load.')
    print('           SAVE')
    print('                 Save route on disk with standard filename pattern.')
    print('           DEFAULT')
    print('                 Do not load a route (a new route will be generated) and do not save it.')
    print('           eg LOAD:<ROUTE_FILENAME>.npz or SAVE:LOAD:<ROUTE_FILENAME>.npz')
    print('    <FIGURE> any combination of:')
    print('           SHOW, SAVE or both, or DEFAULT (: is separator), eg SHOW:SAVE')
    print('           SHOW')
    print('                 Show routes in figure on screen.')
    print('           SAVE')
    print('                 Save the route as figure on disk using standard filename pattern.')
    print('           DEFAULT')
    print('                 Do not show and don\'t save the route.')
    print('    <FILE_NUM>')
    print('           A string that is added in the saved route and figure file names, eg ZV_10')
    print('    <SYN_NOISE>')
    print('           The amound of activation function noise to use (real number).')
    print('    <MOTOR_NOISE>')
    print('           The amound of motor noise to add on agent turning (real number).')
    print('    [FORGET]')
    print('           Whether and in what way to affect the CPU4 activity level before homing.')
    print('           DEFAULT or REMEMBER')
    print('                 Do not alter the memory.')
    print('           <NUMBER>  eg 0, 0.0, 0.5.')
    print('                 Set all elements of CPU4 to this value.')
    print('           =<NUMBER> eg \\=0.5')
    print('                 Set each element of CPU4 to this value. Value is not cliped to [0,1].')
    print('                 Make sure to escape the = character because it has special meaning in UNIX shells.')
    print('           x<NUMBER> eg x0.5')
    print('                 Multiply each element of CPU4 by this value. Resulting values are cliped to [0,1].')
    print('           -<NUMBER> eg -0.5')
    print('                 Subtract from each element of CPU4 this value. Resulting values are cliped to [0,1].')
    print('           n<NUMBER> eg n0.5')
    print('                 Noisify each element of CPU4 by adding values drawn from the Gaussian distribution ')
    print('                 with mean 0 and standard deviation set to the given value. Resulting values are cliped to [0,1].')
    print('           l<NUMBER> eg l1')
    print('                 Loss of memory by reducing each element of CPU4 by a value depended on the given value (number of hours). Reduction is done following the inverse logistic function matching the homing distance loss in ants.  Resulting values are cliped to [0,1].')
    print('           b<NUMBER>,<NUMBER> eg b1,0.005')
    print('                 Loss of memory by reduction combined with addition of noise to each element of CPU4 depending on the given value (number of hours). The numbers specify in order <waiting time in hours>, <the diffusion coefficient>. Reduction is done following the inverse logistic function matching the homing distance loss in ants and noise is added from a Gaussian distribution introducing diffusion dynamics over time.  Resulting values are cliped to [0,1]. Eg b1,0.005 would reduce the CPU4 values by the inverse logistic function as if the agent was waiting before release for 1h and adding Gaussian noise with standard deviation 0.005 per second.')
    print('           p<NUMBER>,<NUMBER>,<NUMBER>,<NUMBER> eg b1,0.005,0.01,-0.02')
    print('                 The numbers specify in order <waiting time in hours>, <the diffusion coefficient>, <population loss parameter Nl>, <populaiton loss parameter r>. Loss of memory by reduction combined with addition of noise to each element of CPU4 depending on the given value (number of hours). Reduction is done following the inverse logistic function matching the homing distance loss in ants and noise is added from a Gaussian distribution introducing diffusion dynamics over time.  Resulting values are cliped to [0,1]. Eg b1,0.005,0.01,-0.02 would reduce the CPU4 values by the inverse logistic function as if the agent was waiting before release for 1h and adding Gaussian noise with standard deviation 0.005 per second.')
    print('    [MOTOR_NOISE_SLOPE]')
    print('           The slope of the exponential function used to shape the motor noise as function of CPU4 amplitude.')
    print('           Default 125.')
    print('')
    print("USAGE EXAMPLE:")
    print("noise_syn=0.1")
    print('for noise_turn in 0.0 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0; do ')
    print('    echo ; ')
    print('    echo "Noise synaptic   ${noise_syn}"; ')
    print('    echo "Noise locomotive ${noise_turn}"; ')
    print('    echo "======================="; ')
    print('    for iter in $(seq 10 29); do ')
    print('        echo ${iter}; ')
    print('        python collect_simulation_path_data.py ZV_release SAVE DEFAULT "ZV_${iter}" "${noise_syn}" "${noise_turn}"; ')
    print('    done; ')
    print('done')        

# Acceptable <MODE> values
modes = ['Generate_outbound_route', 'ZV_release', 'FV_release']

# Default values
show_figs=False
save_figs=False
load_route = False # If true instead of generating a new route it loads one from disk.
save_route = False

def reduce_set(value):
    """ Loss of memory setting to a value """
    def setter(x):
        out = np.ones(x.shape) * value
        return out
    return setter

def reduce_multiply(factor):
    """ Loss of memory multiplying by factor """
    def reducer(x):
        out = np.clip(x * factor, 0, 1)
        return out
    return reducer

def reduce_subtract(value):
    """ Loss of memory subtracting a value """
    def reducer(x):
        out = np.clip(x - value, 0, 1)
        return out
    return reducer

def reduce_additive_noise(value):
    """ Loss of memory by additive noise with standard deviation sigma=value """
    def reducer(x):
        noise = np.random.normal(loc=0.0, scale=value, size=x.shape)
        out = np.clip(x + noise, 0, 1)
        return out
    return reducer

def reduce_pop_loss(time_in_hours):
    """ Loss of memory following a logistic progression depended on waiting time """
    def reducer(x):
        # Fitted parameters from ants homing distance
        a = 1.0
        k = 0.00784362457
        r = 0.0220667871
        y = x * a / (1 + k * np.exp((r * time_in_hours)))
        out = np.clip(y, 0, 1)
        return out
    return reducer

# Data were collected using this function
def reduce_pop_loss_and_additive_noise(time_in_hours, base_sd=0.0055):
    """ Loss of memory following a logistic progression depended on waiting time and additive noise accumulating over time """
    def reducer(x):
        # The logistic reduction
        # Fitted parameters from ants homing distance
        a = 1.0
        k = 0.00784362457
        r = 0.0220667871
        y = x * a / (1 + k * np.exp((r * time_in_hours)))
        # The noise
        sd = np.sqrt(time_in_hours*60*60) * base_sd / 128
        noise = np.random.normal(loc=0.0, scale=sd, size=x.shape)
        out = np.clip(y + noise, 0, 1)
        return out
    return reducer

# This is an equivalent function to the above
def reduce_pop_loss_and_additive_noise(time_in_hours, base_sd=0.0055):
    """ Loss of memory following a logistic progression depended on waiting time and additive noise accumulating over time """
    def reducer(x):
        # The logistic reduction
        # Fitted parameters from ants homing distance
        K = 1.0
        Nl = 0.00771702
        r = -0.02206672
        y = x * K / (1 + Nl/(K - Nl) * np.exp(-r * time_in_hours))
        # The noise
        sd = np.sqrt(time_in_hours*60*60) * base_sd / 128
        noise = np.random.normal(loc=0.0, scale=sd, size=x.shape)
        out = np.clip(y + noise, 0, 1)
        return out
    return reducer

# This is an equivalent function to the above
def reduce_pop_loss_and_additive_noise(time_in_hours, base_sd=0.0055, Nl = 0.00771702, r = -0.02206672):
    """ Loss of memory following a logistic progression depended on waiting time and additive noise accumulating over time """
    def reducer(x):
        # The logistic reduction
        # Fitted parameters from ants homing distance
        K = 1.0
        #Nl = 0.00771702
        #r = -0.02206672
        y = x * K / (1 + Nl/(K - Nl) * np.exp(-r * time_in_hours))
        # The noise
        sd = np.sqrt(time_in_hours*60*60) * base_sd / 128
        noise = np.random.normal(loc=0.0, scale=sd, size=x.shape)
        out = np.clip(y + noise, 0, 1)
        return out
    return reducer

# Check the provided command line arguments
if (len(sys.argv) - 1) >= 6:
    try:
        # Get the mode
        mode = sys.argv[1]
        if mode not in modes:
            print('ERROR: A valid <MODE> value was expected.')
            raise

        # Get the route file actions to take
        s = sys.argv[2]
        if 'DEFAULT' not in s and 'SAVE' not in s and 'LOAD' not in s:
            print('ERROR: A valid <ROUTE> value was expected.')
            raise
        if 'DEFAULT' in s:
            pass
        else:
            if 'SAVE' in s:
                # Remove the SAVE from the string to avoid reprocessing it
                s = s.replace('SAVE', '')
                save_route = True # Save the resulting route to disk with standard pattern.
            if 'LOAD' in s:
                load_route = True # If true instead of generating a new route it loads one from disk.
                # Is there a filename given
                re_res = re.search('LOAD:([^:]+.npz)', s)
                if re_res is not None:
                    load_route_file = re_res.group(1)
                else:
                    print('ERROR: The LOAD argument must be followed by : and a npz filename to load a route from.')
                    raise

        # Get the figures actions to take
        s = sys.argv[3]
        if 'DEFAULT' not in s and 'SAVE' not in s and 'SHOW' not in s:
            print('ERROR: A valid <FIGURE> value was expected.')
            raise
        if 'DEFAULT' in s:
            pass
        else:
            if 'SHOW' in s:
                show_figs = True  # Show figure on screen
            if 'SAVE' in s:
                save_figs = True # Save figure on disk using the standard filename pattern

        # Get the string to use in the route and figure filename
        saved_file_num = sys.argv[4]

        # Get the activation function noise to use
        noise_syn  = float(sys.argv[5])

        # Get the motor turning noise to use
        noise_turn = float(sys.argv[6])
        
        memory_loss_method = None
        memory_loss_method_str = ''
        if (len(sys.argv) - 1) >= 7:
            memory_loss_method = sys.argv[7]
            if memory_loss_method == 'DEFAULT' or memory_loss_method == 'REMEMBER': # Do not alter the memory
                memory_loss_method = reduce_multiply(1.0)
            elif memory_loss_method.startswith('='): # Set all elements of CPU4 to this value
                memory_loss_method_str = 'Ice' + memory_loss_method 
                memory_loss_method = reduce_set(float(memory_loss_method.replace('=', '')))
            elif memory_loss_method.startswith('x'): # Multiply each element of CPU4 by this value
                memory_loss_method_str = 'Ice' + memory_loss_method 
                memory_loss_method = reduce_multiply(float(memory_loss_method.replace('x', '')))
            elif memory_loss_method.startswith('-'): # Subtract from each element of CPU4 this value
                memory_loss_method_str = 'Ice' + memory_loss_method 
                memory_loss_method = reduce_subtract(-float(memory_loss_method))
            elif memory_loss_method.startswith('n'): # Add Gaussian noise to each element of CPU4
                memory_loss_method_str = 'Ice' + memory_loss_method 
                memory_loss_method = reduce_additive_noise(float(memory_loss_method.replace('n', '')))
            elif memory_loss_method.startswith('l'): # Reduce CPU4 by the inverse logistic function depending on the waiting hours
                memory_loss_method_str = 'Wait' + memory_loss_method
                memory_loss_method = reduce_pop_loss(float(memory_loss_method.replace('l', '')))
            elif memory_loss_method.startswith('b'): # Combined 'l' and 'n' effect on CPU4 values
                # Split the parameter values
                try:
                    params_separate = memory_loss_method.replace('b', '').split(',')
                    param_logistic_hours = float(params_separate[0])
                    param_base_noise_sd  = float(params_separate[1])
                except:
                    print('ERROR: Argument ' + memory_loss_method + ' was expected to have the form ' + 'b<NUMBER>,<NUMBER>' + ' eg b1,0.005 where 1 is the inverse logistic function waiting time in hours (1h) and 0.005 is the standard deviation of the Gaussian noise.')
                    exit(1)

                memory_loss_method_str = 'Wait' + str(param_logistic_hours) + 'h' + 'Noise' + str(param_base_noise_sd)
                
                memory_loss_method = reduce_pop_loss_and_additive_noise(param_logistic_hours, base_sd=param_base_noise_sd)
            elif memory_loss_method.startswith('p'): # Combined 'l' and 'n' effect on CPU4 values with parametric PopulationLoss
                # Split the parameter values
                try:
                    params_separate = memory_loss_method.replace('p', '').split(',')
                    param_logistic_hours = float(params_separate[0])
                    param_base_noise_sd  = float(params_separate[1])
                    param_Nl             = float(params_separate[2])
                    param_r              = float(params_separate[3])
                except:
                    print('ERROR: Argument ' + memory_loss_method + ' was expected to have the form ' + 'p<NUMBER>,<NUMBER>,<NUMBER>,<NUMBER>' + ' eg b1,0.005,0.01,-0.02 where 1 is the waiting time in hours (1h), 0.005 is the difffusion coefficient, 0.01 and -0.02 the Nl and r parameters of the inverse logistic function.')
                    exit(1)

                memory_loss_method_str = 'Wait' + str(param_logistic_hours) + 'h' + 'Noise' + str(param_base_noise_sd) + 'Nl' + str(param_Nl) + 'r' + str(param_r)
                
                memory_loss_method = reduce_pop_loss_and_additive_noise(param_logistic_hours, 
                                                                        base_sd = param_base_noise_sd, 
                                                                        Nl      = param_Nl, 
                                                                        r       = param_r)
            else:
                print('ERROR: Argument given has none of the recognised formats:', memory_loss_method)
        

        if (len(sys.argv) - 1) >= 8:
            noise_turn_slope = float(sys.argv[8])
        else: 
            noise_turn_slope = 125.0
        
        # Check for some contradictory combinations of parameter values
        # Not an exhaustive check for inconsistent combinations of values
        if mode == 'Generate_outbound_route' and load_route:
            print('ERROR: Contradictory combination of arguments given: ')
            print('Route loading (LOAD) and mode Generate_outbound_route makes no sense to be used together.')
            raise
    except TypeError as e:
        print("Error", e)
        exit(1)        
    except:
        print("Error", sys.exc_info()[0])
        exit(1)        
else:
    print("ERROR: Not adequate required parameters were provided.")
    usage()
    exit(1)        

output_route_file = 'noiseSyn{:}_noiseRot{:}_noiseSlope{:}_route_{:}{:}_{:}.npz'

# Default
T_outbound = 1500        # FV
T_inbound  = 1500        # For inbound path returning home
T_inbound  = 3000        # For inbound path returning home

if mode == 'ZV_release':
    T_outbound = 1       # ZV

if mode == 'Generate_outbound_route':
    T_inbound = 0        # For collecting outbound only trajectory

print('Trial parameters:')
print('   Mode             : {}'.format(mode))
print('   show_figs        : {}'.format(show_figs))
print('   save_figs        : {}'.format(save_figs))
print('   load_route       : {}'.format(load_route))
if load_route:
    print('   load file        : {}'.format(load_route_file))
print('   save_route       : {}'.format(save_route))
print('   saved_file_num   : {}'.format(saved_file_num))
print('   Noise synaptic   : {}'.format(noise_syn))
print('   Noise locomotive : {}'.format(noise_turn))
print('   Noise slope      : {}'.format(noise_turn_slope))

if load_route:
    h_out, v_out, log_out = trials.load_route(filename=load_route_file)
    T_outbound = log_out.T_outbound
    print("Route loaded        :", load_route_file)
else:
    h_out, v_out = trials.generate_route(T=T_outbound, vary_speed=True)
    print("Route generated")

cxrph = cx_rate.CXRatePontinHolonomic(noise=noise_syn)

cxs = [cxrph]
titles = ['with Pontin Holonomic']
logs = []
hs = []
vs = []

print('   T_outbound       :', T_outbound)
print('   T_inbound        :', T_inbound)

# Only do it if needed because running over ssh gives Error: no display name and no $DISPLAY environment variable
if show_figs or save_figs:
    fig, ax = plt.subplots(1, len(cxs), figsize=(16,4))

# Only do it if needed because running over ssh gives Error: no display name and no $DISPLAY environment variable
if show_figs or save_figs:
    if not isinstance(ax, list):
        ax = [ax]

for i in range(len(cxs)):
    h, v, log, cpu4_snapshot = trials.run_trial(route=(h_out, v_out), cx=cxs[i], logging=True, 
                                                T_outbound=T_outbound, T_inbound=T_inbound, 
                                                noise=noise_syn, noise_turn=noise_turn, 
                                                noise_turn_slope=noise_turn_slope, 
                                                cooling_treatment_method=memory_loss_method)
    logs.append(log)
    hs.append(h)
    vs.append(v)
    # Only do it if needed because running over ssh gives Error: no display name and no $DISPLAY environment variable
    if show_figs or save_figs:
        x, y = analysis.compute_location_estimate(cpu4_snapshot, cxs[i])
        plotter.plot_route(h, v, T_outbound=T_outbound, T_inbound=T_inbound,
                           title=titles[i], ax=ax[i], plot_speed=True, memory_estimate=(x, y))
    
    #if save_route and not load_route:
    if save_route:
        print("saving...")
        trials.save_route(h=h, v=v, cx_log=log, filename=titles[i].replace(' ', '_') + '_' + output_route_file.format(noise_syn, noise_turn, noise_turn_slope, mode.replace('_release', ''), memory_loss_method_str, saved_file_num), data_path=data_path)

if save_figs:
    fig.savefig(data_path + '/all_trajectories' + '_' + output_route_file.format(noise_syn, noise_turn, noise_turn_slope, mode.replace('_release', ''), memory_loss_method_str, saved_file_num).replace('.npz', '.png'))

if show_figs:
    plt.show()
