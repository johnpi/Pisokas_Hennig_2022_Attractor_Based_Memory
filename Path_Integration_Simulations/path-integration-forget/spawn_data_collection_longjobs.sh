#!/bin/bash

echo "Spawning a set of longjobs on the computer to"
echo "collect homing simulation data using the script"
echo "collect_simulation_path_data.py ..."
echo "It will start [mem_noise] x [mem_wait] jobs..."

# Decide which directory the files will be stored in
if [ -e /disk/data/ipisokas/data/ ]; then
   data_path="/disk/data/ipisokas/data"
else 
   if [ -e /disk/scratch/ipisokas/data/ ]; then 
      data_path="/disk/scratch/ipisokas/data"
   else # else use the subdir data in the working directory
      if [ -e /exports/eddie/scratch/s0093128/data/ ]; then
         data_path="/exports/eddie/scratch/s0093128/data"
      else # else use the subdir data in the working directory
         data_path="data"
      fi
   fi
fi

# For different levels of diffusion coefficient (noise)
for mem_noise in 0.0 0.001 0.002 0.003 0.004 0.005 0.0055 0.006 0.0065 0.007 0.008 0.009 0.01 0.015 0.02; do # 13
# For different captivity waiting periods
for mem_wait in 0 1 24 48 96 144 192 240 288 336 384 432; do # 12

# Create a script to run simulations for mem_noise and mem_wait using this here doc
(
cat <<EOF
#!/bin/bash

# Starts longjob for collecting data with mem_wait=${mem_wait} and mem_noise=${mem_noise}...

echo "Starts longjob for collecting data with mem_wait=${mem_wait} and mem_noise=${mem_noise}..."

mem_wait=${mem_wait}
mem_noise=${mem_noise}

outbound_file=outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.npz
ve=FV; noise_slope=9.0; noise_syn=0.1; noise_turn=2.0; 
echo "Noise synaptic   \${noise_syn}"; 
echo "Noise locomotive \${noise_turn}"; 
echo "Noise slope      \${noise_slope}"; 
echo "======================="; 
# For different Nl values (parameter of the logistic function)
# Collected:
#for mem_Nl in 0.0 0.02 0.03 0.04 0.05 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2; do # 13
# Collect Extra Nl with the same r:
#for mem_Nl in     0.001 0.01 0.016 0.018 0.019      0.021 0.022 0.025                                                      0.3 0.5 0.999 1.3; do # 12
# Both together
for mem_Nl in  0.0 0.001 0.01 0.016 0.018 0.019 0.02 0.021 0.022 0.025 0.03 0.04 0.05 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.3 0.5 0.999 1.3; do # 25
	# For different r values (parameter of the logistic function)
	#Collected:
	#for mem_r in -0.008 -0.01 -0.012 -0.014 -0.015 -0.016 -0.017 -0.018 -0.019 -0.02 -0.021 -0.022 -0.023 -0.024 -0.025 -0.026 -0.027 -0.028 -0.029 -0.03 -0.031 -0.032                                 ; do # 22
	# Collect Extra r with all Nl:
	#for mem_r in                                                                                                                                                        -0.035 -0.04 -0.045 -0.05 -0.1; do # 5
	# Both together
	for mem_r in  -0.008 -0.01 -0.012 -0.014 -0.015 -0.016 -0.017 -0.018 -0.019 -0.02 -0.021 -0.022 -0.023 -0.024 -0.025 -0.026 -0.027 -0.028 -0.029 -0.03 -0.031 -0.032 -0.035 -0.04 -0.045 -0.05 -0.1; do # 27
		# The memory manipulation specification string
		mem="p\${mem_wait},\${mem_noise},\${mem_Nl},\${mem_r}"; 
		# Run a number of simulation trials
		for iter in \$(seq 1001 1040); do 
			echo "Iteration           : \$iter"; 
			echo "Memory              : \$mem"; 
			python3 collect_simulation_path_data.py \${ve}_release SAVE:LOAD:\${outbound_file} DEFAULT "\${iter}" "\${noise_syn}" "\${noise_turn}" "\${mem}" "\${noise_slope}"; 
			# Converts the collected file to CSV and deletes the npz file
			echo "Converting to CSV... ${data_path}/with_Pontin_Holonomic_noiseSyn\${noise_syn}_noiseRot\${noise_turn}_noiseSlope\${noise_slope}_route_FVWait\${mem_wait}.0hNoise\${mem_noise}Nl\${mem_Nl}r\${mem_r}_\${iter}.npz"
			python3 npz_to_csv.py -i=${data_path}/with_Pontin_Holonomic_noiseSyn\${noise_syn}_noiseRot\${noise_turn}_noiseSlope\${noise_slope}_route_FVWait\${mem_wait}.0hNoise\${mem_noise}Nl\${mem_Nl}r\${mem_r}_\${iter}.npz -o=${data_path}/ && \
			rm ${data_path}/with_Pontin_Holonomic_noiseSyn\${noise_syn}_noiseRot\${noise_turn}_noiseSlope\${noise_slope}_route_FVWait\${mem_wait}.0hNoise\${mem_noise}Nl\${mem_Nl}r\${mem_r}_\${iter}.npz
		done; 
	done; 
done

echo "longjob for collecting data with mem_wait=${mem_wait} and mem_noise=${mem_noise} done."
exit 0
EOF
) > collect_data_longjob_${mem_wait}_${mem_noise}_$(uname -n).sh 

chmod u+x collect_data_longjob_${mem_wait}_${mem_noise}_$(uname -n).sh

echo "Starting longjob for collecting data with mem_wait=${mem_wait} and mem_noise=${mem_noise}..."
echo "Please, give password:"

longjob -28day -c "./collect_data_longjob_${mem_wait}_${mem_noise}_$(uname -n).sh" >> log_file_${mem_wait}_${mem_noise}_$(uname -n).log

done;
done

echo "Spawning done."

exit 0
