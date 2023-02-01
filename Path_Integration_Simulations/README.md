
Directory structure


The `README.md` file in `path-integration-forget/` explains how to run agent simulations and collect data. This would produce files that contain agent paths (csv files that contain x,y path coordinates). 

`analyse_trajectories_publication_script.py` Used to analyse the collected simulation data (csv files that contain x,y path coordinates) and calculates descriptive statistics for the paths. It creates the path-integration-forget/data/path_analysis_calculation_results_3parameters.npz file with the results. 

`analyse_trajectories_step2_script.py` Reads the path_analysis_calculation_results_3parameters.npz file produced by the previous script and calculates the Mean Squared Errors (MSE) between the values in the file and the actual ants behaviour. The resulting MSE values are stored in the file path-integration-forget/data/3_parameters_results_correct_Nl_range_combined_01Ded2022/path_analysis_calculation_results_3parameters_MSE_values.npz

`visualize_n_minimize.ipynb` Used to search for the combination of parameters that give the minimum MSE.

`analyse_trajectories_step_2_3params.ipynb` is used to produce the plots in Figures 2A, 4B, 6.
