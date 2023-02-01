# Convert data from .npz files to .CSV trajectory files

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trials


def usage():
    print('Converts npz files produced by collect_simulation_path_data.py that')
    print('contain simulation paths to csv files with x,y path coordinates.')
    print('It recreates the same path directory hierarchy for the csv files.')
    print('')
    print('SYNTAX:')
    print('    python npx_to_csv.py -i=/path/to/source/npz/files -o=/path/to/produced/csv/files')
    print('    -i=<DIRECTORY PATH>')
    print('           Path to the npz files to read and process.')
    print('    -o=<DIRECTORY PATH>')
    print('           Path to store the produced csv files.')
    print('')
    print('E.g.')
    print('    python npx_to_csv.py -i=./data/Conditions/Memory/ -o=./data/Converted_to_CSV/Conditions/Memory/')


def check_num_of_entries_in_datafiles(filename):
    """ This is used to check if the collected files are valid """
    
    # Create subdirectory structure if it does not exist
    #Path(path_output_csv).joinpath(Path(filename).parent).mkdir(parents=True, exist_ok=True)

    h, v, log = trials.load_route(filename=filename, data_path='')
    
    return len(v)


def convert_datafiles_to_csv_files(filename, path_output_csv, scaling=1.0, sampling_rate=25, 
                                   cut_once_crossing_x=None, only_inbound=False, keep_first_n_samples=None):
    """ 
        Gets velocity measurements from a npz file and converts it  
        to x,y coordinates series of a trajectory. Result is saved 
        in the path_output_csv .CSV file. 
        The velocity data are assumed to be sampled every 
        1/sampling_rate seconds. 
        If cut_once_crossing_x is a number the trajectory data is 
        cut off as soon as the simulated ant crosses a cut_once_crossing_x
        radius from the release point.
        If only_inbound is True the csv files will contain only the inbound 
        (homing) part of the route. 
        If keep_first_n_samples == None all samples will be stored. 
        If keep_first_n_samples(int) < length(path) then only the first 
        keep_first_n_samples samples will be stored. 
    """
    
    # Create subdirectory structure if it does not exist
    # Use the path_output_csv + the last directory in the filename
    Path(path_output_csv).joinpath(Path(filename).parent.name).mkdir(parents=True, exist_ok=True)
    
    try:
        h, v, log = trials.load_route(filename=filename, data_path='')
        
        # Convert velocity to x,y coordinates
        if not only_inbound:
            # Use the whole route
            xy = np.vstack([np.array([0.0, 0.0]), np.cumsum(v, axis=0)])
        else:
            # Use only the homing part of the route
            xy = np.vstack([np.array([0.0, 0.0]), np.cumsum(v[log.T_outbound:], axis=0)])
        x, y = xy[:, 0], xy[:, 1]
        
        # Create a time stamp vector
        t = np.array(range(0, len(x))) * (1.0 / sampling_rate)
        data = list(zip(x, y, t))
        
        # Construct pandas data frame
        columns = ['x', 'y', 'Time']
        df = pd.DataFrame(data, columns=columns)
        
        # Coordinates scalling to m
        df.x = df.x * scaling
        df.y = df.y * scaling
        
        if cut_once_crossing_x is not None and isinstance(cut_once_crossing_x, (int, float)):
            #cut_off_index = np.argmin(np.abs(np.hypot(df.x, df.y) - cut_once_crossing_x))
            cut_off_index = np.argmax(np.hypot(df.x, df.y) > cut_once_crossing_x)
            if cut_off_index > 0:
                df = df.head(cut_off_index+1)
            filename = filename.replace('.npz', '_platformReleases.npz')
        
        if keep_first_n_samples is not None and isinstance(keep_first_n_samples, int):
            if keep_first_n_samples < len(df):
                df = df.head(keep_first_n_samples)
        
        # Construct the CSV filename
        # Use path_output_csv + the last directory of the filename + the file name and change the extension
        pathfilename = Path(path_output_csv).joinpath(Path(filename).parent.name, Path(filename).name).with_suffix('.csv')
        #print('Writing data frame to file:', pathfilename)
        print('.', end='')
        df.to_csv(pathfilename, index = False)
    except: 
        print('Error while processing ', filename)


# Check the provided command line arguments
if (len(sys.argv) - 1) == 2:
    # The first argument is the path to the source npz files
    input_path = sys.argv[1]
    if input_path.startswith('-i='):
        input_path = input_path.replace('-i=', '')
    else:
        print('ERROR: Expected the input path as the first argument.')
        print()
        usage()
        exit(1)
    if (not os.path.isdir(input_path)) and (not os.path.isfile(input_path)):
        print('ERROR: Path does not exist', input_path)
        exit(1)
    # The second argument is the path to store the produced csv files
    output_path = sys.argv[2]
    if output_path.startswith('-o='):
        output_path = output_path.replace('-o=', '')
    else:
        print('ERROR: Expected the output path as the second argument.')
        print()
        usage()
        exit(1)
elif (len(sys.argv) - 1) == 1 and sys.argv[1] == '-h':
    usage()
    exit(0)
else:
    print('ERROR: Expected 2 arguments.')
    print()
    usage()
    exit(1)

path_input_npz   = input_path
path_output_csv  = output_path


# Convert the trajectory files from .npz to .csv

fps = 25 # in frames per sec
scaling = 0.03  # max distance = 0.03m/step * 1500steps = 45m

# Check if the input path is a file or directory
if path_input_npz.endswith('.npz') and os.path.isfile(path_input_npz):
	# Set the pattern to the single file
	filename_pattern = path_input_npz
else:
	# Process all the subdirectories of the collected data for memory experiments
	filename_pattern = [sub + '/with_Pontin_Holonomic_*_*.npz' for sub in glob.glob(path_input_npz+'*')]; 

only_inbound=True;

if not isinstance(filename_pattern, list):
    filename_pattern = [filename_pattern]

for filename_pattern_i in filename_pattern:
    files_list = glob.glob(filename_pattern_i, recursive=False)

    for f in files_list:
        convert_datafiles_to_csv_files(f, 
                                       path_output_csv=path_output_csv, 
                                       scaling = scaling, 
                                       sampling_rate=fps, 
                                       only_inbound=only_inbound) #, keep_first_n_samples=600)

