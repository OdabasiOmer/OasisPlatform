import csv
import os
import sys
import subprocess
import logging
import json
import configparser
import numpy as np
import shutil
import pandas as pd
from scipy.interpolate import interp1d

FIG_PARAMS      = {"f_size": (9.5, 6),
                   "p_width": 5.5,
                   "p_height": 4.,
                   "label_size": 1.5,
                   "title_size": 0.5}

def append_to_existing_file(existing_file_path, new_content, output_file_path):
    """
    Appends new content to an existing file and saves it as a new file.

    Parameters:
    existing_file_path (str): The path to the existing file.
    new_content (str): The content to be appended to the file.

    Returns:
    str: The path to the new combined file.
    """
    # Read the existing file content
    with open(existing_file_path, 'r') as file:
        existing_content = file.read()

    # Combine the existing content with the new content
    combined_content = existing_content + '\n' + new_content

    # Write the combined content to the new file
    with open(output_file_path, 'w') as file:
        file.write(combined_content)

    return output_file_path

def check_redcat_completion(folder_dir):
    # Search for files with pattern 'GM*.aal' in the current directory
    for file in os.listdir(folder_dir):
        if file.startswith('GM') and file.endswith('.aal'):
            return True
        if file.endswith('.epc'):
            return True
    return False


def copy_files(src_directory, dst_directory):
    """
    Copies all files from the source directory to the destination directory.

    Args:
    src_directory (str): The path to the source directory.
    dst_directory (str): The path to the destination directory.
    """
    # Ensure destination directory exists
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)

    # Copy each file from the source directory to the destination directory
    for filename in os.listdir(src_directory):
        file_path = os.path.join(src_directory, filename)
        
        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            shutil.copy(file_path, dst_directory)

def delete_lines_from_file(file_path, start_line, end_line):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        with open(file_path, 'w') as file:
            for i, line in enumerate(lines):
                if not start_line <= i + 1 <= end_line:
                    file.write(line)
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_bash_script(num_processes, runRI, output_filepath):
    """
    Generates a Bash script based on the number of processes and the runRI flag.
    """
    script = []

    script.append("rm -R -f work/kat/")
    script.append("mkdir -p work/kat/")
    script.append("")
    script.append("# Generate the occurrence file...")
    script.append("occurrencetobin -D -P 30000 < ./input/occurrence.csv > ./input/occurrence.bin")

    # Initial setup: Directory and FIFO creation
    script.append("rm -R -f /tmp/i02QFjyaNF/")
    script.append("mkdir -p /tmp/i02QFjyaNF/fifo/")
    script.append("mkdir -p work/gul_S1_summaryleccalc")
    script.append("mkdir -p work/gul_S1_summaryaalcalc")
    script.append("mkdir -p work/il_S1_summaryleccalc")
    script.append("mkdir -p work/il_S1_summaryaalcalc")
    if runRI:
        script.append("mkdir -p work/ri_S1_summaryleccalc")
        script.append("mkdir -p work/ri_S1_summaryaalcalc")

    script.append(f"# # #")

    # Creating FIFOs for each process
    for i in range(1, num_processes+1):
        script.append(f"# Process ---")
        script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/gul_P{i}")
        script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/gul_S1_summary_P{i}")
        script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/gul_S1_summary_P{i}.idx")
        script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/gul_S1_eltcalc_P{i}")
        script.append(f"# # #")
        script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/il_P{i}")
        script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/il_S1_summary_P{i}")
        script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/il_S1_summary_P{i}.idx")
        script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/il_S1_eltcalc_P{i}")
        script.append(f"# # #")
        if runRI:
            script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/ri_P{i}")
            script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/ri_S1_summary_P{i}")
            script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/ri_S1_summary_P{i}.idx")
            script.append(f"mkfifo /tmp/i02QFjyaNF/fifo/ri_S1_eltcalc_P{i}")
            script.append(f"# # #")

    pid=0
    # <1> Parallel computation commands -> Do reinsurance loss computes
    if runRI:
        script.append("# --- Do reinsurance loss computes ---")
        pid=pid+1
        script.append(f"( eltcalc < /tmp/i02QFjyaNF/fifo/ri_S1_eltcalc_P1 > work/kat/ri_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid{pid}=$!")
        for i in range(2, num_processes+1): # one less because the above line is the first
            pid=pid+1
            script.append(f"( eltcalc -s < /tmp/i02QFjyaNF/fifo/ri_S1_eltcalc_P{i} > work/kat/ri_S1_eltcalc_P{i} ) 2>> $LOG_DIR/stderror.err & pid{pid}=$!")
        for i in range(1, num_processes+1):
            pid=pid+1
            script.append(f"tee < /tmp/i02QFjyaNF/fifo/ri_S1_summary_P{i} /tmp/i02QFjyaNF/fifo/ri_S1_eltcalc_P{i} work/ri_S1_summaryaalcalc/P{i}.bin work/ri_S1_summaryleccalc/P{i}.bin > /dev/null & pid{pid}=$!")
            pid=pid+1
            script.append(f"tee < /tmp/i02QFjyaNF/fifo/ri_S1_summary_P{i}.idx work/ri_S1_summaryaalcalc/P{i}.idx work/ri_S1_summaryleccalc/P{i}.idx > /dev/null & pid{pid}=$!")
        for i in range(1, num_processes+1):
            script.append(f"( summarycalc -m -f -p RI_1 -1 /tmp/i02QFjyaNF/fifo/ri_S1_summary_P{i} < /tmp/i02QFjyaNF/fifo/ri_P{i} ) 2>> $LOG_DIR/stderror.err  &")

    script.append(" ")

    # <2> Parallel computation commands -> Do insured loss computes
    script.append("# --- Do insured loss computes ---")
    pid=pid+1
    script.append(f"( eltcalc < /tmp/i02QFjyaNF/fifo/il_S1_eltcalc_P1 > work/kat/il_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid{pid}=$!")
    for i in range(2, num_processes+1): # one less because the above line is the first
        pid=pid+1
        script.append(f"( eltcalc -s < /tmp/i02QFjyaNF/fifo/il_S1_eltcalc_P{i} > work/kat/il_S1_eltcalc_P{i} ) 2>> $LOG_DIR/stderror.err & pid{pid}=$!")
    for i in range(1, num_processes+1):
        pid=pid+1
        script.append(f"tee < /tmp/i02QFjyaNF/fifo/il_S1_summary_P{i} /tmp/i02QFjyaNF/fifo/il_S1_eltcalc_P{i} work/il_S1_summaryaalcalc/P{i}.bin work/il_S1_summaryleccalc/P{i}.bin > /dev/null & pid{pid}=$!")
        pid=pid+1
        script.append(f"tee < /tmp/i02QFjyaNF/fifo/il_S1_summary_P{i}.idx work/il_S1_summaryaalcalc/P{i}.idx work/il_S1_summaryleccalc/P{i}.idx > /dev/null & pid{pid}=$!")
    for i in range(1, num_processes+1):
        script.append(f"( summarycalc -m -f  -1 /tmp/i02QFjyaNF/fifo/il_S1_summary_P{i} < /tmp/i02QFjyaNF/fifo/il_P{i} ) 2>> $LOG_DIR/stderror.err  &")

    script.append(" ")

    # <3> Parallel computation commands -> Do GUP computes 
    script.append("# --- Do ground up loss computes ---")
    pid=pid+1
    script.append(f"( eltcalc < /tmp/i02QFjyaNF/fifo/gul_S1_eltcalc_P1 > work/kat/gul_S1_eltcalc_P1 ) 2>> $LOG_DIR/stderror.err & pid{pid}=$!")
    for i in range(2, num_processes+1): # one less because the above line is the first
        pid=pid+1
        script.append(f"( eltcalc -s < /tmp/i02QFjyaNF/fifo/gul_S1_eltcalc_P{i} > work/kat/gul_S1_eltcalc_P{i} ) 2>> $LOG_DIR/stderror.err & pid{pid}=$!")
    for i in range(1, num_processes+1):
        pid=pid+1
        script.append(f"tee < /tmp/i02QFjyaNF/fifo/gul_S1_summary_P{i} /tmp/i02QFjyaNF/fifo/gul_S1_eltcalc_P{i} work/gul_S1_summaryaalcalc/P{i}.bin work/gul_S1_summaryleccalc/P{i}.bin > /dev/null & pid{pid}=$!")
        pid=pid+1
        script.append(f"tee < /tmp/i02QFjyaNF/fifo/gul_S1_summary_P{i}.idx work/gul_S1_summaryaalcalc/P{i}.idx work/gul_S1_summaryleccalc/P{i}.idx > /dev/null & pid{pid}=$!")
    for i in range(1, num_processes+1):
        script.append(f"( summarycalc -m -i  -1 /tmp/i02QFjyaNF/fifo/gul_S1_summary_P{i} < /tmp/i02QFjyaNF/fifo/gul_P{i} ) 2>> $LOG_DIR/stderror.err  &")

    # <4> Parallel computation commands -> REDCAT 
    script.append("# --- Do REDCat computes ---")
    script.append("touch output/redloss.log")
    script.append(f"REDLoss -f redloss1.cf 2>> output/redloss.log &")
    for i in range(2, num_processes+1):
        script.append(f"REDLoss -f redloss{i}.cf &")

    script.append(f"ls -a")
    script.append(f"check_fifo_x")
    script.append(f" ")

    # <5> Connect all pipes and start analysis
    for i in range(1, num_processes+1):
        if not runRI:
            script.append(f"( tee < fifo_p{i} /tmp/i02QFjyaNF/fifo/gul_P{i} | fmcalc -a2 > /tmp/i02QFjyaNF/fifo/il_P{i} ) 2>> $LOG_DIR/stderror.err &")
        else:
            script.append(f"( tee < fifo_p{i} /tmp/i02QFjyaNF/fifo/gul_P{i} | fmcalc -a2 | tee /tmp/i02QFjyaNF/fifo/il_P{i} | fmcalc -a3 -n -p RI_1 > /tmp/i02QFjyaNF/fifo/ri_P{i} ) 2>> $LOG_DIR/stderror.err &")

    waitString = "wait"
    
    for p in range(1, pid+1):
        waitString = waitString + f" $pid{p}"
    
    script.append(waitString)
    
    # <6> kats *************************************************************************************
    kpid = 0
    
    # <6a> Reinsurance kats <optional>
    katString = "kat"
    if runRI:
        script.append("# --- Do reinsurance kats ---")
        kpid = kpid + 1
        for i in range(1, num_processes+1):
            katString = katString + f" work/kat/ri_S1_eltcalc_P{i}" 
        katString = katString + f" > output/ri_S1_eltcalc.csv & kpid{kpid}=$!"
        script.append(katString)
        script.append(" ")
    

    # <6b> Insured kats *********
    script.append("# --- Do insurance kats ---")
    katString = "kat"
    kpid = kpid + 1
    for i in range(1, num_processes+1):
        katString = katString + f" work/kat/il_S1_eltcalc_P{i}" 
    katString = katString + f" > output/il_S1_eltcalc.csv & kpid{kpid}=$!"
    script.append(katString)
    script.append(" ")

    # <6c> GUP kats *********
    script.append("# --- Do ground up kats ---")
    katString = "kat"
    kpid = kpid + 1
    for i in range(1, num_processes+1):
        katString = katString + f" work/kat/gul_S1_eltcalc_P{i}" 
    katString = katString + f" > output/gul_S1_eltcalc.csv & kpid{kpid}=$!"
    script.append(katString)
    script.append(" ")

    waitString = "wait"
    for p in range(1, kpid+1):
        waitString = waitString + f" $kpid{p}"
    script.append(waitString)
    script.append(" ")

    # <7> Final -> aalcalc and leccalc **************************************************************************
    lpid = 0
    if runRI:
        lpid = lpid + 1
        script.append(f"( aalcalc -Kri_S1_summaryaalcalc > output/ri_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid{lpid}=$!")
        lpid = lpid + 1
        script.append(f"( leccalc -r -Kri_S1_summaryleccalc -F output/ri_S1_leccalc_full_uncertainty_aep.csv -f output/ri_S1_leccalc_full_uncertainty_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid{lpid}=$!")
    
    # insured
    lpid = lpid + 1
    script.append(f"( aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid{lpid}=$!")
    lpid = lpid + 1
    script.append(f"( leccalc -r -Kil_S1_summaryleccalc -F output/il_S1_leccalc_full_uncertainty_aep.csv -f output/il_S1_leccalc_full_uncertainty_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid{lpid}=$!")

    # ground up
    lpid = lpid + 1
    script.append(f"( aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv ) 2>> $LOG_DIR/stderror.err & lpid{lpid}=$!")
    lpid = lpid + 1
    script.append(f"( leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv ) 2>> $LOG_DIR/stderror.err & lpid{lpid}=$!")
        
    waitString = "wait"
    for p in range(1, lpid+1):
        waitString = waitString + f" $lpid{p}"
    script.append(waitString)
    script.append("")
    
    # <7> Clean up **************************************************************************
    script.append("check_complete")
    script.append("")

    script.append("rm -Rf work/il_S1*")
    script.append("rm -Rf work/gul_S1*")
    script.append("rm -Rf work/kat")
    script.append("rm -Rf fifo*")
    script.append("")
   
    script.append("rm -R -f /tmp/i02QFjyaNF/")
    
    script_to_write = '\n'.join(script)

    with open(output_filepath, "w") as file:
        file.write(script_to_write)

    return script_to_write

def generate_uniform_grid(input_coord_file_path, grid_spacing, output_filepath):
    
    bounds = get_boundary_box(input_coord_file_path)
    
    # Open a CSV file to write the lat-lon data
    with open(output_filepath, 'w', newline='') as csvfile:
        grid_writer = csv.writer(csvfile)

        # Generate grid points
        lat = bounds['latitude_min']
        while lat <= bounds['latitude_max']:
            lon = bounds['longitude_min']
            while lon <= bounds['longitude_max']:
                # Format latitude and longitude with 5 decimal places
                formatted_lat = "{:.5f}".format(lat)
                formatted_lon = "{:.5f}".format(lon)
                grid_writer.writerow([formatted_lat, formatted_lon])
                lon += grid_spacing
            lat += grid_spacing

def get_boundary_box(file_path):
    """
    Finds the minimum and maximum latitude and longitude from a CSV file.

    :param file_path: Path to the CSV file containing latitude and longitude data.
    :return: A dictionary with min and max values for latitude and longitude.
    """
    # Read the CSV file into a DataFrame
    # Assuming the first column is latitude and the second is longitude
    data = pd.read_csv(file_path, header=None, names=['latitude', 'longitude'])

    # Calculate the min and max values
    lat_min = data['latitude'].min()
    lat_max = data['latitude'].max()
    lon_min = data['longitude'].min()
    lon_max = data['longitude'].max()

    return {
        'latitude_min': lat_min,
        'latitude_max': lat_max,
        'longitude_min': lon_min,
        'longitude_max': lon_max
    }

def get_damage_function_file_string(country_code, coverage, LoB):
    return 'df_' + country_code + '_Cov' + coverage + '_' + LoB + '.dfs'

def get_coordinate_limits_from_case(case_name):
    """     
    Keyword arguments:
    case -- test case dictionary containing case parameters
    Return: parsed coordinates in proper format.
    """
    try:
        x , test_cases = read_config('./config.cfg')
    except:
        x , test_cases = read_config('../config.cfg')        

    coordLims = {'lon': parse_longitude_latitude(test_cases[case_name]['longitude_minmax']),
                'lat': parse_longitude_latitude(test_cases[case_name]['latitude_minmax'])}
    return coordLims

def get_full_exposure_filepath(root_dir, country_name, LoB):
    return os.path.join(root_dir, country_name, f'{country_name}_{LoB}.txt')

def get_most_recently_created_folder(directory_path):
    # List all folders in the given directory
    all_folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]

    if not all_folders:
        raise ValueError(f'No folders found in {directory_path}')

    # Sort folders by creation time
    all_folders.sort(key=lambda folder: os.path.getctime(os.path.join(directory_path, folder)), reverse=True)

    # Return the most recently created folder
    return all_folders[0]

def log_to_file(message, file, echo=True):
    if echo:
        print(message)
    original_stdout = sys.stdout  # Save the original stdout stream
    sys.stdout = file  # Redirect stdout to the log file
    print(message)  # Write the message to the log file
    sys.stdout = original_stdout  # Restore the original stdout stream


def log_to_console(message):
    print(message)  # Write the message to the console


def get_non_underscore_folders():
    cwd = os.getcwd()
    folder_names = [name for name in os.listdir(
        cwd) if os.path.isdir(os.path.join(cwd, name))]
    non_underscore_folders = [
        name for name in folder_names if not name.startswith('_')]
    return non_underscore_folders

def parse_analysis_settings_file(file_path, debug=False):
    # Reading the JSON file
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    # Creating a configparser object
    config = configparser.ConfigParser()
    
    # Adding a default section for the JSON data
    config.add_section('default')
    for key, value in json_data.items():
        if isinstance(value, (dict, list)):
            # If the value is a dictionary or list, convert it to a string
            config.set('default', key, json.dumps(value))
        else:
            config.set('default', key, str(value))

    if debug:
        config.write(sys.stdout)

    return config

def partition_events(num_threads, base_fls_file):
    # Read the contents of the input file
    with open(base_fls_file, 'r') as file:
        lines = file.readlines()

    # Calculate the number of lines per chunk
    lines_per_chunk = len(lines) // num_threads

    # Split the file and write the chunks
    for i in range(num_threads):
        chunk = lines[i*lines_per_chunk : (i+1)*lines_per_chunk]

        # Handle the last chunk to include any remaining lines
        if i == num_threads - 1:
            chunk = lines[i*lines_per_chunk :]

        # Write the chunk to a new file
        with open(f'HFL{i+1}.fls', 'w') as chunk_file:
            if not i==0:
                chunk_file.writelines(['./work/portfolio.grd\n'])
            chunk_file.writelines(chunk)

def set_number_of_samples(nSamples, debug=False):
    """
    Function to modify the redloss.cf file in place, setting SIM_ACC to specified number.

    Parameters:
    base_cf_filepath (str): Path to the baseline configuration file (redloss.cf).
    nSamples (int): Number of samples.
    """

    # Import the baseline configuration file
    with open('redloss.cf', 'r') as file:
        content = file.readlines()
    # Iterate through each line of the base file and modify as required
    for line in content:
        if 'OPT_SIMACC' in line:
            key, _ = line.split(',')
            new_line = f"{key},{nSamples}\n"
            if debug:
                logging.info(f'Successfully set the number of samples to {nSamples}')

def partition_redloss_config(num_threads, base_cf_filepath):
    """
    Function to create multiple redloss configuration files based on a baseline file.

    Parameters:
    base_cf_filepath (str): Path to the baseline configuration file (redloss.cf).
    num_threads (int): Number of configuration files to create.
    """

    # Import the baseline configuration file
    with open(base_cf_filepath, 'r') as file:
        base_content = file.readlines()

    # Iterate through the number of threads and create new files
    for i in range(num_threads):
        new_content = []

        # Iterate through each line of the base file and modify as required
        for line in base_content:
            if 'OPT_MAPDATAFILELIST' in line:
                key, _ = line.split(',')
                new_line = f"{key},./HFL{i+1}.fls\n"
            elif 'OPT_FIFO' in line:
                key, _ = line.split(',')
                new_line = f"{key},fifo_p{i+1}\n"
            else:
                new_line = line
            new_content.append(new_line)

        # Write the new content to a file
        with open(f"redloss{i+1}.cf", 'w') as new_file:
            new_file.writelines(new_content)

def read_config(file_path, logfile=None):
    config = configparser.ConfigParser()
    config.read(file_path)
    msg = []
    parameters = {}
    test_cases = {}
    msg.append('Parsing sections from the provided config.cfg file...')
    for section in config.sections():
        if section == "general":
            # Loop through each parameter in a section
            for key, value in config.items(section):
                parameters[key] = value
        else:
            test_cases[section] = {}
            for key, value in config.items(section):
                test_cases[section][key] = value
            assert len(test_cases[section])>=6, f"Missing required arguements for case: {section}"
            msg.append(f'Parsed case <{section}>')
    
    if logfile is not None:
        [log_to_file(m, logfile) for m in msg]
    
    return parameters, test_cases

def run_getzones(zones_path, shortlist_zone_idx, ruptures_path, map_files_dir):
    command = f'getzones work/coordinates.txt {zones_path} {shortlist_zone_idx}'
    process = subprocess.Popen(command,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    if not process.returncode == 0:
        logging.info('stderr: {}'.format(stderr.decode()))
        logging.info('stdout: getzones econountered an error, exiting program!')
        return False

    shortlist_ruptures(rupture_zone_idx_file=ruptures_path,
                            shortlisted_zones_idx=shortlist_zone_idx,
                            map_files_dir=map_files_dir,
                            out_filename='work/mapBins.fls')
    
    process.terminate()

    return True


def make_dirs():
    if not os.path.exists('output'):
        os.makedirs('output')

    if not os.path.exists('figures'):
        os.makedirs('figures')

def make_redloss_build_cf(workDir, generalParams: dict, buildParams: dict):
    msg = []
    filepath_base_cf = os.path.join(workDir, "redloss.cf")
    filepath_build_cf = os.path.join(workDir, "redloss_build.cf")
    shutil.copy(filepath_base_cf, filepath_build_cf)

    # Modify/add OPT provided in 'build':
    for key in buildParams:
        if key[0:3] == 'OPT':
            m = set_opt(filepath_build_cf, key, buildParams[key])
            msg = msg + m

    return msg


def make_redhaz_build_cf(workDir, generalParams: dict, buildParams: dict):
    flags = ['OPT_RUPFILE', 'OPT_COUNTRY']
    msg = []
    filepath_base_cf = os.path.join(workDir, "redhaz.cf")
    filepath_build_cf = os.path.join(workDir, "redhaz_build.cf")
    shutil.copy(filepath_base_cf, filepath_build_cf)

    # Modify/add OPT provided in 'build':
    for key in buildParams:
        if key in flags:
            m = set_opt(filepath_build_cf, key, buildParams[key])
            msg = msg + m

    return msg


def make_redexp_build_cf(workDir, generalParams: dict, buildParams: dict):
    flags = []
    msg = []
    filepath_base_cf = os.path.join(workDir, "redexp.cf")
    filepath_build_cf = os.path.join(workDir, "redexp_build.cf")
    shutil.copy(filepath_base_cf, filepath_build_cf)
    msg.append(
        "INFO: redexp_build.cf is copied from redexp.cf. No changes have been made.")
    return msg


def set_opt(file_name, opt, new_value):
    """Sets requested OPT to <new_value>. If the OPT does not exists, creates and adds.
    Keyword arguments:
    file_name (str) -- .cf file to modify
    opt (str) -- OPT flag to modify (exact parameter name required)
    new_value (str) -- new value for <opt>
    Return: nothing
    """
    msg = []
    lines = []
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if opt in line:
            value = line.split(",")[1].strip()
            msg.append(f"INFO: |---> Current {opt} value: {value}")
            lines[i] = line.replace(value, new_value)
            msg.append(
                f"INFO: |---> {opt} parameter specified in the {file_name} has been successfully set to {new_value}")
            break
    if msg == []:
        if lines == []:
            raise ValueError("Something went wrong with .cf file overwrite!")
        lines.append(f"{opt},{new_value}")
        msg.append(
            f"INFO: |---> {opt} parameter specified in the {file_name} has been successfully set to {new_value}")

    with open(file_name, 'w') as f:
        f.writelines(lines)

    return msg


def set_datadir(target_path):
    """Sets OPT_DATADIR flag to provided value
    Keyword arguments:
    target_path -- Path to data (damage functiosn) files
    Return: nothing
    """
    file_name = "redloss.cf"
    with open(file_name, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "OPT_DATADIR" in line:
            value = line.split(",")[1].strip()
            print(f"INFO: |---> Current datadir: {value}")
            lines[i] = line.replace(value, target_path)
            print(
                f"INFO: |---> OPT_DATADIR parameter specified in the redloss.cf has been successfully set to {target_path}")
            break

    with open(file_name, 'w') as f:
        f.writelines(lines)

def setup_redcat_spatialcorr(coord_filepath, redcat_bins_dir, grid_size=0.025):
    """"Should be called from a model run directory
    
    Keyword arguments:
    argument -- 
    Return: log message
    """
    bins = redcat_bins_dir
    
    #1) Create RF across unif. grid whose limiits are derived based on input portfolio coordinates
    #a) Py function to produce grid csv as input
    generate_uniform_grid(coord_filepath, grid_size, './work/grid.csv')
    
    #b) Create .grd file:                               REDHazOQ -f redhazoq-creategrid.cf -> work/grid.csv
    os.system(f'{bins}/REDHazOQ -f redhazoq-creategrid.cf')
    
    #c) Run REDField (create RF)                        REDField -f redfield-create.cf -> work/random-fields.rfd
    os.system(f'{bins}/REDField -f redfield-create.cf')
    
    #2) Interpolate: run REDField (interpol).               REDField -f redfield-int.cf (Save -> work/random-fields-int.rfd)
    os.system(f'{bins}/REDField -f redfield-int.cf')
    
    #3) Mod redloss.cf to add  OPT_RANDOMDATAFILE,work/random-fields-int.rfd
    msg = set_opt("redloss.cf", "OPT_RANDOMDATAFILE", 'work/random-fields-int.rfd')
    
    return msg

def shortlist_ruptures(rupture_zone_idx_file, shortlisted_zones_idx, map_files_dir, out_filename):
    """Reads in the baseline model ruptures.fls and writes out a shortlistedRuptures.fls 
    based on the shortlist (rupture) indices.
    """
    msg = []
    # 1. Import the 'ruptures.fls' file
    df_rup_zone = pd.read_csv(rupture_zone_idx_file, header=None, names=['rup', 'id'])

    # 2. Import the 'shortlisted.idx' file
    zone_keys = list(pd.read_csv(shortlisted_zones_idx, header=None).squeeze('columns'))

    shortlisted_ruptures = df_rup_zone[df_rup_zone['id'].isin(zone_keys)]['rup'].tolist()

    # 3. Export the shortlisted filenames to the desired directory
    with open(out_filename, 'w') as f:
        for rup in shortlisted_ruptures:
            f.write(f'{map_files_dir}/{rup}.gmap\n')

    msg.append(f"Shortlisted ruptures have been written to {out_filename}")

    return msg

def _read_legacy_epc(filepath,isExposureRun=False):
    num_cols = 4
    loss_col_idx = 3
    with open(filepath, 'r') as f:
        # Read and skip header line
        first_line = f.readline().strip()
        
        if isExposureRun:
            num_cols = len(first_line.split(','))
            loss_col_idx = num_cols - 1

        # Read probability and loss values
        return_period, loss = np.loadtxt(
            f, delimiter=',', usecols=(2, loss_col_idx), unpack=True)
        
    return 1.0 / return_period, loss


def read_oasis_epc(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError()
    with open(filepath) as f:
        # Skip header line
        next(f)
        # Read columns 3 and 4 where column 2 is equal to '2'
        return_period, loss = [], []
        for line in f:
            columns = line.split(',')
            if columns[1] == '2':
                return_period.append(float(columns[2]))
                loss.append(float(columns[3]))
    return return_period, loss

def get_exposure_condensed(df_expo):
    df_expo['TRV'] = df_expo[[col for col in df_expo.columns if '_COST' in col]].sum(axis=1)
    df_condensed = df_expo[['ID', 'LAT', 'LON', 'TRV']] 
    df_condensed = df_condensed.rename(columns={'ID': 'Policy_ID'}) # unfortunately    
    df_condensed.columns = ['Policy_ID','LAT','LON', 'TRV']
    return df_condensed

def get_exposure_TRV(_df_expo):
    _df_expo['TRV'] = _df_expo[[col for col in _df_expo.columns if '_COST' in col]].sum(axis=1)
    df_trv = _df_expo[['ID', 'TRV']] 
    df_trv = df_trv.rename(columns={'ID': 'Policy_ID'}) # unfortunately
    return df_trv

def import_single_chfile(chk_filepath, df_expo_condense, df_expo):
    """
    Single chk file read and process implementation. TRV: Total Replacement Value
    """

    # load the data skipping the first row (row indexing starts from 0 in python)
    data = pd.read_csv(chk_filepath, skiprows=[0], delimiter=',', comment=None)
    ids = data['istat']

    # 1) Exposure DF manipulation
    df_trv = get_exposure_TRV(df_expo)
    
    # 2) chk file - loss sum and index association
    loss_columns = [col for col in data.columns if 'loss' in col]
    loss_data = data[loss_columns]
    loss_data['Loss'] = loss_data.sum(axis=1)
    loss_data.insert(0, "Policy_ID", ids) # Can do this since using the same dataframe. Indexing is the same
    
    # 3) Merge DFs on IDs
    df = pd.merge(loss_data, df_expo_condense, on='Policy_ID')

    return df

def import_aal_maps(condensed_expo_df, read_dir, save_dir):
    """
    Reads all .aal files in specified directory and plots maps
    """
    dfs = {}
    aal_stats = {}
    for dirName, subdirList, fileList in os.walk(read_dir):
        for fname in fileList:         
            if not fname.endswith('.aal'):
                continue
            filepath = os.path.join(read_dir, fname)
            tag = fname[:-5]
            # D a t a  P r o c c e s s i n g --------------------------------
            data = pd.read_csv(filepath, sep=r',')
            data.columns = [col.strip() for col in data.columns]

            # select the columns of interest
            ids = data['Point_id']
            dmg_columns = [col for col in data.columns if r'o]' in col]
            loss_columns = [col for col in data.columns if r'$' in col]
            dmg_data = data[dmg_columns]
            loss_data = data[loss_columns]
            loss_data.insert(0, "Policy_ID", ids)

            # Merge results with portfolio based on Policy_ID
            df = pd.merge(loss_data, condensed_expo_df, on='Policy_ID')
            dfs[tag] = df

            # Finally, compute aal stats by class:
            aal_stats[tag] = df.sum().drop(['Policy_ID', 'LAT', 'LON'])

    return dfs, aal_stats
            
def convert_exposure_to_all_unknown(df_expo, output_filepath):
    # Get all the columns that match the pattern VC*_ID  & Infer N of VVs
    vc_id_cols = [col for col in df_expo.columns if col.startswith('VC') and col.endswith('_ID')]
    N = len(vc_id_cols)
    # Get total expo value before manipulation
    total_cost_before = df_expo[[col for col in df_expo if col.endswith('_COST')]].sum().sum()

    # Perform the operations with the inferred N
    # 1) Modify all values under the last VC*_ID columns to 'UNK'.
    df_expo[f'VC{N}_ID'] = 'UNK'

    # 2) Modify all values under the last VC*_COST to the sum of the values under all VC*_COST columns
    cost_cols = [col for col in df_expo if col.startswith('VC') and col.endswith('COST')]
    df_expo[f'VC{N}_COST'] = df_expo[cost_cols].sum(axis=1)

    # 3) Modify all values under the last VC*_AREA to the sum of the values under all VC*_AREA columns
    area_cols = [col for col in df_expo if col.startswith('VC') and col.endswith('AREA')]
    df_expo[f'VC{N}_AREA'] = df_expo[area_cols].sum(axis=1)

    # 4) Set all other VC*_COST & VC*_AREA  columns to 0.0, excluding the last one.
    for col in cost_cols[:-1]:
        df_expo[col] = 0.0

    for col in area_cols[:-1]:
        df_expo[col] = 0.0
        
    assert abs(total_cost_before - df_expo[f'VC{N}_COST'].sum()) < 0.1, "WARNING: A discrepancy arose upon exposure conversion to all unknown! Review your results!"
    # 5) Finally, write out modified UNK exposure 
    df_expo.to_csv(output_filepath, sep='\t', index=False)

    
def find_and_replace(filename, search_string, replace_string):
    """
    Replaces all occurrences of a specified string in a file with another string.

    Parameters:
        filename (str): The name of the file to edit.
        search_string (str): The string to search for in the file.
        replace_string (str): The string to replace the search_string with.

    Returns:
        None: The function modifies the file in place and does not return any value.
    """
    # Open the file in read mode and read its content into a variable
    with open(filename, 'r') as file:
        file_contents = file.read()

    # Replace all occurrences of search_string with replace_string
    new_contents = file_contents.replace(search_string, replace_string)

    # Open the file in write mode and write the new content back to it
    with open(filename, 'w') as file:
        file.write(new_contents)

def find_and_replace_after_equals(file_path, sought_key, new_value):
    # Open the file and read all its lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Modify the target line
    for index, line in enumerate(lines):
        if line.startswith(sought_key):
            # Extract the key and the value separately
            key, value = line.split('=', 1)
            # Replace the value with the new path
            lines[index] = '{}="{}"\n'.format(key, new_value)
            break
    
    # Write (overwrite) the modified lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

def parse_longitude_latitude(input_str):
    try:
        # Split the input string by a comma and strip any leading/trailing whitespace
        parts = input_str.strip().split(',')
        
        # Check if we have exactly two parts (longitude and latitude)
        if len(parts) != 2:
            raise ValueError("Input must contain exactly two values separated by a comma.")
        
        # Convert the two parts to decimal values
        longitude = float(parts[0].strip())
        latitude = float(parts[1].strip())
        
        # Return the longitude and latitude as a list of decimals
        return [longitude, latitude]
    
    except ValueError as e:
        print('Something wrong with the passed lat-lon string.')
        raise ValueError(e)

def purge_folders(delete_these):
    for p in delete_these:
        if os.path.exists(p):
                shutil.rmtree(p)

def filter_exposure_by_coordinates(filepath, _lon_range, _lat_range, output_filepath, drop_dupes=False):
    """Filters passed exposure file and writes it out. 
    
    Keyword arguments:
    lon_range -- string representation of <lon-min,lon-max>
    lat_range -- string representation of <lat-min,lat-max>
    Return: return_description
    """
    output_dir = os.path.dirname(output_filepath)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Parse lon_range and lat_range inputs
    lon_range = parse_longitude_latitude(_lon_range)
    lat_range = parse_longitude_latitude(_lat_range)

    # Read the exposure file into a pandas dataframe
    df = pd.read_csv(filepath, delimiter='\t')
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Filter the dataframe based on the given latitude range (whitespaces are there on purpose..)
    filtered_df = df[(df['LON'] >= lon_range[0]) & (df['LON'] <= lon_range[1]) & (df['LAT'] >= lat_range[0]) & (df['LAT'] <= lat_range[1])]

    len_og = len(filtered_df)

    if drop_dupes:
        print('CAUTION: DROPPING DUPES from the source file - ' + filepath)
        filtered_df.drop_duplicates(subset=['LON', 'LAT'], keep='first', inplace=True)
        dropped = len_og - len(filtered_df)
        print('---> Dropped ' + str(dropped) + ' lines out of ' + str(len_og) + ' (kept first)!')

        final_dupes = filtered_df.duplicated(subset=['LON', 'LAT'])
        print(f"Remaining duplicates: {final_dupes.sum()}")

    # Write the filtered exposure to a new CSV file
    filtered_df.to_csv(output_filepath, sep='\t', index=False)
    
    return filtered_df


def fetch_coordinates_from_location_file(filepath, out_filepath):
    # Reading the CSV file into a DataFrame
    df = pd.read_csv(filepath, delimiter=',')

    # Filtering only 'LON' and 'LAT' columns. Again whitespaces are there on purpose..,..
    filtered_df = df[['Longitude', 'Latitude']]

    # Writing these columns back to a new CSV file without headers
    filtered_df.to_csv(out_filepath, index=False, header=False)

    return out_filepath

def fetch_coordinates_from_exposure(filepath, out_filepath):
    # Reading the CSV file into a DataFrame
    df = pd.read_csv(filepath, delimiter='\t')

    # Filtering only 'LON' and 'LAT' columns. Again whitespaces are there on purpose..,..
    filtered_df = df[['LON', 'LAT']]

    # Writing these columns back to a new CSV file without headers
    filtered_df.to_csv(out_filepath, index=False, header=False)

    return out_filepath

def write_average_epc(epcs, output_dir, output_filename):
    # Step 1: Generate fixed X values using logarithmic scale
    x_fixed = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    # Step 2: Interpolate Y values for fixed X-values and store results
    interpolated_loss_arrays = []
    for epc in epcs:
        rp_array, loss_values = epc
        interpolation_function = interp1d(rp_array, loss_values, kind='linear', fill_value="extrapolate")
        interpolated_loss = interpolation_function(x_fixed)
        interpolated_loss_arrays.append(interpolated_loss)

    # Step 3: Compute the average loss array
    average_loss_array = np.mean(interpolated_loss_arrays, axis=0)
    non_negative = np.where(average_loss_array < 0, 0, average_loss_array)

    # Step 4: Writing out the interpolated X-Y arrays in a CSV file
    csv_filename = os.path.join(output_dir, output_filename)
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Return Period', 'Average Loss'])
        for x_val, avg_loss in zip(x_fixed, non_negative):
            csvwriter.writerow([x_val, avg_loss])    

def waisman_aal(read_dir, save_dir, lob, isWeisman=False, all_sims=True):
    if not os.path.exists(read_dir):
        raise NotADirectoryError()
    portfolio = pd.read_csv('./inputs/portfolio.csv')
    Total_value = sum(portfolio.CovAValue.values)
    legacy_epcs = []
    aal = []
    if all_sims:
        fig, ax = plt.subplots(figsize=FIG_PARAMS["f_size"])
        for dirName, subdirList, fileList in os.walk(read_dir):
            for fname in fileList:
                if fname.startswith('GM_Sim_0') and fname.endswith('.aal'):
                    aal_file = pd.read_csv(read_dir+'/'+fname, sep=r',')
                    aal.append(sum(aal_file['  GU Losses[$]']))
    aal = sum(aal)/len(aal)/Total_value
    if os.path.exists(save_dir+'/'+'aalr-'+lob+'.txt'):
        os.remove(save_dir+'/'+'aalr-'+lob+'.txt')
    with open(save_dir+'/'+'aalr-'+lob+'.txt', 'w') as aal_file:
        aal_file.write(str(aal))
    aal_file.close()
    return aal
