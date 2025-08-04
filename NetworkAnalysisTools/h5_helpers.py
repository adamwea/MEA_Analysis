# imports
import os
import numpy as np
import h5py
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.core as si
import logging

def extract_raw_h5_filepaths(h5_dir):
    #walk through the directory and find all .h5 files
    h5_subdirs = []
    if h5_dir.endswith('.h5'): return [h5_dir]
    for root, dirs, files in os.walk(h5_dir): 
        for file in files:
            if file.endswith('.h5'):
                h5_subdirs.append(os.path.join(root, file))
    # assert len(h5_subdirs) > 0, "No .h5 files found in the directory."
    # assert len(h5_subdirs) == 1, "Ambiguous file selection. Multiple .h5 files found in the directory."
    return h5_subdirs

def extract_recording_details(h5_dirs):
    """
    This function extracts details about each recording from the given h5 directories.
    The details include the file path, run ID, scan type, chip ID, and recording date.

    Parameters:
    h5_dirs: The list of h5 directories.

    Returns:
    records: The list of dictionaries, where each dictionary contains details about a recording.
    """

    # If h5_dirs is a string, convert it to a list with a single element
    if isinstance(h5_dirs, str):
        h5_dirs = [h5_dirs]

    #logger.info(f"Extracting recording details from h5 directories:")
    print(f"Extracting recording details from h5 directories:")
    records = []
    for h5_dir in h5_dirs:
        try: assert '.h5' in h5_dir, "The input is not a list of h5 directories."
        except: 
            try: 
                h5_subdirs = extract_raw_h5_filepaths(h5_dir)
                assert len(h5_subdirs) > 0, "No .h5 files found in the directory."
                assert len(h5_subdirs) == 1, "Ambiguous file selection. Multiple .h5 files found in the directory."
                #h5_dir = h5_subdirs[0]
            except: 
                #logger.error("Some error occurred during the extraction of .h5 file paths.")
                print("Some error occurred during the extraction of .h5 file paths.")
                continue

        parent_dir = os.path.dirname(h5_dir)
        runID = os.path.basename(parent_dir)

        grandparent_dir = os.path.dirname(parent_dir)
        scan_type = os.path.basename(grandparent_dir)

        great_grandparent_dir = os.path.dirname(grandparent_dir)
        chipID = os.path.basename(great_grandparent_dir)

        ggg_dir = os.path.dirname(great_grandparent_dir)
        date = os.path.basename(ggg_dir)
        
        gggg_dir = os.path.dirname(ggg_dir)
        projectName = os.path.basename(gggg_dir)

        record = {
            'h5_file_path': h5_dir, 
            'runID': runID, 
            'scanType': scan_type, 
            'chipID': chipID,
            'date': date,
            'projectName': projectName,                  
            }
        records.append(record)

    return records

def load_recordings(h5_file_path, stream_select=None, logger=None, max_id=None):
    """
    Loads all recordings from the specified file and identifies whether it's a MaxOne or MaxTwo file type.

    Parameters:
    h5_file_path (str): The path to the file to read from.
    stream_select (int, optional): Specific stream ID to select. Default is None.
    logger (Logger, optional): Logger for logging. Default is None.

    Returns:
    tuple: 
    (1) MaxID indicating file type (1 for MaxOne, 2 for MaxTwo) 
    (2) List of recordings loaded 
    (3) Total number of streams
    (4) List of recording counts per stream
    """
    # if logger is None:
    #     logger = logging.getLogger(__name__)
    print(f"Loading recordings from {h5_file_path}...")
    
    h5_details = extract_recording_details(h5_file_path)
    h5_full_path = h5_details[0]['h5_file_path']
    scanType = h5_details[0]['scanType']
    print(f"Scan Type: {scanType}")
    recordings = {}
    rec_counts = []

    if max_id is None:
        try:
            recording = se.read_maxwell(h5_full_path, rec_name='rec0000', stream_id='well000')
            MaxID = 2
        except:
            try:
                recording = se.read_maxwell(h5_full_path, rec_name='rec0000')
                MaxID = 1
            except:
                logger.error("Error: Unable to read recording. Cannot identify as MaxOne or MaxTwo.")
                raise Exception("Error: Unable to read recording. Cannot identify as MaxOne or MaxTwo.")
                #return 0, [], 0, []
    else:
        assert max_id in [1, 2], "max_id must be either 1 or 2."
        MaxID = max_id

    if MaxID == 1:
        #logger.info("MaxOne Detected.")
        print("MaxOne Detected.")
        expected_rec_count = len(h5py.File(h5_full_path)['recordings'])
        recs = []
        for rec_count in range(expected_rec_count):
            try:
                rec_name_str = f'rec{rec_count:04}'
                recording = se.read_maxwell(h5_full_path, rec_name=rec_name_str)
                recs.append(recording)
            except:
                continue
        #recordings['well000'] = {'recording_segments': recs}
        recordings['well000'] = recs # NOTE: this should be a list of recording segments. Should be a list of 1 for network scans.
                                        # activity scans and axon tracking scans will usually
        return MaxID, recordings, 1, [expected_rec_count]

    elif MaxID == 2:
        logger.info("MaxTwo Detected.")
        with h5py.File(h5_full_path, 'r') as h5_file:
            expected_well_count = len(h5_file['wells'])
            streams_to_process = range(expected_well_count) if stream_select is None else [stream_select]
            for stream_id in streams_to_process:
                stream_id_str = f'well{stream_id:03}'
                try: expected_rec_names = list(h5_file['wells'][stream_id_str].keys())
                except: 
                    print(f"Stream ID: {stream_id_str} not found in h5 file.")
                    continue
                expected_rec_count = len(expected_rec_names)
                valid_rec_count = 0
                recs = []
                for rec_count in range(expected_rec_count):
                    try:
                        rec_name_str = f'rec{rec_count:04}'
                        if rec_name_str not in expected_rec_names:
                            continue
                        recording = se.read_maxwell(h5_full_path, rec_name=rec_name_str, stream_id=stream_id_str)
                        recs.append(recording)
                        valid_rec_count += 1
                        logger.debug(f"Stream ID: {stream_id_str}, Recording: {rec_name_str} loaded.")
                    except Exception as e:
                        logger.warning(e)
                        if "Unable to open object" in str(e):
                            logger.debug("This error may not be an issue. For some reason, some wells say they have segments when they don't.")
                rec_counts.append(valid_rec_count)
                #recordings[stream_id_str] = {'recording_segments': recs}
                recordings[stream_id_str] = recs # NOTE: this should be a list of recording segments. Should be a list of 1 for network scans.
                                                # activity scans and axon tracking scans will usually
                if stream_select is not None:
                    break
        return MaxID, recordings, expected_well_count, rec_counts

# helper functions for loading h5 files and extracting data objects
def get_data_obj_groups_v3(h5_paths, sorted_output_folders):
    # aw 2025-02-11
    # get recording details from each h5 file, check for match where all details match in sorted_output_folders
    # this way, we pair up recordings with their corresponding sorting output
    # get paired data objects - network analysis requires both recording and sorting objects
    
    # Subfunctions ======================================
    def load_three_objects(h5_path, sorted_output_folder, recording_details):
        # this function expects to load one sorting_obj, one recording_obj, and waveform data for the related well.
        # there may be any number of rec_segments in the recording_obj, but only one sorting_obj
        
        # get stream_select from sorted_output_folder path
        # look for the word 'well' in the string. It will beb followed by three digits.
        # get the int value of those digits. Should be 0-5
        stream_select = int(sorted_output_folder.split('well')[1][:3])
        wellid = f'well{str(0).zfill(2)}{stream_select}'
        
        # load recording object
        try:
            _, well_recs, _, _ = mea.load_recordings(h5_path, stream_select=stream_select)
            rec_segments = well_recs[wellid]
        except Exception as e: rec_segments = (e, traceback.format_exc()) # put error in rec_segments for debugging
        
        # load sorting object
        try: sort_obj = mea.load_kilosort2_results(sorted_output_folder)
        except Exception as e: sort_obj = (e, traceback.format_exc()) # put error in sort_obj for debugging
        
        # # load waveform data
        # try:
        #     waveform_output_dir = sorted_output_folder.replace('sorter_output', '')
        #     waveform_output_dir = waveform_output_dir.replace('sorted', 'waveforms')
        #     sort_obj.register_recording(rec_segments[0])
        #     waveform_extractor = mea.load_waveforms(waveform_output_dir, sorting=sort_obj)
        # except Exception as e: waveform_extractor = (e, traceback.format_exc()) # put error in waveform_extractor for debugging
        
        #import traceback
        #from spikeinterface.analysis import load_sorting_analyzer

        # load waveform data
        try:
            #waveform_output_dir = sorted_output_folder.replace('sorter_output', '')
            #waveform_output_dir = waveform_output_dir.replace('sorted', 'waveforms')
            sorting_analyzer_output_dir = sorted_output_folder.replace('sorter_output', '')
            sorting_analyzer_output_dir = sorting_analyzer_output_dir.replace('sorted', 'analyzer')
            
            # register rec
            sort_obj.register_recording(rec_segments[0])

            # load SortingAnalyzer from folder
            sorting_analyzer = si.load_sorting_analyzer(sorting_analyzer_output_dir)

            # get waveforms extension
            # waveform_extractor = sorting_analyzer.get_extension("waveforms")

        except Exception as e:
            #waveform_extractor = (e, traceback.format_exc())  # put error in waveform_extractor for debugging
            sorting_analyzer = (e, traceback.format_exc())  # put error in waveform_extractor for debugging
        
        # return paired objects
        # return (rec_segments, sort_obj, waveform_extractor), recording_details
        return (rec_segments, sort_obj, sorting_analyzer), recording_details
    
    # Main ======================================   
    well_data_list = []
    path_pairs = []
    for h5_path in h5_paths:
        
        # get recording details
        recording_details = mea.extract_recording_details(h5_path)[0] # NOTE: this works for a list of dirs or a single dir - but treats single dir as a list of a single dir
        
        #remove h5_file_path from recording_details - this wont match, this is the old path
        h5_path = recording_details.pop('h5_file_path')
        
        for sorted_output_folder in sorted_output_folders:         
            
            # shortform
            found = all([f'/{recording_details[key]}/' in sorted_output_folder for key in recording_details.keys()])
            
            if found:
                path_pairs.append((h5_path, sorted_output_folder))                       
                well_data, recording_details = load_three_objects(h5_path, sorted_output_folder, recording_details) # NOTE: on error, well_data will be exception information
                well_data_list.append(well_data)
                
    return well_data_list, recording_details, path_pairs
