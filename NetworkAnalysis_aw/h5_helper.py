# imports
import os
import numpy as np


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
