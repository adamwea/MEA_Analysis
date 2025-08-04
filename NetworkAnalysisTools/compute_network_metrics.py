'''
Get network metrics for simulated or experimental data.

Ideally this minimizes changes to computing network metrics for either simulated or experimental data in the future by consolidating the process into a single function.
Here, we focus on pre-processing either type of data into like formats before computing network metrics - which should be mostly identical downstream.

Core philosophies here:
- every subfunction should be able to handle either simulated or experimental data.
- use as many subfuncs as possible to keep the main function clean and readable.
- use as many shared functions as possible to keep the code DRY. (Don't Repeat Yourself)
- keep the main function as simple as possible.
- always put try and except logic in subfunctions to catch errors and print them out. Use traceback to get the full error message.
'''
# Imports =====================================================================
import traceback
import numpy as np
from MEA_Analysis.NetworkAnalysis_aw import network_metrics_helper as nmh
import MEA_Analysis.NetworkAnalysis_aw.compute_spike_metrics as csm
import MEA_Analysis.NetworkAnalysis_aw.compute_burst_metrics as cbm
import os

# Subfunctions =============================================================
def _validate_inputs(kwargs):
    '''Validate input data.'''
    source = kwargs.get('source', None)
    assert source in ['simulated', 'experimental'], 'Invalid data source'
    if source == 'simulated':
        assert 'simData' in kwargs, 'No simData provided'
        assert 'popData' in kwargs, 'No popData provided'
        assert 'cellData' in kwargs, 'No cellData provided'
    elif source == 'experimental':
        assert 'sorting_object' in kwargs, 'No sorting object provided'
        assert 'recording_object' in kwargs, 'No recording object provided'
        #assert 'wf_extractor' in kwargs, 'No waveform extractor provided'
        assert 'sorting_analyzer' in kwargs, 'No sorting analyzer provided'
    #return source, kwargs

def _initialize_data_dict(kwargs):
    '''Initialize dictionary to store network data.'''
    global network_data
    network_data = {}
    
    source = kwargs.get('source', None)
    
    try:
        if source == 'simulated':
            simData = kwargs['simData']
            network_data['source'] = 'simulated'
            network_data['spiking_data'] = {}
            if 'sim_data_path' in kwargs:
                network_data['sim_data_path'] = kwargs['sim_data_path']
            elif 'data_file_path' in kwargs:
                network_data['sim_data_path'] = kwargs['data_file_path']
            if network_data['sim_data_path'] is None:
                raise ValueError('No sim data path provided in kwargs.')
            # network_data['sim_data_path'] = kwargs.get('sim_data_path', None)
            # network_data['sim_data_path'] = kwargs.get('data_file_path', None)
            return network_data
        elif source == 'experimental':
            sorting_object = kwargs['sorting_object']
            recording_object = kwargs['recording_object']
            network_data['source'] = 'experimental'
            assert sorting_object is not None, 'No sorting object provided'
            network_data['recording_path'] = recording_object._kwargs.get('file_path', None)
            network_data['sorting_output'] = sorting_object._kwargs.get('folder_path', None)
            assert network_data['sorting_output'] is not None, 'No sorting object source found'
            #network_data['waveform_output'] = network_data['sorting_output'].replace('sorted', 'waveforms').replace('sorter_output', '')
            network_data['sorting_analyzer_output'] = network_data['sorting_output'].replace('sorted', 'analyzer').replace('sorter_output', '')
            network_data['network_metrics_output'] = network_data['sorting_output'].replace('sorted', 'network_metrics').replace('sorter_output', '')
            network_data['spiking_data'] = {}
            network_data['data_file_path'] = kwargs.get('data_file_path', None)
            if network_data['data_file_path'] is None:
                raise ValueError('No data file path provided in kwargs.')
            #network_data['spiking_metrics'] = {}
            #network_data['bursting_metrics'] = {}
            return network_data
        else:
            raise ValueError('Invalid source specified. Options are: "simulated" or "experimental".')
    except Exception as e:
        print(f'Error initializing network data: {e}')
        traceback.print_exc()
        pass

def _update_kwargs(conv_params, mega_params, source, kwargs):
    '''Update kwargs with additional parameters.'''
    try:
        if source == 'simulated':
            kwargs['source'] = source
            kwargs['conv_params'] = conv_params
            kwargs['mega_params'] = mega_params
        elif source == 'experimental':
            # sorting_object = kwargs['sorting_object']
            # recording_object = kwargs['recording_object']
            # wf_extractor = kwargs['wf_extractor']
            kwargs['source'] = source
            kwargs['conv_params'] = conv_params
            kwargs['mega_params'] = mega_params
            # kwargs['sorting_object'] = sorting_object
            # kwargs['recording_object'] = recording_object
            # kwargs['wf_extractor'] = wf_extractor
        return kwargs
    except Exception as e:
        print(f'Error updating kwargs: {e}')
        traceback.print_exc()
        pass

def _validate_conv_params(kwargs):
    '''
    Validate convolution parameters in kwargs.
    Raises:
        ValueError: If burst_params or hyperburst_params are not defined in kwargs.
    '''
    burst_params = kwargs.get('burst_params', None)
    hyperburst_params = kwargs.get('hyperburst_params', None)
    if burst_params is None:
        raise ValueError('burst_params must be defined in kwargs.')
    if hyperburst_params is None:
        raise ValueError('hyperburst_params must be defined in kwargs.')
    return kwargs

def _trim_simulated_data(network_data, kwargs):
    '''Trim simulated data to remove initial transient period.'''
    try:
        trim_start_time = kwargs['trimed_start_time']
        timeVector = network_data['timeVector']
        spike_times = network_data['spiking_data']['spike_times']
        spike_times_by_unit = network_data['spiking_data']['spiking_times_by_unit']
        # trim data
        timeVector = timeVector[timeVector >= trim_start_time]
        spike_times = spike_times[spike_times >= trim_start_time]
        for unit in spike_times_by_unit.keys():
            spike_times_by_unit[unit] = spike_times_by_unit[unit][spike_times_by_unit[unit] >= trim_start_time]
        network_data['timeVector'] = timeVector
        network_data['spiking_data']['spike_times'] = spike_times
        network_data['spiking_data']['spike_times_by_unit'] = spike_times_by_unit
        return network_data
    except Exception as e:
        print(f'Error trimming simulated data: {e}')
        traceback.print_exc()
        raise ValueError('Failed to trim simulated data.')
        #pass

def _format_raw_spike_data(network_data, kwargs):
    '''Initialize spike data for simulated or experimental data.'''
    try:
        source = kwargs['source']
        if source == 'simulated':
            simData = kwargs['simData']
            rasterData = simData.copy()
            if len(rasterData['spkt']) == 0:
                #print('No spike times found in rasterData')
                raise ValueError('No spike times found in rasterData')
            spike_times = np.array(rasterData['spkt']) / 1000
            timeVector = np.array(rasterData['t']) / 1000
            spike_times_by_unit = {int(i): spike_times[rasterData['spkid'] == i] for i in np.unique(rasterData['spkid'])}
            network_data['timeVector'] = timeVector
            # NOTE: idk, sampling_rate is kindof available in simData if I record full trace of a particular neuron. Not super important for now.
            network_data['sampling_rate'] = 'Not currently implemented for simulated data'
            network_data['spiking_data']['spike_times'] = spike_times
            network_data['spiking_data']['spiking_times_by_unit'] = spike_times_by_unit
            
            # TODO: put this into kwargs options later - for now, default
            kwargs['trim_simulated_data'] = True
            kwargs['trimed_start_time'] = 20
            if kwargs['trim_simulated_data']:
                network_data = _trim_simulated_data(network_data, kwargs)
                spike_times = network_data['spiking_data']['spike_times']
                timeVector = network_data['timeVector']
                spike_times_by_unit = network_data['spiking_data']['spiking_times_by_unit']
                network_data['trimmed'] = True
            else:
                network_data['trimmed'] = False
            
            return spike_times, timeVector, spike_times_by_unit, network_data
        elif source == 'experimental':
            sorting_object = kwargs['sorting_object']
            recording_object = kwargs['recording_object']
            sampling_rate = recording_object.get_sampling_frequency()
            spike_times = get_spike_times(recording_object, sorting_object, sampling_rate=sampling_rate) #seconds
            timeVector = get_time_vector(recording_object, sampling_rate=sampling_rate) #seconds
            spike_times_by_unit = get_spike_times_by_unit(sorting_object, sampling_rate=sampling_rate)
            network_data['timeVector'] = timeVector
            network_data['sampling_rate'] = sampling_rate
            network_data['spiking_data']['spike_times'] = spike_times
            network_data['spiking_data']['spiking_times_by_unit'] = spike_times_by_unit
            return spike_times, timeVector, spike_times_by_unit, network_data
    except Exception as e:
        print(f'Error initializing spike data: {e}')
        # if 'No spike times found in rasterData' in e:
        #     pass
        # traceback.print_exc()
        #pass
        return None, None, None, network_data
    
def _compute_burst_metrics_wrapper(network_data, kwargs):
    #extract bursting metrics from simulated data (but this one works for both simulated and experimental data)
    try: 
        # assert network data has spike_times and spiking_times_by_unit
        assert 'spike_times' in network_data['spiking_data'], 'No spike times found in network data'
        assert 'spiking_times_by_unit' in network_data['spiking_data'], 'No spiking times by unit found in network data'
        #assert 'sampling_rate' in network_data, 'No sampling rate found in network data'
        
        network_data = compute_burst_metrics(network_data, kwargs)
    except Exception as e:
        print(f'Error calculating bursting activity: {e}')
        #traceback.print_exc()
        pass
    
    return network_data
    
def _classify_units(network_data, kwargs):     
    # classify neurons
    source = kwargs['source']
    try:
        if source == 'simulated':
            cellData = kwargs['cellData']
            unit_types = {int(i['gid']): i['tags']['pop'] for i in cellData}
            network_data['unit_types'] = unit_types
        elif source == 'experimental':
            network_data = classify_neurons_v2(network_data, **kwargs)
            classification_data = network_data['classification_output']
            #network_data['unit_types'] = classification_data['classified_units']
            unit_types = {}
            for unit_id, unit in classification_data['classified_units'].items():
                desc = unit['desc']
                if 'inhib' in desc:
                    unit_types[unit_id] = 'I'
                elif 'excit' in desc:
                    unit_types[unit_id] = 'E'
                else:
                    unit_types[unit_id] = 'U'
            network_data['unit_types'] = unit_types
    except Exception as e:
        print(f'Error classifying neurons: {e}')
        #traceback.print_exc()
        pass
    
    return network_data
    
def _compute_dynamic_time_warping(network_data, kwargs):
    # dynamic time warping (dtw) - burst analysis
    try:
        network_data = dtw_burst_analysis(network_data, kwargs)
    except Exception as e:
        print(f'Error in dtw burst analysis: {e}')
        traceback.print_exc()
        pass  
    
    return network_data

def _get_unit_pops(network_data):
    try:
        print('Checking if unit pops provided in network data inputs...')
        unit_types = network_data['inputs'].get('unit_pops', None)
        assert unit_types is not None, 'No unit types found in network data. Please classify units first.'
    except AssertionError as e:
        try:
            print(f'unit_pops not found in network data inputs')
            print(f'Checking if unit pops derived in classification step...')
            unit_types = network_data['classification_output'].get('unit_pops', None)
            assert unit_types is not None, 'No unit types found in network data. Please classify units first.'
        except AssertionError as e:
            print(f'No unit pops found in network data inputs or classification output. Please classify units first.')
            raise ValueError('No unit types found in network data. Please classify units first.')
    
    return unit_types

def _summarize_metrics(network_data, kwargs):
    # TODO: parse this out into seperate functions, optimize.
    
    # fr
    #spiking_data = network_data['spiking_data']
    spiking_data = network_data.get('spk_metrics', None)
    if spiking_data is None:
        raise ValueError('No spiking data found in network data. Please compute spiking metrics first.')
    # frs = [i['fr'] for i in spiking_data['by_unit'].values()]
    # frs = [np.nan if i == np.inf else i for i in frs] #replace inf with nan
    # spiking_data['frs'] = {
    #     'data': frs if len(frs) > 0 else None,
    #     'mean': np.nanmean(frs) if len(frs) > 0 else None,
    #     'std': np.nanstd(frs) if len(frs) > 0 else None,
    #     'median': np.nanmedian(frs) if len(frs) > 0 else None,
    #     'cov': np.nanstd(frs) / np.nanmean(frs) if np.nanmean(frs) and len(frs) > 0 else None,
    #     'max': np.nanmax(frs) if len(frs) > 0 else None,
    #     'min': np.nanmin(frs) if len(frs) > 0 else None,
    # }
    
    # E/I/U frs
    # unit_types = network_data['unit_types']
    unit_types = _get_unit_pops(network_data)
    i_frs = [metrics['fr'] for i, metrics in spiking_data['by_unit'].items() if i in unit_types and unit_types[i] == 'I']
    e_frs = [metrics['fr'] for i, metrics in spiking_data['by_unit'].items() if i in unit_types and unit_types[i] == 'E']
    u_frs = [metrics['fr'] for i, metrics in spiking_data['by_unit'].items() if i not in unit_types]
    i_frs = [np.nan if i == np.inf else i for i in i_frs] #replace inf with nan
    e_frs = [np.nan if i == np.inf else i for i in e_frs]
    u_frs = [np.nan if i == np.inf else i for i in u_frs]
    spiking_data['i_frs'] = {
        'data': i_frs if len(i_frs) > 0 else None,
        'mean': np.nanmean(i_frs) if len(i_frs) > 0 else None,
        'std': np.nanstd(i_frs) if len(i_frs) > 0 else None,
        'median': np.nanmedian(i_frs) if len(i_frs) > 0 else None,
        'cov': np.nanstd(i_frs) / np.nanmean(i_frs) if np.nanmean(i_frs) and len(i_frs) > 0 else None,
        'max': np.nanmax(i_frs) if len(i_frs) > 0 else None,
        'min': np.nanmin(i_frs) if len(i_frs) > 0 else None,
    }
    spiking_data['e_frs'] = {
        'data': e_frs if len(e_frs) > 0 else None,
        'mean': np.nanmean(e_frs) if len(e_frs) > 0 else None,
        'std': np.nanstd(e_frs) if len(e_frs) > 0 else None,
        'median': np.nanmedian(e_frs) if len(e_frs) > 0 else None,
        'cov': np.nanstd(e_frs) / np.nanmean(e_frs) if np.nanmean(e_frs) and len(e_frs) > 0 else None,
        'max': np.nanmax(e_frs) if len(e_frs) > 0 else None,
        'min': np.nanmin(e_frs) if len(e_frs) > 0 else None,
    }
    spiking_data['u_frs'] = {
        'data': u_frs if len(u_frs) > 0 else None,
        'mean': np.nanmean(u_frs) if len(u_frs) > 0 else None,
        'std': np.nanstd(u_frs) if len(u_frs) > 0 else None,
        'median': np.nanmedian(u_frs) if len(u_frs) > 0 else None,
        'cov': np.nanstd(u_frs) / np.nanmean(u_frs) if np.nanmean(u_frs) and len(u_frs) > 0 else None,
        'max': np.nanmax(u_frs) if len(u_frs) > 0 else None,
        'min': np.nanmin(u_frs) if len(u_frs) > 0 else None,
    }

    # unit spiking summary metrics
    #sim_spike_data = simulated_metrics.get('spiking_data', {})
    sim_spike_data = network_data['spk_metrics']
    e_frs = sim_spike_data.get('e_frs', {}).get('data', [])
    i_frs = sim_spike_data.get('i_frs', {}).get('data', [])

    # get number of non-nan values in e_frs and i_frs
    num_e_frs = np.count_nonzero(~np.isnan(e_frs))
    num_i_frs = np.count_nonzero(~np.isnan(i_frs))
    network_data['spk_metrics']['num_e_firing'] = num_e_frs
    network_data['spk_metrics']['num_i_firing'] = num_i_frs

    # total expected number of excitatory and inhibitory units
    tot = len(e_frs) + len(i_frs)
    total_firing = num_e_frs + num_i_frs
    network_data['spk_metrics']['num_total_firing'] = total_firing

    #E/I total spike ratio
    # by_unit = sim_spike_data.get('by_unit', {})
    # Espikes = 0
    # Ispikes = 0
    # for unit, unit_type in network_data.get('unit_types', {}).items():
    #     if unit_type == 'E':
    #         # if unit not in e_frs: e_frs.append(unit)
    #         num_spikes = by_unit.get(unit, {}).get('num_spikes', 0)
    #         Espikes += num_spikes
    #     elif unit_type == 'I':
    #         #if unit not in i_frs: i_frs.append(unit)
    #         num_spikes = by_unit.get(unit, {}).get('num_spikes', 0)
    #         Ispikes += num_spikes
    # Espikes_per_unit = Espikes / num_e_frs if num_e_frs > 0 else 0
    # Ispikes_per_unit = Ispikes / num_i_frs if num_i_frs > 0 else 0
    # spiking_data['E_I_spike_ratio'] = Espikes_per_unit / Ispikes_per_unit if Ispikes_per_unit > 0 else None
    
    #E/I/U total spikes NOTE: this data is not super useful unless simulation time = experiment time
    E_spikes = [metrics['num_spikes'] for i, metrics in spiking_data['by_unit'].items() if i in unit_types and unit_types[i] == 'E']
    I_spikes = [metrics['num_spikes'] for i, metrics in spiking_data['by_unit'].items() if i in unit_types and unit_types[i] == 'I']
    U_spikes = [metrics['num_spikes'] for i, metrics in spiking_data['by_unit'].items() if i not in unit_types]
    E_spikes = [np.nan if i == np.inf else i for i in E_spikes]  # replace inf with nan
    I_spikes = [np.nan if i == np.inf else i for i in I_spikes]
    U_spikes = [np.nan if i == np.inf else i for i in U_spikes]
    spiking_data['E_spikes'] = {
        'data': E_spikes if len(E_spikes) > 0 else None,
        'mean': np.nanmean(E_spikes) if len(E_spikes) > 0 else None,
        'std': np.nanstd(E_spikes) if len(E_spikes) > 0 else None,
        'median': np.nanmedian(E_spikes) if len(E_spikes) > 0 else None,
        'cov': np.nanstd(E_spikes) / np.nanmean(E_spikes) if np.nanmean(E_spikes) and len(E_spikes) > 0 else None,
        'max': np.nanmax(E_spikes) if len(E_spikes) > 0 else None,
        'min': np.nanmin(E_spikes) if len(E_spikes) > 0 else None,
    }
    spiking_data['I_spikes'] = {
        'data': I_spikes if len(I_spikes) > 0 else None,
        'mean': np.nanmean(I_spikes) if len(I_spikes) > 0 else None,
        'std': np.nanstd(I_spikes) if len(I_spikes) > 0 else None,
        'median': np.nanmedian(I_spikes) if len(I_spikes) > 0 else None,
        'cov': np.nanstd(I_spikes) / np.nanmean(I_spikes) if np.nanmean(I_spikes) and len(I_spikes) > 0 else None,
        'max': np.nanmax(I_spikes) if len(I_spikes) > 0 else None,
        'min': np.nanmin(I_spikes) if len(I_spikes) > 0 else None,
    }
    spiking_data['U_spikes'] = {
        'data': U_spikes if len(U_spikes) > 0 else None,
        'mean': np.nanmean(U_spikes) if len(U_spikes) > 0 else None,
        'std': np.nanstd(U_spikes) if len(U_spikes) > 0 else None,
        'median': np.nanmedian(U_spikes) if len(U_spikes) > 0 else None,
        'cov': np.nanstd(U_spikes) / np.nanmean(U_spikes) if np.nanmean(U_spikes) and len(U_spikes) > 0 else None,
        'max': np.nanmax(U_spikes) if len(U_spikes) > 0 else None,
        'min': np.nanmin(U_spikes) if len(U_spikes) > 0 else None,
    }


    # E/I fr ratio metrics
    # EI_fr_ratios = {
    #     'mean': np.nanmean(spiking_data['e_frs']['mean'] / spiking_data['i_frs']['mean']) if spiking_data['e_frs']['mean'] and spiking_data['i_frs']['mean'] else None,
    #     'std': np.nanstd(spiking_data['e_frs']['mean'] / spiking_data['i_frs']['mean']) if spiking_data['e_frs']['mean'] and spiking_data['i_frs']['mean'] else None,
    #     'median': np.nanmedian(spiking_data['e_frs']['mean'] / spiking_data['i_frs']['mean']) if spiking_data['e_frs']['mean'] and spiking_data['i_frs']['mean'] else None,
    #     'cov': np.nanstd(spiking_data['e_frs']['mean'] / spiking_data['i_frs']['mean']) / np.nanmean(spiking_data['e_frs']['mean'] / spiking_data['i_frs']['mean']) if spiking_data['e_frs']['mean'] and spiking_data['i_frs']['mean'] else None,
    #     'max': np.nanmax(spiking_data['e_frs']['mean'] / spiking_data['i_frs']['mean']) if spiking_data['e_frs']['mean'] and spiking_data['i_frs']['mean'] else None,
    #     'min': np.nanmin(spiking_data['e_frs']['mean'] / spiking_data['i_frs']['mean']) if spiking_data['e_frs']['mean'] and spiking_data['i_frs']['mean'] else None,
    # }
    E_mean = spiking_data['e_frs']['mean']
    I_mean = spiking_data['i_frs']['mean']
    E_std = spiking_data['e_frs']['std']
    I_std = spiking_data['i_frs']['std']
    E_median = spiking_data['e_frs']['median']
    I_median = spiking_data['i_frs']['median']
    E_cov = spiking_data['e_frs']['cov']
    I_cov = spiking_data['i_frs']['cov']
    E_max = spiking_data['e_frs']['max']
    I_max = spiking_data['i_frs']['max']
    E_min = spiking_data['e_frs']['min']
    I_min = spiking_data['i_frs']['min']
    EI_fr_ratios = {
        'mean': E_mean / I_mean if E_mean is not None and I_mean is not None and I_mean != 0 else None,
        'std': E_std / I_std if E_std is not None and I_std is not None and I_std != 0 else None,
        'median': E_median / I_median if E_median is not None and I_median is not None and I_median != 0 else None,
        'cov': E_cov / I_cov if E_cov is not None and I_cov is not None and I_cov != 0 else None,
        'max': E_max / I_max if E_max is not None and I_max is not None and I_max != 0 else None,
        'min': E_min / I_min if E_min is not None and I_min is not None and I_min != 0 else None,
    }
    spiking_data['EI_fr_ratios'] = EI_fr_ratios

    #E/I spiking ratio metrics
    E_spikes_mean = spiking_data['E_spikes']['mean']
    I_spikes_mean = spiking_data['I_spikes']['mean']
    E_std = spiking_data['E_spikes']['std']
    I_std = spiking_data['I_spikes']['std']
    E_median = spiking_data['E_spikes']['median']
    I_median = spiking_data['I_spikes']['median']
    E_cov = spiking_data['E_spikes']['cov']
    I_cov = spiking_data['I_spikes']['cov']
    E_max = spiking_data['E_spikes']['max']
    I_max = spiking_data['I_spikes']['max']
    E_min = spiking_data['E_spikes']['min']
    I_min = spiking_data['I_spikes']['min']
    EI_spike_ratios = {
        'mean': E_spikes_mean / I_spikes_mean if E_spikes_mean is not None and I_spikes_mean is not None and I_spikes_mean != 0 else None,
        'std': E_std / I_std if E_std is not None and I_std is not None and I_std != 0 else None,
        'median': E_median / I_median if E_median is not None and I_median is not None and I_median != 0 else None,
        'cov': E_cov / I_cov if E_cov is not None and I_cov is not None and I_cov != 0 else None,
        'max': E_max / I_max if E_max is not None and I_max is not None and I_max != 0 else None,
        'min': E_min / I_min if E_min is not None and I_min is not None and I_min != 0 else None,
    }
    spiking_data['EI_spike_ratios'] = EI_spike_ratios
    
    # isi
    isis = [i['isi']['mean'] for i in spiking_data['by_unit'].values()]
    isis = [np.nan if i == np.inf else i for i in isis]
    spiking_data['isi'] = {
        'data': isis if len(isis) > 0 else None,
        'mean': np.nanmean(isis) if len(isis) > 0 else None,
        'std': np.nanstd(isis) if len(isis) > 0 else None,
        'median': np.nanmedian(isis) if len(isis) > 0 else None,
        'cov': np.nanstd(isis) / np.nanmean(isis) if np.nanmean(isis) and len(isis) > 0 else None,
        'max': np.nanmax(isis) if len(isis) > 0 else None,
        'min': np.nanmin(isis) if len(isis) > 0 else None,
    }
    
    # E/I/U isi
    i_isis = [metrics['isi']['mean'] for i, metrics in spiking_data['by_unit'].items() if i in unit_types and unit_types[i] == 'I'] 
    e_isis = [metrics['isi']['mean'] for i, metrics in spiking_data['by_unit'].items() if i in unit_types and unit_types[i] == 'E']
    u_isis = [metrics['isi']['mean'] for i, metrics in spiking_data['by_unit'].items() if i not in unit_types]  
    i_isis = [np.nan if i == np.inf else i for i in i_isis]
    e_isis = [np.nan if i == np.inf else i for i in e_isis]
    u_isis = [np.nan if i == np.inf else i for i in u_isis]
    spiking_data['i_isi'] = {
        'data': i_isis if len(i_isis) > 0 else None,
        'mean': np.nanmean(i_isis) if len(i_isis) > 0 else None,
        'std': np.nanstd(i_isis) if len(i_isis) > 0 else None,
        'median': np.nanmedian(i_isis) if len(i_isis) > 0 else None,
        'cov': np.nanstd(i_isis) / np.nanmean(i_isis) if np.nanmean(i_isis) and len(i_isis) > 0 else None,
        'max': np.nanmax(i_isis) if len(i_isis) > 0 else None,
        'min': np.nanmin(i_isis) if len(i_isis) > 0 else None,
    }
    
    spiking_data['e_isi'] = {
        'data': e_isis if len(e_isis) > 0 else None,
        'mean': np.nanmean(e_isis) if len(e_isis) > 0 else None,
        'std': np.nanstd(e_isis) if len(e_isis) > 0 else None,
        'median': np.nanmedian(e_isis) if len(e_isis) > 0 else None,
        'cov': np.nanstd(e_isis) / np.nanmean(e_isis) if np.nanmean(e_isis) and len(e_isis) > 0 else None,
        'max': np.nanmax(e_isis) if len(e_isis) > 0 else None,
        'min': np.nanmin(e_isis) if len(e_isis) > 0 else None,
    }
    
    spiking_data['u_isi'] = {
        'data': u_isis if len(u_isis) > 0 else None,
        'mean': np.nanmean(u_isis) if len(u_isis) > 0 else None,
        'std': np.nanstd(u_isis) if len(u_isis) > 0 else None,
        'median': np.nanmedian(u_isis) if len(u_isis) > 0 else None,
        'cov': np.nanstd(u_isis) / np.nanmean(u_isis) if np.nanmean(u_isis) and len(u_isis) > 0 else None,
        'max': np.nanmax(u_isis) if len(u_isis) > 0 else None,
        'min': np.nanmin(u_isis) if len(u_isis) > 0 else None,
    }
    
    # burst metrics
    #burst_metrics = network_data['burst_metrics']
    #burst_metrics = burst_metrics['burst_metrics']

    #baseline metrics
    bursting_baseline = network_data['burst_metrics']['convolved_data']['baseline']
    mega_baseline = network_data['hyperburst_metrics']['convolved_data']['baseline']
    network_data['burst_metrics']['baseline'] = bursting_baseline
    network_data['hyperburst_metrics']['baseline'] = mega_baseline

    #spike in/out burst summary metrics
    # burst and quiet part rates
    burst_part_rates=[]
    quiet_part_rates=[]
    E_burst_part_rates=[]
    I_burst_part_rates=[]
    U_burst_part_rates=[]
    E_quiet_part_rates=[]
    I_quiet_part_rates=[]
    U_quiet_part_rates=[]
    # unit_types = by_unit
    unit_types = _get_unit_pops(network_data)
    hyperburst_metrics = network_data['hyperburst_metrics']
    for unit, metrics in hyperburst_metrics['unit_metrics'].items():
        #print(f'Unit {unit} burst metrics:')
        burst_part_rate = metrics['burst_part_rate']
        quiet_part_rate = metrics['quiet_part_rate']
        burst_part_rates.append(burst_part_rate)
        quiet_part_rates.append(quiet_part_rate)
        unit_type = unit_types.get(unit, 'U')  # Default to 'U' if not found
        if unit_type == 'E':
            E_burst_part_rates.append(burst_part_rate)
            E_quiet_part_rates.append(quiet_part_rate)
        elif unit_type == 'I':
            I_burst_part_rates.append(burst_part_rate)
            I_quiet_part_rates.append(quiet_part_rate)
        elif unit_type == 'U':
            U_burst_part_rates.append(burst_part_rate)
            U_quiet_part_rates.append(quiet_part_rate)
    burst_part_rates = [np.nan if i == np.inf else i for i in burst_part_rates]
    quiet_part_rates = [np.nan if i == np.inf else i for i in quiet_part_rates]
    E_burst_part_rates = [np.nan if i == np.inf else i for i in E_burst_part_rates]
    I_burst_part_rates = [np.nan if i == np.inf else i for i in I_burst_part_rates]
    U_burst_part_rates = [np.nan if i == np.inf else i for i in U_burst_part_rates]
    E_quiet_part_rates = [np.nan if i == np.inf else i for i in E_quiet_part_rates]
    I_quiet_part_rates = [np.nan if i == np.inf else i for i in I_quiet_part_rates]
    U_quiet_part_rates = [np.nan if i == np.inf else i for i in U_quiet_part_rates]
    hyperburst_metrics['burst_part_rates'] = {
        'data': burst_part_rates if len(burst_part_rates) > 0 else None,
        'mean': np.nanmean(burst_part_rates) if len(burst_part_rates) > 0 else None,
        'std': np.nanstd(burst_part_rates) if len(burst_part_rates) > 0 else None,
        'median': np.nanmedian(burst_part_rates) if len(burst_part_rates) > 0 else None,
        'cov': np.nanstd(burst_part_rates) / np.nanmean(burst_part_rates) if np.nanmean(burst_part_rates) and len(burst_part_rates) > 0 else None,
        'max': np.nanmax(burst_part_rates) if len(burst_part_rates) > 0 else None,
        'min': np.nanmin(burst_part_rates) if len(burst_part_rates) > 0 else None,
    }
    hyperburst_metrics['quiet_part_rates'] = {
        'data': quiet_part_rates if len(quiet_part_rates) > 0 else None,
        'mean': np.nanmean(quiet_part_rates) if len(quiet_part_rates) > 0 else None,
        'std': np.nanstd(quiet_part_rates) if len(quiet_part_rates) > 0 else None,
        'median': np.nanmedian(quiet_part_rates) if len(quiet_part_rates) > 0 else None,
        'cov': np.nanstd(quiet_part_rates) / np.nanmean(quiet_part_rates) if np.nanmean(quiet_part_rates) and len(quiet_part_rates) > 0 else None,
        'max': np.nanmax(quiet_part_rates) if len(quiet_part_rates) > 0 else None,
        'min': np.nanmin(quiet_part_rates) if len(quiet_part_rates) > 0 else None,
    }
    hyperburst_metrics['E_burst_part_rates'] = {
        'data': E_burst_part_rates if len(E_burst_part_rates) > 0 else None,
        'mean': np.nanmean(E_burst_part_rates) if len(E_burst_part_rates) > 0 else None,
        'std': np.nanstd(E_burst_part_rates) if len(E_burst_part_rates) > 0 else None,
        'median': np.nanmedian(E_burst_part_rates) if len(E_burst_part_rates) > 0 else None,
        'cov': np.nanstd(E_burst_part_rates) / np.nanmean(E_burst_part_rates) if np.nanmean(E_burst_part_rates) and len(E_burst_part_rates) > 0 else None,
        'max': np.nanmax(E_burst_part_rates) if len(E_burst_part_rates) > 0 else None,
        'min': np.nanmin(E_burst_part_rates) if len(E_burst_part_rates) > 0 else None,
    }
    hyperburst_metrics['I_burst_part_rates'] = {
        'data': I_burst_part_rates if len(I_burst_part_rates) > 0 else None,
        'mean': np.nanmean(I_burst_part_rates) if len(I_burst_part_rates) > 0 else None,
        'std': np.nanstd(I_burst_part_rates) if len(I_burst_part_rates) > 0 else None,
        'median': np.nanmedian(I_burst_part_rates) if len(I_burst_part_rates) > 0 else None,
        'cov': np.nanstd(I_burst_part_rates) / np.nanmean(I_burst_part_rates) if np.nanmean(I_burst_part_rates) and len(I_burst_part_rates) > 0 else None,
        'max': np.nanmax(I_burst_part_rates) if len(I_burst_part_rates) > 0 else None,
        'min': np.nanmin(I_burst_part_rates) if len(I_burst_part_rates) > 0 else None,
    }
    hyperburst_metrics['U_burst_part_rates'] = {
        'data': U_burst_part_rates if len(U_burst_part_rates) > 0 else None,
        'mean': np.nanmean(U_burst_part_rates) if len(U_burst_part_rates) > 0 else None,
        'std': np.nanstd(U_burst_part_rates) if len(U_burst_part_rates) > 0 else None,
        'median': np.nanmedian(U_burst_part_rates) if len(U_burst_part_rates) > 0 else None,
        'cov': np.nanstd(U_burst_part_rates) / np.nanmean(U_burst_part_rates) if np.nanmean(U_burst_part_rates) and len(U_burst_part_rates) > 0 else None,
        'max': np.nanmax(U_burst_part_rates) if len(U_burst_part_rates) > 0 else None,
        'min': np.nanmin(U_burst_part_rates) if len(U_burst_part_rates) > 0 else None,
    }

    # in/out burst fr and spikes (unit-wise metrics)
    all_burst_frs = []
    all_quiet_frs = []
    E_burst_frs = []
    I_burst_frs = []
    U_burst_frs = []
    E_quiet_frs = []
    I_quiet_frs = []
    U_quiet_frs = []
    all_in_burst_spikes = []
    all_out_burst_spikes = []
    E_in_burst_spikes = []
    I_in_burst_spikes = []
    U_in_burst_spikes = []
    E_out_burst_spikes = []
    I_out_burst_spikes = []
    U_out_burst_spikes = []
    for unit, metrics in hyperburst_metrics['unit_metrics'].items():
        in_burst_fr = metrics['fr']['in_burst'].get('data', None)
        out_burst_fr = metrics['fr']['out_burst'].get('data', None)
        in_burst_spikes = metrics['spike_counts']['in_burst'].get('data', None)
        out_burst_spikes = metrics['spike_counts']['out_burst'].get('data', None)
        #break

        # Convert dicts to lists if needed
        def dict_to_list(val):
            if isinstance(val, dict):
                return list(val.values())
            elif val is None:
                return None
            return val

        in_burst_fr = dict_to_list(in_burst_fr)
        out_burst_fr = dict_to_list(out_burst_fr)
        in_burst_spikes = dict_to_list(in_burst_spikes)
        out_burst_spikes = dict_to_list(out_burst_spikes)

        # burst_frs.extend(in_burst_fr if in_burst_fr is not None else [])
        # quiet_frs.extend(out_burst_fr if out_burst_fr is not None else [])
        # in_burst_spikes.extend(in_burst_spikes if in_burst_spikes is not None else [])
        # out_burst_spikes.extend(out_burst_spikes if out_burst_spikes is not None else [])
        if in_burst_fr is not None: all_burst_frs.extend(in_burst_fr)
        if out_burst_fr is not None: all_quiet_frs.extend(out_burst_fr)
        if in_burst_spikes is not None: all_in_burst_spikes.extend(in_burst_spikes)
        if out_burst_spikes is not None: all_out_burst_spikes.extend(out_burst_spikes)
        unit_type = unit_types.get(unit, 'U')  # Default to 'U' if not found
        if unit_type == 'E':
            E_burst_frs.extend(in_burst_fr if in_burst_fr is not None else [])
            E_quiet_frs.extend(out_burst_fr if out_burst_fr is not None else [])
            E_in_burst_spikes.extend(in_burst_spikes if in_burst_spikes is not None else [])
            E_out_burst_spikes.extend(out_burst_spikes if out_burst_spikes is not None else [])
        elif unit_type == 'I':
            I_burst_frs.extend(in_burst_fr if in_burst_fr is not None else [])
            I_quiet_frs.extend(out_burst_fr if out_burst_fr is not None else [])
            I_in_burst_spikes.extend(in_burst_spikes if in_burst_spikes is not None else [])
            I_out_burst_spikes.extend(out_burst_spikes if out_burst_spikes is not None else [])
        elif unit_type == 'U':
            U_burst_frs.extend(in_burst_fr if in_burst_fr is not None else [])
            U_quiet_frs.extend(out_burst_fr if out_burst_fr is not None else [])
            U_in_burst_spikes.extend(in_burst_spikes if in_burst_spikes is not None else [])
            U_out_burst_spikes.extend(out_burst_spikes if out_burst_spikes is not None else [])  
    all_burst_frs = [np.nan if i == np.inf else i for i in all_burst_frs]
    all_quiet_frs = [np.nan if i == np.inf else i for i in all_quiet_frs]
    E_burst_frs = [np.nan if i == np.inf else i for i in E_burst_frs]
    I_burst_frs = [np.nan if i == np.inf else i for i in I_burst_frs]
    U_burst_frs = [np.nan if i == np.inf else i for i in U_burst_frs]
    E_quiet_frs = [np.nan if i == np.inf else i for i in E_quiet_frs]
    I_quiet_frs = [np.nan if i == np.inf else i for i in I_quiet_frs]
    U_quiet_frs = [np.nan if i == np.inf else i for i in U_quiet_frs]
    if all_burst_frs is None: all_burst_frs = []
    hyperburst_metrics['burst_frs'] = {
        'data': all_burst_frs if len(all_burst_frs) > 0 else None,
        'mean': np.nanmean(all_burst_frs) if len(all_burst_frs) > 0 else None,
        'std': np.nanstd(all_burst_frs) if len(all_burst_frs) > 0 else None,
        'median': np.nanmedian(all_burst_frs) if len(all_burst_frs) > 0 else None,
        'cov': np.nanstd(all_burst_frs) / np.nanmean(all_burst_frs) if np.nanmean(all_burst_frs) and len(all_burst_frs) > 0 else None,
        'max': np.nanmax(all_burst_frs) if len(all_burst_frs) > 0 else None,
        'min': np.nanmin(all_burst_frs) if len(all_burst_frs) > 0 else None,
    }
    if all_quiet_frs is None: all_quiet_frs = []
    hyperburst_metrics['quiet_frs'] = {
        'data': all_quiet_frs if len(all_quiet_frs) > 0 else None,
        'mean': np.nanmean(all_quiet_frs) if len(all_quiet_frs) > 0 else None,
        'std': np.nanstd(all_quiet_frs) if len(all_quiet_frs) > 0 else None,
        'median': np.nanmedian(all_quiet_frs) if len(all_quiet_frs) > 0 else None,
        'cov': np.nanstd(all_quiet_frs) / np.nanmean(all_quiet_frs) if np.nanmean(all_quiet_frs) and len(all_quiet_frs) > 0 else None,
        'max': np.nanmax(all_quiet_frs) if len(all_quiet_frs) > 0 else None,
        'min': np.nanmin(all_quiet_frs) if len(all_quiet_frs) > 0 else None,
    }
    if E_burst_frs is None: E_burst_frs = []
    hyperburst_metrics['E_burst_frs'] = {
        'data': E_burst_frs if len(E_burst_frs) > 0 else None,
        'mean': np.nanmean(E_burst_frs) if len(E_burst_frs) > 0 else None,
        'std': np.nanstd(E_burst_frs) if len(E_burst_frs) > 0 else None,
        'median': np.nanmedian(E_burst_frs) if len(E_burst_frs) > 0 else None,
        'cov': np.nanstd(E_burst_frs) / np.nanmean(E_burst_frs) if np.nanmean(E_burst_frs) and len(E_burst_frs) > 0 else None,
        'max': np.nanmax(E_burst_frs) if len(E_burst_frs) > 0 else None,
        'min': np.nanmin(E_burst_frs) if len(E_burst_frs) > 0 else None,
    }
    if I_burst_frs is None: I_burst_frs = []
    hyperburst_metrics['I_burst_frs'] = {
        'data': I_burst_frs if len(I_burst_frs) > 0 else None,
        'mean': np.nanmean(I_burst_frs) if len(I_burst_frs) > 0 else None,
        'std': np.nanstd(I_burst_frs) if len(I_burst_frs) > 0 else None,
        'median': np.nanmedian(I_burst_frs) if len(I_burst_frs) > 0 else None,
        'cov': np.nanstd(I_burst_frs) / np.nanmean(I_burst_frs) if np.nanmean(I_burst_frs) and len(I_burst_frs) > 0 else None,
        'max': np.nanmax(I_burst_frs) if len(I_burst_frs) > 0 else None,
        'min': np.nanmin(I_burst_frs) if len(I_burst_frs) > 0 else None,
    }
    if U_burst_frs is None: U_burst_frs = []
    hyperburst_metrics['U_burst_frs'] = {
        'data': U_burst_frs if len(U_burst_frs) > 0 else None,
        'mean': np.nanmean(U_burst_frs) if len(U_burst_frs) > 0 else None,
        'std': np.nanstd(U_burst_frs) if len(U_burst_frs) > 0 else None,
        'median': np.nanmedian(U_burst_frs) if len(U_burst_frs) > 0 else None,
        'cov': np.nanstd(U_burst_frs) / np.nanmean(U_burst_frs) if np.nanmean(U_burst_frs) and len(U_burst_frs) > 0 else None,
        'max': np.nanmax(U_burst_frs) if len(U_burst_frs) > 0 else None,
        'min': np.nanmin(U_burst_frs) if len(U_burst_frs) > 0 else None,
    }
    if E_quiet_frs is None: E_quiet_frs = []
    hyperburst_metrics['E_quiet_frs'] = {
        'data': E_quiet_frs if len(E_quiet_frs) > 0 else None,
        'mean': np.nanmean(E_quiet_frs) if len(E_quiet_frs) > 0 else None,
        'std': np.nanstd(E_quiet_frs) if len(E_quiet_frs) > 0 else None,
        'median': np.nanmedian(E_quiet_frs) if len(E_quiet_frs) > 0 else None,  
        'cov': np.nanstd(E_quiet_frs) / np.nanmean(E_quiet_frs) if np.nanmean(E_quiet_frs) and len(E_quiet_frs) > 0 else None,
        'max': np.nanmax(E_quiet_frs) if len(E_quiet_frs) > 0 else None,
        'min': np.nanmin(E_quiet_frs) if len(E_quiet_frs) > 0 else None,
    }
    if I_quiet_frs is None: I_quiet_frs = []
    hyperburst_metrics['I_quiet_frs'] = {
        'data': I_quiet_frs if len(I_quiet_frs) > 0 else None,
        'mean': np.nanmean(I_quiet_frs) if len(I_quiet_frs) > 0 else None,
        'std': np.nanstd(I_quiet_frs) if len(I_quiet_frs) > 0 else None,
        'median': np.nanmedian(I_quiet_frs) if len(I_quiet_frs) > 0 else None,
        'cov': np.nanstd(I_quiet_frs) / np.nanmean(I_quiet_frs) if np.nanmean(I_quiet_frs) and len(I_quiet_frs) > 0 else None,
        'max': np.nanmax(I_quiet_frs) if len(I_quiet_frs) > 0 else None,
        'min': np.nanmin(I_quiet_frs) if len(I_quiet_frs) > 0 else None,
    }
    if U_quiet_frs is None: U_quiet_frs = []
    hyperburst_metrics['U_quiet_frs'] = {
        'data': U_quiet_frs if len(U_quiet_frs) > 0 else None,
        'mean': np.nanmean(U_quiet_frs) if len(U_quiet_frs) > 0 else None,
        'std': np.nanstd(U_quiet_frs) if len(U_quiet_frs) > 0 else None,
        'median': np.nanmedian(U_quiet_frs) if len(U_quiet_frs) > 0 else None,
        'cov': np.nanstd(U_quiet_frs) / np.nanmean(U_quiet_frs) if np.nanmean(U_quiet_frs) and len(U_quiet_frs) > 0 else None,
        'max': np.nanmax(U_quiet_frs) if len(U_quiet_frs) > 0 else None,
        'min': np.nanmin(U_quiet_frs) if len(U_quiet_frs) > 0 else None,
    }
    if all_in_burst_spikes is None: all_in_burst_spikes = []
    hyperburst_metrics['in_burst_spikes'] = {
        'data': all_in_burst_spikes if len(all_in_burst_spikes) > 0 else None,
        'mean': np.nanmean(all_in_burst_spikes) if len(all_in_burst_spikes) > 0 else None,
        'std': np.nanstd(all_in_burst_spikes) if len(all_in_burst_spikes) > 0 else None,
        'median': np.nanmedian(all_in_burst_spikes) if len(all_in_burst_spikes) > 0 else None,
        'cov': np.nanstd(all_in_burst_spikes) / np.nanmean(all_in_burst_spikes) if np.nanmean(all_in_burst_spikes) and len(all_in_burst_spikes) > 0 else None,
        'max': np.nanmax(all_in_burst_spikes) if len(all_in_burst_spikes) > 0 else None,
        'min': np.nanmin(all_in_burst_spikes) if len(all_in_burst_spikes) > 0 else None,
    }
    if all_out_burst_spikes is None: all_out_burst_spikes = []
    hyperburst_metrics['out_burst_spikes'] = {
        'data': all_out_burst_spikes if len(all_out_burst_spikes) > 0 else None,
        'mean': np.nanmean(all_out_burst_spikes) if len(all_out_burst_spikes) > 0 else None,
        'std': np.nanstd(all_out_burst_spikes) if len(all_out_burst_spikes) > 0 else None,
        'median': np.nanmedian(all_out_burst_spikes) if len(all_out_burst_spikes) > 0 else None,
        'cov': np.nanstd(all_out_burst_spikes) / np.nanmean(all_out_burst_spikes) if np.nanmean(all_out_burst_spikes) and len(all_out_burst_spikes) > 0 else None,
        'max': np.nanmax(all_out_burst_spikes) if len(all_out_burst_spikes) > 0 else None,
        'min': np.nanmin(all_out_burst_spikes) if len(all_out_burst_spikes) > 0 else None,
    }
    if E_in_burst_spikes is None: E_in_burst_spikes = []
    hyperburst_metrics['E_in_burst_spikes'] = {
        'data': E_in_burst_spikes if len(E_in_burst_spikes) > 0 else None,
        'mean': np.nanmean(E_in_burst_spikes) if len(E_in_burst_spikes) > 0 else None,
        'std': np.nanstd(E_in_burst_spikes) if len(E_in_burst_spikes) > 0 else None,
        'median': np.nanmedian(E_in_burst_spikes) if len(E_in_burst_spikes) > 0 else None,
        'cov': np.nanstd(E_in_burst_spikes) / np.nanmean(E_in_burst_spikes) if np.nanmean(E_in_burst_spikes) and len(E_in_burst_spikes) > 0 else None,
        'max': np.nanmax(E_in_burst_spikes) if len(E_in_burst_spikes) > 0 else None,
        'min': np.nanmin(E_in_burst_spikes) if len(E_in_burst_spikes) > 0 else None,
    }
    if I_in_burst_spikes is None: I_in_burst_spikes = []
    hyperburst_metrics['I_in_burst_spikes'] = {
        'data': I_in_burst_spikes if len(I_in_burst_spikes) > 0 else None,
        'mean': np.nanmean(I_in_burst_spikes) if len(I_in_burst_spikes) > 0 else None,
        'std': np.nanstd(I_in_burst_spikes) if len(I_in_burst_spikes) > 0 else None,
        'median': np.nanmedian(I_in_burst_spikes) if len(I_in_burst_spikes) > 0 else None,
        'cov': np.nanstd(I_in_burst_spikes) / np.nanmean(I_in_burst_spikes) if np.nanmean(I_in_burst_spikes) and len(I_in_burst_spikes) > 0 else None,
        'max': np.nanmax(I_in_burst_spikes) if len(I_in_burst_spikes) > 0 else None,
        'min': np.nanmin(I_in_burst_spikes) if len(I_in_burst_spikes) > 0 else None,
    }
    if U_in_burst_spikes is None: U_in_burst_spikes = []
    hyperburst_metrics['U_in_burst_spikes'] = {
        'data': U_in_burst_spikes if len(U_in_burst_spikes) > 0 else None,
        'mean': np.nanmean(U_in_burst_spikes) if len(U_in_burst_spikes) > 0 else None,
        'std': np.nanstd(U_in_burst_spikes) if len(U_in_burst_spikes) > 0 else None,
        'median': np.nanmedian(U_in_burst_spikes) if len(U_in_burst_spikes) > 0 else None,
        'cov': np.nanstd(U_in_burst_spikes) / np.nanmean(U_in_burst_spikes) if np.nanmean(U_in_burst_spikes) and len(U_in_burst_spikes) > 0 else None,
        'max': np.nanmax(U_in_burst_spikes) if len(U_in_burst_spikes) > 0 else None,
        'min': np.nanmin(U_in_burst_spikes) if len(U_in_burst_spikes) > 0 else None,
    }
    if E_out_burst_spikes is None: E_out_burst_spikes = []
    hyperburst_metrics['E_out_burst_spikes'] = {
        'data': E_out_burst_spikes if len(E_out_burst_spikes) > 0 else None,
        'mean': np.nanmean(E_out_burst_spikes) if len(E_out_burst_spikes) > 0 else None,
        'std': np.nanstd(E_out_burst_spikes) if len(E_out_burst_spikes) > 0 else None,
        'median': np.nanmedian(E_out_burst_spikes) if len(E_out_burst_spikes) > 0 else None,
        'cov': np.nanstd(E_out_burst_spikes) / np.nanmean(E_out_burst_spikes) if np.nanmean(E_out_burst_spikes) and len(E_out_burst_spikes) > 0 else None,
        'max': np.nanmax(E_out_burst_spikes) if len(E_out_burst_spikes) > 0 else None,
        'min': np.nanmin(E_out_burst_spikes) if len(E_out_burst_spikes) > 0 else None,
    }
    if I_out_burst_spikes is None: I_out_burst_spikes = []
    hyperburst_metrics['I_out_burst_spikes'] = {
        'data': I_out_burst_spikes if len(I_out_burst_spikes) > 0 else None,
        'mean': np.nanmean(I_out_burst_spikes) if len(I_out_burst_spikes) > 0 else None,
        'std': np.nanstd(I_out_burst_spikes) if len(I_out_burst_spikes) > 0 else None,
        'median': np.nanmedian(I_out_burst_spikes) if len(I_out_burst_spikes) > 0 else None,
        'cov': np.nanstd(I_out_burst_spikes) / np.nanmean(I_out_burst_spikes) if np.nanmean(I_out_burst_spikes) and len(I_out_burst_spikes) > 0 else None,
        'max': np.nanmax(I_out_burst_spikes) if len(I_out_burst_spikes) > 0 else None,
        'min': np.nanmin(I_out_burst_spikes) if len(I_out_burst_spikes) > 0 else None,
    }
    if U_out_burst_spikes is None: U_out_burst_spikes = []
    hyperburst_metrics['U_out_burst_spikes'] = {
        'data': U_out_burst_spikes if len(U_out_burst_spikes) > 0 else None,
        'mean': np.nanmean(U_out_burst_spikes) if len(U_out_burst_spikes) > 0 else None,
        'std': np.nanstd(U_out_burst_spikes) if len(U_out_burst_spikes) > 0 else None,
        'median': np.nanmedian(U_out_burst_spikes) if len(U_out_burst_spikes) > 0 else None,
        'cov': np.nanstd(U_out_burst_spikes) / np.nanmean(U_out_burst_spikes) if np.nanmean(U_out_burst_spikes) and len(U_out_burst_spikes) > 0 else None,
        'max': np.nanmax(U_out_burst_spikes) if len(U_out_burst_spikes) > 0 else None,
        'min': np.nanmin(U_out_burst_spikes) if len(U_out_burst_spikes) > 0 else None,
    }
    
    # in/out burst spikes per burst (burst-wise metrics)
    try:
        burst_spike_counts={}
        burst_spike_counts_E = {}
        burst_spike_counts_I = {}
        burst_spike_counts_U = {}
        quiet_spike_counts = {}
        quiet_spike_counts_E = {}
        quiet_spike_counts_I = {}
        quiet_spike_counts_U = {}
        # all_in_burst_spikes_per_burst = []
        # all_out_burst_spikes_per_burst = []
        # E_in_burst_spikes_per_burst = []
        # I_in_burst_spikes_per_burst = []
        # U_in_burst_spikes_per_burst = []
        # E_out_burst_spikes_per_burst = []
        # I_out_burst_spikes_per_burst = []
        # U_out_burst_spikes_per_burst = []
        for unit, metrics in hyperburst_metrics['unit_metrics'].items():
            print(f'Processing unit {unit}...')
            burst_ids = metrics['burst_id']
            quiet_ids = metrics['quiet_id']
            unit_type = unit_types.get(unit, 'U')  # Default to 'U' if not found
            for burst_id in burst_ids:
                in_burst_spikes = metrics['spike_counts']['in_burst'].get('data', {}).get(burst_id, None)
                if burst_id not in burst_spike_counts:
                    burst_spike_counts[burst_id] = in_burst_spikes if in_burst_spikes is not None else 0
                    # if unit_type == 'E':
                    #     burst_spike_counts_E[burst_id] = in_burst_spikes if in_burst_spikes is not None else 0
                    # elif unit_type == 'I':
                    #     burst_spike_counts_I[burst_id] = in_burst_spikes if in_burst_spikes is not None else 0
                    # elif unit_type == 'U':
                    #     burst_spike_counts_U[burst_id] = in_burst_spikes if in_burst_spikes is not None else 0
                else:
                    burst_spike_counts[burst_id] += in_burst_spikes if in_burst_spikes is not None else 0
                    # if unit_type == 'E':
                    #     burst_spike_counts_E[burst_id] += in_burst_spikes if in_burst_spikes is not None else 0
                    # elif unit_type == 'I':
                    #     burst_spike_counts_I[burst_id] += in_burst_spikes if in_burst_spikes is not None else 0
                    # elif unit_type == 'U':
                    #     burst_spike_counts_U[burst_id] += in_burst_spikes if in_burst_spikes is not None else 0
                if unit_type == 'E':
                    if burst_id not in burst_spike_counts_E:
                        burst_spike_counts_E[burst_id] = in_burst_spikes if in_burst_spikes is not None else 0
                    else:
                        burst_spike_counts_E[burst_id] += in_burst_spikes if in_burst_spikes is not None else 0
                elif unit_type == 'I':
                    if burst_id not in burst_spike_counts_I:
                        burst_spike_counts_I[burst_id] = in_burst_spikes if in_burst_spikes is not None else 0
                    else:
                        burst_spike_counts_I[burst_id] += in_burst_spikes if in_burst_spikes is not None else 0
                elif unit_type == 'U':
                    if burst_id not in burst_spike_counts_U:
                        burst_spike_counts_U[burst_id] = in_burst_spikes if in_burst_spikes is not None else 0
                    else:
                        burst_spike_counts_U[burst_id] += in_burst_spikes if in_burst_spikes is not None else 0

            for quiet_id in quiet_ids:
                out_burst_spikes = metrics['spike_counts']['out_burst'].get('data', {}).get(quiet_id, None)
                if quiet_id not in quiet_spike_counts:
                    quiet_spike_counts[quiet_id] = out_burst_spikes if out_burst_spikes is not None else 0
                else:
                    quiet_spike_counts[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                    #burst_spike_counts[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                    # if unit_type == 'E':
                    #     burst_spike_counts_E[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                    # elif unit_type == 'I':
                    #     burst_spike_counts_I[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                    # elif unit_type == 'U':
                    #     burst_spike_counts_U[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                if unit_type == 'E':
                    if quiet_id not in quiet_spike_counts_E:
                        #burst_spike_counts_E[quiet_id] = out_burst_spikes if out_burst_spikes is not None else 0
                        quiet_spike_counts_E[quiet_id] = out_burst_spikes if out_burst_spikes is not None else 0
                    else:
                        #burst_spike_counts_E[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                        quiet_spike_counts_E[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                elif unit_type == 'I':
                    if quiet_id not in quiet_spike_counts_I:
                        #burst_spike_counts_I[quiet_id] = out_burst_spikes if out_burst_spikes is not None else 0
                        quiet_spike_counts_I[quiet_id] = out_burst_spikes if out_burst_spikes is not None else 0
                    else:
                        #burst_spike_counts_I[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                        quiet_spike_counts_I[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                elif unit_type == 'U':
                    if quiet_id not in quiet_spike_counts_U:
                        #burst_spike_counts_U[quiet_id] = out_burst_spikes if out_burst_spikes is not None else 0
                        quiet_spike_counts_U[quiet_id] = out_burst_spikes if out_burst_spikes is not None else 0
                    else:
                        #burst_spike_counts_U[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
                        quiet_spike_counts_U[quiet_id] += out_burst_spikes if out_burst_spikes is not None else 0
        # convert burst_spike_counts to lists
        burst_spike_counts_list = list(burst_spike_counts.values())
        burst_spike_counts_E_list = list(burst_spike_counts_E.values())
        burst_spike_counts_I_list = list(burst_spike_counts_I.values())
        burst_spike_counts_U_list = list(burst_spike_counts_U.values())
        quiet_spike_counts_list = list(quiet_spike_counts.values())
        quiet_spike_counts_E_list = list(quiet_spike_counts_E.values())
        quiet_spike_counts_I_list = list(quiet_spike_counts_I.values())
        quiet_spike_counts_U_list = list(quiet_spike_counts_U.values())
        if burst_spike_counts_list is None: burst_spike_counts_list = []
        hyperburst_metrics['burst_spike_counts'] = {
            'data': burst_spike_counts_list if len(burst_spike_counts_list) > 0 else None,
            'mean': np.nanmean(burst_spike_counts_list) if len(burst_spike_counts_list) > 0 else None,
            'std': np.nanstd(burst_spike_counts_list) if len(burst_spike_counts_list) > 0 else None,
            'median': np.nanmedian(burst_spike_counts_list) if len(burst_spike_counts_list) > 0 else None,
            'cov': np.nanstd(burst_spike_counts_list) / np.nanmean(burst_spike_counts_list) if np.nanmean(burst_spike_counts_list) and len(burst_spike_counts_list) > 0 else None,
            'max': np.nanmax(burst_spike_counts_list) if len(burst_spike_counts_list) > 0 else None,
            'min': np.nanmin(burst_spike_counts_list) if len(burst_spike_counts_list) > 0 else None,
        }
        if burst_spike_counts_E_list is None: burst_spike_counts_E_list = []
        hyperburst_metrics['burst_spike_counts_E'] = {
            'data': burst_spike_counts_E_list if len(burst_spike_counts_E_list) > 0 else None,
            'mean': np.nanmean(burst_spike_counts_E_list) if len(burst_spike_counts_E_list) > 0 else None,
            'std': np.nanstd(burst_spike_counts_E_list) if len(burst_spike_counts_E_list) > 0 else None,
            'median': np.nanmedian(burst_spike_counts_E_list) if len(burst_spike_counts_E_list) > 0 else None,
            'cov': np.nanstd(burst_spike_counts_E_list) / np.nanmean(burst_spike_counts_E_list) if np.nanmean(burst_spike_counts_E_list) and len(burst_spike_counts_E_list) > 0 else None,
            'max': np.nanmax(burst_spike_counts_E_list) if len(burst_spike_counts_E_list) > 0 else None,
            'min': np.nanmin(burst_spike_counts_E_list) if len(burst_spike_counts_E_list) > 0 else None,
        }
        if burst_spike_counts_I_list is None: burst_spike_counts_I_list = []
        hyperburst_metrics['burst_spike_counts_I'] = {
            'data': burst_spike_counts_I_list if len(burst_spike_counts_I_list) > 0 else None,
            'mean': np.nanmean(burst_spike_counts_I_list) if len(burst_spike_counts_I_list) > 0 else None,
            'std': np.nanstd(burst_spike_counts_I_list) if len(burst_spike_counts_I_list) > 0 else None,
            'median': np.nanmedian(burst_spike_counts_I_list) if len(burst_spike_counts_I_list) > 0 else None,
            'cov': np.nanstd(burst_spike_counts_I_list) / np.nanmean(burst_spike_counts_I_list) if np.nanmean(burst_spike_counts_I_list) and len(burst_spike_counts_I_list) > 0 else None,
            'max': np.nanmax(burst_spike_counts_I_list) if len(burst_spike_counts_I_list) > 0 else None,
            'min': np.nanmin(burst_spike_counts_I_list) if len(burst_spike_counts_I_list) > 0 else None,
        }
        if burst_spike_counts_U_list is None: burst_spike_counts_U_list = []
        hyperburst_metrics['burst_spike_counts_U'] = {
            'data': burst_spike_counts_U_list if len(burst_spike_counts_U_list) > 0 else None,
            'mean': np.nanmean(burst_spike_counts_U_list) if len(burst_spike_counts_U_list) > 0 else None,
            'std': np.nanstd(burst_spike_counts_U_list) if len(burst_spike_counts_U_list) > 0 else None,
            'median': np.nanmedian(burst_spike_counts_U_list) if len(burst_spike_counts_U_list) > 0 else None,
            'cov': np.nanstd(burst_spike_counts_U_list) / np.nanmean(burst_spike_counts_U_list) if np.nanmean(burst_spike_counts_U_list) and len(burst_spike_counts_U_list) > 0 else None,
            'max': np.nanmax(burst_spike_counts_U_list) if len(burst_spike_counts_U_list) > 0 else None,
            'min': np.nanmin(burst_spike_counts_U_list) if len(burst_spike_counts_U_list) > 0 else None,
        }
        if quiet_spike_counts_list is None: quiet_spike_counts_list = []
        hyperburst_metrics['quiet_spike_counts'] = {
            'data': quiet_spike_counts_list if len(quiet_spike_counts_list) > 0 else None,
            'mean': np.nanmean(quiet_spike_counts_list) if len(quiet_spike_counts_list) > 0 else None,
            'std': np.nanstd(quiet_spike_counts_list) if len(quiet_spike_counts_list) > 0 else None,
            'median': np.nanmedian(quiet_spike_counts_list) if len(quiet_spike_counts_list) > 0 else None,
            'cov': np.nanstd(quiet_spike_counts_list) / np.nanmean(quiet_spike_counts_list) if np.nanmean(quiet_spike_counts_list) and len(quiet_spike_counts_list) > 0 else None,
            'max': np.nanmax(quiet_spike_counts_list) if len(quiet_spike_counts_list) > 0 else None,
            'min': np.nanmin(quiet_spike_counts_list) if len(quiet_spike_counts_list) > 0 else None,
        }
        if quiet_spike_counts_E_list is None: quiet_spike_counts_E_list = []
        hyperburst_metrics['quiet_spike_counts_E'] = {
            'data': quiet_spike_counts_E_list if len(quiet_spike_counts_E_list) > 0 else None,
            'mean': np.nanmean(quiet_spike_counts_E_list) if len(quiet_spike_counts_E_list) > 0 else None,
            'std': np.nanstd(quiet_spike_counts_E_list) if len(quiet_spike_counts_E_list) > 0 else None,
            'median': np.nanmedian(quiet_spike_counts_E_list) if len(quiet_spike_counts_E_list) > 0 else None,
            'cov': np.nanstd(quiet_spike_counts_E_list) / np.nanmean(quiet_spike_counts_E_list) if np.nanmean(quiet_spike_counts_E_list) and len(quiet_spike_counts_E_list) > 0 else None,
            'max': np.nanmax(quiet_spike_counts_E_list) if len(quiet_spike_counts_E_list) > 0 else None,
            'min': np.nanmin(quiet_spike_counts_E_list) if len(quiet_spike_counts_E_list) > 0 else None,
        }
        if quiet_spike_counts_I_list is None: quiet_spike_counts_I_list = []
        hyperburst_metrics['quiet_spike_counts_I'] = {
            'data': quiet_spike_counts_I_list if len(quiet_spike_counts_I_list) > 0 else None,
            'mean': np.nanmean(quiet_spike_counts_I_list) if len(quiet_spike_counts_I_list) > 0 else None,
            'std': np.nanstd(quiet_spike_counts_I_list) if len(quiet_spike_counts_I_list) > 0 else None,
            'median': np.nanmedian(quiet_spike_counts_I_list) if len(quiet_spike_counts_I_list) > 0 else None,
            'cov': np.nanstd(quiet_spike_counts_I_list) / np.nanmean(quiet_spike_counts_I_list) if np.nanmean(quiet_spike_counts_I_list) and len(quiet_spike_counts_I_list) > 0 else None,
            'max': np.nanmax(quiet_spike_counts_I_list) if len(quiet_spike_counts_I_list) > 0 else None,
            'min': np.nanmin(quiet_spike_counts_I_list) if len(quiet_spike_counts_I_list) > 0 else None,
        }
        if quiet_spike_counts_U_list is None: quiet_spike_counts_U_list = []
        hyperburst_metrics['quiet_spike_counts_U'] = {
            'data': quiet_spike_counts_U_list if len(quiet_spike_counts_U_list) > 0 else None,
            'mean': np.nanmean(quiet_spike_counts_U_list) if len(quiet_spike_counts_U_list) > 0 else None,
            'std': np.nanstd(quiet_spike_counts_U_list) if len(quiet_spike_counts_U_list) > 0 else None,
            'median': np.nanmedian(quiet_spike_counts_U_list) if len(quiet_spike_counts_U_list) > 0 else None,
            'cov': np.nanstd(quiet_spike_counts_U_list) / np.nanmean(quiet_spike_counts_U_list) if np.nanmean(quiet_spike_counts_U_list) and len(quiet_spike_counts_U_list) > 0 else None,
            'max': np.nanmax(quiet_spike_counts_U_list) if len(quiet_spike_counts_U_list) > 0 else None,
            'min': np.nanmin(quiet_spike_counts_U_list) if len(quiet_spike_counts_U_list) > 0 else None,
        }
        

    except Exception as e:
        traceback.print_exc()
        print(f'Error processing unit {unit}: {e}')
    
    
    
    # add to network_data
    network_data['hyperburst_metrics'] = hyperburst_metrics


    #TODO: add in to out of burst fr ratio metrics
    
    
    #raise NotImplementedError('This function is not yet implemented.')
    return network_data

def _locate_units(network_data, kwargs):
    source = kwargs['source']
    if source == 'simulated':
        # TODO: EXTRACT from SIMDATA
        # raise NotImplementedError('This function is not yet implemented.')
        cellData = kwargs['cellData']
        unit_locations_dict = {}
        for cell in cellData:
            #cell['location'] = None
            gid = int(cell['gid'])
            x = cell['tags']['x']
            y = cell['tags']['y']
            z = cell['tags']['z']
            unit_locations_dict[gid] = (x, y, z) # TODO: extract x, y, z from cell
        
        network_data['unit_locations'] = unit_locations_dict
        
        return network_data
    elif source == 'experimental':
        #we = kwargs.get('wf_extractor', None)
        sa = kwargs.get('sorting_analyzer', None)
        recording_object = kwargs.get('recording_object', None)
        rec_path = recording_object.neo_reader.filename
        #wfs = sa.get_extension("waveforms")
        
        classification_output = network_data['classification_output']
        include_unit_ids = classification_output['include_units']
        # classified_units = classification_output['classified_units']

        print(f'Getting unit locations for {len(include_unit_ids)} units...')            
        #sa.load('recording')
        # print recording path sa is looking for
        #prop = sa.get_recording_property('recording_path')
        #sa.set_temporary_recording(recording_object)

        if "templates" not in sa.extensions:
            sa.compute("templates")
        unit_locations = spost.compute_unit_locations(sa)
        unit_locations_dict = {unit_id: unit_locations[i] for i, unit_id in enumerate(include_unit_ids)}
        
        network_data['unit_locations'] = unit_locations_dict
        
        return network_data
        # inhib_neuron_locs = np.array([unit_locations_dict[i] for i in include_unit_ids if classified_units[i]['desc'] == 'inhib'])
        # excit_neuron_locs = np.array([unit_locations_dict[i] for i in include_unit_ids if classified_units[i]['desc'] == 'excit'])

def _add_sim_specific_data(network_data, kwargs):
    # add simData, popData, cellData to network_data - stuff that should be in kwargs
    network_data['simData'] = kwargs['simData']
    network_data['popData'] = kwargs['popData']
    network_data['cellData'] = kwargs['cellData']
    return network_data

def _validate_input_format(kwargs):
    source = kwargs.get('source', None)
    if source not in ['simulated', 'experimental']:
        raise ValueError("Invalid source. Must be 'simulated' or 'experimental'.")
    
    if source == 'simulated':
        required_keys = ['spike_times', 'spike_gids', 'simData', 'popData', 'cellData']
    elif source == 'experimental':
        required_keys = ['spike_times', 'spike_gids', 'recording_object', 'sorting_analyzer']
    
    missing_keys = [key for key in required_keys if key not in kwargs]
    if missing_keys:
        raise ValueError(f'Missing required keys for source {source}: {missing_keys}')
    
    return True

def _default_options():
    '''
    Returns a dictionary of default analysis options.
    '''
    return {
        # time unit
        'time_unit': 's',  # default time unit is seconds
        
        # convolution params
        'burst_params': {
            'bin_size': 0.075,  # bin size in seconds
            'gaussian_sigma': 0.075,  # standard deviation for Gaussian kernel
            'thresholdBurst': None,  # threshold for burst detection
            'min_peak_distance': None,  # minimum distance between peaks in seconds
            'prominence': 1,  # prominence for peak detection
        },
        'hyperburst_params': {
            'bin_size': 0.3,  # bin size in seconds
            'gaussian_sigma': 0.3,  # standard deviation for Gaussian kernel
            'thresholdBurst': None,  # threshold for hyperburst detection
            'min_peak_distance': None,  # minimum distance between peaks in seconds
            'prominence': 1,  # prominence for peak detection
        },
        
        # analysis options
        'compute_spike_metrics': True,
        'compute_burst_metrics': True,
        'classify_units': True,
        'compute_dynamic_time_warping': False,  # default to False for now
        'compute_summary_metrics': True,
        
        # a priori classification
        'unit_pops': {}, # dictionary to store unit pops, e.g., {'unit1': 'E', 'unit2': 'I', ...}
        
        # runtime
        'parallel': True,  # default to True for parallel processing
    }

def _parse_options(kwargs):
    '''
    Parses the analysis options from kwargs and returns a dictionary of options.
    '''
    options = _default_options()
    
    # update with any provided options
    for key in options.keys():
        if key in kwargs:
            options[key] = kwargs[key]
            
    # update kwargs with options
    kwargs.update(options)
    
    # make sure time_unit is set to seconds, anything else wont work for now..
    if 'time_unit' in options and options['time_unit'] != 's':
        raise ValueError("Only 's' (seconds) is supported for time_unit at the moment."
                         " Please set 'time_unit' to 's' and convert all time inputs to seconds.")
    
    return kwargs

def _unpack_required_inputs(kwargs):
    '''
    Unpacks the required inputs from kwargs and returns them.
    '''
    spkt = kwargs.get('spkt', None)  # spike times
    spkt_by_unit = kwargs.get('spkt_by_unit', None)  # spike times by unit
    t = kwargs.get('t', None)  # time vector
    
    # raise errors as needed
    if spkt is None or spkt_by_unit is None or t is None:
        raise ValueError("Missing required inputs: 'spkt', 'spkt_by_unit', or 't'.")
    
    return spkt, spkt_by_unit, t

def _initialize_network_data(kwargs, spkt, spkt_by_unit, t):
    '''
    Initializes the network data dictionary with required keys and values.
    '''
    network_data = {
        # 'spkt': spkt,  # spike times
        # 'spkt_by_unit': spkt_by_unit,  # spike times by unit
        # 't': t,  # time vector
        # 'source': kwargs.get('source', None),  # data source (simulated or experimental)
        # 'classification_output': {},  # to store classification results
        # 'burst_metrics': {},  # to store bursting data
        # 'hyperburst_metrics': {},  # to store mega bursting data
        # 'unit_locations': {},  # to store unit locations if available
        'inputs': {
            'spkt': spkt,  # raw spike times
            'spkt_by_unit': spkt_by_unit,  # raw spike times by unit
            't': t,  # raw time vector
            't_unit': kwargs.get('time_unit', 's'),  # time unit, default is seconds
            'unit_pops': kwargs.get('unit_pops', {}),  # unit populations if provided
            'burst_params': kwargs.get('burst_params', {}),  # burst detection parameters
            'hyperburst_params': kwargs.get('hyperburst_params', {}),  # hyperburst
        },
        'spk_metrics': {},  # to store spike metrics
        'wf_metrics': {},  # to store waveform metrics
        'burst_metrics': {},  # to store burst metrics
        'hyperburst_metrics': {},  # to store hyperburst metrics
        #'hfburst_metrics': {},  # to store high-frequency burst metrics
        'classification_output': {},  # to store classification results
        #'summary_metrics': {},  # to store summary metrics
    }
    
    return network_data

# main function to compute network metrics from spike data ================================
#def compute_network_metrics(conv_params, mega_params, source, **kwargs):
def compute_network_metrics(kwargs): # aw 2025-07-03 13:47:04 updating nomenclature
    '''
    main function to compute network metrics from spike data. Simulated or experimental data.
    '''
    
    # set up save_path
    output_dir = kwargs.get('output_dir', None)  # output directory for saving results
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f'Output directory set to: {output_dir}')
        save_path = os.path.join(output_dir, 'network_data.npy')
        print(f'Results will be saved to: {save_path}')
    else:
        print('No output directory specified. Results will not be saved.')
        save_path = None
        
    
    # try_load
    try_load = kwargs.get('try_load', True)  # whether to try loading existing data if available
    if try_load:
        try:
            # check if save_path exists
            if save_path is not None and os.path.exists(save_path):
                print(f'Loading existing network data from {save_path}...')
                network_data = np.load(save_path, allow_pickle=True).item()
                print('Network data loaded successfully.')
                return network_data
        except Exception as e:
            print(f'Error loading existing network data: {e}')
            print('Proceeding to compute network metrics from scratch...')
            
    
    # init
    kwargs = _parse_options(kwargs) # parse analysis options, load defaults if not provided
    
    # unpack required inputs from kwargs
    spkt, spkt_by_unit, t = _unpack_required_inputs(kwargs)
    
    # init network_data_dict
    network_data = _initialize_network_data(kwargs, spkt, spkt_by_unit, t)
    
    # parallel?
    run_parallel = kwargs.get('parallel', True)  # default to True for parallel processing
    
    try:
        
        '''main computation steps'''
        
        ## compute spike metrics by unit
        if kwargs.get('compute_spike_metrics', True):
            #run_parallel = kwargs.get('parallel', True)
            network_data = csm.compute_spike_metrics(
                network_data, 
                run_parallel=False,  # for now, run spike metrics in serial
                verbose=True)
        
        ## compute burst metrics
        if kwargs.get('compute_burst_metrics', True):
            #run_parallel = kwargs.get('parallel', True)
            network_data = cbm.compute_burst_metrics(
                network_data, 
                run_parallel,
                max_workers=kwargs.get('max_workers', None),  # max workers for parallel processing 
                verbose=True,
                burst_sequencing = kwargs.get('burst_sequencing', True),  # whether to compute burst sequencing metrics
                )
        
        #if compute_HFBursting_metrics_flag:
            #network_data = compute_HFBursting_metrics(network_data, kwargs)
        
        if kwargs.get('classify_units', True):
            network_data = _classify_units(network_data, kwargs)
            network_data = _locate_units(network_data, kwargs)
        
        #if compute_dynamic_time_warping_flag and source in ['experimental']:
        if kwargs.get('compute_dynamic_time_warping', False):
            #if source == 'experimental': # for now only do with with experimental data
            network_data = _compute_dynamic_time_warping(network_data, kwargs)
        
        # if kwargs.get('source', None) == 'simulated' or kwargs.get('source', None) == 'netpyne':
        #     network_data = _add_sim_specific_data(network_data, kwargs) # for now only do with simulated data
        
        if kwargs.get('compute_summary_metrics', True):
            network_data = _summarize_metrics(network_data, kwargs)
        
        # return the final network data with all computed metrics
        print('Network metrics computation completed successfully.')
        
        # save
        if save_path is not None:
            np.save(save_path, network_data)
            print(f'Network metrics saved to {save_path}')
        
        # output_dir = kwargs.get('output_dir', None)
        # if output_dir is not None:
        #     save_path = os.path.join(output_dir, 'network_data.npy')
        #     np.save(save_path, network_data)
        #     print(f'Network data saved to {save_path}')
        # else:
        #     print('No output directory specified. Network data not saved.')
        
        return network_data
    except Exception as e:
        print(f'Error in compute_network_metrics: {e}')
        traceback.print_exc()
        #raise ValueError('Failed to compute network metrics.')
        #pass
        return None