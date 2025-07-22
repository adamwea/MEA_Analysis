# imports ==========
import time
import traceback
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.signal import convolve, find_peaks
from concurrent.futures import ProcessPoolExecutor, as_completed

# sub functions ===========
def process_burst(burst_id, burst_part):
    """Compute metrics for a given burst."""
    
    try:
        result = {}

        # Number of participating units
        result['num_units_participating'] = len(burst_part['participating_units'])
        
        # Spike counts by unit
        spike_counts = burst_part['spike_counts_by_unit']
        result['spike_counts_by_unit'] = {
            'data': spike_counts,
            'mean': np.nanmean(spike_counts),
            'std': np.nanstd(spike_counts),
            'cov': np.nanstd(spike_counts) / np.nanmean(spike_counts) if np.nanmean(spike_counts) > 0 else np.nan,
            'median': np.nanmedian(spike_counts),
            'min': np.nanmin(spike_counts),
            'max': np.nanmax(spike_counts),
        }
        
        # Burst duration
        sorted_spike_times = sorted(list(burst_part['spike_times']))
        duration = sorted_spike_times[-1] - sorted_spike_times[0]
        result['duration'] = duration
        
        # Spiking rate
        result['spike_rate'] = np.sum(spike_counts) / duration if duration > 0 else np.nan
        
        # ISI calculations
        isi_values = np.diff(sorted_spike_times)
        result['isi'] = {
            'data': isi_values,
            'mean': np.nanmean(isi_values),
            'std': np.nanstd(isi_values),
            'cov': np.nanstd(isi_values) / np.nanmean(isi_values) if np.nanmean(isi_values) > 0 else np.nan,
            'median': np.nanmedian(isi_values),
            'min': np.nanmin(isi_values),
            'max': np.nanmax(isi_values),
        }
        
        # # Firing sequence
        # sequence = []
        # sequence_times = []
        # for spike_time in sorted_spike_times:
        #     for unit, spike_times in zip(burst_part['participating_units'], burst_part['spike_times_by_unit']):
        #         if spike_time in spike_times:
        #             sequence.append(unit)
        #             sequence_times.append(spike_time)
        #             #break  # Ensure each spike is only counted once
        # #result['firing_sequence'] = sequence
        # result['unit_seqeunce'] = sequence
        # result['time_sequence'] = sequence_times
        # result['relative_time_sequence'] = [time - sequence_times[0] for time in sequence_times]
        
        # Firing sequence - more efficient - dict lookup
        # Build a flat lookup: spike_time -> unit
        # spike_lookup = {}
        # for unit, spike_times in zip(burst_part['participating_units'], burst_part['spike_times_by_unit']):
        #     for t in spike_times:
        #         spike_lookup[t] = unit  # assume spike_times are unique across units in a burst

        # # Now reconstruct sequence efficiently
        # sequence = []
        # sequence_times = []

        # for spike_time in sorted_spike_times:
        #     unit = spike_lookup.get(spike_time)
        #     if unit is not None:
        #         sequence.append(unit)
        #         sequence_times.append(spike_time)

        # # Store results
        # result['unit_seqeunce'] = sequence
        # result['time_sequence'] = sequence_times
        # result['relative_time_sequence'] = [t - sequence_times[0] for t in sequence_times] if sequence_times else []

        # Even more efficient for large datasets - vectorized approach # aw 2025-04-23 02:46:26
        # Step 1: Flatten all spike times and match to units
        all_spike_times = []
        all_unit_ids = []

        for unit, times in zip(burst_part['participating_units'], burst_part['spike_times_by_unit']):
            all_spike_times.extend(times)
            all_unit_ids.extend([unit] * len(times))

        # Step 2: Convert to arrays
        all_spike_times = np.array(all_spike_times)
        all_unit_ids = np.array(all_unit_ids)

        # Step 3: Sort spike times
        sort_idx = np.argsort(all_spike_times)
        sorted_spike_times = all_spike_times[sort_idx]
        sorted_units = all_unit_ids[sort_idx]

        # Step 4: Save to result
        result['unit_seqeunce'] = sorted_units.tolist()
        result['time_sequence'] = sorted_spike_times.tolist()
        result['relative_time_sequence'] = (sorted_spike_times - sorted_spike_times[0]).tolist() if sorted_spike_times.size else []
        
        print(f'Burst {burst_id} sequenced')
        
        return burst_id, result
    except Exception as e:
        print(f'Error in processing burst {burst_id}: {e}')
        return burst_id, None

def compute_unit_burst_participation(unit_metrics, convolved_data, max_workers=4, debug_mode=False):
    '''For each burst, compute which units do and don't participate.'''
    #indent_increase()
    left_base_times, right_base_times = convolved_data['left_base_times'], convolved_data['right_base_times']
    time_range = convolved_data['time_vector'][-1] - convolved_data['time_vector'][0]
    burst_starts = convolved_data['left_base_times']
    burst_ends = convolved_data['right_base_times']
    
    
    # Metrics ===============================================================
    participation_by_burst = {}
    for unit, metrics in unit_metrics.items():
        for burst_id, burst in metrics['bursts'].items():
            if burst_id not in participation_by_burst:
                participation_by_burst[burst_id] = {
                    #'duration': (burst[-1] - burst[0]),
                    'burst_start': burst_starts[burst_id],
                    'burst_end': burst_ends[burst_id],
                    'participating_units': [unit,],
                    'spike_count': len(burst),
                    'spike_times': set(burst),
                    'spike_counts_by_unit': [len(burst)],
                    'spike_times_by_unit': [burst,],
                    #'spike_times': {unit: burst,}                
                }
            else:
                participation_by_burst[burst_id]['participating_units'].append(unit)
                participation_by_burst[burst_id]['spike_count'] += len(burst)
                participation_by_burst[burst_id]['spike_times'].update(burst)
                participation_by_burst[burst_id]['spike_counts_by_unit'].append(len(burst))
                #participation_by_burst[burst_id]['spike_times'].append(set(burst))
                participation_by_burst[burst_id]['spike_times_by_unit'].append(burst)
    
    # sort participation by burst by key (burst_id)
    participation_by_burst = dict(sorted(participation_by_burst.items()))    
    
    # set max_workers
    if max_workers is None:
        max_workers = 1
    else:
        max_workers = max_workers
    
    # debug_mode
    #debug_mode = True
    if debug_mode:
        print(f'Debug mode: only processing first 2 burst parts')
        participation_by_burst = dict(list(participation_by_burst.items())[:2])
        max_workers = 2
    
    if max_workers is None:
        max_workers = 1
        
    if len(participation_by_burst) < max_workers:
        max_workers = len(participation_by_burst)
    
    if len(participation_by_burst) >= 1:    
        print(f'using {max_workers} cpus to process bursts')
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_burst_id = {executor.submit(process_burst, burst_id, burst_part): burst_id for burst_id, burst_part in participation_by_burst.items()}
            
            for future in as_completed(future_to_burst_id):
                burst_id, burst_result = future.result()
                participation_by_burst[burst_id].update(burst_result)
        
        # confirm that all bursts are accounted for and in order
        last_id = None
        for burst_id, burst_part in participation_by_burst.items():
            if last_id is None: 
                last_id = burst_id
                continue
            else:       
                if last_id > burst_id:
                    print(f'Error: Bursts are not in order')
                expected_val = last_id + 1
                if burst_id != expected_val:
                    print(f'Error: Missing burst {expected_val}')            
                last_id = burst_id                
        
        # return
        #indent_decrease()
        return participation_by_burst
    else:
        print(f'No bursts to process')
        #indent_decrease()
        return {}        

def burstwise_analysis(unit_metrics, convolved_data, max_workers=4, debug_mode=False, **kwargs):
    burst_sequencing = kwargs.get('burst_sequencing', False)
    warnings = None

    peak_times = convolved_data.get('peak_times', np.array([]))
    burst_parts = "Burst sequencing not enabled"
    if burst_sequencing:
        try:
            burst_parts = compute_unit_burst_participation(unit_metrics, convolved_data, max_workers=max_workers, debug_mode=debug_mode)
            burst_parts = {**burst_parts}
        except Exception as e:
            burst_parts = {}
            print(f"⚠️ Error computing burst participation: {e}")

    burst_metrics = {}

    try:
        burst_metrics['num_bursts'] = len(peak_times)
        burst_metrics['burst_rate'] = len(peak_times) / (convolved_data['time_vector'][-1] - convolved_data['time_vector'][0]) if len(peak_times) > 1 else np.nan
        burst_metrics['burst_ids'] = list(range(len(peak_times)))
        print("✔️ Basic burst counts and rates computed")
    except Exception as e:
        print(f"❌ Error computing basic burst stats: {e}")
        burst_metrics['num_bursts'] = 0
        burst_metrics['burst_rate'] = np.nan
        burst_metrics['burst_ids'] = []

    # IBI stats
    try:
        ibi_values = np.diff(peak_times)
        burst_metrics['ibi'] = {
            'data': ibi_values,
            'mean': np.nanmean(ibi_values) if ibi_values.size else np.nan,
            'std': np.nanstd(ibi_values) if ibi_values.size else np.nan,
            'cov': np.nanstd(ibi_values) / np.nanmean(ibi_values) if np.nanmean(ibi_values) > 0 else np.nan,
            'median': np.nanmedian(ibi_values) if ibi_values.size else np.nan,
            'min': np.nanmin(ibi_values) if ibi_values.size else np.nan,
            'max': np.nanmax(ibi_values) if ibi_values.size else np.nan,
        }
        print("✔️ Inter-burst interval (IBI) stats computed")
    except Exception as e:
        print(f"❌ Error computing IBI: {e}")
        burst_metrics['ibi'] = {'data': [], 'mean': np.nan, 'std': np.nan, 'cov': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan}

    # Amplitude
    try:
        amp_values = convolved_data.get('peak_values', np.array([]))
        burst_metrics['burst_amp'] = {
            'data': amp_values,
            'mean': np.nanmean(amp_values) if amp_values.size else np.nan,
            'std': np.nanstd(amp_values) if amp_values.size else np.nan,
            'cov': np.nanstd(amp_values) / np.nanmean(amp_values) if np.nanmean(amp_values) > 0 else np.nan,
            'median': np.nanmedian(amp_values) if amp_values.size else np.nan,
            'min': np.nanmin(amp_values) if amp_values.size else np.nan,
            'max': np.nanmax(amp_values) if amp_values.size else np.nan,
        }
        print("✔️ Burst amplitude stats computed")
    except Exception as e:
        print(f"❌ Error computing amplitude: {e}")
        burst_metrics['burst_amp'] = {'data': [], 'mean': np.nan, 'std': np.nan, 'cov': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan}

    # Duration
    try:
        left = convolved_data.get('left_base_times', np.array([]))
        right = convolved_data.get('right_base_times', np.array([]))
        durations = right - left if left.size and right.size else np.array([])
        burst_metrics['burst_duration'] = {
            'data': durations,
            'mean': np.nanmean(durations) if durations.size else np.nan,
            'std': np.nanstd(durations) if durations.size else np.nan,
            'cov': np.nanstd(durations) / np.nanmean(durations) if np.nanmean(durations) > 0 else np.nan,
            'median': np.nanmedian(durations) if durations.size else np.nan,
            'min': np.nanmin(durations) if durations.size else np.nan,
            'max': np.nanmax(durations) if durations.size else np.nan,
        }
        print("✔️ Burst duration stats computed")
    except Exception as e:
        print(f"❌ Error computing duration: {e}")
        burst_metrics['burst_duration'] = {'data': [], 'mean': np.nan, 'std': np.nan, 'cov': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan}

    burst_metrics['burst_parts'] = burst_parts
    print("✔️ Burst parts registered")

    # Units per burst
    try:
        if isinstance(burst_parts, dict):
            num_units_list = [bp.get('num_units_participating', np.nan) for bp in burst_parts.values()]
            burst_metrics['num_units_per_burst'] = {
                'data': num_units_list,
                'mean': np.nanmean(num_units_list),
                'std': np.nanstd(num_units_list),
                'cov': np.nanstd(num_units_list) / np.nanmean(num_units_list) if np.nanmean(num_units_list) > 0 else np.nan,
                'median': np.nanmedian(num_units_list),
                'min': np.nanmin(num_units_list),
                'max': np.nanmax(num_units_list),
            }
            print("✔️ Num units per burst stats computed")
        else:
            raise ValueError("Burst sequencing not enabled")
    except Exception as e:
        print(f"⚠️ Num units per burst skipped: {e}")
        burst_metrics['num_units_per_burst'] = "Burst sequencing not enabled"

    # In-burst firing rate
    try:
        if isinstance(burst_parts, dict):
            spike_rates = [bp.get('spike_rate', np.nan) for bp in burst_parts.values()]
            burst_metrics['in_burst_fr'] = {
                'data': spike_rates,
                'mean': np.nanmean(spike_rates),
                'std': np.nanstd(spike_rates),
                'cov': np.nanstd(spike_rates) / np.nanmean(spike_rates) if np.nanmean(spike_rates) > 0 else np.nan,
                'median': np.nanmedian(spike_rates),
                'min': np.nanmin(spike_rates),
                'max': np.nanmax(spike_rates),
            }
            print("✔️ In-burst firing rate stats computed")
        else:
            raise ValueError("Burst sequencing not enabled")
    except Exception as e:
        print(f"⚠️ In-burst firing rate stats skipped: {e}")
        burst_metrics['in_burst_fr'] = "Burst sequencing not enabled"

    print("✅ Burst Metrics Computed")
    return burst_metrics, warnings

def analyze_unit_activity(spike_times_by_unit, convolved_data):
    """Analyze bursts, quiet periods, firing rates, and ISIs for each unit, safely handling empty spike_times."""

    if convolved_data.get('time_vector') is None or len(convolved_data['time_vector']) == 0:
        print("[Info] convolved_data['time_vector'] is empty. Skipping unit activity analysis.")
        return {}, None  # no units, no warnings
    
    total_number_of_bursts_in_convolved_data = len(convolved_data['peak_times'])
    left_base_times, right_base_times = convolved_data['left_base_times'], convolved_data['right_base_times']
    time_range = convolved_data['time_vector'][-1] - convolved_data['time_vector'][0]

    # 2025-05-30 16:00:18 due to overlapping bursts, need to consider all left and right base times to get left and right quiet periods
    # merged = []
    # intervals = np.column_stack((left_base_times, right_base_times))

    # # remove idx 2 in intervals
    # intervals = np.delete(intervals, 2, axis=0) # 2025-06-02 01:48:02 for debugging purposes, remove idx 2 in intervals

    # earliest_start = None
    # lastest_end = None
    # for start, end in intervals:
    #     if lastest_end is not None:
    #         if start > lastest_end:
    #             print('found quiet period')
    #             print(f'Quiet period from {lastest_end} to {start}')
    #     if earliest_start is None or start < earliest_start:
    #         earliest_start = start
    #     if lastest_end is None or end > lastest_end:
    #         lastest_end = end

    # 2025-05-30 16:00:25 the above method might work for hyper bursting... unsure... but it appears bursting periods start and end right ontop of eachother in regular bursting so there arent any quiet periods...which is not right.
    # insteady I'm just got to find periods where we're below baseline + 10% in the meantime.
    baseline = convolved_data.get('baseline', None)
    if baseline is None:
        print("[Warning] No baseline found in convolved_data. Using 0 as baseline.")
        baseline = 0.0  # Default to 0 if no baseline is provided
    threshold = baseline * 1.1  # 10% above baseline
    convolved_FR = convolved_data.get('convolved_FR', None)
    time_vector = convolved_data.get('time_vector', None)

    #get times where threshold is crossed
    if convolved_FR is None or time_vector is None:
        raise ValueError("convolved_FR and time_vector must be provided in convolved_data for burst analysis.")
    
    #quiet_times = time_vector[np.where(convolved_FR <= threshold)[0]]
    quiet_time_idx = np.where(convolved_FR <= threshold)[0]

    # get continuous quiet periods
    quiet_idx_intervals = []
    if len(quiet_time_idx) > 0:
        # Find continuous quiet periods
        start_idx = quiet_time_idx[0]
        for i in range(1, len(quiet_time_idx)):
            if quiet_time_idx[i] != quiet_time_idx[i - 1] + 1:  # Check for discontinuity
                quiet_idx_intervals.append((start_idx, quiet_time_idx[i - 1]))
                start_idx = quiet_time_idx[i]
        # Add the last period
        quiet_idx_intervals.append((start_idx, quiet_time_idx[-1]))

        # convert to time intervals
        left_quiet_times = time_vector[[start for start, _ in quiet_idx_intervals]]
        right_quiet_times = time_vector[[end for _, end in quiet_idx_intervals]]

    else:
        print("[Info] No quiet periods found in convolved data.")
    
    bursts_by_unit, non_bursts_by_unit, burst_durations, quiet_durations, warnings = {}, {}, {}, {}, []
    unit_data = {}

    for unit, spike_times in spike_times_by_unit.items():
        # if unit == 9:
        #     print(f"Skipping unit {unit} for debugging purposes.")
        spike_times = np.asarray(spike_times)
        bursts, non_bursts = {}, {}
        burst_id, quiet_id = 0, 0

        if spike_times.size == 0:
            bursts_by_unit[unit] = {}
            non_bursts_by_unit[unit] = {}
            unit_data[unit] = {
                'burst_id': [],
                'quiet_id': [],
                'bursts': {},
                'quiets': {},
                'burst_durations': {},
                'quiet_durations': {},
                'burst_part_rate': 0.0,
                'quiet_part_rate': 0.0,
                'burst_part_perc': 0.0,
                'fr': {'in_burst': {}, 'out_burst': {}},
                'isi': {'in_burst': {}, 'out_burst': {}},
                'spike_counts': {'in_burst': {}, 'out_burst': {}},
                'fano_factor': {'in_burst': np.nan, 'out_burst': np.nan},
                'note': "No spikes in this unit."
            }
            continue

        for left, right in zip(left_base_times, right_base_times):
            burst_spikes = spike_times[(spike_times >= left) & (spike_times <= right)]
            burst_durations[burst_id] = right - left
            if burst_spikes.size > 0:
                bursts[burst_id] = burst_spikes
                if np.any(np.diff(burst_spikes) < 0):
                    warnings.append(f'Negative ISI in unit {unit} (burst {burst_id})')
            burst_id += 1

        # quiet_left = np.concatenate([[0], right_base_times[:-1]])
        # last_time = spike_times[-1] if spike_times.size > 0 else convolved_data['time_vector'][-1]
        # quiet_right = np.concatenate([left_base_times[1:], [last_time]])

        # rewriting this for testing, but maybe comment out soon.
        # left_quiet_times = np.concatenate([[0], right_base_times[:-1]])
        # last_time = spike_times[-1] if spike_times.size > 0 else convolved_data['time_vector'][-1]
        # right_quiet_times = np.concatenate([left_base_times[1:], [last_time]])

        #for left, right in zip(quiet_left, quiet_right):
        for left, right in zip(left_quiet_times, right_quiet_times):
            quiet_spikes = spike_times[(spike_times >= left) & (spike_times <= right)]
            quiet_durations[quiet_id] = right - left
            if quiet_spikes.size > 0:
                non_bursts[quiet_id] = quiet_spikes
                if np.any(np.diff(quiet_spikes) < 0):
                    warnings.append(f'Negative ISI in unit {unit} (non-burst {quiet_id})')
            quiet_id += 1

        bursts_by_unit[unit] = bursts
        non_bursts_by_unit[unit] = non_bursts

        # Define analytics functions inside the loop
        def compute_isi_metrics(isis):
            try:
                all_isis = np.concatenate([np.diff(burst) for burst in isis.values() if len(burst) > 1])
                return {
                    'data': all_isis,
                    'mean': np.nanmean(all_isis),
                    'std': np.nanstd(all_isis),
                    'cov': np.nanstd(all_isis) / np.nanmean(all_isis) if np.nanmean(all_isis) > 0 else np.nan,
                    'median': np.nanmedian(all_isis),
                    'min': np.nanmin(all_isis),
                    'max': np.nanmax(all_isis),
                }
            except Exception:
                return dict.fromkeys(['data', 'mean', 'std', 'cov', 'median', 'min', 'max'], np.nan)

        def compute_fr(bursts, durations):
            #fr_data = {i: len(b) / durations[i] if durations[i] > 0 else np.nan for i, b in bursts.items()}
            fr_data ={}
            for i, b in bursts.items():
                if durations[i] > 0:
                    if len(b) > 1: # need at least 2 spikes to compute a firing rate
                        fr_data[i] = len(b) / durations[i]
                    else:
                        fr_data[i] = np.nan  # Not enough spikes to compute firing rate
                else:
                    fr_data[i] = np.nan
                #if i==9:break
            valid_values = [v for v in fr_data.values() if not np.isnan(v)]
            return {
                'data': fr_data,
                'mean': np.nanmean(valid_values) if valid_values else np.nan,
                'std': np.nanstd(valid_values) if valid_values else np.nan,
                'cov': np.nanstd(valid_values) / np.nanmean(valid_values) if valid_values and np.nanmean(valid_values) > 0 else np.nan,
                'median': np.nanmedian(valid_values) if valid_values else np.nan,
                'min': np.nanmin(valid_values) if valid_values else np.nan,
                'max': np.nanmax(valid_values) if valid_values else np.nan,
            }

        def compute_spike_counts(bursts):
            counts = {i: len(b) for i, b in bursts.items()}
            values = np.array(list(counts.values()))
            return {
                'data': counts,
                'mean': np.mean(values) if values.size > 0 else np.nan,
                'std': np.std(values) if values.size > 0 else np.nan,
                'cov': np.std(values) / np.mean(values) if values.size > 0 and np.mean(values) > 0 else np.nan,
                'median': np.median(values) if values.size > 0 else np.nan,
                'min': np.min(values) if values.size > 0 else np.nan,
                'max': np.max(values) if values.size > 0 else np.nan,
            }

        def compute_fano(counts):
            values = list(counts.values())
            return np.var(values) / np.mean(values) if values and np.mean(values) > 0 else np.nan

        # Compose all metrics
        fr_analytics_dict = {
            'in_burst': compute_fr(bursts, burst_durations),
            'out_burst': compute_fr(non_bursts, quiet_durations)
        }
        isi_analytics_dict = {
            'in_burst': compute_isi_metrics(bursts),
            'out_burst': compute_isi_metrics(non_bursts)
        }
        spike_count_analytics_dict = {
            'in_burst': compute_spike_counts(bursts),
            'out_burst': compute_spike_counts(non_bursts)
        }
        fano_factor_dict = {
            'in_burst': compute_fano(spike_count_analytics_dict['in_burst']['data']),
            'out_burst': compute_fano(spike_count_analytics_dict['out_burst']['data'])
        }

        unit_data[unit] = {
            'burst_id': list(bursts.keys()),
            'quiet_id': list(non_bursts.keys()),
            'bursts': bursts,
            'quiets': non_bursts,
            'burst_durations': burst_durations,
            'quiet_durations': quiet_durations,
            'burst_part_rate': len(bursts) / time_range if time_range > 0 else np.nan,
            'quiet_part_rate': len(non_bursts) / time_range if time_range > 0 else np.nan,
            'burst_part_perc': len(bursts) / total_number_of_bursts_in_convolved_data if total_number_of_bursts_in_convolved_data > 0 else np.nan,
            'fr': fr_analytics_dict,
            'isi': isi_analytics_dict,
            'spike_counts': spike_count_analytics_dict,
            'fano_factor': fano_factor_dict,
            'note': "The same spike may be represented in multiple bursts if they compound in overlapping epochs",
        }

    if not warnings:
        warnings = None

    return unit_data, warnings

def convolve_network_activity(ax, SpikeTimes, min_peak_distance=1.0, 
                             binSize=0.1, 
                             gaussianSigma=0.16, 
                             thresholdBurst=1.2, 
                             prominence=1, 
                             figSize=(10, 6),
                             title='Network Activity'
                             ):
    relativeSpikeTimes = []
    units = 0
    for unit_id, spike_times in SpikeTimes.items():
        temp_spike_times = spike_times
        if isinstance(temp_spike_times, np.ndarray):
            relativeSpikeTimes.extend(temp_spike_times.tolist())
        elif isinstance(temp_spike_times, (float, int)):
            relativeSpikeTimes.append(temp_spike_times)
        else:
            print(f'[Warning] Unit {unit_id} has spike times that are not float/int/array.')
            continue
        units += 1

    if not relativeSpikeTimes:
        print("[Info] No spike times provided. Skipping network activity plot.")
        return ax, {
            'convolved_FR': np.array([]),
            'peak_idxs': np.array([]),
            'peak_times': np.array([]),
            'peak_values': np.array([]),
            'prominences': np.array([]),
            'left_base_idxs': np.array([]),
            'right_base_idxs': np.array([]),
            'left_base_times': np.array([]),
            'right_base_times': np.array([]),
            'time_vector': np.array([]),
        }

    relativeSpikeTimes = np.sort(np.array(relativeSpikeTimes))

    # Step 1: Bin all spike times into small time windows
    timeVector = np.arange(min(relativeSpikeTimes), max(relativeSpikeTimes), binSize)
    binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)
    binnedTimes = np.append(binnedTimes, 0)

    # Step 2: Smooth the binned spike times with a Gaussian kernel
    kernelRange = np.arange(-3 * gaussianSigma, 3 * gaussianSigma + binSize, binSize)
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)
    kernel *= binSize
    firingRate = convolve(binnedTimes, kernel, mode='same') / binSize
    firingRate = firingRate / max(units, 1)

    # Plot
    ax.plot(timeVector, firingRate, color='royalblue')
    ax.set_xlim([min(relativeSpikeTimes), max(relativeSpikeTimes)])
    ax.set_ylim([min(firingRate) * 0.85, max(firingRate) * 1.15])
    ax.set_ylabel('Firing Rate [Hz]')
    ax.set_xlabel('Time [ms]')
    ax.set_title(title, fontsize=11)

    peaks, properties = find_peaks(firingRate, prominence=prominence, distance=min_peak_distance)
    burstPeakTimes = timeVector[peaks]
    burstPeakValues = firingRate[peaks]

    # post process peaks, left and right bases
    left_base_idxs = properties.get('left_bases', np.array([]))
    right_base_idxs = properties.get('right_bases', np.array([]))
    left_base_times = timeVector[left_base_idxs] if left_base_idxs.size > 0 else np.array([])
    right_base_times = timeVector[right_base_idxs] if right_base_idxs.size > 0 else np.array([])


    # # post process peaks, left and right bases
    # peaks = peaks.copy()
    # L = properties.get('left_bases', np.array([], dtype=int))
    # R = properties.get('right_bases', np.array([], dtype=int))
    # L_times = timeVector[L] if L.size > 0 else np.array([])
    # R_times = timeVector[R] if R.size > 0 else np.array([])

    # # 1. compute interval widths and sort by width (smallest first)
    # widths = R - L
    # widths_times = R_times - L_times
    # order  = np.argsort(widths)

    # kept = []
    # for i in order:
    #     li, ri = L[i], R[i]
    #     # if this interval contains any already-kept peak, skip it
    #     if any(li < peaks[j] < ri for j in kept):
    #         continue
    #     kept.append(i)

    # kept = np.array(kept, dtype=int)


    ax.plot(burstPeakTimes, burstPeakValues, 'or')

    baseline = np.mean(firingRate)

    convolved_data = {
        'convolved_FR': firingRate,
        'baseline': baseline,
        'peak_idxs': peaks,
        'peak_times': burstPeakTimes,
        'peak_values': burstPeakValues,
        'prominences': properties.get('prominences', np.array([])),
        #'left_base_idxs': properties.get('left_bases', np.array([])),
        #'right_base_idxs': properties.get('right_bases', np.array([])),
        #'left_base_times': timeVector[properties['left_bases']] if 'left_bases' in properties else np.array([]),
        #'right_base_times': timeVector[properties['right_bases']] if 'right_bases' in properties else np.array([]),
        'left_base_idxs': left_base_idxs,
        'right_base_idxs': right_base_idxs,
        'left_base_times': left_base_times,
        'right_base_times': right_base_times,
        'time_vector': timeVector,
    }

    return ax, convolved_data

def _line_debug(ax):
    # make sure there are lines in the ax
    lines = ax.get_lines()
    if len(lines) == 0:
        print("No lines in ax")
    else:
        print(f"Number of lines in ax: {len(lines)}")
        for line in lines:
            print(f"Line: {line.get_label()}")
    # check if there are any lines in the ax

def analyze_burst_activity(
    spike_times, 
    spike_times_by_unit, 
    min_peak_distance=1.0, 
    binSize=0.001,
    gaussianSigma=0.01, 
    thresholdBurst=0.5, 
    prominence=1, 
    title='Network Activity',
    debug_mode=False,
    **kwargs
    ):
    
    # init warnings
    unit_warnings = None
    burst_warnings = None
         
    #
    try:
        # 
        start = time.time()

        # Step 1: Generate convolved data from network activity plot
        try:
            fig, ax = plt.subplots()
            ax, convolved_data = convolve_network_activity(
                ax, 
                spike_times_by_unit,
                binSize=binSize, 
                gaussianSigma=gaussianSigma,
                thresholdBurst=thresholdBurst, 
                prominence=prominence, 
                title=title
            )
        except Exception as e:
            traceback.print_exc()
            print(f'Error in convolving network activity: {e}')
            plt.close(fig)           
            return None

        # Step 2: Analyze individual units
        try:
            unit_metrics, unit_warnings = analyze_unit_activity(spike_times_by_unit, convolved_data)
            if unit_warnings:
                print("\n".join(unit_warnings))
        except Exception as e:
            traceback.print_exc()
            print(f'Error in analyzing unit activity: {e}')
            plt.close(fig)
            return {
                'ax' : ax,
                'convolved_data': convolved_data,
                'unit_metrics': None,
                'burst_metrics': None,
                'warnings': {
                    'unit': unit_warnings,
                    'burst': None
                }
            }
            
        # Step 3: Burst Summary Metrics
        try:
            max_workers = kwargs.get('max_workers', 4)
            burst_metrics, burst_warnings = burstwise_analysis(
                unit_metrics, 
                convolved_data, 
                debug_mode=debug_mode, 
                **kwargs
                )
            if burst_warnings:
                print("\n".join(burst_warnings))
        except Exception as e:
            traceback.print_exc()
            print(f'Error in computing burst metrics: {e}')
            plt.close(fig) # NOTE: for some reason this is required for ax to persist later in the code -- I dont fully understand why
            return {
                'ax' : ax,
                'convolved_data': convolved_data,
                'unit_metrics': unit_metrics,
                'burst_metrics': None,
                'warnings': {
                    'unit': unit_warnings,
                    'burst': burst_warnings
                }
            }
        
        # return
        print(f'Elapsed time: {time.time() - start}')
        
        #debug - check if zero lines in ax
        #_line_debug(ax)
        
        plt.close(fig)
        return {
            'ax': ax,
            'convolved_data': convolved_data,            
            'unit_metrics': unit_metrics,
            'burst_metrics': burst_metrics,
            'warnings': {
                'unit': unit_warnings,
                'burst': burst_warnings,
            }
        }
    except Exception as e:
        traceback.print_exc()
        print(f'Error in bursting activity analysis: {e}')
        return None

#def compute_burst_metrics(network_data, kwargs):
def compute_burst_metrics(network_data, run_parallel=False, verbose=False, **kwargs):

    #
    spkt = network_data['inputs']['spkt']
    spkt_by_unit = network_data['inputs']['spkt_by_unit']
    
    # requred parameters
    burst_params = network_data['inputs'].get('burst_params', None)
    assert burst_params is not None, 'No burst parameters provided'
    assert 'binSize' in burst_params, 'Burst parameters must include "binSize"'
    assert 'gaussianSigma' in burst_params, 'Burst parameters must include "gaussianSigma"'
    assert 'prominence' in burst_params, 'Burst parameters must include "prominence"'
    
    #
    hyperburst_params = network_data['inputs'].get('hyperburst_params', None)
    assert hyperburst_params is not None, 'No hyperburst parameters provided'
    assert 'binSize' in hyperburst_params, 'Hyperburst parameters must include "binSize"'
    assert 'gaussianSigma' in hyperburst_params, 'Hyperburst parameters must include "gaussianSigma"'
    assert 'prominence' in hyperburst_params, 'Hyperburst parameters must include "prominence"'
    
    
    try:
        kwargs['spike_times'] = spkt
        kwargs['spike_times_by_unit'] = spkt_by_unit
        kwargs['binSize'] = burst_params['binSize']
        kwargs['gaussianSigma'] = burst_params['gaussianSigma']
        kwargs['thresholdBurst'] = burst_params['thresholdBurst']
        kwargs['min_peak_distance'] = burst_params['min_peak_distance']
        kwargs['prominence'] = burst_params['prominence']
        bursting_data = analyze_burst_activity(**kwargs)
        
    except Exception as e:
        print(f'Error analyzing bursting activity: {e}')
        traceback.print_exc()
        bursting_data = None
    
    try:
        kwargs['spike_times'] = spkt
        kwargs['spike_times_by_unit'] = spkt_by_unit
        kwargs['binSize'] = hyperburst_params['binSize']
        kwargs['gaussianSigma'] = hyperburst_params['gaussianSigma']
        kwargs['thresholdBurst'] = hyperburst_params['thresholdBurst']
        kwargs['min_peak_distance'] = hyperburst_params['min_peak_distance']
        kwargs['prominence'] = hyperburst_params['prominence']
        hyperbursting_data = analyze_burst_activity(**kwargs)
    except Exception as e:
        print(f'Error analyzing hyper bursting activity: {e}')
        traceback.print_exc()
        hyperbursting_data = None
    
    # update network data
    network_data.update({
        #'bursting_data': bursting_data,
        #'mega_bursting_data': mega_bursting_data,
        'burst_metrics': bursting_data,
        'hyperburst_metrics': hyperbursting_data,
    })
    
    return network_data