# imports
import numpy as np
import spikeinterface.full as si
#from spikeinterface.core import compute_firing_rate, compute_isi_metrics
#from spikeinterface.core.waveform_tools import load_best_channel_waveforms, compute_waveform_metrics_safe
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor, as_completed  # safer than ProcessPool in many I/O-heavy cases

# sub functions
def interpolate_and_smooth(unit_wfs, num_interp_samples, sigma=9):
    """Interpolate and apply Gaussian smoothing to waveforms."""
    num_samples = unit_wfs.shape[1]
    x = np.arange(num_samples)
    x_new = np.linspace(0, num_samples - 1, num_interp_samples)
    interpolated = np.stack([np.interp(x_new, x, wf) for wf in unit_wfs])
    smoothed = gaussian_filter1d(interpolated, sigma=sigma, axis=1)
    return smoothed

def compute_wf_metrics_modular(best_channel_waveforms, sampling_rate, plot_wf=False, save_fig=False, fig_name="waveform_debug.png", unit=None):
    unit_wfs = best_channel_waveforms
    num_samples = unit_wfs.shape[1]
    sampling_duration = num_samples / sampling_rate * 1000  # ms
    num_interp_samples = 1000
    time_conversion_factor = sampling_duration / num_interp_samples

    # Step 1: Interpolation & Smoothing
    interpolated_wfs = interpolate_and_smooth(unit_wfs, num_interp_samples)

    # Step 2: Compute mean and Mahalanobis-based weights
    mean_vector = np.nanmean(interpolated_wfs, axis=0)
    if len(interpolated_wfs) < 2:
        return {'excluded': True, 'reason': 'Not enough waveforms'}

    try:
        weights, inv_covariance = compute_mahalanobis_weights(interpolated_wfs, mean_vector)
    except Exception:
        return {'excluded': True, 'reason': 'Covariance computation failed'}

    weighted_avg_wf = np.average(interpolated_wfs, axis=0, weights=weights)

    # Step 3: Biological Variability
    weighted_variability = np.average(np.abs(interpolated_wfs - weighted_avg_wf), axis=0, weights=weights)
    weighted_coefficient_of_variation = weighted_variability / np.abs(weighted_avg_wf)
    bio_variability_metrics = {
        'weighted_variability': weighted_variability,
        'mean_variance': np.mean(weighted_variability),
        'std_variance': np.std(weighted_variability),
        'cov_variance': np.std(weighted_variability) / np.mean(weighted_variability),
        'weighted_coefficient_of_variation': weighted_coefficient_of_variation,
        'mean_cv': np.mean(weighted_coefficient_of_variation),
        'std_cv': np.std(weighted_coefficient_of_variation),
        'cov_cv': np.std(weighted_coefficient_of_variation) / np.mean(weighted_coefficient_of_variation),
    }

    # Step 4: Smearing metrics
    vertical_smearing, horizontal_smearing, trough_times, peak_times = compute_smearing_metrics(interpolated_wfs, time_conversion_factor)

    # Step 5: Key points
    trough_idx = np.argmin(weighted_avg_wf)
    peak_idx = np.argmax(weighted_avg_wf[trough_idx:]) + trough_idx
    ap_start_idx, ap_end_idx, refractory_end_idx = compute_zero_crossings(weighted_avg_wf, trough_idx)

    # Step 6: Waveform feature extraction
    waveform_features = compute_waveform_features(
        weighted_avg_wf, trough_idx, peak_idx,
        ap_start_idx, ap_end_idx, refractory_end_idx,
        time_conversion_factor, sampling_duration
    )

    # Step 7: Plotting
    if plot_wf:
        time_axis = np.linspace(0, sampling_duration, num_interp_samples)
        plt.figure(figsize=(8, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time_axis, interpolated_wfs.T, color='gray', alpha=0.75, linewidth=0.25)
        plt.plot(time_axis, weighted_avg_wf, color='red', linewidth=1.0, label="weighted mean")
        plt.fill_between(time_axis, weighted_avg_wf - vertical_smearing["std_across_time"],
                         weighted_avg_wf + vertical_smearing["std_across_time"], color='red', alpha=0.5)
        plt.title(f"Unit {unit}")
        plt.legend(fontsize='small')
        plt.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.5)
        plt.xlim(0, max(time_axis))
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.subplot(2, 1, 2)
        bins = np.linspace(0, max(time_axis), 100)
        plt.hist(trough_times * time_conversion_factor, bins=bins, color="blue", alpha=0.6, label="trough timing")
        plt.hist(peak_times * time_conversion_factor, bins=bins, color="green", alpha=0.6, label="peak timing")
        plt.xlabel("time (ms)")
        plt.ylabel("spikes")
        plt.legend(fontsize='small')
        plt.xlim(0, max(time_axis))
        plt.tight_layout()

        if save_fig:
            fig_dir = os.path.dirname(fig_name)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            plt.savefig(fig_name, dpi=300)
            plt.savefig(fig_name.replace('.png', '.pdf'))
        else:
            plt.show()
        plt.close()

    return {
        'weighted_avg_wf': weighted_avg_wf,
        'bio_variability_metrics': bio_variability_metrics,
        **waveform_features,
        'vertical_smearing': vertical_smearing,
        'horizontal_smearing': horizontal_smearing,
    }

def compute_waveform_metrics_safe(waveforms, sampling_rate, unit, plot_wf, sa_well_folder):
    try:
        fig_path = sa_well_folder.replace('analyzer', 'wf_plots') + f"/unit_{unit}_waveforms.png"
        return compute_wf_metrics_modular(
            waveforms, sampling_rate,
            plot_wf=plot_wf, save_fig=True,
            unit=unit, fig_name=fig_path
        )
    except Exception as e:
        print(f"⚠️ Error computing waveform metrics: {e}")
        return 'Error computing waveform metrics'

def load_best_channel_waveforms(sa, unit):
    try:
        unit_wfs = sa.get_extension("waveforms").get_waveforms_one_unit(unit)
        avg_waveform = np.nanmean(unit_wfs, axis=0)
        best_channel_idx = np.argmax(np.max(np.abs(avg_waveform), axis=0))
        return unit_wfs[:, :, best_channel_idx]
    except Exception as e:
        print(f"⚠️ Error loading waveform data: {e}")
        return None

def compute_firing_rate(spike_times):
    try:
        if len(spike_times) > 1:
            return len(spike_times) / (spike_times[-1] - spike_times[0])
    except Exception as e:
        print(f"⚠️ Error computing firing rate: {e}")
    return np.nan

def compute_isi_metrics(spike_times):
    isi_diffs = np.diff(spike_times) if len(spike_times) > 1 else np.array([])
    if isi_diffs.size == 0:
        return {
            'data': isi_diffs,
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'cov': np.nan, 'max': np.nan, 'min': np.nan
        }
    try:
        mean = np.nanmean(isi_diffs)
        return {
            'data': isi_diffs,
            'mean': mean,
            'std': np.nanstd(isi_diffs),
            'median': np.nanmedian(isi_diffs),
            'cov': np.nanstd(isi_diffs) / mean if mean > 0 else np.nan,
            'max': np.nanmax(isi_diffs),
            'min': np.nanmin(isi_diffs)
        }
    except Exception as e:
        print(f"⚠️ Error computing ISI stats: {e}")
        return {
            'data': isi_diffs,
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'cov': np.nan, 'max': np.nan, 'min': np.nan
        }

def get_time_vector(recording_object, sampling_rate=10000):
    duration = recording_object.get_total_duration() #seconds
    time_vector = np.linspace(0, duration, int(duration * sampling_rate))
    assert len(time_vector) == int(duration * sampling_rate), 'Time vector length mismatch'
    return time_vector

def get_spike_times_by_unit(sorting_object, sampling_rate=10000):
    spike_times_by_unit = {}
    units = sorting_object.get_unit_ids()
    for unit in units:
        spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
        assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
        spike_times_by_unit[unit] = spike_train
    return spike_times_by_unit

def get_spike_times(recording_object, sorting_object, sampling_rate=10000):
    spike_times = []
    units = sorting_object.get_unit_ids()
    total_duration = recording_object.get_total_duration() #seconds
    for unit in units:
        spike_train = sorting_object.get_unit_spike_train(unit) #in samples
        spike_times.extend(spike_train)
    spike_times.sort()
    spike_times = np.array(spike_times) / sampling_rate #convert to seconds
    
    # quality control
    # spike_times_max need to be rounded to the nearest thousandths place to avoid floating point errors
    spike_times_max = np.round(max(spike_times), 3)
    
    print(f'Max Spike Time: {max(spike_times)}')
    #assert max(spike_times) <= total_duration, 'Spike times are not in seconds'
    assert spike_times_max <= total_duration, 'Spike times are not in seconds'
    assert all(spike_time >= 0 for spike_time in spike_times), 'Spike times contain negative values'
    
    #return
    return spike_times

def compute_unit_spike_metrics(
    unit, 
    spkt_by_unit, 
    #sampling_rate, plot_wfs, recording_object, sorting_object, sa_well_folder, 
    verbose=False, 
    #**pkwargs
    ):
    
    if verbose:
        print(f'Processing unit {unit}...')
    try:
        spike_times = spkt_by_unit
        num_spikes = len(spike_times)

        fr = compute_firing_rate(spike_times)
        isi_metrics = compute_isi_metrics(spike_times)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'❌ Fatal error in waveform computation for unit {unit}: {e}')
        wf_metrics = 'Error computing waveform metrics'
        return unit, None

    return unit, {
        'num_spikes': num_spikes,
        'fr': fr,
        'isi': isi_metrics,
        'spike_times': spike_times,
    }

def _get_units_and_analyzer(source, network_data, kwargs):
    sorting_analyzer = kwargs.get('sorting_analyzer', None)
    if source == 'experimental':
        if sorting_analyzer is None:
            raise ValueError("No sorting analyzer provided for experimental data.")
        units = kwargs['sorting_object'].get_unit_ids()
        network_data['unit_ids'] = units
    elif source == 'simulated':
        units = [int(i['gid']) for i in kwargs['cellData']]
        network_data['gids'] = units
    else:
        raise ValueError("Unknown data source.")
    return units, sorting_analyzer

# main function to compute spike metrics by unit =====================================
def compute_spike_metrics_by_unit(network_data, run_parallel, verbose=False):

    print("⚡ Computing spiking metrics by unit...")    
    try:        
        # parse list of units
        spkt_by_unit = network_data['inputs']['spkt_by_unit']
        units = list(spkt_by_unit.keys())        

        results = {}
        if run_parallel:
            print(f"🚀 Running in parallel with {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(compute_unit_spike_metrics, spkt_by_unit[unit], verbose=verbose): unit for unit in units}
                for future in as_completed(futures):
                    unit, result = future.result()
                    if result:
                        results[unit] = result
        else:
            print("🔁 Running sequentially...")
            for unit in units:
                unit_id, result = compute_unit_spike_metrics(unit, spkt_by_unit[unit], verbose=verbose)
                if result:
                    results[unit_id] = result
                    
        # count successful computations
        successful_units = len(results)
        print(f"✅ Completed spiking metrics for {successful_units}/{len(units)} units successfully.")

        #
        network_data['spk_metrics']['by_unit'] = results
        print("✅ Spiking metrics by unit computed.")
        return network_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error in compute_spike_metrics_by_unit: {e}")
        return network_data

def _compute_spikecount_stats(network_data, verbose=False):
    """
    Compute the number of spikes per unit and add it to the network data.
    
    Parameters:
    - network_data: dictionary containing network data.
    - verbose: boolean, if True, print detailed information.
    
    Returns:
    - network_data: updated dictionary with spike counts per unit.
    """
    # if verbose:
    #     print("🔍 Computing spikes per unit...")

    spk_metrics = network_data['spk_metrics']['by_unit']
    
    # Count spikes per unit
    #spikes_per_unit = {unit: metrics['num_spikes'] for unit, metrics in spk_metrics.items()}
    spikes_per_unit = [metrics['num_spikes'] for metrics in spk_metrics.values()]
    
    # Add to network data
    network_data['spk_metrics']['count_stats'] = {}
    network_data['spk_metrics']['count_stats']['data'] = spikes_per_unit
    network_data['spk_metrics']['count_stats']['mean'] = np.nanmean(spikes_per_unit)
    network_data['spk_metrics']['count_stats']['std'] = np.nanstd(spikes_per_unit)
    network_data['spk_metrics']['count_stats']['median'] = np.nanmedian(spikes_per_unit)
    network_data['spk_metrics']['count_stats']['cov'] = np.nanstd(spikes_per_unit) / np.nanmean(spikes_per_unit) if np.nanmean(spikes_per_unit) > 0 else np.nan
    network_data['spk_metrics']['count_stats']['max'] = np.nanmax(spikes_per_unit)
    network_data['spk_metrics']['count_stats']['min'] = np.nanmin(spikes_per_unit)
    
    # if verbose:
    #     print(f"✅ Computed spikes per unit: {spikes_per_unit}")
    
    return network_data

def _compute_fr_stats(network_data, verbose=False):
    """
    Compute firing rates for each unit and add it to the network data.
    
    Parameters:
    - network_data: dictionary containing network data.
    - verbose: boolean, if True, print detailed information.
    
    Returns:
    - network_data: updated dictionary with firing rates per unit.
    """
    #if verbose:
        #print("🔍 Computing firing rates per unit...")

    spk_metrics = network_data['spk_metrics']['by_unit']
    
    # Compute firing rates per unit
    firing_rates = [metrics['fr'] for metrics in spk_metrics.values()]
    
    # Add to network data
    network_data['spk_metrics']['fr_stats'] = {}
    network_data['spk_metrics']['fr_stats']['data'] = firing_rates
    network_data['spk_metrics']['fr_stats']['mean'] = np.nanmean(firing_rates)
    network_data['spk_metrics']['fr_stats']['std'] = np.nanstd(firing_rates)
    network_data['spk_metrics']['fr_stats']['median'] = np.nanmedian(firing_rates)
    network_data['spk_metrics']['fr_stats']['cov'] = np.nanstd(firing_rates) / np.nanmean(firing_rates) if np.nanmean(firing_rates) > 0 else np.nan
    network_data['spk_metrics']['fr_stats']['max'] = np.nanmax(firing_rates)
    network_data['spk_metrics']['fr_stats']['min'] = np.nanmin(firing_rates)
    
    # if verbose:
    #     print(f"✅ Computed firing rates per unit: {firing_rates}")
    
    return network_data

def compute_spike_metrics(network_data, run_parallel=True, verbose=False):
    """
    Compute spike metrics for each unit in the dataset.
    
    Parameters:
    - kwargs: dictionary containing necessary parameters and data.
    
    Returns:
    - network_data: dictionary with computed spike metrics added.
    """
    print("⚡ Starting spike metrics computation...")
    
    # Compute spike metrics by unit
    network_data = compute_spike_metrics_by_unit(network_data, run_parallel, verbose=verbose)
    
    # summary spike stats
    ## spike counts
    network_data = _compute_spikecount_stats(network_data, verbose=verbose)
    
    ## firing rates
    network_data = _compute_fr_stats(network_data, verbose=verbose)
    
    
    print("✅ Spike metrics computation completed.")
    return network_data