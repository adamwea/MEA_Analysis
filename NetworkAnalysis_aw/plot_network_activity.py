import os
import json
import glob
import numpy as np
from multiprocessing import Pool
#from MEA_Analysis.NetworkAnalysis_aw.plot_network_activity import plot_network_summary_v3
import traceback
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Sub-Functions ==============================================

def _load_metrics_file(fpath):
    metrics = np.load(fpath, allow_pickle=True).item()
    print(f'Loaded {fpath}')
    return metrics

def _get_metrics_files(batch_paths):
    metrics_files = []
    for batch_path in batch_paths:
        metrics_files += glob.glob(f'{batch_path}/**/*_metrics.npy', recursive=True)
    return metrics_files

def _get_data_files(target_dirs, query='network_data.npy'):
    data_files = []
    for target_dir in target_dirs:
        data_files += glob.glob(f'{target_dir}/**/{query}', recursive=True)
    return data_files

def _sort_by_fitness(metrics_files, top_n):
    fitness_values = []
    valid_files = []

    for fpath in metrics_files:
        json_path = fpath.replace('_metrics.npy', '_fitness.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                fit_value = data.get('fit', 1000)
                fitness_values.append(fit_value)
                valid_files.append(fpath)

    sorted_idx = np.argsort(fitness_values)
    top_files = np.array(valid_files)[sorted_idx][:top_n]
    top_fitness = np.array(fitness_values)[sorted_idx][:top_n]

    return top_files.tolist(), top_fitness.tolist()

def _extract_network_data(network_data):
    
    t = None
    try:
        t = network_data['inputs']['t']
    except Exception:
        raise ValueError("time vector not found in network_data['inputs']")
    
    b_ax = None
    try:
        b_ax = network_data['burst_metrics'].get('ax', None)
    except Exception:
        raise ValueError("burst_metrics not found in network_data")
    
    # get hyperbursting data
    hb_ax = None
    try:
        hb_ax = network_data['hyperburst_metrics'].get('ax', None)
    except Exception:
        raise ValueError("hyperburst_metrics not found in network_data")
    
    # get spiking data by unit
    spk_data = None
    try:
        spk_metrics = network_data['spk_metrics'].get('by_unit', None)
    except Exception:
        raise ValueError("spk_metrics not found in network_data")

    # get unit populations
    unit_pops = None
    try:
        if 'unit_pops' in network_data['inputs']:
            unit_pops = network_data['inputs']['unit_pops']
        elif 'unit_pops' in network_data['class_output']:
            unit_pops = network_data['class_output']['unit_pops']
        else:
            print("[WARNING] No unit_pops found in network_data, 'inputs' or 'class_output'.")
            raise ValueError("unit_pops not found in network_data")
    except Exception:
        #raise ValueError("unit_types not found in network_data")
        print("[WARNING] No unit_pops found in network_data. Proceeding without unit pops.")
        unit_pops = None

    return b_ax, hb_ax, spk_metrics, unit_pops, t

def _auto_limit(axes, getter, setter, direction='x'):
    """
    Automatically set the limits of the axes based on the minimum and maximum values
    of the data in the axes.
    Parameters:
        axes: list of axes to set limits for
        getter: function to get the current limits of the axes
        setter: function to set the limits of the axes
        direction: 'x' or 'y' to specify which axis to limit
    """
    values_min, values_max = [], []
    for ax in axes:
        # always skip ax[0] for the raster plot - rasterplot will have unique y axis from network bursting
        if ax == axes[0] and direction == 'y':
            continue            
        try:
            mn, mx = getter(ax)
        except Exception:
            mn, mx = np.nan, np.nan
        values_min.append(mn)
        values_max.append(mx)
    overall_min = np.nanmin(values_min)
    overall_max = np.nanmax(values_max)
    for ax in axes:
        # skip ax[0] for the raster plot - rasterplot will have unique y axis from network bursting
        if ax == axes[0] and direction == 'y':
            continue            
        try:
            setter(ax, overall_min, overall_max)
        except Exception:
            pass

# Main Plotting Functions ====================================
def plot_convolved(ax, bursting_ax, x_lim=None, y_lim=None, label=None):
    """Plot network bursting activity with clear differentiation between lines."""
    
    # Copy ax features to the new ax
    ax.set_xlim(bursting_ax.get_xlim())
    ax.set_ylim(bursting_ax.get_ylim())
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(bursting_ax.get_title())
    
    # Plot Line 1 (Blue)
    ax.plot(
        bursting_ax.get_lines()[0].get_xdata(),
        bursting_ax.get_lines()[0].get_ydata(),
        color='blue',
        #label=label if label else 'Bursts'  # Default label if none provided
    )
    
    # plot bursting peaks
    # ax.plot(
    #     bursting_ax.get_lines()[1].get_xdata(),
    #     bursting_ax.get_lines()[1].get_ydata(),
    #     'o', 
    #     color='orange',
    #     label=label if label else 'Burst Peaks'
    # )
    
    # plot vertical lines for burst peaks
    # Mark hyper peaks with vertical dotted lines
    label = label if label else 'Burst Peaks'
    for i, x_peak in enumerate(bursting_ax.get_lines()[1].get_xdata()):
        ax.axvline(x=x_peak, linestyle='dotted', color='red',
                label=label if i == 0 else None)

    # get max y-axis value
    max_y = np.nanmax(bursting_ax.get_lines()[0].get_ydata())
    if y_lim is not None:
        y_max_too_high = y_lim[1] < max_y
    
    # x and y limits
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None and not y_max_too_high:
        ax.set_ylim(y_lim)
    else:
        # add line at set y_max prior to auto-scaling
        target_max_y = y_lim[1]
        ax.axhline(y=target_max_y, color='black', linestyle='--', linewidth=1, label=f'Target Max: {target_max_y:.2f}')
        
        y_lim = (0, max_y * 1.1)  # auto-scale y-axis to 10% above max
        ax.set_ylim(y_lim)
        
    # Add a legend for clarity
    ax.legend()
    
    return ax

def plot_convolved_superimposed(ax, bursting_ax, hyperbursting_ax, x_lim=None, y_lim=None):
    """Plot network bursting activity with clear differentiation between lines."""
    
    # Copy ax features to the new ax
    ax.set_xlim(bursting_ax.get_xlim())
    ax.set_ylim(bursting_ax.get_ylim())
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(bursting_ax.get_title())
    
    ## Regular Bursting
    # Plot Line 1 (Blue)
    ax.plot(
        bursting_ax.get_lines()[0].get_xdata(),
        bursting_ax.get_lines()[0].get_ydata(),
        color='blue',
        #label='Mega Bursts'
    )
    
    # plot bursting peaks
    # ax.plot(
    #     bursting_ax.get_lines()[1].get_xdata(),
    #     bursting_ax.get_lines()[1].get_ydata(),
    #     'o', 
    #     #color='orange',
    #     color='red',
    #     #label='Hyper Bursts'
    #     label='Bursts'
    # )
    
    ## Hyper Bursting
    ax.fill_between(
        hyperbursting_ax.get_lines()[0].get_xdata(),
        hyperbursting_ax.get_lines()[0].get_ydata(),
        color='red',
        alpha=0.2,  # Transparency for the filled area
        #label='Hyper Burst Peaks'  # Label for the filled area
    )

    #outline the filled area, dotted line
    ax.plot(
        hyperbursting_ax.get_lines()[0].get_xdata(),
        hyperbursting_ax.get_lines()[0].get_ydata(),
        color='red',
        linestyle='dotted',
        #label='Hyper Burst Peaks'  # Label for the filled area
    )

    # Mark hyper peaks with vertical dotted lines
    for i, x_peak in enumerate(hyperbursting_ax.get_lines()[1].get_xdata()):
        ax.axvline(x=x_peak, linestyle='dotted', color='red',
                label='Hyper Burst Peaks' if i == 0 else None)

    # get max y-axis value
    max_y = np.nanmax(bursting_ax.get_lines()[0].get_ydata())
    if y_lim is not None:
        y_max_too_high = y_lim[1] < max_y
    
    # x and y limits
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None and not y_max_too_high:
        ax.set_ylim(y_lim)
    else:
        # add line at set y_max prior to auto-scaling
        target_max_y = y_lim[1]
        ax.axhline(y=target_max_y, color='black', linestyle='--', linewidth=1, label=f'Target Max: {target_max_y:.2f}')
        
        y_lim = (0, max_y * 1.1)  # auto-scale y-axis to 10% above max
        ax.set_ylim(y_lim)
        
    # Add a legend for clarity
    ax.legend()
    
    return ax

def plot_raster(ax, spiking_data_by_unit, unit_types=None, x_lim=None):
    """Plot a raster plot for spiking data."""
    
    # Calculate the average firing rate for each unit
    firing_rates = {}
    for gid in spiking_data_by_unit:
        spike_times = spiking_data_by_unit[gid]['spike_times']
        spike_times = [spike_times] if isinstance(spike_times, (int, float)) else spike_times
        if spike_times is None: firing_rate = 0
        else: firing_rate = len(spike_times) / (max(spike_times) - min(spike_times)) if len(spike_times) > 1 else 0
        firing_rates[gid] = firing_rate
        
    # Sort the units based on their average firing rates
    sorted_units = sorted(firing_rates, key=firing_rates.get)
    
    # Create a mapping from original gid to new y-axis position
    gid_to_ypos = {gid: pos for pos, gid in enumerate(sorted_units)}
    
    # Plot the units in the sorted order
    if unit_types is None:
        for gid in sorted_units:
            spike_times = spiking_data_by_unit[gid]['spike_times']
            spike_times = [spike_times] if isinstance(spike_times, (int, float)) else spike_times
            if spike_times is not None:
                ax.plot(spike_times, [gid_to_ypos[gid]] * len(spike_times), 'b.', markersize=2)
    else:
        # Define legend markers for excitatory (E) and inhibitory (I) units
        exc_marker, = ax.plot([], [], 'b.', markersize=2, label='Excitatory (E)')
        inh_marker, = ax.plot([], [], 'r.', markersize=2, label='Inhibitory (I)')

        for gid in sorted_units:
            spike_times = spiking_data_by_unit[gid]['spike_times']
            spike_times = [spike_times] if isinstance(spike_times, (int, float)) else spike_times
            if spike_times is not None:
                try:
                    if unit_types[gid] == 'E':
                        ax.plot(spike_times, [gid_to_ypos[gid]] * len(spike_times), 'b.', markersize=2)
                    elif unit_types[gid] == 'I':
                        ax.plot(spike_times, [gid_to_ypos[gid]] * len(spike_times), 'r.', markersize=2)
                except:
                    print(f"Warning: Unit {gid} not found in unit_types. Skipping.")

        # Add legend with proxy markers
        ax.legend(handles=[exc_marker, inh_marker])

    # Set x-axis limits if provided
    if x_lim is not None:
        ax.set_xlim(x_lim)
        
    # Set y-axis limits based on the number of units
    ax.set_ylim(-0.5, len(sorted_units) + 0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Unit ID (sorted by firing rate)')
    ax.set_title('Raster Plot')
    plt.tight_layout()
    return ax

def plot_network_activity(network_data, x_lim=None, y_lim=None, **kwargs):
    """
    Plot a 2-panel or 3-panel summary of network data,
    auto-scaling x and y limits (nan-safe) unless overridden.

    Parameters:
      network_data: dict with keys 'sim_data_path', 'bursting_data', 'mega_bursting_data', 'spiking_data', 'unit_types'
      x_lim: tuple (xmin, xmax) to override x-axis limits for all axes
      y_lim: tuple (ymin, ymax) to override y-axis limits for all axes
      limit_seconds: within kwargs, time to zoom x-axis to [0, limit_seconds]
      network_summary_mode: within kwargs, '2p' or '3p'
      output_dir: within kwargs, directory to save figures
    """
    # initialize parameters
    mode = kwargs.get('num_panels', 2)  # 2 or 3 panels

    # safely extract axes/data
    b_ax, hb_ax, spk_data, unit_pops, t = _extract_network_data(network_data)
    max_time = np.nanmax(t) if t is not None else 0
    if max_time == 0:
        raise ValueError("No valid time data found in network_data['inputs']['t']")
    
    # parse limits
    x_lim_max = x_lim[1] if x_lim is not None else max_time
    x_lim_min = x_lim[0] if x_lim is not None else 0
    y_lim_max = y_lim[1] if y_lim is not None else None
    y_lim_min = y_lim[0] if y_lim is not None else 0
    if x_lim_max is None: x_lim_max = max_time
    assert x_lim_max <= max_time, f"x_lim_max {x_lim_max} must be less than max_time {max_time}"
    x_lim = (x_lim_min, x_lim_max)
    y_lim = (y_lim_min, y_lim_max)

    # define output paths
    label = kwargs.get('label', None)
    x_lim_min_rnd = int(np.floor(x_lim_min))
    x_lim_max_rnd = int(np.ceil(x_lim_max))
    combined_label = f"{label}_conv{x_lim_min_rnd}-{x_lim_max_rnd}" if label else f"conv{x_lim_min_rnd}-{x_lim_max_rnd}"
    output_path = os.path.join(kwargs.get('output_dir', os.getcwd()), combined_label)
    output_types = kwargs.get('output_types', ['png'])
    paths = {}
    if 'pdf' in output_types:
        paths['2p_pdf'] = f'{output_path}_2p.pdf'
        paths['3p_pdf'] = f'{output_path}_3p.pdf'
    if 'png' in output_types:
        paths['2p_png'] = f'{output_path}_2p.png'
        paths['3p_png'] = f'{output_path}_3p.png'
    if 'svg' in output_types:
        paths['2p_svg'] = f'{output_path}_2p.svg'
        paths['3p_svg'] = f'{output_path}_3p.svg'

    if mode == 2: # 2-panel mode
        fig, axs = plt.subplots(2, 1, figsize=(16, 9))
        
        print("Generating raster plot...")
        axs[0] = plot_raster(axs[0], spk_data, unit_types=unit_pops, x_lim=x_lim)

        print("Generating network bursting plot...")        
        axs[1] = plot_convolved_superimposed(axs[1], b_ax, hb_ax, x_lim=x_lim, y_lim=y_lim)
        
        # tighten layout
        plt.tight_layout()
                
        #debug
        #fig.savefig(f'conv_debug.png', dpi=100)

        for path in paths.values():
            if '2p' in path:
                if path.endswith('.pdf'):
                    fig.savefig(path)
                    print(f"Saved: {path}")
                elif path.endswith('.png'):
                    fig.savefig(path, dpi=100)
                    print(f"Saved: {path}")
                elif path.endswith('.svg'):
                    fig.savefig(path, dpi=300)
                    print(f"Saved: {path}")
                
        print(f"Generated {mode}-panel activity plot for {combined_label}.")

    elif mode == 3:
        fig, axs = plt.subplots(3, 1, figsize=(16, 9))

        print("Generating raster plot...")
        #axs[0] = plot_raster(axs[0], spiking_data, unit_types)
        axs[0] = plot_raster(axs[0], spk_data, unit_types=unit_pops, x_lim=x_lim)

        print("Generating network bursting plot...")
        #axs[1] = plot_network_bursting_v2(axs[1], bursting_ax)
        axs[1] = plot_convolved(axs[1], b_ax, x_lim=x_lim, y_lim=y_lim, label='Burst Peaks')

        print("Generating hyper bursting plot...")
        #axs[2] = plot_network_bursting_v2(axs[2], mega_ax, mode='mega')
        axs[2] = plot_convolved(axs[2], hb_ax, x_lim=x_lim, y_lim=y_lim, label='Hyper Burst Peaks')
        
        # tighten layout
        plt.tight_layout()
        
        #debug
        #fig.savefig(f'conv_debug.png', dpi=100)
        
        for path in paths.values():
            if '3p' in path:
                if path.endswith('.pdf'):
                    fig.savefig(path)
                    print(f"Saved: {path}")
                elif path.endswith('.png'):
                    fig.savefig(path, dpi=100)
                    print(f"Saved: {path}")
                elif path.endswith('.svg'):
                    fig.savefig(path, dpi=300)
                    print(f"Saved: {path}")
                
        print(f"Generated {mode}-panel activity plot for {combined_label}.")

def generate_activity_plots(network_data, kwargs):
    """
    Generate standard set of activity plots for a given network data.
    """
    
    # network data can be passed as a path to .npy dict, or as the dict itself
    if isinstance(network_data, str):
        network_data_path = network_data
        try:
            # load the network data from the file
            print(f"Loading network data from {network_data}...")
            with open(network_data, 'rb') as f:
                network_data = np.load(f, allow_pickle=True).item()
            
            # since we loaded from a file, check if output_dir is set
            if 'output_dir' not in kwargs:
                print(f"[WARNING] No output_dir provided in kwargs, will set output_dir to the directory of {network_data_path}")
                output_dir = os.path.dirname(network_data_path)
                kwargs['output_dir'] = output_dir
            
            if 'label' not in kwargs:
                print(
                    f"[WARNING] No label provided in kwargs, will set label to the basename of {network_data_path}. "
                    "Note, if there are multiple network_data files in the same directory, lead to non-unique output filepaths.")
                kwargs['label'] = os.path.basename(os.path.dirname(network_data_path))
                
        except Exception as e:
            print(f"[ERROR] Failed to load network data from {network_data_path}: {e}")
            return
    elif isinstance(network_data, dict):
        # if a dict is passed, ensure output_dir is set
        if 'output_dir' not in kwargs:
            raise ValueError("output_dir must be provided in kwargs when network_data is passed as a dict.")
    elif not isinstance(network_data, dict):
        raise ValueError("network_data must be a path to a .npy file or a dict containing network data.")
    
    
    try:  

        # run both 2p and 3p modes
        for mode in [2, 3]:
            kwargs['num_panels'] = mode
            plot_network_activity(network_data, **kwargs)
            print(f"Generated activity plots for {network_data_path} in {mode}-panel mode.")

    except Exception as e:
        #print(f"[ERROR] Network {sim_data_path}: {e}")
        traceback.print_exc()
        print(f"[ERROR] Failed to generate activity plots for {network_data_path}: {e}")

def batch_plot(target_dirs, parallel=False, num_workers=4, **kwargs):
    """
    Plot network summaries for each item in network_data_list.
    
    Args:
      network_data_list: list of dicts, each with 'sim_data_path', etc.
      kwargs: base kwargs to forward into plot_network_summary_v3
      parallel: if True, use a ProcessPoolExecutor
      num_workers: number of processes to spawn when parallel=True
    """

    # get all network data files
    data_files = _get_data_files(target_dirs)

    # plot network activity for each network data
    try:
        if parallel:
            # spin up processes, map helper 
            with ProcessPoolExecutor(max_workers=num_workers) as exe:
                # map takes care of ordering; exceptions are printed in helper
                exe.map(generate_activity_plots,
                        data_files,
                        [kwargs]*len(data_files))
        else:
            # simple serial loop
            for data_file in data_files:
                generate_activity_plots(data_file, kwargs)
    except:
        print("[ERROR] Exception in plot_network_metrics_v2")
        traceback.print_exc()