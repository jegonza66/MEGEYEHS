import functions_general
import functions_analysis
import load
import mne
import numpy as np
import os
import plot_general
import save
import setup
from paths import paths
import matplotlib.pyplot as plt

#----- Path -----#
save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = False
save_fig = False
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# Select channels
chs_id = 'frontal'
# ICA / RAW
use_ica_data = True
# MSS
mss = None
# Id
# save_id = f'mss{mss}_cross1_ms_cross2'
save_id = 'l_sac'
# epoch_id = 'ms_'
epoch_id = 'l_sac'
# Power frequency range
l_freq = 1
h_freq = 20
# Baseline method
bline_mode = 'logratio'
#----------#

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 0}
cross1_dur = 0.75
cross2_dur = 1
if 'cross1' in epoch_id and mss:
    dur = cross1_dur + mss_duration[mss] + cross2_dur  # seconds
elif 'cross2' in epoch_id:
    dur = cross2_dur + 5  # seconds
else:
    dur = None

# Direction
dir = None
# Screen
screen = None
# Item
tgt = functions_general.get_item(epoch_id=epoch_id)

# Get time windows from epoch_id name
map_times = dict(cross1={'tmin': 0, 'tmax': dur, 'plot_xlim': (0, dur)},
                 ms={'tmin': -cross1_dur, 'tmax': mss_duration[mss] + cross2_dur, 'plot_xlim': (-cross1_dur, mss_duration[mss] + cross2_dur)},
                 cross2={'tmin': 0, 'tmax': dur, 'plot_xlim': (0, dur)},
                 sac={'tmin': -0.2, 'tmax': 0.2, 'plot_xlim': (-0.05, 0.1)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

# Baseline duration
if 'cross1' in epoch_id or 'ms' in epoch_id and mss:
    baseline = (tmin, tmin+cross1_dur)
elif 'cross2' in epoch_id:
    baseline = (tmin, tmin+cross2_dur)
else:
    baseline = (None, 0)

# Specific run path for saving data and plots
run_path = f'/{save_id}_{tmin}_{tmax}/'

# Grand average data variable
averages = []

for subject_code in exp_info.subjects_ids:

    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if use_ica_data:
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        # Load meg data
        meg_data = load.ica_data(subject=subject)

        # Save data paths
        trf_save_path = save_path + f'Time_Frequency_ICA/' + run_path
        os.makedirs(trf_save_path, exist_ok=True)
        epochs_save_path = save_path + f'Epochs_ICA/' + run_path
        os.makedirs(epochs_save_path, exist_ok=True)
        # Save figures paths
        trf_fig_path = plot_path + f'Time_Frequency_ICA/' + run_path + f'{chs_id}/'
        os.makedirs(trf_fig_path, exist_ok=True)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        # Load meg data
        meg_data = subject.load_preproc_meg()

        # Save data paths
        trf_save_path = save_path + f'Time_Frequency_RAW/' + run_path
        os.makedirs(trf_save_path, exist_ok=True)
        epochs_save_path = save_path + f'Epochs_RAW/' + run_path
        os.makedirs(epochs_save_path, exist_ok=True)
        # Save figures paths
        trf_fig_path = plot_path + f'Time_Frequency_RAW/' + run_path + f'{chs_id}/'
        os.makedirs(trf_fig_path, exist_ok=True)

    # Data filenames
    trf_data_fname = f'Subject_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    grand_avg_data_fname = f'Grand_Average_{l_freq}_{h_freq}_tfr.h5'

    try:
        # Load previous data
        power = mne.time_frequency.read_tfrs(trf_save_path + trf_data_fname, condition=0)
        picks = functions_general.pick_chs(chs_id=chs_id, info=power.info)
    except:

        try:
            # Load epoched data
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
            # Pick MEG channels to plot
            picks = functions_general.pick_chs(chs_id=chs_id, info=epochs.info)
        except:

            # Define events
            metadata, events, events_id, metadata_sup = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                                         screen=screen, mss=mss, meg_data=meg_data,
                                                                                         dur=None, tgt=None, dir=None)
            # Reject based on channel amplitude
            reject = dict(mag=subject.config.general.reject_amp)
            # reject = dict(mag=2.5e-12)
            # reject = dict(mag=1)

            # Epoch data
            epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                                baseline=baseline, event_repeated='drop', metadata=metadata, picks='mag', preload=True)
            # Drop bad epochs
            epochs.drop_bad()

            if metadata_sup is not None:
                metadata_sup = metadata_sup.loc[(metadata_sup['id'].isin(epochs.metadata['event_name']))].reset_index(
                    drop=True)
                epochs.metadata = metadata_sup

            if save_data:
                # Save epoched data
                epochs.reset_drop_log_selection()
                os.makedirs(epochs_save_path, exist_ok=True)
                epochs.save(epochs_save_path + epochs_data_fname, overwrite=True)

        # No previous data. Compute
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)

        # Compute power over frequencies
        freqs = np.logspace(*np.log10([l_freq, h_freq]), num=40)
        n_cycles = freqs / 4.  # different number of cycle per frequency
        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                              return_itc=False, decim=3, n_jobs=None)

        if save_data:
            # Save trf data
            power.save(trf_save_path + trf_data_fname, overwrite=True)

    # Append data for GA
    averages.append(power)

    # Define figure
    fig, axes_topo, ax_tf = plot_general.fig_time_frequency(fontsize=14)

    # Plot time-frequency
    power.plot(picks=picks, baseline=None, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
               combine='mean', cmap='jet', axes=ax_tf, show=display_figs)

    # Plot time markers as vertical lines
    if 'cross1' in epoch_id and mss:
        ax_tf.vlines(x=cross1_dur, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
        ax_tf.vlines(x=cross1_dur + mss_duration[mss], ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1],
                     linestyles='--', colors='black')
    elif 'ms' in epoch_id and mss:
        ax_tf.vlines(x=0, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
        ax_tf.vlines(x=mss_duration[mss], ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1],
                     linestyles='--', colors='black')
    elif 'cross2' in epoch_id:
        ax_tf.vlines(x=cross2_dur, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
    else:
        ax_tf.vlines(x=0, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')

    # Topomaps parameters
    topomap_kw = dict(ch_type='mag', tmin=plot_xlim[0], tmax=plot_xlim[1], baseline=baseline,
                      mode=bline_mode, show=display_figs)
    plot_dict = dict(Delta=dict(fmin=1, fmax=4), Theta=dict(fmin=4, fmax=8), Alpha=dict(fmin=8, fmax=12),
                     Beta=dict(fmin=12, fmax=30), Gamma=dict(fmin=30, fmax=45), HGamma=dict(fmin=45, fmax=100))

    # Plot topomaps
    for ax, (title, fmin_fmax) in zip(axes_topo, plot_dict.items()):
        try:
            power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
        except:
            ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
            ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title)
    fig.suptitle(subject.subject_id + f'_{chs_id}_{bline_mode}')
    fig.tight_layout()

    if save_fig:
        fname = 'Time_Freq_' + subject.subject_id + f'_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)

# Grand Average
try:
    # Load previous data
    grand_avg = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_data_fname, condition=0)
except:
    # Compute grand average
    grand_avg = mne.grand_average(averages)

    if save_data:
        # Save trf data
        grand_avg.save(trf_save_path + grand_avg_data_fname, overwrite=True)

# Define figure
fig, axes_topo, ax_tf = plot_general.fig_time_frequency(fontsize=14)

# Plot time-frequency
grand_avg.plot(picks=picks, baseline=baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1], combine='mean', cmap='jet',
               axes=ax_tf, show=display_figs)
# Plot time markers as vertical lines
if 'cross1' in epoch_id and mss:
    ax_tf.vlines(x=cross1_dur, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
    ax_tf.vlines(x=cross1_dur + mss_duration[mss], ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
elif 'ms' in epoch_id and mss:
    ax_tf.vlines(x=0, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
    ax_tf.vlines(x=mss_duration[mss], ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1],
                 linestyles='--', colors='black')
elif 'cross2' in epoch_id:
    ax_tf.vlines(x=cross2_dur, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
else:
    ax_tf.vlines(x=0, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')

# Topomaps parameters
topomap_kw = dict(ch_type='mag', tmin=plot_xlim[0], tmax=plot_xlim[1], baseline=baseline,
                  mode=bline_mode, show=display_figs)
plot_dict = dict(Delta=dict(fmin=1, fmax=4), Theta=dict(fmin=4, fmax=8), Alpha=dict(fmin=8, fmax=12),
                 Beta=dict(fmin=12, fmax=30), Gamma=dict(fmin=30, fmax=45), HGamma=dict(fmin=45, fmax=100))

# Plot topomaps
for ax, (title, fmin_fmax) in zip(axes_topo, plot_dict.items()):
    try:
        grand_avg.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
    except:
        ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
        ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title)
fig.suptitle(f'Grand_average_{chs_id}_{bline_mode}')
fig.tight_layout()

if save_fig:
    fname = 'Time_Freq_' + f'Grand_average_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    save.fig(fig=fig, path=trf_fig_path, fname=fname)