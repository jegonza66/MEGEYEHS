import functions_general
import functions_analysis
import load
import mne
import numpy as np
import os

import matplotlib.pyplot as plt
import plot_general
import save
import setup
from paths import paths

#----- Path -----#
save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()


#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Select MEG channels -----#
chs_id = 'frontal_L'

#----- Select events -----#
evt_from_df = False
evt_from_annot = True

# MSS
mss = 1
mss_duration = {1: 2, 2: 3.5, 4: 5}
cross1_dur = 0.75
cross2_dur = 1
# Id
epoch_id = f'mss{mss}_cross1_ms_cross2'
# Duration
dur = cross1_dur + mss_duration[mss] + cross2_dur # seconds
# Direction
dir = None
# Screen
screen = epoch_id.split('_')[-1]
# Item
tgt = functions_general.get_item(epoch_id=epoch_id)

# Get time windows from epoch_id name
map_times = dict(cross={'tmin': 0, 'tmax': dur, 'plot_xlim': (dur-1, dur-0.2)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

baseline = (tmin, tmin+cross1_dur)
bline_mode = 'logratio'

# Specific run path for saving data and plots
run_path = f'/{epoch_id}_{tmin}_{tmax}/'

averages = []
for subject_code in exp_info.subjects_ids:

    # Load subject, meg data and pick channels
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    try:
        # Load previous data
        tf_save_path = save_path + f'Time_Frequency/' + run_path + subject.subject_id + '/'
        tf_data_fname = f'Subject_{subject.subject_id}_tfr.h5'
        power = mne.time_frequency.read_tfrs(tf_save_path + tf_data_fname, condition=0)
        picks = functions_general.pick_chs(chs_id=chs_id, info=power.info)
    except:
        # No previous data. Compute
        meg_data = subject.load_preproc_meg()
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)

        # Exclude bad channels
        bads = subject.bad_channels
        meg_data.info['bads'].extend(bads)

        metadata, events, events_id = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                       evt_from_df=evt_from_df, evt_from_annot=evt_from_annot,
                                                                       screen=screen, mss=mss, dur=dur, tgt=tgt, dir=dir,
                                                                       meg_data=meg_data)

        # Reject based on channel amplitude
        reject = dict(mag=subject.config.general.reject_amp)
        reject = dict(mag=2.5e-12)
        reject = dict(mag=1)

        # Epoch data
        epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                            baseline=baseline, event_repeated='drop', metadata=metadata, picks='mag', preload=True)
        # Drop bad epochs
        epochs.drop_bad()

        # Compute power over frequencies
        l_freq = 4
        h_freq = 100
        freqs = np.logspace(*np.log10([l_freq, h_freq]), num=40)
        n_cycles = freqs / 2.  # different number of cycle per frequency
        power = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                              return_itc=False, decim=3, n_jobs=None)

        if save_data:
            # Save evoked data
            tf_save_path = save_path + f'Time_Frequency/' + run_path + subject.subject_id + '/'
            os.makedirs(tf_save_path, exist_ok=True)
            tf_data_fname = f'Subject_{subject.subject_id}_tfr.h5'
            power.save(tf_save_path + tf_data_fname, overwrite=True)

    averages.append(power)

    # Define figure
    fig, axes_topo, ax_tf = plot_general.fig_time_frequency(fontsize=14)

    # Plot time-frequency
    power.plot(picks=picks, baseline=baseline, mode=bline_mode, tmin=tmin, tmax=plot_xlim[1], combine='mean', cmap='jet',
               axes=ax_tf, show=display_figs)
    ax_tf.vlines(x=cross1_dur, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
    ax_tf.vlines(x=cross1_dur + mss_duration[mss], ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')

    # Topomaps parameters
    topomap_kw = dict(ch_type='mag', tmin=plot_xlim[0], tmax=plot_xlim[1], baseline=baseline,
                      mode=bline_mode, show=display_figs)
    plot_dict = dict(Delta=dict(fmin=1, fmax=4), Theta=dict(fmin=4, fmax=8), Alpha=dict(fmin=8, fmax=12),
                     Beta=dict(fmin=12, fmax=30), Gamma=dict(fmin=30, fmax=45), HGamma=dict(fmin=45, fmax=100))

    # Plot topomaps
    for ax, (title, fmin_fmax) in zip(axes_topo, plot_dict.items()):
        power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
        ax.set_title(title)
    fig.suptitle(subject.subject_id + f'_{chs_id}_{bline_mode}')
    fig.tight_layout()

    if save_fig:
        fig_path = plot_path + f'Time_Frequency/' + run_path + f'{subject.subject_id}/'
        fname = 'Time_Freq_' + subject.subject_id + f'_{chs_id}_{bline_mode}'
        save.fig(fig=fig, path=fig_path, fname=fname)

# Grand Average
try:
    # Load previous data
    tf_save_path = save_path + f'Time_Frequency/' + run_path
    tf_data_fname = f'Grand_Average_tfr.h5'
    grand_avg = mne.time_frequency.read_tfrs(tf_save_path + tf_data_fname, condition=0)
except:
    # Compute grand average
    grand_avg = mne.grand_average(averages)

    if save_data:
        # Save evoked data
        tf_save_path = save_path + f'Time_Frequency/' + run_path
        os.makedirs(tf_save_path, exist_ok=True)
        tf_data_fname = f'Grand_average_tfr.h5'
        grand_avg.save(tf_save_path + tf_data_fname, overwrite=True)

# Define figure
fig, axes_topo, ax_tf = plot_general.fig_time_frequency(fontsize=14)

# Plot time-frequency
grand_avg.plot(picks=picks, baseline=baseline, mode=bline_mode, tmin=tmin, tmax=plot_xlim[1], combine='mean', cmap='jet',
               axes=ax_tf, show=display_figs)
ax_tf.vlines(x=cross1_dur, ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')
ax_tf.vlines(x=cross1_dur + mss_duration[mss], ymin=ax_tf.get_ylim()[0], ymax=ax_tf.get_ylim()[1], linestyles='--', colors='black')


# Topomaps parameters
topomap_kw = dict(ch_type='mag', tmin=plot_xlim[0], tmax=plot_xlim[1], baseline=baseline,
                  mode=bline_mode, show=False)
plot_dict = dict(Delta=dict(fmin=1, fmax=4), Theta=dict(fmin=4, fmax=8), Alpha=dict(fmin=8, fmax=12),
                 Beta=dict(fmin=12, fmax=30), Gamma=dict(fmin=30, fmax=45), HGamma=dict(fmin=45, fmax=100))

# Plot topomaps
for ax, (title, fmin_fmax) in zip(axes_topo, plot_dict.items()):
    grand_avg.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
    ax.set_title(title)
fig.suptitle(f'Grand_average_{chs_id}_{bline_mode}')
fig.tight_layout()

if save_fig:
    fig_path = plot_path + f'Time_Frequency/' + run_path
    fname = 'Time_Freq_' + f'Grand_average_{chs_id}_{bline_mode}'
    save.fig(fig=fig, path=fig_path, fname=fname)