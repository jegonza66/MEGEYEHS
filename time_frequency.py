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
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# Select channels
chs_id = 'parietal'
# ICA / RAW
use_ica_data = True
corr_ans = True
tgt_pres = None
# MSS
mss = 4
# Id
# save_id = f'mss{mss}_cross1_ms_cross2_Corr_{corr_ans}_tgt_{tgt_pres}'
save_id = f'sac_ms_Corr_{corr_ans}_tgt_{tgt_pres}'
# epoch_id = 'ms_'
epoch_id = 'sac_ms'
# Power frequency range
l_freq = 1
h_freq = 100
# Baseline method
bline_mode = 'logratio'
#----------#

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 0}
cross1_dur = 0.75
cross2_dur = 1
vs_dur = 4
plot_edge = 0.1

if 'cross1' in epoch_id and mss:
    dur = cross1_dur + mss_duration[mss] + cross2_dur + vs_dur # seconds
elif 'ms' in epoch_id:
    dur = mss_duration[mss] + cross2_dur + vs_dur
elif 'cross2' in epoch_id:
    dur = cross2_dur + vs_dur  # seconds
else:
    dur = 0

# Get time windows from epoch_id name
map_times = dict(cross1={'tmin': 0, 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                 ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
                 cross2={'tmin': 0, 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                 sac={'tmin': -0.2, 'tmax': 0.2, 'plot_xlim': (-0.05, 0.1)},
                 fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

# Baseline duration
if 'sac' in epoch_id:
    baseline = (plot_xlim[0], 0)
elif 'fix' in epoch_id or 'fix' in epoch_id:
    baseline = (tmin, -0.05)
elif 'cross1' in epoch_id or 'ms' in epoch_id and mss:
    baseline = (tmin, tmin+cross1_dur)
elif 'cross2' in epoch_id:
    baseline = (tmin, tmin+cross2_dur)
else:
    baseline = (plot_xlim[0], 0)

# Specific run path for saving data and plots
save_path = f'/{save_id}_{tmin}_{tmax}/'
plot_path = f'/{save_id}_{plot_xlim[0]}_{plot_xlim[1]}/'
if use_ica_data:
    # Save data paths
    trf_save_path = paths().save_path() + f'Time_Frequency_ICA/' + save_path
    os.makedirs(trf_save_path, exist_ok=True)
    epochs_save_path = paths().save_path() + f'Epochs_ICA/Band_None/' + save_path
    os.makedirs(epochs_save_path, exist_ok=True)
    # Save figures paths
    trf_fig_path = paths().plots_path() + f'Time_Frequency_ICA/' + plot_path + f'{chs_id}/'
    os.makedirs(trf_fig_path, exist_ok=True)
else:
    # Save data paths
    trf_save_path = paths().save_path() + f'Time_Frequency_RAW/' + save_path
    os.makedirs(trf_save_path, exist_ok=True)
    epochs_save_path = paths().save_path() + f'Epochs_RAW/Band_None/' + save_path
    os.makedirs(epochs_save_path, exist_ok=True)
    # Save figures paths
    trf_fig_path = paths().plots_path() + f'Time_Frequency_RAW/' + plot_path + f'{chs_id}/'
    os.makedirs(trf_fig_path, exist_ok=True)

# Grand average data variable
grand_avg_power_fname = f'Grand_Average_power_{l_freq}_{h_freq}_tfr.h5'
grand_avg_itc_fname = f'Grand_Average_itc_{l_freq}_{h_freq}_tfr.h5'
averages_power = []
averages_itc = []

for subject_code in exp_info.subjects_ids:

    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if use_ica_data:
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        # Load meg data
        meg_data = load.ica_data(subject=subject)

    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        # Load meg data
        meg_data = subject.load_preproc_meg()

    # Data filenames
    power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
    itc_data_fname = f'ITC_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

    try:
        # Load previous data
        power = mne.time_frequency.read_tfrs(trf_save_path + power_data_fname, condition=0)
        itc = mne.time_frequency.read_tfrs(trf_save_path + itc_data_fname, condition=0)
    except:
        try:
            # Load epoched data
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        except:
            # Define events
            # Trials
            cond_trials, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=mss,
                                                                              corr_ans=corr_ans, tgt_pres=tgt_pres)
            metadata, events, events_id, metadata_sup = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                                         trials=cond_trials, meg_data=meg_data)
            # Reject based on channel amplitude
            if 'sac' in epoch_id or 'fix' in epoch_id:
                reject = dict(mag=subject.config.general.reject_amp)
                # reject = dict(mag=2.5e-12)
            else:
                reject = None

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

        # Compute power over frequencies
        # freqs = np.logspace(*np.log10([l_freq, h_freq]), num=40)
        freqs = np.linspace(l_freq, h_freq, num=h_freq)
        n_cycles = freqs / 4.  # different number of cycle per frequency
        power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                   return_itc=True, decim=3, n_jobs=None)

        if save_data:
            # Save trf data
            power.save(trf_save_path + power_data_fname, overwrite=True)
            itc.save(trf_save_path + itc_data_fname, overwrite=True)

    # Append data for GA
    averages_power.append(power)
    averages_itc.append(itc)

    # Plot power time-frequency
    fname = f'{subject.subject_id}/Power_tf_' + subject.subject_id + f'_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.trf(subject=subject, trf=power, chs_id=chs_id, baseline=baseline, bline_mode=bline_mode,
                           plot_xlim=plot_xlim, epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur,
                           mss_duration=mss_duration, cross2_dur=cross2_dur, display_figs=display_figs,
                           save_fig=save_fig, fig_path=trf_fig_path, fname=fname)

    # Plot ITC time-frequency
    fname = f'{subject.subject_id}/ITC_tf_' + subject.subject_id + f'_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.trf(subject=subject, trf=itc, chs_id=chs_id, baseline=baseline, bline_mode=bline_mode,
                           plot_xlim=plot_xlim, epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur,
                           mss_duration=mss_duration, cross2_dur=cross2_dur, display_figs=display_figs,
                           save_fig=save_fig, fig_path=trf_fig_path, fname=fname)

    # Power topoplot
    fig = power.plot_topo(baseline=baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
                    cmap='jet', show=display_figs)
    if save_fig:
        fname = f'{subject.subject_id}/Power_topoch_' + subject.subject_id + f'_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)

    # ITC topoplot
    fig = itc.plot_topo(tmin=plot_xlim[0], tmax=plot_xlim[1], cmap='jet', show=display_figs, title='Inter-Trial coherence')
    if save_fig:
        fname = f'{subject.subject_id}/ITC_topoch_' + subject.subject_id + f'_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)

# Grand Average
try:
    # Load previous power data
    grand_avg_power = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_power_fname)[0]
    # Load previous itc data
    grand_avg_itc = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_itc_fname)[0]
    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power.info)
except:
    # Compute grand average
    grand_avg_power = mne.grand_average(averages_power)
    grand_avg_itc = mne.grand_average(averages_itc)

    if save_data:
        # Save trf data
        grand_avg_power.save(trf_save_path + grand_avg_power_fname, overwrite=True)
        grand_avg_itc.save(trf_save_path + grand_avg_itc_fname, overwrite=True)


# Plot ITC time-frequency
fname = f'GA_Power_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.trf(trf=grand_avg_power, chs_id=chs_id, baseline=baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                 epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                 subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname)

# Plot ITC time-frequency
fname = f'GA_ITC_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.trf(trf=grand_avg_itc, chs_id=chs_id, baseline=baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                 epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                 subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname)

# Power topoplot
fig = power.plot_topo(baseline=baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
                      cmap='jet', show=display_figs)
if save_fig:
    fname = f'GA_Power_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    save.fig(fig=fig, path=trf_fig_path, fname=fname)

# ITC topoplot
fig = itc.plot_topo(tmin=plot_xlim[0], tmax=plot_xlim[1], cmap='jet', show=display_figs, title='Inter-Trial coherence')
if save_fig:
    fname = f'GA_ITC_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    save.fig(fig=fig, path=trf_fig_path, fname=fname)