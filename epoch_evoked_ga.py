import pandas as pd
import os
import load
import mne
import save
import matplotlib.pyplot as plt
import setup
from paths import paths
import numpy as np

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

# Save data and display figures
save_data = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

# Pick MEG chs (Select channels or set picks = 'mag')
pick_chs = 'mag'
if pick_chs == 'mag':
    picks = 'mag'
elif pick_chs == 'LR':
    right_chs = ['MRT51', 'MRT52', 'MRT53']
    left_chs = ['MLT51', 'MLT52', 'MLT53']
    picks = right_chs + left_chs

# Filter evoked
filter_evoked = False
l_freq = 0.5
h_freq = 100

# Select events
epoch_ids = ['l_sac']

if any('fix' in id for id in epoch_ids):
    tmin = -0.1
    tmax = 0.4
    plot_xlim = (tmin, tmax)
elif any('sac' in id for id in epoch_ids):
    tmin = -0.1
    tmax = 0.1
    plot_xlim = (-0.05, 0.1)

evokeds = []
for subject_code in exp_info.subjects_ids:

    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg()

    # Exclude bad channels
    bads = subject.bad_channels
    meg_data.info['bads'].extend(bads)

    # Get events from annotations
    all_events, all_event_id = mne.events_from_annotations(meg_data, verbose=False)
    # Select epochs
    epoch_keys = [key for epoch_id in epoch_ids for key in all_event_id.keys() if epoch_id in key]

    # Get events and ids matchig selection
    metadata, events, event_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id,
                                                          row_events=epoch_keys, tmin=0, tmax=0,
                                                          sfreq=meg_data.info['sfreq'])

    # Reject based on channel amplitude
    reject = dict(mag=subject.config.general.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=event_id, tmin=tmin, tmax=tmax, reject=reject,
                        event_repeated='drop', metadata=metadata, preload=True)
    # Drop bad epochs
    epochs.drop_bad()

    if any('sac' in id for id in epoch_ids):
        saccades = subject.saccades
        metadata = saccades.loc[(~pd.isna(saccades['screen'])) &
                                (saccades['id'].isin(epochs.metadata['event_name']))].reset_index(drop=True)
        epochs.metadata = metadata
    elif any('fix' in id for id in epoch_ids):
        fixations = subject.fixations
        metadata = fixations.loc[(~pd.isna(fixations['screen'])) &
                                 (fixations['id'].isin(epochs.metadata['event_name']))].reset_index(drop=True)
        epochs.metadata = metadata

    # Define evoked and append for GA
    evoked = epochs.average(picks=['mag', 'misc'])
    evokeds.append(evoked)

    # Parameters for plotting
    overlay = epochs.metadata['duration'] # Overlay duration
    order = overlay.argsort()  # Sorting from longer to shorter
    group_by = {}

    # Plot epochs
    fig_ep = epochs.plot_image(picks=picks, order=order, sigma=5, cmap='jet', overlay_times=overlay, combine='std',
                               title=subject.subject_id, show=display_figs)
    fig_path = plot_path + f'Epochs/{"-".join(epoch_ids)}/'

    # Save figure
    if len(fig_ep) == 1:
        fig = fig_ep[0]
        fname = 'Epochs_' + subject.subject_id + f'_{pick_chs}.png'
        save.fig(fig=fig, path=fig_path, fname=fname)
    else:
        for i in range(len(fig_ep)):
            fig = fig_ep[i]
            group = group_by.keys()[i]
            fname = f'Epochs_{group}' + subject.subject_id + f'_{pick_chs}.png'
            save.fig(fig=fig, path=fig_path, fname=fname)

    # Plot evoked
    # Separete MEG and misc channels
    evoked_meg = evoked.copy().pick('mag')
    evoked_misc = evoked.copy().pick('misc')

    # Filter MEG evoked
    if filter_evoked:
        evoked_meg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

    # Get Gaze x ch
    gaze_x_ch_idx = np.where(np.array(evoked_misc.ch_names) == 'ET_gaze_x')[0][0]

    fig_ev, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    axs[1].plot(evoked_misc.times, evoked_misc.data[gaze_x_ch_idx, :])
    axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
    axs[1].set_ylabel('Gaze x')
    axs[1].set_xlabel('Time')
    evoked_meg.plot(picks=picks, gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim, show=display_figs)
    axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')
    fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}/'
    fname = 'Evoked_' + subject.subject_id + f'_{pick_chs}'
    if filter_evoked:
        fname += f'_lfreq{l_freq}_hfreq{h_freq}.png'
    else:
        fname += '.png'
    save.fig(fig=fig_ev, path=fig_path, fname=fname)

    # Save data
    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        epoch_save_path = save_path + f'Epochs/{"-".join(epoch_ids)}/' + subject.subject_id + '/'
        os.makedirs(epoch_save_path, exist_ok=True)
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
        epochs.save(epoch_save_path + epochs_data_fname, overwrite=True)

        # Save evoked data
        evoked_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}/' + subject.subject_id + '/'
        os.makedirs(evoked_save_path, exist_ok=True)
        evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
        evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    ga_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}/'
    os.makedirs(ga_save_path, exist_ok=True)
    grand_avg_data_fname = f'Grand_average_ave.fif'
    grand_avg.save(ga_save_path + grand_avg_data_fname, overwrite=True)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')
grand_avg_misc = grand_avg.copy().pick('misc')

# Filter MEG data
if filter_evoked:
    grand_avg_meg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

# Get Gaze x ch
gaze_x_ch_idx = np.where(np.array(grand_avg_misc.ch_names) == 'ET_gaze_x')[0][0]

# Plot
fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
axs[1].plot(grand_avg_misc.times, grand_avg_misc.data[gaze_x_ch_idx, :])
axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
axs[1].set_ylabel('Gaze x')
axs[1].set_xlabel('Time')
grand_avg_meg.plot(picks=picks, gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim,
                   titles=f'Grand average', show=display_figs)
axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')
fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}/'
fname = f'Grand_average_{pick_chs}'
if filter_evoked:
    fname += f'_lfreq{l_freq}_hfreq{h_freq}.png'
else:
    fname += '.png'
save.fig(fig, fig_path, fname)

# Plot Saccades frontal channels
if any('sac' in id for id in epoch_ids):

    sac_chs = ['MLF14', 'MLF13', 'MLF12', 'MLF11', 'MRF11', 'MRF12', 'MRF13',
               'MRF14', 'MZF01']

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    axs[1].plot(grand_avg_misc.times, grand_avg_misc.data[gaze_x_ch_idx, :])
    axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
    axs[1].set_ylabel('Gaze x')
    axs[1].set_xlabel('Time')
    grand_avg_meg.plot(picks=sac_chs, gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim,
                       titles=f'Grand average', show=display_figs)
    axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')
    fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}/'
    fname = f'Grand_average_front_ch'
    if filter_evoked:
        fname += f'_lfreq{l_freq}_hfreq{h_freq}.png'
    else:
        fname += '.png'
    save.fig(fig, fig_path, fname)