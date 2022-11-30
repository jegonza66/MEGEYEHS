import os
import functions_general
import functions_analysis
import load
import mne

import plot_general
import save
import matplotlib.pyplot as plt
import setup
from paths import paths
import numpy as np

#----- Path -----#
save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()


#----- Save data and display figures -----#
save_data = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Select MEG channels -----#
'''
Select channels or set picks = 'mag')
'''
chs_id = 'mag'
if chs_id == 'mag':
    picks = 'mag'
elif chs_id == 'LR':
    right_chs = ['MRT51', 'MRT52', 'MRT53']
    left_chs = ['MLT51', 'MLT52', 'MLT53']
    picks = right_chs + left_chs

#-----  Filter evoked -----#
filter_evoked = False
l_freq = 0.5
h_freq = 100

#----- Select events -----#
'''
Id Format
Saccades: f'{dir}_sac_{screen}_t{trial}_{n_sacs[-1]}'
Fixations: f'{prefix}_fix_{screen}_t{trial}_{n_fix}' prefix (tgt/it/none)only if vs screen
'''
evt_from_df = True
evt_from_annot = False

# MSS
mss = None
# Id
epoch_id = f'fix_ms'
# Screen
screen = 'ms'
# Duration
dur = 0.2  # seconds
# Direction
dir = None
# Item
if 'tgt' in epoch_id:  # 1 for target, 0 for item, None for none
    tgt = 1
elif 'it' in epoch_id:
    tgt = 0
else:
    tgt = None

if 'fix' in epoch_id:
    tmin = -0.1
    tmax = 0.2
    plot_xlim = (tmin, tmax)
elif 'sac' in epoch_id:
    tmin = -0.1
    tmax = 0.1
    plot_xlim = (-0.05, 0.1)
else:
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

    metadata, events, events_id = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                   evt_from_df=evt_from_df, evt_from_annot=evt_from_annot,
                                                                   screen=screen, mss=mss, dur=dur, tgt=tgt, dir=dir,
                                                                   meg_data=meg_data)

    # Reject based on channel amplitude
    reject = dict(mag=subject.config.general.reject_amp)

    # Epoch data
    epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                        event_repeated='drop', metadata=metadata, preload=True)
    # Drop bad epochs
    epochs.drop_bad()

    # Parameters for plotting
    overlay = None
    if overlay:
        order = overlay.argsort()  # Sorting from longer to shorter
    else:
        order = None
    combine = 'mean'
    group_by = {}

    # Plot epochs
    plot_general.epochs(subject=subject, epochs=epochs, picks=picks, epoch_id=epoch_id, chs_id=chs_id,
                        order=order, overlay=overlay, combine=combine, display_figs=display_figs, group_by=group_by)

    #----- Evoked -----#
    # Define evoked and append for GA
    evoked = epochs.average(picks=['mag', 'misc'])
    evokeds.append(evoked)

    # Separete MEG and misc channels
    evoked_meg = evoked.copy().pick('mag')
    evoked_misc = evoked.copy().pick('misc')

    # Filter MEG evoked
    if filter_evoked:
        evoked_meg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

    # Plot evoked
    fig_path = plot_path + f'Evoked/{epoch_id}/'
    fname = 'Evoked_' + subject.subject_id + f'_{chs_id}'
    plot_general.evoked(subject=subject, evoked_meg=evoked_meg, evoked_misc=evoked_misc, picks=picks,
                        filter_evoked=filter_evoked, l_freq=l_freq, h_freq=h_freq, plot_gaze=True,
                        plot_xlim=plot_xlim, display_figs=display_figs, save_fig=True, fig_path=fig_path, fname=fname)

    # Save data
    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        epoch_save_path = save_path + f'Epochs/{epoch_id}/' + subject.subject_id + '/'
        os.makedirs(epoch_save_path, exist_ok=True)
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
        epochs.save(epoch_save_path + epochs_data_fname, overwrite=True)

        # Save evoked data
        evoked_save_path = save_path + f'Evoked/{epoch_id}/' + subject.subject_id + '/'
        os.makedirs(evoked_save_path, exist_ok=True)
        evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
        evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    ga_save_path = save_path + f'Evoked/{epoch_id}/'
    os.makedirs(ga_save_path, exist_ok=True)
    grand_avg_data_fname = f'Grand_average_ave.fif'
    grand_avg.save(ga_save_path + grand_avg_data_fname, overwrite=True)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')
grand_avg_misc = grand_avg.copy().pick('misc')

# Filter MEG data
if filter_evoked:
    grand_avg_meg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

# Plot evoked
save_fig = True
fig_path = plot_path + f'Evoked/{epoch_id}/'
fname = f'Grand_average_{chs_id}'

plot_general.evoked(subject=None, evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                    filter_evoked=filter_evoked, l_freq=l_freq, h_freq=h_freq, plot_gaze=True,
                    plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=fname)

# Plot Saccades frontal channels
if 'sac' in epoch_id:

    save_fig = True
    fig_path = plot_path + f'Evoked/{epoch_id}/'
    fname = f'Grand_average_front_ch'
    sac_chs = ['MLF14', 'MLF13', 'MLF12', 'MLF11', 'MRF11', 'MRF12', 'MRF13',
               'MRF14', 'MZF01']

    plot_general.evoked(subject=None, evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc,
                        picks=sac_chs, filter_evoked=filter_evoked, l_freq=l_freq, h_freq=h_freq,
                        plot_gaze=True, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path,
                        fname=fname)