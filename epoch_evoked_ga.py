import os
import matplotlib.pyplot as plt
import mne

import functions_general
import functions_analysis
import load
import plot_general
import setup
from paths import paths


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


#----- Parameters -----#
# Frequency band
band_id = 'Theta'
# Id
epoch_id = f'it_fix_vs'
# MSS
mss = None
# Duration
dur = 0.2  # seconds
# Saccades direction
dir = None
# Event definition
evt_from_df = True
evt_from_annot = False
# Plot channels
chs_id = 'mag'


# Screen
screen = epoch_id.split('_')[-1]
# Item
tgt = functions_general.get_item(epoch_id=epoch_id)
# Get time windows from epoch_id name
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)
# Specific run path for saving data and plots
run_path = f'/{band_id}/{epoch_id}_{tmin}_{tmax}/'


#----- Run -----#
evokeds = []
for subject_code in exp_info.subjects_ids:

    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    if band_id:
        meg_data = load.filtered_data(subject=subject, band_id=band_id)
    else:
        meg_data = subject.load_preproc_meg()

    # Pick MEG channels to plot
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

    save_fig = True
    fig_path = plot_path + f'Epochs/' + run_path
    fname = 'Epochs_' + subject.subject_id + f'_{chs_id}_{combine}'

    # Plot epochs
    plot_general.epochs(subject=subject, epochs=epochs, picks=picks, order=order, overlay=overlay, combine=combine,
                        group_by=group_by, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=fname)

    #----- Evoked -----#
    # Define evoked and append for GA
    evoked = epochs.average(picks=['mag', 'misc'])
    evokeds.append(evoked)

    # Separete MEG and misc channels
    evoked_meg = evoked.copy().pick('mag')
    evoked_misc = evoked.copy().pick('misc')

    # Plot evoked
    fig_path = plot_path + f'Evoked/' + run_path
    fname = 'Evoked_' + subject.subject_id + f'_{chs_id}'
    plot_general.evoked(evoked_meg=evoked_meg, evoked_misc=evoked_misc, picks=picks,
                        plot_gaze=True, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                        fig_path=fig_path, fname=fname)

    # Save data
    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        epoch_save_path = save_path + f'Epochs/' + run_path + subject.subject_id + '/'
        os.makedirs(epoch_save_path, exist_ok=True)
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
        epochs.save(epoch_save_path + epochs_data_fname, overwrite=True)

        # Save evoked data
        evoked_save_path = save_path + f'Evoked/' + run_path + subject.subject_id + '/'
        os.makedirs(evoked_save_path, exist_ok=True)
        evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
        evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    ga_save_path = save_path + f'Evoked/' + run_path
    os.makedirs(ga_save_path, exist_ok=True)
    grand_avg_data_fname = f'Grand_average_ave.fif'
    grand_avg.save(ga_save_path + grand_avg_data_fname, overwrite=True)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')
grand_avg_misc = grand_avg.copy().pick('misc')

# Plot evoked
save_fig = True
fig_path = plot_path + f'Evoked/' + run_path
fname = f'Grand_average_{chs_id}'

plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                    plot_gaze=True, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                    fig_path=fig_path, fname=fname)

# Plot Saccades frontal channels
if 'sac' in epoch_id:
    save_fig = True
    fig_path = plot_path + f'Evoked/' + run_path
    fname = f'Grand_average_front_ch'

    # Pick MEG channels to plot
    chs_id = 'sac_chs'
    picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_meg.info)

    plot_general.evoked(subject=None, evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                        plot_gaze=True, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                        fig_path=fig_path, fname=fname)