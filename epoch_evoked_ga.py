import os
import matplotlib.pyplot as plt
import mne
import functions_general
import functions_analysis
import load
import plot_general
import setup
from paths import paths


#----- Paths -----#
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


#----- Parameters -----#
# ICA vs raw data
use_ica_data = True
# Frequency band
band_id = None
# Id
epoch_id = 'l_sac'
# Plot channels
chs_id = 'mag'
# PLots
plot_epochs = True
plot_gaze = True
corr_ans = None
tgt_pres = None
# MSS
mss = None

# Get time windows from epoch_id name
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)
# Baseline
baseline = (tmin, 0)

# Specific run path for saving data and plots
save_id = f'{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}'
run_path = f'/Band_{band_id}/{epoch_id}_{tmin}_{tmax}_bline{baseline}/'

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'


#----- Run -----#
evokeds = []
for subject_code in exp_info.subjects_ids:

    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if use_ica_data:
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Save data paths
    epochs_save_path = save_path + f'Epochs_{data_type}/' + run_path
    evoked_save_path = save_path + f'Evoked_{data_type}/' + run_path
    # Save figures paths
    epochs_fig_path = plot_path + f'Epochs_{data_type}/' + run_path
    evoked_fig_path = plot_path + f'Evoked_{data_type}/' + run_path

    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
    grand_avg_data_fname = f'Grand_average_ave.fif'

    try:
        # Load epoched data
        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
        # Pick MEG channels to plot
        picks = functions_general.pick_chs(chs_id=chs_id, info=epochs.info)
    except:
        # Compute
        if band_id:
            meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=False)
        elif use_ica_data:
            meg_data = load.ica_data(subject=subject)
        else:
            meg_data = subject.load_preproc_meg()

        # Trials
        cond_trials, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=mss,
                                                                          corr_ans=corr_ans, tgt_pres=tgt_pres)

        metadata, events, events_id, metadata_sup = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                                     trials=cond_trials,
                                                                                     meg_data=meg_data)

        # Reject based on channel amplitude
        reject = dict(mag=subject.config.general.reject_amp)

        # Epoch data
        epochs = mne.Epochs(raw=meg_data, events=events, event_id=events_id, tmin=tmin, tmax=tmax, reject=reject,
                            event_repeated='drop', metadata=metadata, preload=True)
        # Drop bad epochs
        epochs.drop_bad()

        if metadata_sup is not None:
            metadata_sup = metadata_sup.loc[(metadata_sup['id'].isin(epochs.metadata['event_name']))].reset_index(drop=True)
            epochs.metadata = metadata_sup

        if save_data:
            # Save epoched data
            epochs.reset_drop_log_selection()
            os.makedirs(epochs_save_path, exist_ok=True)
            epochs.save(epochs_save_path + epochs_data_fname, overwrite=True)

    # Pick MEG channels to plot
    picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
    if plot_epochs:
        # Parameters for plotting
        overlay = epochs.metadata['duration']
        if overlay is not None:
            order = overlay.argsort()  # Sorting from longer to shorter
        else:
            order = None
        sigma = 5
        combine = 'std'
        group_by = {}

        # Figure file name
        fname = 'Epochs_' + subject.subject_id + f'_{chs_id}_{combine}'

        # Plot epochs
        plot_general.epochs(subject=subject, epochs=epochs, picks=picks, order=order, overlay=overlay, combine=combine, sigma=sigma,
                            group_by=group_by, display_figs=display_figs, save_fig=save_fig, fig_path=epochs_fig_path, fname=fname)

    #----- Evoked -----#
    # Define evoked and append for GA
    evoked = epochs.average(picks=['mag', 'misc'])
    evokeds.append(evoked)

    # Separete MEG and misc channels
    evoked_meg = evoked.copy().pick('mag')
    evoked_misc = evoked.copy().pick('misc')

    # Plot evoked
    fname = 'Evoked_' + subject.subject_id + f'_{chs_id}'
    plot_general.evoked(evoked_meg=evoked_meg, evoked_misc=evoked_misc, picks=picks,
                        plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                        fig_path=evoked_fig_path, fname=fname)

    if save_data:
        # Save evoked data
        os.makedirs(evoked_save_path, exist_ok=True)
        evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)


# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    os.makedirs(evoked_save_path, exist_ok=True)
    grand_avg.save(evoked_save_path + grand_avg_data_fname, overwrite=True)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')
grand_avg_misc = grand_avg.copy().pick('misc')

# Plot evoked
fname = f'Grand_average_{chs_id}'
# ylim = dict({'mag': (-150, 200)})
plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                    plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                    fig_path=evoked_fig_path, fname=fname)

# Plot Saccades frontal channels
if 'sac' in epoch_id:
    fname = f'Grand_average_front_ch'

    # Pick MEG channels to plot
    chs_id = 'sac_chs'
    picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_meg.info)

    plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                        plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                        fig_path=evoked_fig_path, fname=fname)