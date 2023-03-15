import load
import os
import mne
import matplotlib.pyplot as plt
import setup
from paths import paths
import plot_general
import functions_general

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

#-----  Select frequency band -----#
# ICA vs raw data
use_ica_data = False
band_id = None
# Id
epoch_id = 'l_sac'
# Duration
dur = None  # seconds
# Pick MEG chs (Select channels or set picks = 'mag')
chs_id = 'mag'
# Plot eye movements
plot_gaze = False

# Get time windows from epoch_id name
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)

# Specific run path for saving data and plots
run_path = f'/Band_{band_id}/{epoch_id}_{tmin}_{tmax}/'

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

evokeds = []
for subject_code in exp_info.subjects_ids:

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
        # Load evoked data
        evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]

    except:
        # Load epoched data
        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

        # Get evoked by averaging epochs
        evoked = epochs.average(picks=['mag', 'misc'])

        # Save data
        if save_data:
            # Save evoked data
            evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

    # Apend to evokeds list to pass to grand average
    evokeds.append(evoked)

    # Separete MEG and misc channels
    evoked_meg = evoked.copy().pick('mag')
    evoked_misc = evoked.copy().pick('misc')

    # Plot evoked
    picks = functions_general.pick_chs(chs_id=chs_id, info=evoked_meg.info)

    # Save plot
    fname = 'Evoked_' + subject.subject_id + f'_{chs_id}'
    plot_general.evoked(evoked_meg=evoked_meg, evoked_misc=evoked_misc,
                        picks=picks, plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs,
                        save_fig=save_fig, fig_path=evoked_fig_path, fname=fname)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    grand_avg.save(evoked_save_path + grand_avg_data_fname, overwrite=True)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')
grand_avg_misc = grand_avg.copy().pick('misc')

# Plot evoked
fname = f'Grand_average_{chs_id}'
plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                    plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs,
                    save_fig=save_fig, fig_path=evoked_fig_path, fname=fname)

# Plot Saccades frontal channels
if 'sac' in epoch_id:
    fname = f'Grand_average_front_ch'

    # Pick MEG channels to plot
    chs_id = 'sac_chs'
    picks = functions_general.pick_chs(chs_id=chs_id, info=evoked_meg.info)
    plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                        plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                        fig_path=evoked_fig_path, fname=fname)