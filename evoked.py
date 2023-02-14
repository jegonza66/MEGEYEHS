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
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

plot_gaze = False

# Pick MEG chs (Select channels or set picks = 'mag')
chs_id = 'mag'

#-----  Select frequency band -----#
band_id = None

epoch_id = 'sac'

# Get time windows from epoch_id name
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)

# Specific run path for saving data and plots
run_path = f'/{band_id}/{epoch_id}_{tmin}_{tmax}/'

evokeds = []
for subject_code in exp_info.subjects_ids:

    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    try:
        # Load evoked data
        # ACTUALIZAR A LA DATA CON ICA
        evoked_save_path = save_path + f'Evoked/' + run_path
        evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
        evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)

    except:
        # Load epoched data
        epochs_save_path = save_path + f'Epochs/' + run_path
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

        # Get evoked by averaging epochs
        evoked = epochs.average(picks=['mag', 'misc'])

        # Save data
        if save_data:
            # Save evoked data
            evoked_save_path = save_path + f'Evoked/' + run_path
            os.makedirs(evoked_save_path, exist_ok=True)
            evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
            evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

    # Apend to evokeds list to pass to grand average
    evokeds.append(evoked)

    # Separete MEG and misc channels
    evoked_meg = evoked.copy().pick('mag')
    evoked_misc = evoked.copy().pick('misc')

    # Plot evoked
    picks = functions_general.pick_chs(chs_id=chs_id, info=evoked_meg.info)

    # Save plot
    fig_path = plot_path + f'Evoked/' + run_path
    fname = 'Evoked_' + subject.subject_id + f'_{chs_id}'

    plot_general.evoked(evoked_meg=evoked_meg, evoked_misc=evoked_misc,
                        picks=picks, plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs,
                        save_fig=save_fig, fig_path=fig_path, fname=fname)

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
fig_path = plot_path + f'Evoked/' + run_path
fname = f'Grand_average_{chs_id}'
plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                    plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs,
                    save_fig=save_fig, fig_path=fig_path, fname=fname)

# Plot Saccades frontal channels
if 'sac' in epoch_id:
    fig_path = plot_path + f'Evoked/' + run_path
    fname = f'Grand_average_front_ch'

    # Pick MEG channels to plot
    chs_id = 'sac_chs'
    picks = functions_general.pick_chs(chs_id=chs_id, info=evoked_meg.info)
    plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                        plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                        fig_path=fig_path, fname=fname)