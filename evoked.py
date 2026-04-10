import load
import save
import os
import mne
import matplotlib.pyplot as plt
import setup
from paths import paths
import plot_general
import functions_general
import functions_analysis

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

#-----  Select frequency band -----#
# ICA vs raw data
use_ica_data = True
band_id = None
# Id
epoch_id = 'it_fix_vs+tgt_fix_vs'
# Pick MEG chs (Select channels or set picks = 'mag')
chs_id = 'mag'
# Plot eye movements
plot_gaze = False
corr_ans = None
tgt_pres = None
mss = None
reject = None
evt_dur = None

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
if 'vs' in epoch_id and 'fix' not in epoch_id:
    trial_dur = vs_dur[mss]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_dur = None

# Get time windows from epoch_id name
map = dict(tgt_fix={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3, 0.6)},
           it_fix_vs={'tmin': -0.2, 'tmax': 0.5, 'plot_xlim': (-0.15, 0.45)},
           sac_emap={'tmin': -0.5, 'tmax': 3, 'plot_xlim': (-0.3, 2.5)},
           hl_start={'tmin': -3, 'tmax': 35, 'plot_xlim': (-2.5, 33)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, mss=mss, plot_edge=0, map=map)

# Baseline
baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=epoch_id, mss=mss, tmin=tmin, tmax=tmax, cross1_dur=cross1_dur,
                                                                  mss_duration=mss_duration, cross2_dur=cross2_dur)

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Specific run path for saving data and plots
save_id = f'{epoch_id}_mss{mss}_Corr{corr_ans}_tgt{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}'
run_path = f'/Band_{band_id}/{save_id}_{tmin}_{tmax}_bline{baseline}/'

# Save data paths
epochs_save_path = save_path + f'Epochs_{data_type}/' + run_path
evoked_save_path = save_path + f'Evoked_{data_type}/' + run_path
grand_avg_data_fname = f'Grand_average_ave.fif'
# Save figures paths
epochs_fig_path = plot_path + f'Epochs_{data_type}/' + run_path
evoked_fig_path = plot_path + f'Evoked_{data_type}/' + run_path + f'{chs_id}/'


evokeds = []

for subject_code in exp_info.subjects_ids:

    if use_ica_data:
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

    try:
        # Load evoked data
        evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]

    except:
        try:
            # Load epoched data
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

            # Get evoked by averaging epochs
            evoked = epochs.average(picks=['mag', 'misc'])

            # Save data
            if save_data:
                # Save evoked data
                evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)
        except:

            # Define save path and file name for loading and saving epoched, evoked, and GA data
            if use_ica_data:
                # Load subject object
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
                if band_id:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=False)
                else:
                    meg_data = load.ica_data(subject=subject)
            else:
                # Load subject object
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
                if band_id:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=False)
                else:
                    meg_data = subject.load_preproc_meg_data()

            # Epoch data
            epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans,
                                                           tgt_pres=tgt_pres, epoch_id=epoch_id, meg_data=meg_data,
                                                           tmin=tmin, tmax=tmax, save_data=save_data,
                                                           epochs_save_path=epochs_save_path, evt_dur=evt_dur,
                                                           epochs_data_fname=epochs_data_fname, reject=reject,
                                                           baseline=baseline)

            # ----- Evoked -----#
            # Define evoked and append for GA
            evoked = epochs.average(picks=['mag', 'misc'])
            evokeds.append(evoked)

        if save_data:
            # Save evoked data
            os.makedirs(evoked_save_path, exist_ok=True)
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
grand_avg = mne.grand_average(evokeds, interpolate_bads=True)

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
                    save_fig=save_fig, fig_path=evoked_fig_path, fname=fname, ticksize=18)


# Plot topographies
topo_times = [grand_avg_meg.times[grand_avg_meg._data.std(axis=0).argmax()]]
fig_topo = grand_avg_meg.plot_topomap(times=topo_times, cmap='coolwarm', show=display_figs, vlim=(-75, 75), size=2)
# Increase colorbar font size (last axis is the colorbar in MNE topomap figures)
cbar_ax = fig_topo.axes[-1]
cbar_ax.tick_params(labelsize=18)
if cbar_ax.get_ylabel():
    cbar_ax.yaxis.label.set_size(18)
if save_fig:
    fname = f'Grand_average_topomap_{chs_id}'
    save.fig(fig=fig_topo, path=evoked_fig_path, fname=fname)

# Plot Saccades frontal channels
if 'sac' in epoch_id:
    fname = f'Grand_average_front_ch'

    # Pick MEG channels to plot
    chs_id = 'sac_chs'
    picks = functions_general.pick_chs(chs_id=chs_id, info=evoked_meg.info)
    plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                        plot_gaze=plot_gaze, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                        fig_path=evoked_fig_path, fname=fname)