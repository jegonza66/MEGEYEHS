import os
import matplotlib.pyplot as plt
import mne
import functions_general
import plot_general
import load
import numpy as np
import save
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
chs_id = 'mag'
# Id
epoch_id = 'it_fix'
corr_ans = True
tgt_pres = True
mss = None
trial_dur = None
evt_dur = None
reject = None


# Get time windows from epoch_id name
tmin, tmax, plot_xlim = tmin, tmax, plot_xlim = -0.3, 0.6, (-0.1, 0.5)
# Baseline
baseline = (-0.3, -0.05)

# Specific run path for saving data and plots
save_id = f'{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}'
run_path = f'/Band_{band_id}/{save_id}_{tmin}_{tmax}_bline{baseline}/'

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

#----- Run -----#
epochs_list_it_sub = []
epochs_list_tgt = []
evokeds = []
ga_subjects = []
for subject_code in exp_info.subjects_ids[:12]:

    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if use_ica_data:
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Load data paths
    epochs_tgt_load_path = save_path + f'Epochs_{data_type}/' + run_path.replace('it_fix', 'tgt_fix')
    epochs_it_load_path = save_path + f'Epochs_{data_type}/' + run_path

    # Save data paths
    epochs_it_save_path = epochs_it_load_path.replace('it_fix', 'it_fix_downsampled')
    evoked_it_save_path = save_path + f'Evoked_{data_type}/' + run_path.replace('it_fix', 'it_fix_subsampled')

    # Save figures paths
    epochs_fig_path = plot_path + f'Epochs_{data_type}/' + run_path.replace('it_fix', 'it_fix_subsampled')
    evoked_fig_path = plot_path + f'Evoked_{data_type}/' + run_path.replace('it_fix', 'it_fix_subsampled')

    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
    grand_avg_data_fname = f'Grand_average_ave.fif'

    # Load epoched data
    epochs_tgt = mne.read_epochs(epochs_tgt_load_path + epochs_data_fname)
    epochs_it = mne.read_epochs(epochs_it_load_path + epochs_data_fname)

    # Extract metadata and subsample
    tgt_metadata = epochs_tgt.metadata
    it_metadata = epochs_it.metadata
    it_metadata_subsampled = it_metadata.sample(n=len(tgt_metadata))
    # it_metadata_subsampled = it_metadata.sort_values(by='duration', ascending=False).iloc[:len(tgt_metadata)]

    # Plot duration histogram
    hist_bins = 10
    fig, ax = plt.subplots()
    plt.title('Fixation duration distribution')
    tgt_metadata['duration'].plot.hist(bins=np.linspace(0, max(tgt_metadata['duration']), hist_bins), ax=ax, alpha=0.5, label='Target')
    it_metadata_subsampled['duration'].plot.hist(bins=np.linspace(0, max(tgt_metadata['duration']), hist_bins), ax=ax, alpha=0.5, label='Distractor')
    plt.legend()
    if save_fig:
        save.fig(fig=fig, path=epochs_fig_path, fname=f'{subject_code}_fixation_duration_distribution')

    # Get epoch ids of epochs to keep
    epochs_subsampled_ids = it_metadata_subsampled['id']

    # Get epochs mask of epochs to drop
    drop_epochs_mask = [False if id in epochs_subsampled_ids.values else True for id in it_metadata['id']]

    # Drop extra epochs
    epochs_subsampled = epochs_it.copy()
    epochs_subsampled.drop(drop_epochs_mask)

    # Append to list to compare GA results
    epochs_list_it_sub.append(epochs_subsampled)
    epochs_list_tgt.append(epochs_tgt)

    # Average epochs and append to evokeds list
    evoked_subsampled = epochs_subsampled.average()
    evokeds.append(evoked_subsampled)

    if save_data:
        # Save epoched data
        os.makedirs(epochs_it_save_path, exist_ok=True)
        epochs_subsampled.save(epochs_it_save_path + epochs_data_fname, overwrite=True)

        # Save evoked data
        os.makedirs(evoked_it_save_path, exist_ok=True)
        evoked_subsampled.save(evoked_it_save_path + evoked_data_fname, overwrite=True)

# Define epochs for GA
ga_tgt_epochs = mne.concatenate_epochs(epochs_list_tgt)
ga_it_sub_epochs = mne.concatenate_epochs(epochs_list_it_sub)

# Get metadata
ga_tgt_metadata = ga_tgt_epochs.metadata
ga_it_sub_metadata = ga_it_sub_epochs.metadata

# Plot GA duration histogram
hist_bins = 10
fig, ax = plt.subplots()
plt.title('Fixation duration distribution')
ga_tgt_metadata['duration'].plot.hist(bins=np.linspace(0, max(ga_tgt_metadata['duration']), hist_bins), ax=ax, alpha=0.5, label='Target')
ga_it_sub_metadata['duration'].plot.hist(bins=np.linspace(0, max(ga_tgt_metadata['duration']), hist_bins), ax=ax, alpha=0.5, label='Distractor')
plt.legend()
if save_fig:
    save.fig(fig=fig, path=epochs_fig_path, fname='GA_fixation_duration_distribution')

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    os.makedirs(evoked_it_save_path, exist_ok=True)
    grand_avg.save(evoked_it_save_path + grand_avg_data_fname, overwrite=True)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')
grand_avg_misc = grand_avg.copy().pick('misc')

# Plot evoked
fname = f'Grand_average_{chs_id}'
picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_meg.info)
plot_general.evoked(evoked_meg=grand_avg_meg, evoked_misc=grand_avg_misc, picks=picks,
                    plot_gaze=False, plot_xlim=plot_xlim, display_figs=display_figs, save_fig=save_fig,
                    fig_path=evoked_fig_path, fname=fname)



