import os
import matplotlib.pyplot as plt
import mne
import functions_analysis
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

#----- Parameters -----#

# Trial selection
trial_params = {'epoch_id': 'tgt_fix_ms',  # use'+' to mix conditions (red+blue)
                'corrans': True,
                'tgtpres': True,
                'mss': None,
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evtdur': None,
                'trialdur': None,
                'rel_sac': None,
                'tmin': -0.3,
                'tmax': 0.6,
                'baseline': (-0.3, -0.05)}

# Base condition
trial_params_base = {'epoch_id': 'tgt_fix_ms',  # use'+' to mix conditions (red+blue)
                     'corrans': False,
                     'tgtpres': True,
                     'mss': None,
                     'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                     'evtdur': None,
                     'trialdur': None,
                     'rel_sac': None,
                     'tmin': -0.3,
                     'tmax': 0.6,
                     'baseline': (-0.3, -0.05)}

meg_params = {'chs_id': 'mag',
              'band_id': None,
              'filter_sensors': True,
              'filter_method': 'iir',
              'data_type': 'ICA'
              }

# Specific run path for saving data and plots
run_path = (f'/Band_{meg_params["band_id"]}/{trial_params["epoch_id"]}_mss{trial_params["mss"]}_corrans{trial_params["corrans"]}_tgtpres{trial_params["tgtpres"]}_'
            f'trialdur{trial_params["trialdur"]}_evtdur{trial_params["evtdur"]}_{trial_params["tmin"]}_{trial_params["tmax"]}_bline{trial_params["baseline"]}/')
# Specific run path for saving data and plots
run_path_base = (f'/Band_{meg_params["band_id"]}/{trial_params_base["epoch_id"]}_mss{trial_params_base["mss"]}_corrans{trial_params_base["corrans"]}_tgtpres{trial_params_base["tgtpres"]}_'
            f'trialdur{trial_params_base["trialdur"]}_evtdur{trial_params_base["evtdur"]}_{trial_params_base["tmin"]}_{trial_params_base["tmax"]}_bline{trial_params_base["baseline"]}/')

# Load data paths
epochs_base_path = save_path + f'Epochs_{meg_params["data_type"]}/' + run_path_base
epochs_path = save_path + f'Epochs_{meg_params["data_type"]}/' + run_path

#----- Run -----#
epochs_list_sub = []
for subject_code in exp_info.subjects_ids:

    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if meg_params['data_type'] == 'ICA':
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    elif meg_params['data_type'] == 'RAW':
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Save data paths
    epochs_new_path = epochs_path.replace(trial_params["epoch_id"], f'{trial_params["epoch_id"]}_subsampled')
    evoked_new_path = epochs_path.replace('Epochs', 'Evoked').replace(trial_params["epoch_id"], f'{trial_params["epoch_id"]}_subsampled')
    epochs_new_base_path = epochs_base_path.replace(trial_params["epoch_id"], f'{trial_params["epoch_id"]}_subsampled')

    # Save figures paths
    epochs_fig_path = epochs_new_path.replace(save_path, plot_path)

    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

    # Load epoched data
    try:
        epochs_base = mne.read_epochs(epochs_base_path + epochs_data_fname)
    except:
        # Load MEG
        meg_data = load.meg(subject=subject, meg_params=meg_params, save_data=save_data)
        epochs_base, _ = functions_analysis.epoch_data(subject=subject, meg_data=meg_data, mss=trial_params_base['mss'], corr_ans=trial_params_base['corrans'],
                                                       tgt_pres=trial_params_base['tgtpres'], epoch_id=trial_params_base['epoch_id'], rel_sac=trial_params_base['rel_sac'],
                                                       tmin=trial_params_base["tmin"], trial_dur=trial_params_base['trialdur'], evt_dur=trial_params_base['evtdur'],
                                                       tmax=trial_params_base["tmax"], reject=trial_params_base['reject'], baseline=trial_params_base["baseline"],
                                                       save_data=save_data, epochs_save_path=epochs_base_path,
                                                       epochs_data_fname=epochs_data_fname)
    try:
        epochs = mne.read_epochs(epochs_path + epochs_data_fname)
    except:
        # Load MEG
        meg_data = load.meg(subject=subject, meg_params=meg_params, save_data=save_data)
        epochs, _ = functions_analysis.epoch_data(subject=subject, meg_data=meg_data, mss=trial_params['mss'], corr_ans=trial_params['corrans'],
                                                       tgt_pres=trial_params['tgtpres'], epoch_id=trial_params['epoch_id'],
                                                       rel_sac=trial_params['rel_sac'],
                                                       tmin=trial_params["tmin"], trial_dur=trial_params['trialdur'], evt_dur=trial_params['evtdur'],
                                                       tmax=trial_params["tmax"], reject=trial_params['reject'], baseline=trial_params["baseline"],
                                                       save_data=save_data, epochs_save_path=epochs_path,
                                                       epochs_data_fname=epochs_data_fname)

    swap = False
    if len(epochs) < len(epochs_base):
        tmp_epoch = epochs_base.copy()
        epochs_base = epochs.copy()
        epochs = tmp_epoch.copy()
        swap = True

    # Extract metadata and subsample
    base_metadata = epochs_base.metadata
    epochs_metadata = epochs.metadata
    epochs_metadata_subsampled = epochs_metadata.sample(n=len(base_metadata))
    # it_metadata_subsampled = it_metadata.sort_values(by='duration', ascending=False).iloc[:len(tgt_metadata)]

    # Plot duration histogram
    hist_bins = 15
    fig, ax = plt.subplots()
    plt.title('Fixation duration distribution')
    base_metadata['duration'].plot.hist(bins=np.linspace(0, max(base_metadata['duration']), hist_bins), ax=ax, alpha=0.5, label='Target')
    epochs_metadata_subsampled['duration'].plot.hist(bins=np.linspace(0, max(base_metadata['duration']), hist_bins), ax=ax, alpha=0.5, label='Distractor')
    plt.legend()
    if save_fig:
        save.fig(fig=fig, path=epochs_fig_path, fname=f'{subject_code}_fixation_duration_distribution')

    # Get epoch ids of epochs to keep
    epochs_subsampled_ids = epochs_metadata_subsampled['id']

    # Get epochs mask of epochs to drop
    drop_epochs_mask = [False if id in epochs_subsampled_ids.values else True for id in epochs_metadata['id']]

    # Drop extra epochs
    epochs_subsampled = epochs.copy()
    epochs_subsampled.drop(drop_epochs_mask)

    # Append to list to compare GA results
    epochs_list_sub.append(epochs_subsampled)

    # Average epochs and append to evokeds list
    evoked_subsampled = epochs_subsampled.average()

    if save_data:
        # Save epoched data
        if not swap:
            print(f"Saving epochs to: {epochs_new_path}")
            os.makedirs(epochs_new_path, exist_ok=True)
            epochs_subsampled.save(epochs_new_path + epochs_data_fname, overwrite=True)
            print(f"Saving epochs to: {epochs_new_base_path}")
            os.makedirs(epochs_new_base_path, exist_ok=True)
            epochs_base.save(epochs_new_base_path + epochs_data_fname, overwrite=True)
        else:
            print(f"Saving epochs to: {epochs_new_path}")
            os.makedirs(epochs_new_path, exist_ok=True)
            epochs_base.save(epochs_new_path + epochs_data_fname, overwrite=True)
            print(f"Saving epochs to: {epochs_new_path}")
            os.makedirs(epochs_new_base_path, exist_ok=True)
            epochs_subsampled.save(epochs_new_base_path + epochs_data_fname, overwrite=True)