import os.path
from tensorpac import Pac
from tensorpac import EventRelatedPac
import functions_general
import mne
import save
from paths import paths
import load
import setup
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plot_general
import functions_analysis

# Load experiment info
exp_info = setup.exp_info()

#--------- Define Parameters ---------#

# Save
save_fig = True
save_data = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#----- Parameters -----#
# Trial selection
trial_params = {'epoch_id': 'it_fix_vs_sub',  # use'+' to mix conditions (red+blue)
                'corrans': True,
                'tgtpres': None,
                'mss': [1, 4],
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evtdur': None,
                'trialdur': None}

meg_params = {'band_id': None,
              'regions_id': 'all',
              'data_type': 'ICA'
              }

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'surface'
ico = 4
spacing = 5.  # Only for volume source estimation
pick_ori = None
parcelation='aparc'
labels_mode = 'pca_flip'

# Define PAC parameters
l_freq_amp, h_freq_amp = 15, 30
width_amp = 5
step_amp = 1

l_freq_pha, h_freq_pha = 7, 13
width_pha = 1
step_pha = .1

#--------- Setup ---------#

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

if 'vs' in trial_params['epoch_id'] and 'fix' not in trial_params['epoch_id'] and 'sac' not in trial_params['epoch_id']:
    trial_dur = vs_dur[trial_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_dur = None

# Adapt to hfreq model
if meg_params['band_id'] == 'HGamma' or ((isinstance(meg_params['band_id'], list) or isinstance(meg_params['band_id'], tuple)) and meg_params['band_id'][0] > 40):
    model_name = 'hfreq-' + model_name

# --------- Freesurfer Path ---------#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Surface labels id by region
aparc_region_labels = {'occipital': ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine'],
                       'parietal': ['postcentral', 'superiorparietal', 'supramarginal', 'inferiorparietal', 'precuneus'],
                       'temporal': ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'transversetemporal', 'fusiform', 'entorhinal', 'parahippocampal', 'temporalpole'],
                       'frontal': ['precentral', 'caudalmiddlefrontal', 'superiorfrontal', 'rostralmiddlefrontal', 'lateralorbitofrontal', 'parstriangularis', 'parsorbitalis', 'parsopercularis', 'medialorbitofrontal', 'paracentral', 'frontalpole'],
                       'insula': ['insula'],
                       'cingulate': ['isthmuscingulate', 'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate']}

aparc_region_labels['all'] = [value for key in aparc_region_labels.keys() for value in aparc_region_labels[key]]

# Extract region labels data
region_labels = [element for region in meg_params['regions_id'].split('_') for element in aparc_region_labels[region]]

# Get parcelation labels
fsaverage_labels = functions_analysis.get_labels(parcelation=parcelation, subjects_dir=subjects_dir, surf_vol=surf_vol)
fsaverage_labels = [label for label in fsaverage_labels for label_id in region_labels if label.name.startswith(label_id + '-')]

# Define PAC computer
p_obj = Pac(idpac=(6, 0, 0), f_pha=(l_freq_pha, h_freq_pha, width_pha, step_pha), f_amp=(l_freq_amp, h_freq_amp, width_amp, step_amp))

# Define ERPAC computer
rp_obj = EventRelatedPac(f_pha=[l_freq_pha, h_freq_pha], f_amp=(l_freq_amp, h_freq_amp, width_amp, step_amp))

# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}

# Save source estimates time courses on FreeSurfer
stcs_default_dict = {}
GA_stcs = {}

# --------- Run ---------#
for param in param_values.keys():
    stcs_default_dict[param] = {}
    GA_stcs[param] = {}
    for param_value in param_values[param]:

        # Get run parameters from trial params including all comparison between different parameters
        run_params = trial_params
        # Set first value of parameters comparisons to avoid having lists in run params
        if len(param_values.keys()) > 1:
            for key in param_values.keys():
                run_params[key] = param_values[key][0]
        # Set comparison key value
        run_params[param] = param_value

        run_params['tmin'], run_params['tmax'], plot_xlim = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'])

        # Get baseline duration for epoch_id
        run_params['baseline'], run_params['plot_baseline'] = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'],
                                                                                                      tmin=run_params['tmin'], tmax=run_params['tmax'],
                                                                                                      cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                                                      cross2_dur=cross2_dur)

        # Paths
        run_path = (f"{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}_"
                    f"trialdur{run_params['trialdur']}_evtdur{run_params['evtdur']}_{run_params['tmin']}_{run_params['tmax']}_bline{run_params['baseline']}/")
        # Redefine save id
        if 'rel_sac' in run_params.keys() and run_params['rel_sac'] != None:
            run_path = run_params['rel_sac'] + '_' + run_path

        # Source paths
        source_model_path = f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}"
        labels_model_path = source_model_path + f"_{parcelation}_{labels_mode}/"

        epochs_save_path = paths().save_path() + f"Epochs_{meg_params['data_type']}/Band_None/{run_path}/"
        label_ts_save_path = paths().save_path() + f"Source_labels_{meg_params['data_type']}/Band_{meg_params['band_id']}/" + run_path + labels_model_path

        # Save figures paths
        fig_path = paths().plots_path() + f"PAC_{meg_params['data_type']}/" + run_path + f"{l_freq_pha}-{h_freq_pha}_{l_freq_amp}-{h_freq_amp}/{labels_model_path}"

        # Save pac across subjects
        pac_subjects = []
        erpac_subjects = []

        # Save source estimates time courses on default's subject source space
        stcs_default_dict[param][param_value] = []
        GA_stcs[param][param_value] = []

        # Iterate over subjects
        for subject_code in exp_info.subjects_ids:

            # Define save path and file name for loading and saving epoched, evoked, and GA data
            if meg_params['data_type'] == 'ICA':
                # Load subject object
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            else:
                # Load subject object
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

            # Data filenames
            epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
            labels_ts_data_fname = f'Subject_{subject.subject_id}.pkl'

            # Load labels ts data
            if os.path.isfile(label_ts_save_path + labels_ts_data_fname):
                label_ts = load.var(file_path=label_ts_save_path + labels_ts_data_fname)
                epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
            else:
                # Source data path
                sources_path_subject = paths().sources_path() + subject.subject_id

                # Load forward model
                fname_fwd = paths().fwd_path(subject=subject, subject_code=subject_code, ico=ico, spacing=spacing, surf_vol=surf_vol)
                fwd = mne.read_forward_solution(fname_fwd)
                src = fwd['src']

                # Load filter
                fname_filter = paths().filter_path(subject=subject, subject_code=subject_code, ico=ico, spacing=spacing, surf_vol=surf_vol, pick_ori=pick_ori,
                                                   model_name=model_name)
                filters = mne.beamformer.read_beamformer(fname_filter)

                if os.path.isfile(epochs_save_path + epochs_data_fname):
                    # Load epoched data
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                else:
                    # Load meg data
                    if meg_params['data_type']:
                        meg_data = load.ica_data(subject=subject)
                    else:
                        meg_data = subject.load_preproc_meg_data()

                    # Epoch data
                    epochs, events = functions_analysis.epoch_data(subject=subject, mss=run_params['mss'], corr_ans=run_params['corrans'], trial_dur=trial_dur,
                                                                   tgt_pres=run_params['tgtpres'], baseline=run_params['baseline'], reject=run_params['reject'],
                                                                   evt_dur=run_params['evtdur'], epoch_id=run_params['epoch_id'],
                                                                   meg_data=meg_data, tmin=run_params['tmin'], tmax=run_params['tmax'], save_data=save_data,
                                                                   epochs_save_path=epochs_save_path, epochs_data_fname=epochs_data_fname)

                # --------- Source estimation ---------#
                # Redefine stc epochs to iterate over
                stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

                # Extract region labels data

                if surf_vol == 'volume':
                    labels_path = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
                    labels = mne.get_volume_labels_from_aseg(labels_path, return_colors=False)
                    used_labels = [label for label in labels for label_id in region_labels if label_id in label]
                    used_labels = [labels_path, used_labels]

                elif subject_code != 'fsaverage':
                    # Get labels for FreeSurfer cortical parcellation
                    labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
                    used_labels = [label for label in labels for label_id in region_labels if label.name.startswith(label_id + '-')]

                else:
                    labels = fsaverage_labels
                    used_labels = [label for label in labels for label_id in region_labels if label.name.startswith(label_id + '-')]

                # Average the source estimates within each label using sign-flips to reduce signal cancellations
                label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=used_labels, src=src, mode=labels_mode, return_generator=False)

            label_ts_array = np.array(label_ts)
            pac_regions = []
            # Compute PAC over regions
            print('Computing PAC')
            for e in tqdm(range(len(label_ts[0]))):

                region_data = label_ts_array[:, e, :]
                # epochs_data = epochs.get_data()[:, e, :].squeeze()

                # Extract all phases and amplitudes
                epochs_pha = p_obj.filter(epochs.info['sfreq'], region_data, ftype='phase')
                epochs_amp = p_obj.filter(epochs.info['sfreq'], region_data, ftype='amplitude')

                # Compute PAC inside rest, planning, and execution
                pac = p_obj.fit(epochs_pha, epochs_amp, verbose=False).mean(-1)

                # Append
                pac_regions.append(pac)

            # Average electrodes
            pac_subject = np.mean(np.array(pac_regions), axis=0)

            # Plot
            fname = f'{subject.subject_id}_pac_cross1'
            fig = plt.figure()
            p_obj.comodulogram(pac_subject)
            if save_fig:
                save.fig(fig=fig, path=fig_path, fname=fname)

            erpac_regions = []
            # Compute ERPAC over regions
            print('Computing ERPAC')
            for e in tqdm(range(len(label_ts[0]))):

                region_data = label_ts_array[:, e, :]

                # Compute ERPac
                erpac = rp_obj.filterfit(epochs.info['sfreq'], region_data, method='gc', smooth=100, verbose=False)

                # Append
                erpac_regions.append(erpac)

            # Avergae electrodes
            erpac_subject = np.mean(np.array(erpac_regions), axis=0).squeeze()

            # Plot
            fname = f'{subject.subject_id}_erpac'
            fig = plt.figure()
            rp_obj.pacplot(erpac_subject, epochs.times, rp_obj.yvec, xlabel='Time', ylabel='Amplitude frequency (Hz)',
                           title=f'Event-Related PAC occurring for {(l_freq_pha, h_freq_pha)} phase', fz_labels=15, fz_title=18)
            plot_general.add_task_lines(y_text=(h_freq_amp - l_freq_amp) * 95/100)
            if save_fig:
                save.fig(fig=fig, path=fig_path, fname=fname)

            # Append to subjects data
            pac_subjects.append(pac_subject)
            erpac_subjects.append(erpac_subject)


        # Convert to array
        pac_subjects = np.array(pac_subjects)
        erpac_subjects = np.array(erpac_subjects)

        pac_ga = np.mean(pac_subjects, axis=0)
        erpac_ga = np.mean(erpac_subjects, axis=0)

        fname = 'GA_comodulogram'
        fig = plt.figure()
        p_obj.comodulogram(pac_ga)
        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

        fname = 'GA_erpac'
        fig = plt.figure()
        rp_obj.pacplot(erpac_ga, epochs.times, rp_obj.yvec, xlabel='Time', ylabel='Amplitude frequency (Hz)',
                       title=f'GA Event-Related PAC occurring for {(l_freq_pha, h_freq_pha)} phase', fz_labels=15, fz_title=18)
        plot_general.add_task_lines(y_text=(h_freq_amp - l_freq_amp) * 95/100)
        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)