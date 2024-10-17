## Surface Time-Frequency

import os
import functions_analysis
import functions_general
import mne
import mne.beamformer as beamformer
import save
from paths import paths
import load
import setup
import numpy as np
import plot_general
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import itertools
import scipy
from mne.stats import permutation_cluster_1samp_test

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
trial_params = {'epoch_id': ['tgt_fix_vs', 'it_fix_vs_subsampled'],  # use'+' to mix conditions (red+blue)
                'corrans': True,
                'tgtpres': True,
                'mss': None,
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evtdur': None}

meg_params = {'regions_id': 'all',
              'band_id': None,
              'filter_sensors': False,
              'filter_method': 'iir',
              'data_type': 'ICA'
              }

# TRF parameters
trf_params = {'input_features': ['tgt_fix_vs', 'it_fix_vs_subsampled', 'blue', 'red'],   # Select features (events)
              'standarize': False,
              'fit_power': False,
              'alpha': None,
              'tmin': -0.3,
              'tmax': 0.6,
              'baseline': (-0.3, -0.05)
              }

l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])
l_freq, h_freq = (1, 40)

# Compare features
run_comparison = True

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'surface'
labels_mode = 'pca_flip'
ico = 5
spacing = 5.  # Only for volume source estimation
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
source_estimation = 'epo'  # 'epo' / 'evk' / 'cov' / 'trf'
estimate_source_tf = True
visualize_alignment = False

# Baseline
bline_mode_subj = 'db'
bline_mode_ga = 'mean'
plot_edge = 0.15

# Plot
initial_time = 0.1
difference_initial_time = [0.25, 0.35]
positive_cbar = None  # None for free determination, False to include negative values
plot_individuals = True
plot_ga = True

# Permutations test
run_permutations_GA = True
run_permutations_diff = True
n_permutations = 1024
desired_tval = 0.01
degrees_of_freedom = len(exp_info.subjects_ids) - 1
p_threshold = 0.05
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# t_thresh = dict(start=0, step=0.2)
mask_negatives = False


#--------- Setup ---------#

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

# Adapt to hfreq model
if meg_params['band_id'] == 'HGamma' or ((isinstance(meg_params['band_id'], list) or isinstance(meg_params['band_id'], tuple)) and meg_params['band_id'][0] > 40):
    model_name = 'hfreq-' + model_name


# --------- Freesurfer Path ---------#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Get Source space for default subject
if surf_vol == 'volume':
    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_volume_ico{ico}_{int(spacing)}-src.fif'
elif surf_vol == 'surface':
    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_surface_ico{ico}-src.fif'
elif surf_vol == 'mixed':
    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_mixed_ico{ico}_{int(spacing)}-src.fif'

src_default = mne.read_source_spaces(fname_src)

# Surface labels id by region
aparc_region_labels = {'occipital': ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine'],
                       'parietal': ['postcentral', 'superiorparietal', 'supramarginal', 'inferiorparietal', 'precuneus'],
                       'temporal': ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'transversetemporal', 'fusiform', 'entorhinal', 'parahippocampal', 'temporalpole'],
                       'frontal': ['precentral', 'caudalmiddlefrontal', 'superiorfrontal', 'rostralmiddlefrontal', 'lateralorbitofrontal', 'parstriangularis', 'parsorbitalis', 'parsopercularis', 'medialorbitofrontal', 'paracentral', 'frontalpole'],
                       'insula': ['insula'],
                       'cingulate': ['isthmuscingulate', 'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate']}

aparc_region_labels['all'] = [value for key in aparc_region_labels.keys() for value in aparc_region_labels[key]]

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

        # Define trial duration for VS screen analysis
        if 'vs' in run_params['epoch_id'] and 'fix' not in run_params['epoch_id'] and 'sac' not in run_params['epoch_id']:
            run_params['trialdur'] = vs_dur[run_params['mss']]
        else:
            run_params['trialdur'] = None  # Change to trial_dur = None to use all trials for no 'vs' epochs

        # Get time windows from epoch_id name
        run_params['tmin'], run_params['tmax'], _ = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'], plot_edge=plot_edge)

        # Get baseline duration for epoch_id
        # map = dict(sac={'tmin': -0.0, 'tmax': 0.15, 'plot_xlim': (-0.2 + plot_edge, 0.3 - plot_edge)})
        run_params['baseline'], run_params['plot_baseline'] = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'],
                                                                                                      tmin=run_params['tmin'], tmax=run_params['tmax'],
                                                                                                      cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                                                      cross2_dur=cross2_dur, plot_edge=plot_edge)

        # Paths
        run_path = (f"Band_{meg_params['band_id']}/{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}_"
                    f"trialdur{run_params['trialdur']}_evtdur{run_params['evtdur']}_{run_params['tmin']}_{run_params['tmax']}_bline{run_params['baseline']}/")

        # Data paths
        epochs_save_path = paths().save_path() + f"Epochs_{meg_params['data_type']}/" + run_path
        evoked_save_path = paths().save_path() + f"Evoked_{meg_params['data_type']}/" + run_path
        cov_save_path = paths().save_path() + f"Cov_Epochs_{meg_params['data_type']}/" + run_path

        # Source plots paths
        if surf_vol == 'volume' or surf_vol == 'mixed':
            source_path = f"{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}_{bline_mode_subj}_{source_estimation}/"
        else:
            if labels_mode:
                source_path = f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}_{bline_mode_subj}_{source_estimation}_{labels_mode}/"
            else:
                source_path = f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}_{bline_mode_subj}_{source_estimation}/"

        # Define fig path

        fig_path = paths().plots_path() + f"Source_Space_{meg_params['data_type']}_TF/" + run_path + source_path


        # Get parcelation labels
        fsaverage_labels = functions_analysis.get_labels(parcelation='aparc', subjects_dir=subjects_dir, surf_vol=surf_vol)

        # Save source estimates time courses on default's subject source space
        stcs_default_dict[param][param_value] = []

        # Iterate over participants
        for subject_code in exp_info.subjects_ids:
            # Load subject
            if meg_params['data_type'] == 'ICA':
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            elif meg_params['data_type'] == 'RAW':
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

            # Data filenames
            epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
            evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

            # --------- Coord systems alignment ---------#
            if force_fsaverage:
                subject_code = 'fsaverage'
                dig = False
            else:
                # Check if subject has MRI data
                try:
                    fs_subj_path = os.path.join(subjects_dir, subject.subject_id)
                    os.listdir(fs_subj_path)
                    dig = True
                except:
                    subject_code = 'fsaverage'
                    dig = False

            # Plot alignment visualization
            if visualize_alignment:
                plot_general.mri_meg_alignment(subject=subject, subject_code=subject_code, dig=dig, subjects_dir=subjects_dir)

            # Source data path
            sources_path_subject = paths().sources_path() + subject.subject_id
            # Load forward model
            if surf_vol == 'volume':
                fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
            elif surf_vol == 'surface':
                fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
            elif surf_vol == 'mixed':
                fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
            fwd = mne.read_forward_solution(fname_fwd)
            src = fwd['src']

            # Load filter
            if surf_vol == 'volume':
                fname_filter = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
            elif surf_vol == 'surface':
                fname_filter = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}-{model_name}.fif'
            elif surf_vol == 'mixed':
                fname_filter = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
            filters = mne.beamformer.read_beamformer(fname_filter)

            # Get epochs and evoked
            try:
                # Load data
                if source_estimation == 'trf':
                    # Load MEG
                    meg_data = load.meg(subject=subject, meg_params=meg_params, save_data=save_data)

                else:
                    if source_estimation == 'epo' or estimate_source_tf:
                        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                        # Pick meg channels for source modeling
                        epochs.pick('mag')
                    evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]
                    # Pick meg channels for source modeling
                    evoked.pick('mag')

                    if source_estimation == 'cov':
                        channel_types = evoked.get_channel_types()
                        bad_channels = evoked.info['bads']

            except:
                # Get epochs
                try:
                    # load epochs
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

                    if source_estimation == 'cov':
                        channel_types = epochs.get_channel_types()
                        bad_channels = epochs.info['bads']
                    else:
                        # Define evoked from epochs
                        evoked = epochs.average()

                        # Save evoked data
                        if save_data:
                            os.makedirs(evoked_save_path, exist_ok=True)
                            evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

                        # Pick meg channels for source modeling
                        evoked.pick('mag')
                        epochs.pick('mag')

                except:
                    # Load MEG
                    meg_data = load.meg(subject=subject, meg_params=meg_params, save_data=save_data)

                    if source_estimation == 'cov':
                        channel_types = meg_data.get_channel_types()
                        bad_channels = meg_data.info['bads']

                    else:
                        # Epoch data
                        epochs, events = functions_analysis.epoch_data(subject=subject, meg_data=meg_data, mss=run_params['mss'], corr_ans=run_params['corrans'], tgt_pres=run_params['tgtpres'],
                                                                       epoch_id=run_params['epoch_id'],  tmin=run_params['tmin'], trial_dur=run_params['trialdur'],
                                                                       tmax=run_params['tmax'], reject=run_params['reject'], baseline=run_params['baseline'],
                                                                       save_data=save_data, epochs_save_path=epochs_save_path,
                                                                       epochs_data_fname=epochs_data_fname)

                        # Define evoked from epochs
                        evoked = epochs.average()

                        # Save evoked data
                        if save_data:
                            os.makedirs(evoked_save_path, exist_ok=True)
                            evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

                        # Pick meg channels for source modeling
                        evoked.pick('mag')
                        epochs.pick('mag')

            # --------- Source estimation ---------#
            # Source TF
            # Extract labels
            region_labels = [element for region in meg_params['regions_id'].split('_') for element in aparc_region_labels[region]]

            if surf_vol == 'volume':
                labels_path = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
                labels = mne.get_volume_labels_from_aseg(labels_path, return_colors=False)
                used_labels = [label for label in labels for label_id in region_labels if label_id in label]
                used_labels = [labels_path, used_labels]

            elif subject_code != 'fsaverage':
                # Get labels for FreeSurfer cortical parcellation
                labels = mne.read_labels_from_annot(subject=subject_code, parc='aparc', subjects_dir=subjects_dir)
                used_labels = [label for label in labels for label_id in region_labels if label.name.startswith(label_id + '-')]

            else:
                labels = fsaverage_labels
                used_labels = [label for label in labels for label_id in region_labels if label.name.startswith(label_id + '-')]

            # Redefine stc epochs to iterate over
            stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

            # Average the source estimates within each label using sign-flips to reduce signal cancellations
            label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=used_labels, src=src, mode=labels_mode, return_generator=False)

            # Estimate TF
            region_source_array = np.array(label_ts)
            freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)
            n_cycles = freqs / 4.
            source_tf_array = mne.time_frequency.tfr_array_morlet(region_source_array, sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles, output='avg_power')
            source_tf = mne.time_frequency.AverageTFR(info=epochs.copy().pick(epochs.ch_names[:region_source_array.shape[1]]).info, data=source_tf_array,
                                                      times=epochs.times, freqs=freqs, nave=len(label_ts))

            # Trim edge times
            source_tf.crop(tmin=epochs.tmin + plot_edge, tmax=epochs.tmax - plot_edge)

            # Apply baseline
            if bline_mode_subj == 'db':
                source_tf.data = 10 * np.log10(
                    source_tf.data / np.expand_dims(
                        source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1))
            elif bline_mode_subj == 'ratio':
                source_tf.data = source_tf.data / np.expand_dims(
                    source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)
            elif bline_mode_subj == 'mean':
                source_tf.data = source_tf.data - np.expand_dims(
                    source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)

            # Append data for GA
            stcs_default_dict[param][param_value].append(source_tf)

            # Plot power time-frequency
            if plot_individuals:

                fname = f"Power_{subject.subject_id}_{meg_params['regions_id']}_{bline_mode_subj}_{l_freq}_{h_freq}"
                fig, ax = plt.subplots(figsize=(10, 7))
                fig = source_tf.plot(axes=ax, show=display_figs)[0]
                ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--', colors='gray')
                fig.suptitle(fname)

                if save_fig:
                    save.fig(fig=fig, path=fig_path, fname=fname)


        # Grand Average: Average evoked stcs from this epoch_id
        all_subj_source_data = np.zeros(
            tuple([len(stcs_default_dict[param][param_value])] + [size for size in stcs_default_dict[param][param_value][0].data.shape]))
        for j, stc in enumerate(stcs_default_dict[param][param_value]):
            all_subj_source_data[j] = stcs_default_dict[param][param_value][j].data
        if mask_negatives:
            all_subj_source_data[all_subj_source_data < 0] = 0

        # Define GA data
        GA_stc_data = all_subj_source_data.mean(0)

        # Copy Source Time Course from default subject morph to define GA STC
        GA_stc = source_tf.copy()

        # Reeplace data
        GA_stc.data = GA_stc_data
        GA_stc.subject = 'fsaverage'

        # Save GA from epoch id
        GA_stcs[param][param_value] = GA_stc

        # --------- Plot GA ---------#
        if plot_ga:
            for i, region in enumerate(used_labels):

                ga_tf_region = GA_stc.pick(GA_stc.ch_names[0])
                ga_tf_region.data = np.expand_dims(GA_stc_data[i], axis=0)

                fname = f'GA_{region}'
                fig, ax = plt.subplots(figsize=(10, 7))
                fig = ga_tf_region.plot(axes=ax, show=display_figs)[0]
                ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--', colors='gray')
                fig.suptitle(fname)

                if save_fig:
                    save.fig(fig=fig, path=fig_path, fname=fname)



#----- Difference between conditions -----#
for param in param_values.keys():
    if len(param_values[param]) > 1 and run_comparison:
        for comparison in list(itertools.combinations(param_values[param], 2)):

            if all(type(element) == int for element in comparison):
                comparison = sorted(comparison, reverse=True)

            # Figure difference save path
            if param == 'epoch_id':
                fig_path_diff = fig_path.replace(f'{param_values[param][-1]}', f'{comparison[0]}-{comparison[1]}')
            else:
                fig_path_diff = fig_path.replace(f'{param}{param_values[param][-1]}', f'{param}{comparison[0]}-{comparison[1]}')

            print(f'Taking difference between conditions: {param} {comparison[0]} - {comparison[1]}')

            # Get subjects difference
            stcs_diff = []
            for i in range(len(stcs_default_dict[param][comparison[0]])):
                stcs_diff.append(stcs_default_dict[param][comparison[0]][i] - stcs_default_dict[param][comparison[1]][i])

            # Average evoked stcs
            all_subj_diff_data = np.zeros(tuple([len(stcs_diff)]+[size for size in stcs_diff[0].data.shape]))
            for i, stc in enumerate(stcs_diff):
                all_subj_diff_data[i] = stcs_diff[i].data

            if mask_negatives:
                all_subj_diff_data[all_subj_diff_data < 0] = 0

            GA_stc_diff_data = all_subj_diff_data.mean(0)

            # Copy Source Time Course from default subject morph to define GA STC
            GA_stc_diff = GA_stc.copy()

            # Reeplace data
            GA_stc_diff.data = GA_stc_diff_data
            GA_stc_diff.subject = 'fsaverage'

            # Iterate, test and plot each region
            for i, region in enumerate(used_labels):

                #--------- Cluster permutations test ---------#
                if run_permutations_diff and pick_ori != 'vector':

                    # Define data to test
                    region_subj_diff_data = all_subj_diff_data[:, i, :, :]

                    # Run clusters permutations test
                    clusters_mask, clusters_mask_plot = functions_analysis.run_time_frequency_test(data=region_subj_diff_data, pval_threshold=p_threshold, t_thresh=t_thresh, n_permutations=n_permutations)

                    if isinstance(t_thresh, dict):
                        fname = f'GA_{region.name}_{l_freq}_{h_freq}_tTFCE_pval{p_threshold}'
                    else:
                        fname = f'GA_{region.name}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{p_threshold}'

                else:
                    clusters_mask_plot = None
                    clusters_mask = None
                    if isinstance(t_thresh, dict):
                        fname = f'GA_{region.name}_{l_freq}_{h_freq}'
                    else:
                        fname = f'GA_{region.name}_{l_freq}_{h_freq}'


                # --------- Plot --------- #
                ga_tf_diff_region = GA_stc_diff.pick(GA_stc_diff.ch_names[0])
                ga_tf_diff_region.data = np.expand_dims(GA_stc_diff_data[i], axis=0)

                fig, ax = plt.subplots(figsize=(10, 7))
                ga_tf_diff_region.plot(axes=ax, mask=clusters_mask_plot, mask_style='contour', show=display_figs)[0]
                ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--', colors='gray')
                fig.suptitle(fname)

                if save_fig:
                    save.fig(fig=fig, path=fig_path_diff, fname=fname)
