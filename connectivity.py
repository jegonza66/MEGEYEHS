import os
import functions_analysis
import functions_general
import mne
import mne.beamformer as beamformer
from paths import paths
import load
import setup
import mne_connectivity
import plot_general
import matplotlib.pyplot as plt
import numpy as np
import itertools
import save
import mne_rsa
from scipy.stats import ttest_ind, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import copy
import pandas as pd


# --------- Define Parameters --------- #
save_fig = True
display_figs = False
plot_individuals = False
save_data = True

# Turn on/off show figures
if display_figs:
    plt.ion()
else:
    plt.ioff()

# Trial selection and filters parameters. A field with 2 values will compute the difference between the conditions specified
trial_params = {'epoch_id': ['tgt_fix_vs_sub', 'it_fix_vs_sub'],  # 'tgt_fix_vs_sub' / 'tgt_fix_vs_sac' / 'sub_vs_sac' / 'tgt_fix_vs_sub_sac'
                'corrans': True,
                'tgtpres': True,
                'mss': None,
                'reject': None,
                'evtdur': None,
                }
run_comparison = True

meg_params = {'band_id': 'Alpha',  # Frequency band (filter sensor space)
              'filter_sensors': True,  # connectivity computation method includes filtering in desired frequency range (Not envelope correlation method)
              'filter_method': 'iir',  # Only for envelope connectivity
              'data_type': 'ICA'
              }

# Source estimation parameters
force_fsaverage = False
# Model
model_name = 'lcmv'
ico = 4
spacing = 10.
# Souce model ('volume'/'surface'/'mixed')
surf_vol = 'surface'
pick_ori = None  # 'vector' For dipoles, 'max_power' for
# Parcelation (aparc / aparc.a2009s)
if surf_vol == 'volume':
    parcelation_segmentation = 'aseg'  # aseg / aparc+aseg / aparc.a2009s+aseg
elif surf_vol == 'surface':
    parcelation_segmentation = 'aparc.a2009s'  # aparc / aparc.a2009s

multiple_comparisons = True

# Connectivity parameters
compute_tmin = None
compute_tmax = None
if surf_vol == 'volume':
    labels_mode = 'mean'
elif surf_vol == 'surface':
    labels_mode = 'pca_flip'
n_top_connections = 150

# Envelope or PLI
envelope_connectivity = False
downsample_ts = True
if envelope_connectivity:
    connectivity_method = 'corr'
    orthogonalization = 'pair'  # 'pair' for pairwise leakage correction / 'sym' for symmetric leakage correction
    desired_sfreq = 120
else:
    connectivity_method = 'pli'
standarize_con = True

if envelope_connectivity:
    meg_params['filter_sensors'] = True  # Just in case

# Surface labels id by region
region_labels_csv = os.path.join(paths().save_path(), 'aparc.a2009s_regions.csv')  # Path to the CSV file containing region mappings
region_labels = {'aparc': {'occipital': ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine'],
                       'parietal': ['postcentral', 'superiorparietal', 'supramarginal', 'inferiorparietal', 'precuneus'],
                       'temporal': ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'transversetemporal', 'fusiform', 'entorhinal', 'parahippocampal', 'temporalpole'],
                       'frontal': ['precentral', 'caudalmiddlefrontal', 'superiorfrontal', 'rostralmiddlefrontal', 'lateralorbitofrontal', 'parstriangularis', 'parsorbitalis', 'parsopercularis', 'medialorbitofrontal', 'paracentral', 'frontalpole'],
                       'insula': ['insula'],
                       'cingulate': ['isthmuscingulate', 'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate']
                           },
                 'aparc.a2009s': functions_general.read_region_labels_csv(csv_path=region_labels_csv, parcellation='aparc.a2009s')
                 }

#----- Setup -----#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir
# Get Source space for default subject
if surf_vol == 'volume':
    fname_src_default = paths().sources_path() + 'fsaverage' + f'/fsaverage_volume_ico{ico}_{int(spacing)}-src.fif'
elif surf_vol == 'surface':
    fname_src_default = paths().sources_path() + 'fsaverage' + f'/fsaverage_surface_ico{ico}-src.fif'
src_default = mne.read_source_spaces(fname_src_default)

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

# Path for envelope or signal connectivity
if envelope_connectivity:
    main_path = 'Connectivity_Env'
    # Modify path if downsample ts
    if downsample_ts:
        downsample_path = f'ds{desired_sfreq}'
    else:
        downsample_path = f'dsFalse'
    final_path = f'{orthogonalization}_{downsample_path}_{labels_mode}_{connectivity_method}'
else:
    main_path = 'Connectivity'
    final_path = f'{labels_mode}_{connectivity_method}'

# Save data of each id
subj_matrices = {}
subj_matrices_no_std = {}
ga_matrices = {}
mean_global_con = {}

# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}

# --------- Run ---------#
for param in param_values.keys():
    mean_global_con[param] = {}
    subj_matrices[param] = {}
    subj_matrices_no_std[param] = {}
    ga_matrices[param] = {}
    for param_value in param_values[param]:
        # Get run parameters from
        run_params = trial_params
        # Set first value of parameters comparisons to avoid having lists in run params
        if len(param_values.keys()) > 1:
            for key in param_values.keys():
                run_params[key] = param_values[key][0]
        # Set comparison key value
        run_params[param] = param_value

        # Define trial duration for VS screen analysis
        if 'vs' in run_params['epoch_id'] and 'fix' not in run_params['epoch_id'] and 'sac' not in run_params['epoch_id']:
            trialdur = vs_dur[run_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
        else:
            trialdur = None

        # Frequencies from band
        fmin, fmax = functions_general.get_freq_band(band_id=meg_params['band_id'])

        # Get time windows from epoch_id name
        map_times = dict(cross1={'tmin': 0, 'tmax': cross1_dur, 'plot_xlim': (None, None)},
                   ms={'tmin': 0, 'tmax': mss_duration[run_params['mss']], 'plot_xlim': (None, None)},
                   cross2={'tmin': 0, 'tmax': cross2_dur, 'plot_xlim': (None, None)},
                   vs={'tmin': 0, 'tmax': vs_dur[run_params['mss']][0], 'plot_xlim': (None, None)})
        tmin, tmax, _ = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'], map=map_times)

        # Get baseline duration for epoch_id
        baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'], tmin=tmin, tmax=tmax,
                                                                          cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                          cross2_dur=cross2_dur, plot_edge=None)

        # Connectivity tmin relative to epochs tmin time (label_ts restarts time to 0)
        if compute_tmin == None:
            con_tmin = 0
        else:
            con_tmin = compute_tmin - tmin
        if compute_tmax == None:
            con_tmax = tmax - tmin
        else:
            con_tmax = compute_tmax - tmin

        # Load experiment info
        exp_info = setup.exp_info()

        # Define Subjects_dir as Freesurfer output folder
        subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
        os.environ["SUBJECTS_DIR"] = subjects_dir

        # Load data paths
        if envelope_connectivity:
            band_path = meg_params['band_id']
        elif not envelope_connectivity:
            band_path = 'None'

        # Run path
        run_path_data = f"Band_{band_path}/{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}" \
                        f"_trialdur{trialdur}_evtdur{run_params['evtdur']}_{tmin}_{tmax}"

        # Epochs path
        epochs_save_path = paths().save_path() + f"Epochs_{meg_params['data_type']}/{run_path_data}_bline{baseline}/"

        # Source plots and data paths
        run_path_plot = run_path_data.replace('Band_None', f"Band_{meg_params['band_id']}") + "/"
        if compute_tmin or compute_tmax:
            run_path_plot = run_path_plot.replace(f"{tmin}_{tmax}", f"{compute_tmin}_{compute_tmax}") # Replace band id for None because Epochs are the same on all bands

        # Source estimation path
        if surf_vol == 'volume':
            source_model_path = f"{model_name}_{surf_vol}_ico{ico}_spacing{spacing}_{pick_ori}"
        elif surf_vol == 'surface':
            source_model_path = f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}"
        labels_model_path = source_model_path + f"_{parcelation_segmentation}_{labels_mode}/"
        label_ts_save_path = paths().save_path() + f"Source_labels_{meg_params['data_type']}/{run_path_data}_bline{baseline}/" + labels_model_path

        # Connectivity matrices plots and save paths
        fig_path = paths().plots_path() + f"{main_path}_{meg_params['data_type']}/" + run_path_plot + source_model_path + f"_{parcelation_segmentation}_{final_path}_std{standarize_con}/"
        save_path = paths().save_path() + f"{main_path}_{meg_params['data_type']}/" + run_path_plot + source_model_path + f"_{parcelation_segmentation}_{final_path}/"

        # Save conectivity matrices
        subj_matrices[param][param_value] = []
        subj_matrices_no_std[param][param_value] = []
        ga_matrices[param][param_value] = []
        mean_global_con[param][param_value] = {'lh': [], 'rh': [], 'global': []}

        # Get parcelation labels and set up connectivity matrix
        if surf_vol == 'surface':  # or surf_vol == 'mixed':
            labels_fname = None
            # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
            fsaverage_labels = mne.read_labels_from_annot(subject='fsaverage', parc=parcelation_segmentation, subjects_dir=subjects_dir)
            # Remove 'unknown' label for fsaverage aparc labels
            if 'aparc' in parcelation_segmentation:
                print("Dropping extra 'unkown' label from lh.")
                drop_idxs = [i for i, label in enumerate(fsaverage_labels) if 'unknown' in label.name.lower()]
                for drop_idx in drop_idxs[::-1]:
                    fsaverage_labels.pop(drop_idx)
            # con_matrix = np.zeros((len(exp_info.subjects_ids), len(fsaverage_labels), len(fsaverage_labels)))

        elif surf_vol == 'volume':
            # fsaverage labels fname
            fsaverage_labels_fname = subjects_dir + f'/fsaverage/mri/{parcelation_segmentation}.mgz'
            # Get bem model
            fname_bem_fsaverage = paths().sources_path() + 'fsaverage' + f'/fsaverage_bem_ico{ico}-sol.fif'
            bem_fsaverage = mne.read_bem_solution(fname_bem_fsaverage)

            # Get labels for FreeSurfer 'aseg' segmentation
            label_names_fsaverage = mne.get_volume_labels_from_aseg(fsaverage_labels_fname, return_colors=False)
            vol_labels_src_fsaverage = mne.setup_volume_source_space(subject='fsaverage', subjects_dir=subjects_dir, bem=bem_fsaverage, pos=spacing,
                                                                     sphere_units='m', add_interpolator=True, volume_label=label_names_fsaverage)
            fsaverage_labels = mne.get_volume_labels_from_src(vol_labels_src_fsaverage, subject='fsaverage', subjects_dir=subjects_dir)

        con_matrix = []
        con_matrix_no_std = []
        # --------- Run ---------#
        for subj_num, subject_code in enumerate(exp_info.subjects_ids):

            # Load subject
            if meg_params['data_type'] == 'ICA':
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            else:
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

            # --------- Coord systems alignment ---------#
            if force_fsaverage:
                subject_code = 'fsaverage'
                fs_subj_path = os.path.join(subjects_dir, subject_code)
                dig = False
            else:
                # Check if subject has MRI data
                try:
                    fs_subj_path = os.path.join(subjects_dir, subject_code)
                    os.listdir(fs_subj_path)
                    dig = True
                except:
                    subject_code = 'fsaverage'
                    fs_subj_path = os.path.join(subjects_dir, subject_code)
                    dig = False

            # --------- Paths ---------#
            # Data filenames
            epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
            labels_ts_data_fname = f'Subject_{subject.subject_id}.pkl'

            # Save figures path
            fig_path_subj = fig_path + f'{subject.subject_id}/'
            # Connectivity data fname
            fname_con = save_path + f'{subject.subject_id}'

            # Source data path
            sources_path_subject = paths().sources_path() + subject.subject_id
            # Load forward model
            if surf_vol == 'volume':
                fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
            elif surf_vol == 'surface':
                fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
            fwd = mne.read_forward_solution(fname_fwd)
            # Get sources from forward model
            src = fwd['src']
            # Get bem model
            fname_bem = paths().sources_path() + subject_code + f'/{subject_code}_bem_ico{ico}-sol.fif'
            bem = mne.read_bem_solution(fname_bem)

            # Parcellation labels
            if surf_vol == 'volume':
                labels_fname = subjects_dir + f'/{subject_code}/mri/{parcelation_segmentation}.mgz'
                # Get labels for FreeSurfer 'aseg' segmentation
                label_names = mne.get_volume_labels_from_aseg(labels_fname, return_colors=False)
                vol_labels_src = mne.setup_volume_source_space(subject=subject_code, subjects_dir=subjects_dir, bem=bem, pos=spacing,
                                                               sphere_units='m', add_interpolator=True, volume_label=label_names)
                labels = mne.get_volume_labels_from_src(vol_labels_src, subject=subject_code, subjects_dir=subjects_dir)

                label_names_segmentation = []
                for label in labels:
                    if 'rh' in label.name and 'ctx' not in label.name:
                        label_name = 'Right-' + label.name.replace('-rh', '')
                    elif 'lh' in label.name and 'ctx' not in label.name:
                        label_name = 'Left-' + label.name.replace('-lh', '')
                    else:
                        label_name = label.name
                    label_names_segmentation.append(label_name)
            elif subject_code != 'fsaverage':
                # Get labels for FreeSurfer cortical parcellation
                labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation_segmentation, subjects_dir=subjects_dir)

                print("Dropping extra 'unkown' label from lh.")
                drop_idxs = [i for i, label in enumerate(labels) if 'unknown' in label.name.lower()]
                for drop_idx in drop_idxs[::-1]:
                    labels.pop(drop_idx)

                label_names_segmentation = None

            else:
                labels = fsaverage_labels
                label_names_segmentation = None

            # Load connectivity matrix
            if os.path.isfile(fname_con):
                con_subj = mne_connectivity.read_connectivity(fname_con)

            else:
                # Load labels ts data
                if os.path.isfile(label_ts_save_path + labels_ts_data_fname):
                    label_ts = load.var(file_path=label_ts_save_path + labels_ts_data_fname)
                    try:
                        # Use defined sfreq from pprevious ppt
                        sfreq = meg_data.info['sfreq']
                    except:
                        meg_data = load.meg(subject=subject, meg_params=meg_params)
                        sfreq = meg_data.info['sfreq']
                else:
                    # Load epochs data
                    if os.path.isfile(epochs_save_path + epochs_data_fname):
                        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                    elif 'sub' in run_params['epoch_id']:
                        if os.path.isfile(epochs_save_path.replace(f"Band_{band_path}", "Band_None") + epochs_data_fname):
                            epochs = mne.read_epochs(epochs_save_path.replace(f"Band_{band_path}", "Band_None") + epochs_data_fname)

                            # Filter data
                            l_freq, h_freq = functions_general.get_freq_band(meg_params['band_id'])
                            epochs.filter(l_freq=l_freq, h_freq=h_freq, method='iir')
                        else:
                            raise ValueError(f'No saved subsampled epochs found in {epochs_save_path.replace(f"Band_{band_path}", "Band_None")}')
                    else:
                        # Load MEG data
                        meg_data = load.meg(subject=subject, meg_params=meg_params)

                        # Epoch data
                        # Load data as is
                        epochs, events = functions_analysis.epoch_data(subject=subject, mss=run_params['mss'], corr_ans=run_params['corrans'], tgt_pres=run_params['tgtpres'],
                                                                       epoch_id=run_params['epoch_id'], meg_data=meg_data, trial_dur=trialdur,
                                                                       tmin=tmin, tmax=tmax, baseline=baseline, reject=run_params['reject'],
                                                                       save_data=save_data, epochs_save_path=epochs_save_path,
                                                                       epochs_data_fname=epochs_data_fname)

                    # Pick meg channels for source modeling
                    epochs.pick('meg')

                    # Extract sfreq
                    sfreq = epochs.info['sfreq']

                    # --------- Source estimation ---------#
                    # Load filter
                    fname_filter = paths().filter_path(subject=subject, subject_code=subject_code, ico=ico, spacing=spacing, surf_vol=surf_vol, pick_ori=pick_ori,
                                                       model_name=model_name)
                    filters = mne.beamformer.read_beamformer(fname_filter)

                    # Apply filter and get source estimates
                    stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

                    # Average the source estimates within each label using sign-flips to reduce signal cancellations
                    if surf_vol == 'volume':
                        label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=(labels_fname, label_names_segmentation), src=src, mode=labels_mode,
                                                                 return_generator=False)
                    elif surf_vol == 'surface':
                        label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=labels, src=src, mode=labels_mode, return_generator=False)

                    # Save
                    if save_data:
                        os.makedirs(label_ts_save_path, exist_ok=True)
                        save.var(var=label_ts, path=label_ts_save_path, fname=labels_ts_data_fname)

                if envelope_connectivity:
                    if downsample_ts:
                        for i, ts in enumerate(label_ts):
                            samples_interval = int(sfreq/desired_sfreq)
                            # Taking jumping windows average of samples
                            label_ts[i] = np.array([np.mean(ts[:, j*samples_interval:(j+1)*samples_interval], axis=-1) for j in range(int(len(ts[0])/samples_interval) + 1)]).T
                            # Subsampling
                            # label_ts[i] = ts[:, ::samples_interval]

                    # Compute envelope connectivity (automatically computes hilbert transform to extract envelope)
                    if orthogonalization == 'pair':
                        con_subj = mne_connectivity.envelope_correlation(data=label_ts, names=[label.name for label in labels])

                    elif orthogonalization == ' sym':
                        label_ts_orth = mne_connectivity.envelope.symmetric_orth(label_ts)
                        con_subj = mne_connectivity.envelope_correlation(label_ts_orth, orthogonalize=False)
                        # Take absolute value of correlations (orthogonalize False does not take abs by default)
                        con_subj.xarray.data = abs(con_subj.get_data())

                    # Average across epochs
                    con_subj = con_subj.combine()

                else:
                    con_subj = mne_connectivity.spectral_connectivity_epochs(label_ts, method=connectivity_method, mode='multitaper', sfreq=sfreq,
                                                                        fmin=fmin, fmax=fmax, tmin=con_tmin, tmax=con_tmax, faverage=True, mt_adaptive=True)
                # Save
                if save_data:
                    os.makedirs(save_path, exist_ok=True)
                    con_subj.save(fname_con)

            # Get connectivity matrix
            con_subj_data = con_subj.get_data(output='dense')[:, :, 0]
            con_subj_data = np.maximum(con_subj_data, con_subj_data.transpose())  # make symetric

            # Save for comparisons
            con_matrix_no_std.append(con_subj_data)

            # Identify left and right hemisphere indices
            lh_indices = [i for i, label in enumerate(fsaverage_labels) if label.hemi == 'lh']
            rh_indices = [i for i, label in enumerate(fsaverage_labels) if label.hemi == 'rh']

            # Compute mean connectivity within each hemisphere (excluding diagonal)
            lh_matrix = con_subj_data[np.ix_(lh_indices, lh_indices)]
            rh_matrix = con_subj_data[np.ix_(rh_indices, rh_indices)]

            # Remove diagonal elements for hemisphere calculations
            lh_matrix_no_diag = lh_matrix.copy()
            np.fill_diagonal(lh_matrix_no_diag, 0)
            rh_matrix_no_diag = rh_matrix.copy()
            np.fill_diagonal(rh_matrix_no_diag, 0)

            # Global matrix without diagonal
            global_matrix_no_diag = con_subj_data.copy()
            np.fill_diagonal(global_matrix_no_diag, 0)

            # Get top N connections for each hemisphere and global
            if n_top_connections is not None:
                lh_top_values = np.sort(lh_matrix_no_diag.flatten())[-int(n_top_connections/2):]
                rh_top_values = np.sort(rh_matrix_no_diag.flatten())[-int(n_top_connections/2):]
                global_top_values = np.sort(global_matrix_no_diag.flatten())[-int(n_top_connections/2):]
            else:
                # Use all non-zero connections
                lh_top_values = lh_matrix_no_diag.flatten()
                rh_top_values = rh_matrix_no_diag.flatten()
                global_top_values = global_matrix_no_diag.flatten()

            # Append subjects global connectivity using top N connections
            mean_global_con[param][param_value]['global'].append(global_top_values.mean())
            mean_global_con[param][param_value]['lh'].append(lh_top_values.mean())
            mean_global_con[param][param_value]['rh'].append(rh_top_values.mean())

            # Standarize
            if standarize_con:
                con_subj_data = copy.copy((con_subj_data - np.mean(con_subj_data)) / np.std(con_subj_data))

            # Save for GA
            con_matrix.append(con_subj_data)

            if plot_individuals:
                # Plot circle
                plot_general.connectivity_circle(subject=subject, labels=labels, surf_vol=surf_vol, con=con_subj_data, connectivity_method=connectivity_method,
                                                 region_labels=region_labels, parcellation=parcelation_segmentation, n_lines=n_top_connections,
                                                 subject_code=subject_code, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_subj, fname=None)

                # Plot connectome
                plot_general.connectome(subject=subject, labels=labels, adjacency_matrix=con_subj_data, subject_code=subject_code, connections_num=n_top_connections,
                                        save_fig=save_fig, fig_path=fig_path_subj, fname=None)

                # Plot connectivity matrix
                plot_con_subj_data = con_subj_data.copy()
                np.fill_diagonal(plot_con_subj_data, 0)
                plot_general.plot_con_matrix(subject=subject, labels=labels, adjacency_matrix=plot_con_subj_data, region_labels=region_labels, parcellation=parcelation_segmentation,
                                             save_fig=save_fig, fig_path=fig_path_subj, fname=None, n_ticks=5)

                # Plot connectivity strength (connections from each region to other regions)
                plot_general.connectivity_strength(subject=subject, subject_code=subject_code, con=con_subj, src=src, labels=labels, surf_vol=surf_vol,
                                                   labels_fname=labels_fname, label_names_segmentation=label_names_segmentation,
                                                   subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_subj, fname=None)

        # --------- Grand Average ---------#
        # Get connectivity matrix for GA
        ga_con_matrix = np.array(con_matrix).mean(0)
        # Fill diagonal with 0
        np.fill_diagonal(ga_con_matrix, 0)

        # Plot circle
        plot_general.connectivity_circle(subject='GA', labels=fsaverage_labels, surf_vol=surf_vol, con=ga_con_matrix, connectivity_method=connectivity_method, subject_code='fsaverage',
                                         region_labels=region_labels, parcellation=parcelation_segmentation, n_lines=n_top_connections,
                                         display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname='GA_circle')

        # Plot connectome
        plot_general.connectome(subject='GA', labels=fsaverage_labels, adjacency_matrix=ga_con_matrix, connections_num=n_top_connections, subject_code='fsaverage',
                                save_fig=save_fig, fig_path=fig_path, fname='GA_connectome')

        # Plot matrix
        ga_sorted_matrix = plot_general.plot_con_matrix(subject='GA', labels=fsaverage_labels, adjacency_matrix=ga_con_matrix, region_labels=region_labels, parcellation=parcelation_segmentation,
                                                        save_fig=save_fig, fig_path=fig_path, fname='GA_matrix')

        # Plot connectivity strength (connections from each region to other regions)
        plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=ga_con_matrix, src=src, labels=fsaverage_labels, surf_vol=surf_vol,
                                           labels_fname=labels_fname, label_names_segmentation=label_names_segmentation,
                                           subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path, fname='GA_strength')

        # Plot global connectivity
        data_dict = {
            param_value: {'lh': mean_global_con[param][param_value]['lh'], 'rh': mean_global_con[param][param_value]['rh'],
                            'global': mean_global_con[param][param_value]['global']}}
        plot_general.global_connectivity(data_dict=data_dict, categories=[param_value], save_fig=save_fig, fig_path=fig_path, n_lines=n_top_connections)

        # Get connectivity matrices for comparisson
        subj_matrices[param][param_value] = np.array(con_matrix)
        subj_matrices_no_std[param][param_value] = np.array(con_matrix_no_std)
        ga_matrices[param][param_value] = ga_sorted_matrix

# ----- Difference between conditions ----- #
# Take difference of conditions if applies
for param in param_values.keys():
    if len(param_values[param]) > 1 and run_comparison:
        for comparison in list(itertools.combinations(param_values[param], 2)):
            if all(type(element) == int for element in comparison):
                comparison = sorted(comparison, reverse=True)

            # Redefine figure save path
            if param == 'epoch_id':
                fig_path_diff = fig_path.replace(f'{param_values[param][-1]}', f'{comparison[0]}-{comparison[1]}')
            else:
                fig_path_diff = fig_path.replace(f'{param}{param_values[param][-1]}', f'{param}{comparison[0]}-{comparison[1]}')

            print(f'Comparing conditions {param} {comparison[0]} - {comparison[1]}')

            #------ RSA ------#
            # Compute RSA between GA matrices from both conditions
            rsa_result = mne_rsa.rsa(ga_matrices[param][comparison[0]], ga_matrices[param][comparison[1]], metric="spearman")
            # Plot Connectivity matrices from both conditions
            fig = mne_rsa.plot_rdms([ga_matrices[param][comparison[0]], ga_matrices[param][comparison[1]]], names=[comparison[0], comparison[1]])
            fig.suptitle(f'RSA: {round(rsa_result, 2)}')

            # Save
            if save_fig:
                fname = f'GA_rsa'
                save.fig(fig=fig, path=fig_path_diff, fname=fname)

            #------ t-test ------#
            # Connectivity t-values variable
            t_values, p_values = wilcoxon(x=subj_matrices[param][comparison[0]], y=subj_matrices[param][comparison[1]], axis=0)

            # Significance thresholds
            p_threshold = 0.05
            ravel_p_values = p_values.ravel()

            # Make 1D arrays to run FDR correction
            if multiple_comparisons:
                rejected, corrected_pval = fdrcorrection(pvals=ravel_p_values, alpha=p_threshold)  # rejected refers to null hypothesis
            else:
                rejected = ravel_p_values < p_threshold
                corrected_pval = ravel_p_values

            # Reshape to regions x regions array
            corrected_pval = np.reshape(corrected_pval, newshape=p_values.shape)
            rejected = np.reshape(rejected, newshape=p_values.shape)

            # Take significant links (in case asymetric results)
            rejected = np.maximum(rejected, rejected.transpose())

            # Discard diagonal
            np.fill_diagonal(rejected, False)

            # Mask p-values by significance
            corrected_pval[~rejected.astype(bool)] = 1
            log_p_values = -np.log10(corrected_pval)

            # Mask t-values by significance
            t_values[~rejected] = 0

            # Plot significant t-values
            if t_values.any() > 0:
                min_value = sorted(set(np.sort(t_values, axis=None)))[0]/2
                max_value = sorted(set(np.sort(t_values, axis=None)))[-1]

                # Plot circle
                plot_general.connectivity_circle(subject='GA', labels=fsaverage_labels, surf_vol=surf_vol, con=t_values, connectivity_method=connectivity_method, vmin=min_value,
                                                 region_labels=region_labels, parcellation=parcelation_segmentation, colormap='coolwarm', n_lines=n_top_connections,
                                                 vmax=max_value, subject_code='fsaverage', display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff,
                                                 fname='GA_circle_t')

                # Plot p-values connectome
                plot_general.connectome(subject='GA', labels=fsaverage_labels, adjacency_matrix=t_values, subject_code='fsaverage',
                                        save_fig=save_fig, fig_path=fig_path_diff, fname=f'GA_t_con', connections_num=(log_p_values > 0).sum())

                # Plot matrix
                plot_general.plot_con_matrix(subject='GA', labels=fsaverage_labels, adjacency_matrix=t_values, region_labels=region_labels, parcellation=parcelation_segmentation,
                                             save_fig=save_fig, fig_path=fig_path_diff, fname='GA_matrix_t')

                # Plot connectivity strength (connections from each region to other regions)
                plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=t_values, src=src, labels=fsaverage_labels, surf_vol=surf_vol,
                                                   labels_fname=labels_fname, label_names_segmentation=label_names_segmentation,
                                                   subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_diff, fname=f'GA_strength_t')

            #----- Difference -----#
            con_diff_list = []
            mean_global_con_diff = {'lh': [], 'rh': [], 'global': []}  # Global mean connectivity difference
            # Compute difference
            for i in range(len(subj_matrices_no_std[param][comparison[0]])):

                # Matrix differences
                subj_dif = subj_matrices_no_std[param][comparison[0]][i] - subj_matrices_no_std[param][comparison[1]][i]
                if standarize_con:
                    subj_dif = (subj_dif - np.mean(subj_dif)) / np.std(subj_dif)
                con_diff_list.append(subj_dif)

            # Make array
            con_diff = np.array(con_diff_list)

            # Take Grand Average of connectivity differences
            con_diff_ga = con_diff.mean(0)

            # Fill diagonal with 0
            np.fill_diagonal(con_diff_ga, 0)

            # Get indices of top N absolute values
            con_diff_ga_abs = np.abs(con_diff_ga)
            top_indices = np.unravel_index(np.argpartition(con_diff_ga_abs.ravel(), -n_top_connections)[-n_top_connections:], con_diff_ga_abs.shape)

            # Compute difference
            for i in range(len(subj_matrices_no_std[param][comparison[0]])):

                # Get absolute values for ranking but use original values for averaging
                subj_dif = con_diff[i]
                subj_dif_abs = np.abs(subj_dif)
                np.fill_diagonal(subj_dif_abs, 0)  # Remove diagonal

                # Get top N connections globally based on absolute values
                if n_top_connections is not None:

                    # Split top connections by hemisphere
                    lh_top_mask = np.zeros_like(subj_dif, dtype=bool)
                    rh_top_mask = np.zeros_like(subj_dif, dtype=bool)
                    global_top_mask = np.zeros_like(subj_dif, dtype=bool)

                    for idx in range(len(top_indices[0])):
                        i, j = top_indices[0][idx], top_indices[1][idx]
                        global_top_mask[i, j] = True

                        # Check if both indices are in left hemisphere
                        if i in lh_indices and j in lh_indices:
                            lh_top_mask[i, j] = True
                        # Check if both indices are in right hemisphere
                        elif i in rh_indices and j in rh_indices:
                            rh_top_mask[i, j] = True

                    # Get original (non-absolute) values for the top connections
                    lh_diff_values = subj_dif[lh_top_mask]
                    rh_diff_values = subj_dif[rh_top_mask]
                    global_diff_values = subj_dif[global_top_mask]

                    # Compute means (handle empty arrays)
                    lh_mean = lh_diff_values.mean() if len(lh_diff_values) > 0 else 0
                    rh_mean = rh_diff_values.mean() if len(rh_diff_values) > 0 else 0
                    global_mean = global_diff_values.mean()

                else:
                    # Use all non-diagonal connections
                    lh_matrix_diff = subj_dif[np.ix_(lh_indices, lh_indices)]
                    rh_matrix_diff = subj_dif[np.ix_(rh_indices, rh_indices)]

                    np.fill_diagonal(lh_matrix_diff, 0)
                    np.fill_diagonal(rh_matrix_diff, 0)

                    lh_mean = lh_matrix_diff.mean()
                    rh_mean = rh_matrix_diff.mean()
                    global_mean = subj_dif[~np.eye(subj_dif.shape[0], dtype=bool)].mean()

                # Save to dictionary
                mean_global_con_diff['lh'].append(lh_mean)
                mean_global_con_diff['rh'].append(rh_mean)
                mean_global_con_diff['global'].append(global_mean)

            # Plot circle
            plot_general.connectivity_circle(subject='GA', labels=fsaverage_labels, surf_vol=surf_vol, con=con_diff_ga, connectivity_method=connectivity_method, subject_code='fsaverage',
                                             region_labels=region_labels, parcellation=parcelation_segmentation, colormap='coolwarm', n_lines=n_top_connections,
                                             display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_circle_dif')

            # Plot connectome
            plot_general.connectome(subject='GA', labels=fsaverage_labels, adjacency_matrix=con_diff_ga, connections_num=n_top_connections, subject_code='fsaverage',
                                    edge_thresholddirection='absabove', save_fig=save_fig, fig_path=fig_path_diff, fname='GA_connectome_dif')

            # Plot matrix
            plot_general.plot_con_matrix(subject='GA', labels=fsaverage_labels, adjacency_matrix=con_diff_ga, region_labels=region_labels, parcellation=parcelation_segmentation,
                                         save_fig=save_fig, fig_path=fig_path_diff, fname='GA_matrix_dif')

            # Plot connectivity strength (connections from each region to other regions)
            plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=con_diff_ga, src=src, labels=fsaverage_labels, surf_vol=surf_vol,
                                               labels_fname=labels_fname, label_names_segmentation=label_names_segmentation,
                                               subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_strength_dif')

            # Boxplot of mean global connectivity and their difference
            data_dict = {
                'diff': {'lh': mean_global_con_diff['lh'], 'rh': mean_global_con_diff['rh'], 'global': mean_global_con_diff['global']}
            }
            categories = [key for key in data_dict.keys()]
            plot_general.global_connectivity(data_dict=data_dict, categories=categories, save_fig=save_fig, fig_path=fig_path_diff, n_lines=n_top_connections)

            # Compute regional connectivity averages
            regional_conn_averages = functions_analysis.compute_regional_connectivity_averages(
                con=con_diff_ga,
                labels=fsaverage_labels,
                region_labels=region_labels,
                parcellation=parcelation_segmentation,
                n_connections=None
            )
