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


# --------- Define Parameters ---------#
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
trial_params = {'epoch_id': ['vs', 'cross1'],
                'corrans': None,
                'tgtpres': None,
                'mss': None,
                'reject': None,
                'evtdur': None,
                }

run_comparison = True

use_ica_data = True
band_id = 'Theta'  # Frequency band (filter sensor space)
filter_method = 'iir'  # Only for envelope connectivity

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
parcelation = 'aparc'

# Connectivity parameters
envelope_connectivity = False
downsample_ts = False
if envelope_connectivity:
    connectivity_method = 'corr'
    orthogonalization = 'pair'  # 'pair' for pairwise leakage correction / 'sym' for symmetric leakage correction
    desired_sfreq = 10
else:
    connectivity_method = 'pli'
standarize_con = True

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

# Save data of each id
subj_matrices = {}
ga_matrices = {}

# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}

# --------- Run ---------#
for param in param_values.keys():
    subj_matrices[param] = {}
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
            trialdur = vs_dur[run_params['mss']]

        # Frequencies from band
        fmin, fmax = functions_general.get_freq_band(band_id=band_id)

        # Get time windows from epoch_id name
        map = dict(cross1={'tmin': 0, 'tmax': cross1_dur, 'plot_xlim': (None, None)},
                   ms={'tmin': 0, 'tmax': mss_duration[run_params['mss']], 'plot_xlim': (None, None)},
                   cross2={'tmin': 0, 'tmax': cross2_dur, 'plot_xlim': (None, None)},
                   vs={'tmin': 0, 'tmax': vs_dur[run_params['mss']][0], 'plot_xlim': (None, None)})

        tmin, tmax, _ = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'], map=map)

        # Get baseline duration for epoch_id
        baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'], tmin=tmin, tmax=tmax,
                                                                          cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                          cross2_dur=cross2_dur, plot_edge=None)

        # Load experiment info
        exp_info = setup.exp_info()

        # Define Subjects_dir as Freesurfer output folder
        subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
        os.environ["SUBJECTS_DIR"] = subjects_dir

        # Data type
        if use_ica_data:
            data_type = 'ICA'
        else:
            data_type = 'RAW'

        # Load data paths
        if envelope_connectivity:
            band_path = band_id
        elif not envelope_connectivity:
            band_path = 'None'

        run_path_data = f"Band_{band_path}/{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}" \
                        f"_trialdur{trialdur}_evtdur{run_params['evtdur']}_{tmin}_{tmax}"

        epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path_data + f'_bline{baseline}/'

        # Source plots and data paths
        run_path_plot = run_path_data.replace('Band_None', f'Band_{band_id}')  # Replace band id for None because Epochs are the same on all bands

        # Set path for envelope or signal connectivity
        if envelope_connectivity:
            main_path = 'Connectivity_Env'
            # Modify path if downsample ts
            if downsample_ts:
                downsample_path = f'ds{desired_sfreq}'
            else:
                downsample_path = f'dsFalse'
            final_path = f'{orthogonalization}_{downsample_path}_{connectivity_method}'
        else:
            main_path = 'Connectivity'
            final_path = f'{connectivity_method}'

        if surf_vol == 'volume':
            fig_path = paths().plots_path() + f'{main_path}_{data_type}/' + run_path_plot + \
                       f'/{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}_{parcelation}_{final_path}_std{standarize_con}/'
            save_path = paths().save_path() + f'{main_path}_{data_type}/' + run_path_plot + \
                        f'/{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}_{parcelation}_{final_path}/'
        elif surf_vol == 'surface':
            fig_path = paths().plots_path() + f'{main_path}_{data_type}/' + run_path_plot + \
                       f'/{model_name}_{surf_vol}_ico{ico}_{pick_ori}_{parcelation}_{final_path}_std{standarize_con}/'
            save_path = paths().save_path() + f'{main_path}_{data_type}/' + run_path_plot + \
                        f'/{model_name}_{surf_vol}_ico{ico}_{pick_ori}_{parcelation}_{final_path}/'

        # Save conectivity matrices
        subj_matrices[param][param_value] = []
        ga_matrices[param][param_value] = []

        # Get parcelation labels and set up connectivity matrix
        if surf_vol == 'surface':  # or surf_vol == 'mixed':
            # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
            fsaverage_labels = mne.read_labels_from_annot(subject='fsaverage', parc=parcelation, subjects_dir=subjects_dir)
            # Remove 'unknown' label for fsaverage aparc labels
            if parcelation == 'aparc':
                print("Dropping extra 'unkown' label from lh.")
                drop_idxs = [i for i, label in enumerate(fsaverage_labels) if 'unknown' in label.name]
                for drop_idx in drop_idxs:
                    fsaverage_labels.pop(drop_idx)
            con_matrix = np.zeros((len(exp_info.subjects_ids), len(fsaverage_labels), len(fsaverage_labels)))

        if surf_vol == 'volume':
            labels_fname = subjects_dir + f'/fsaverage/mri/aparc+aseg.mgz'
            fsaverage_labels = mne.get_volume_labels_from_aseg(labels_fname, return_colors=True)
            # Drop extra labels in fsaverage
            drop_idxs = [i for i, label in enumerate(fsaverage_labels[0]) if (label == 'ctx-lh-corpuscallosum' or label == 'ctx-rh-corpuscallosum')]
            for drop_idx in drop_idxs:
                fsaverage_labels[0].pop(drop_idx)
                fsaverage_labels[1].pop(drop_idx)
            con_matrix = np.zeros((len(exp_info.subjects_ids), len(fsaverage_labels[0]), len(fsaverage_labels[0])))


        # --------- Run ---------#
        for subj_num, subject_code in enumerate(exp_info.subjects_ids):
            # Load subject
            if use_ica_data:
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

            try:
                # Load connectivity matrix
                con = mne_connectivity.read_connectivity(fname_con)

                # Parcellation labels
                if surf_vol == 'volume':
                    labels = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
                elif subject_code != 'fsaverage':
                    # Get labels for FreeSurfer cortical parcellation
                    labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
                else:
                    labels = fsaverage_labels

            except:
                try:
                    # Load data
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                except:
                    if use_ica_data:
                        if band_id and envelope_connectivity:
                            meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=save_data, method=filter_method)
                        else:
                            meg_data = load.ica_data(subject=subject)
                    else:
                        if band_id and envelope_connectivity:
                            meg_data = load.filtered_data(subject=subject, band_id=band_id, use_ica_data=False, save_data=save_data, method=filter_method)
                        else:
                            meg_data = subject.load_preproc_meg_data()
                    try:
                        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                    except:
                        # Epoch data
                        epochs, events = functions_analysis.epoch_data(subject=subject, mss=run_params['mss'], corr_ans=run_params['corrans'], tgt_pres=run_params['tgtpres'],
                                                                       epoch_id=run_params['epoch_id'], meg_data=meg_data, trial_dur=trialdur,
                                                                       tmin=tmin, tmax=tmax, baseline=baseline, reject=run_params['reject'],
                                                                       save_data=save_data, epochs_save_path=epochs_save_path,
                                                                       epochs_data_fname=epochs_data_fname)

                # Pick meg channels for source modeling
                epochs.pick('meg')

                # --------- Source estimation ---------#
                # Load filter
                if surf_vol == 'volume':
                    fname_filter = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
                elif surf_vol == 'surface':
                    fname_filter = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}-{model_name}.fif'
                filters = mne.beamformer.read_beamformer(fname_filter)

                # Apply filter and get source estimates
                stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

                # --------- Connectivity ---------#
                if surf_vol == 'volume':
                    labels = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
                elif subject_code != 'fsaverage':
                    # Get labels for FreeSurfer cortical parcellation
                    labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
                else:
                    labels = fsaverage_labels

                # Average the source estimates within each label using sign-flips to reduce signal cancellations
                label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=labels, src=src, mode='auto', return_generator=False)

                if envelope_connectivity:
                    if downsample_ts:
                        for i, ts in enumerate(label_ts):
                            sfreq = epochs.info['sfreq']
                            samples_interval = int(sfreq/desired_sfreq)
                            # Taking jumping windows average of samples
                            label_ts[i] = np.array([np.mean(ts[:, j*samples_interval:(j+1)*samples_interval], axis=-1) for j in range(int(len(ts[0])/samples_interval) + 1)]).T
                            # Subsampling
                            # label_ts[i] = ts[:, ::samples_interval]

                    # Compute envelope connectivity (automatically computes hilbert transform to extract envelope)
                    if orthogonalization == 'pair':
                        label_names = [label.name for label in labels]
                        con = mne_connectivity.envelope_correlation(data=label_ts, names=label_names)

                    elif orthogonalization == ' sym':
                        label_ts_orth = mne_connectivity.envelope.symmetric_orth(label_ts)
                        con = mne_connectivity.envelope_correlation(label_ts_orth, orthogonalize=False)
                        # Take absolute value of correlations (orthogonalize False does not take abs by default)
                        con.xarray.data = abs(con.get_data())

                    # Average across epochs
                    con = con.combine()

                else:
                    con = mne_connectivity.spectral_connectivity_epochs(label_ts, method=connectivity_method, mode='multitaper', sfreq=epochs.info['sfreq'],
                                                                        fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, faverage=True, mt_adaptive=True)
                # Save
                if save_data:
                    os.makedirs(save_path, exist_ok=True)
                    con.save(fname_con)

            # Get connectivity matrix
            con_subj = con.get_data(output='dense')[:, :, 0]
            con_subj = np.maximum(con_subj, con_subj.transpose())  # make symetric

            # Standarize
            if standarize_con:
                con_subj = (con_subj - np.mean(con_subj)) / np.std(con_subj)

            # Save for GA
            con_matrix[subj_num] = con_subj

            if plot_individuals:
                # Plot circle
                plot_general.connectivity_circle(subject=subject, labels=labels, surf_vol=surf_vol, con=con_matrix[subj_num], connectivity_method=connectivity_method,
                                                 subject_code=subject_code, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_subj, fname=None)

                # Plot connectome
                plot_general.connectome(subject=subject, labels=labels, adjacency_matrix=con_matrix[subj_num], subject_code=subject_code,
                                        save_fig=save_fig, fig_path=fig_path_subj, fname=None)

                # Plot connectivity matrix
                plot_general.plot_con_matrix(subject=subject, labels=labels, adjacency_matrix=con_matrix[subj_num], subject_code=subject_code,
                                             save_fig=save_fig, fig_path=fig_path_subj, fname=None)

                # Plot connectivity strength (connections from each region to other regions)
                plot_general.connectivity_strength(subject=subject, subject_code=subject_code, con=con, src=src, labels=labels, surf_vol=surf_vol,
                                                   subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_subj, fname=None)

        # --------- Grand Average ---------#
        # Get connectivity matrix for GA
        ga_con_matrix = con_matrix.mean(0)
        # Fill diagonal with 0
        np.fill_diagonal(ga_con_matrix, 0)

        # Plot circle
        plot_general.connectivity_circle(subject='GA', labels=labels, surf_vol=surf_vol, con=ga_con_matrix, connectivity_method=connectivity_method, subject_code='fsaverage',
                                         display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname='GA_circle')

        # Plot connectome
        plot_general.connectome(subject='GA', labels=labels, adjacency_matrix=ga_con_matrix, subject_code='fsaverage', save_fig=save_fig, fig_path=fig_path,
                                fname='GA_connectome')

        # Plot matrix
        ga_sorted_matrix = plot_general.plot_con_matrix(subject='GA', labels=labels, adjacency_matrix=ga_con_matrix, subject_code='fsaverage',
                                                              save_fig=save_fig, fig_path=fig_path, fname='GA_matrix')

        # Plot connectivity strength (connections from each region to other regions)
        plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=ga_con_matrix, src=src, labels=labels, surf_vol=surf_vol,
                                               subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path, fname='GA_strength')

        # Get connectivity matrices for comparisson
        subj_matrices[param][param_value] = con_matrix
        ga_matrices[param][param_value] = ga_sorted_matrix


# ----- Difference between conditions ----- #
# Take difference of conditions if applies
for param in param_values.keys():
    if len(param_values[param]) > 1 and run_comparison:
        for comparison in list(itertools.combinations(param_values[param], 2)):

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

            # Make 1D arrays to run FDR correction
            ravel_p_values = p_values.ravel()
            rejected, corrected_pval = fdrcorrection(pvals=ravel_p_values, alpha=p_threshold)

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

                # Plot matrix
                plot_general.plot_con_matrix(subject='GA', labels=labels, adjacency_matrix=t_values, subject_code='fsaverage',
                                             save_fig=save_fig, fig_path=fig_path_diff, fname='GA_matrix_t')

                # Plot circle
                plot_general.connectivity_circle(subject='GA', labels=labels, surf_vol=surf_vol, con=t_values, connectivity_method=connectivity_method, vmin=min_value,
                                                 vmax=max_value, subject_code='fsaverage', display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_circle_t')

                # Plot p-values connectome
                plot_general.connectome(subject='GA', labels=labels, adjacency_matrix=t_values, subject_code='fsaverage', save_fig=save_fig, fig_path=fig_path_diff,
                                        fname=f'GA_t_con', cmap='Reds', edge_vmin=min_value, edge_vmax=max_value, connections_num=(log_p_values > 0).sum())

                # Plot connectivity strength (connections from each region to other regions)
                plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=t_values, src=src, labels=labels, surf_vol=surf_vol,
                                                   subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_diff, fname=f'GA_strength_t')

            #----- Difference -----#
            con_diff = []
            # Compute difference for cross2
            for i in range(len(subj_matrices[param][comparison[0]])):
                con_diff.append(subj_matrices[param][comparison[0]][i] - subj_matrices[param][comparison[1]][i])

            # Make array
            con_diff = np.array(con_diff)

            # Take Grand Average of connectivity differences
            con_diff_ga = con_diff.mean(0)

            # Fill diagonal with 0
            np.fill_diagonal(con_diff_ga, 0)

            # Plot circle
            plot_general.connectivity_circle(subject='GA', labels=labels, surf_vol=surf_vol, con=con_diff_ga, connectivity_method=connectivity_method, subject_code='fsaverage',
                                             display_figs=display_figs, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_circle_dif')

            # Plot connectome
            plot_general.connectome(subject='GA', labels=labels, adjacency_matrix=con_diff_ga, subject_code='fsaverage',
                                    save_fig=save_fig, fig_path=fig_path_diff, fname='GA_connectome_dif')

            # Plot matrix
            plot_general.plot_con_matrix(subject='GA', labels=labels, adjacency_matrix=con_diff_ga, subject_code='fsaverage',
                                         save_fig=save_fig, fig_path=fig_path_diff, fname='GA_matrix_dif')

            # Plot connectivity strength (connections from each region to other regions)
            plot_general.connectivity_strength(subject='GA', subject_code='fsaverage', con=con_diff_ga, src=src, labels=labels, surf_vol=surf_vol,
                                               subjects_dir=subjects_dir, save_fig=save_fig, fig_path=fig_path_diff, fname='GA_strength_dif')