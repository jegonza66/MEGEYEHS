## Surface Time-Frequency
import os
import shutil
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
import itertools
import scipy
import pandas as pd


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
trial_params = {'epoch_id': 'tgt_fix_ms_subsampled',  # use'+' to mix conditions (red+blue)
                'corrans': [True, False],
                'tgtpres': True,
                'mss': None,
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evtdur': None,
                'rel_sac': 'prev'}

meg_params = {'regions_id': 'all',
              'band_id': None,
              'filter_sensors': None,
              'filter_method': 'iir',
              'data_type': 'ICA'
              }

# Define active time limits
active_times = None

# Get TF frequency limits
if meg_params['band_id'] is None:
    l_freq, h_freq = (1, 40)
else:
    l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])

# Compare features
run_comparison = True

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'surface'
labels_mode = 'pca_flip'
ico = 4
spacing = 5.  # Only for volume source estimation
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
source_estimation = 'epo'
estimate_source_tf = True
visualize_alignment = False
parcelation='aparc'

# Time freqcuency computation
tf_output = 'phase'  # 'phase' / 'avg_power'
n_cycles_div = 4.
# Baseline
bline_mode_subj = False  # 'db'
plot_edge = 0.15

# Plot
plot_individuals = False
plot_ga = False
fontsize = 22
params = {'font.size': fontsize}
plt.rcParams.update(params)

# Permutations test
run_permutations_GA = False
run_permutations_diff = True
n_permutations = 1024
desired_tval = 0.01
degrees_of_freedom = len(exp_info.subjects_ids) - 1
p_threshold = 0.05
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# t_thresh = dict(start=0, step=0.2)

# Regions clustering
run_clustering = False
n_clusters = 3
run_quadrants = True
time_limit = 0.2
freq_limit = 15
sig_chs_percent = 0.5


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

# Extract region labels data
region_labels = [element for region in meg_params['regions_id'].split('_') for element in aparc_region_labels[region]]

# Get parcelation labels
fsaverage_labels = functions_analysis.get_labels(parcelation=parcelation, subjects_dir=subjects_dir, surf_vol=surf_vol)
fsaverage_labels = [label for label in fsaverage_labels for label_id in region_labels if label.name.startswith(label_id + '-')]

# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}

# Save source estimates time courses on FreeSurfer
stcs_default_dict = {}
GA_stcs = {}
epochs_num = {}
fixations_duration = {}
hist_data = None
# --------- Run ---------#
for param in param_values.keys():
    stcs_default_dict[param] = {}
    GA_stcs[param] = {}
    epochs_num[param] = {}
    fixations_duration[param] = {}
    for param_value in param_values[param]:

        # Get run parameters from trial params including all comparison between different parameters
        run_params = trial_params
        # Set first value of parameters comparisons to avoid having lists in run params
        if len(param_values.keys()) > 1:
            for key in param_values.keys():
                run_params[key] = param_values[key][0]
        # Set comparison key value
        run_params[param] = param_value

        # Redefine epoch id
        if 'rel_sac' in run_params.keys() and run_params['rel_sac'] != None:
            run_params['epoch_id'] = run_params['epoch_id'].replace('fix', 'sac')

        # Define trial duration for VS screen analysis
        if 'vs' in run_params['epoch_id'] and 'fix' not in run_params['epoch_id'] and 'sac' not in run_params['epoch_id']:
            run_params['trialdur'] = vs_dur[run_params['mss']]
        else:
            run_params['trialdur'] = None  # Change to trial_dur = None to use all trials for no 'vs' epochs

        # Get time windows from epoch_id name
        map_times = {'it_sac_vs': {'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
                     'it_sac_vs_subsampled': {'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)}}
        run_params['tmin'], run_params['tmax'], _ = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'], plot_edge=plot_edge, map=map_times)

        # Get baseline duration for epoch_id
        bline_map = dict(red={'baseline': (run_params['tmax'] - cross1_dur - plot_edge, run_params['tmax']),
                              'plot_baseline': (run_params['tmax'] - cross1_dur - plot_edge, run_params['tmax'] - plot_edge)})
        run_params['baseline'], run_params['plot_baseline'] = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'],
                                                                                                      tmin=run_params['tmin'], tmax=run_params['tmax'],
                                                                                                      cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                                                      cross2_dur=cross2_dur, plot_edge=plot_edge, map=bline_map)

        # Paths
        run_path = (f"{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}_"
                    f"trialdur{run_params['trialdur']}_evtdur{run_params['evtdur']}_{run_params['tmin']}_{run_params['tmax']}_bline{run_params['baseline']}/")
        # Redefine save id
        if 'rel_sac' in run_params.keys() and run_params['rel_sac'] != None:
            run_path = run_params['rel_sac'] + '_' + run_path

        # Source paths
        source_model_path = f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}"
        labels_model_path = source_model_path + f"_{parcelation}_{labels_mode}/"
        source_tf_path = source_model_path + f"_{bline_mode_subj}_{source_estimation}_{labels_mode}/"

        # Data paths
        epochs_save_path = paths().save_path() + f"Epochs_{meg_params['data_type']}/Band_{meg_params['band_id']}/" + run_path
        evoked_save_path = paths().save_path() + f"Evoked_{meg_params['data_type']}/Band_{meg_params['band_id']}/" + run_path
        cov_save_path = paths().save_path() + f"Cov_Epochs_{meg_params['data_type']}/Band_{meg_params['band_id']}/" + run_path
        label_ts_save_path = paths().save_path() + f"Source_labels_{meg_params['data_type']}/Band_{meg_params['band_id']}/" + run_path + labels_model_path
        source_tf_save_path = paths().save_path() + f"Source_Space_{meg_params['data_type']}_TF/Band_{meg_params['band_id']}/" + run_path + source_tf_path

        # Define fig path
        fig_path = paths().plots_path() + f"Source_Space_{meg_params['data_type']}_TF/Band_{meg_params['band_id']}/" + run_path + source_tf_path + f'{tf_output}/'

        # Save source estimates time courses on default's subject source space
        stcs_default_dict[param][param_value] = []
        epochs_num[param][param_value] = []
        fixations_duration[param][param_value] = []

        # Iterate over participants
        for subject_code in exp_info.subjects_ids:

            # Load subject
            if meg_params['data_type'] == 'ICA':
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            elif meg_params['data_type'] == 'RAW':
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

            # Subject's data fname
            subj_source_data_fname = f"Power_{subject_code}_{l_freq}_{h_freq}_{meg_params['regions_id']}_tfr.h5"
            if tf_output != 'avg_power':
                subj_source_data_fname = subj_source_data_fname.replace('Power', tf_output)

            # Data filenames
            epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
            evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

            # Load subjects data
            if os.path.isfile(source_tf_save_path + subj_source_data_fname):
                print(f'Loading TF data from participant {subject_code}')
                # Load data
                source_tf = mne.time_frequency.read_tfrs(source_tf_save_path + subj_source_data_fname, condition=0)

                # Get next fixation start time and save
                if 'fix' in run_params['epoch_id']:
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

                # Append data for GA
                stcs_default_dict[param][param_value].append(source_tf)
                epochs_num[param][param_value].append(source_tf.nave)

            # Compute subjects data
            else:
                print(f'Computing TF for participant {subject_code}')

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
                fname_fwd = paths().fwd_path(subject=subject, subject_code=subject_code, ico=ico, spacing=spacing, surf_vol=surf_vol)
                fwd = mne.read_forward_solution(fname_fwd)
                src = fwd['src']

                # Load filter
                fname_filter = paths().filter_path(subject=subject, subject_code=subject_code, ico=ico, spacing=spacing, surf_vol=surf_vol, pick_ori=pick_ori,
                                                   model_name=model_name)
                filters = mne.beamformer.read_beamformer(fname_filter)

                # Get epochs and evoked
                try:
                    # Load data
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                    # Pick meg channels for source modeling
                    epochs.pick('mag')
                    evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]

                    # Pick meg channels for source modeling
                    evoked.pick('mag')

                except:
                    # Get epochs
                    try:
                        # load epochs
                        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

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

                        # Epoch data
                        epochs, events = functions_analysis.epoch_data(subject=subject, meg_data=meg_data, mss=run_params['mss'], corr_ans=run_params['corrans'],
                                                                       tgt_pres=run_params['tgtpres'], epoch_id=run_params['epoch_id'], rel_sac=run_params['rel_sac'],
                                                                       tmin=run_params['tmin'], trial_dur=run_params['trialdur'], evt_dur=run_params['evtdur'],
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
                # Redefine stc epochs to iterate over
                stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

                # Extract region labels data
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

                # Average the source estimates within each label using sign-flips to reduce signal cancellations
                label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=used_labels, src=src, mode=labels_mode, return_generator=False)

                # Time-Frequency computation
                region_source_array = np.array(label_ts)
                freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)
                n_cycles = freqs / n_cycles_div
                source_tf_array = mne.time_frequency.tfr_array_morlet(region_source_array, sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles, output=tf_output)
                if tf_output == 'phase':
                    source_tf_array = source_tf_array.mean(axis=0)
                source_tf = mne.time_frequency.AverageTFR(info=epochs.copy().pick(epochs.ch_names[:region_source_array.shape[1]]).info, data=source_tf_array,
                                                          times=epochs.times, freqs=freqs, nave=len(label_ts))

                # Trim edge times
                source_tf.crop(tmin=epochs.tmin + plot_edge, tmax=epochs.tmax - plot_edge)

                # Apply baseline
                baseline_mean_subj = source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1)
                if bline_mode_subj == 'db':
                    source_tf.data = 10 * np.log10(source_tf.data / np.expand_dims(baseline_mean_subj, axis=-1))
                elif bline_mode_subj == 'ratio':
                    source_tf.data = source_tf.data / np.expand_dims(baseline_mean_subj, axis=-1)
                elif bline_mode_subj == 'mean':
                    source_tf.data = source_tf.data - np.expand_dims(baseline_mean_subj, axis=-1)

                if save_data:
                    # Save trf data
                    os.makedirs(source_tf_save_path, exist_ok=True)
                    source_tf.save(source_tf_save_path + subj_source_data_fname, overwrite=True)

                # Append data for GA
                stcs_default_dict[param][param_value].append(source_tf)
                epochs_num[param][param_value].append(len(epochs))

            # Get next fixation start time and save
            if 'fix' in run_params['epoch_id']:
                epochs.metadata['next_sac_dur'] = epochs.metadata['next_sac'].apply(lambda x: subject.saccades.loc[int(x), 'duration'] if pd.notna(x) else 0)
                epochs.metadata['total_duration'] = epochs.metadata['duration'] + epochs.metadata['next_sac_dur']
                fixations_duration[param][param_value].append(epochs.metadata['total_duration'])
            elif 'sac' in run_params['epoch_id']:
                epochs.metadata['total_duration'] = epochs.metadata['duration']
                fixations_duration[param][param_value].append(epochs.metadata['total_duration'])

            # Plot power time-frequency
            if plot_individuals:

                fname = f"Power_{subject.subject_id}_{meg_params['regions_id']}_{bline_mode_subj}_{l_freq}_{h_freq}"
                if tf_output != 'Power':
                    fname = fname.replace('Power', tf_output)

                hist_data = epochs.metadata['total_duration']
                fig = plot_general.source_tf(tf=source_tf, hist_data=hist_data, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=fname)

                # Free up memory
                plt.close(fig)

        # Grand Average: Average evoked stcs from this epoch_id
        all_subj_source_data = np.zeros(tuple([len(stcs_default_dict[param][param_value])] + [size for size in stcs_default_dict[param][param_value][0].data.shape]))
        for j, stc in enumerate(stcs_default_dict[param][param_value]):
            all_subj_source_data[j] = stc.data

        # Define GA data
        GA_stc_data = all_subj_source_data.mean(0)

        # Copy Source Time Course from default subject morph to define GA STC
        GA_stc = source_tf.copy()

        # Reeplace data
        GA_stc.data = GA_stc_data
        GA_stc.subject = 'fsaverage'

        # Save GA from epoch id
        GA_stcs[param][param_value] = GA_stc

        # --------- Permutations test and plot --------- #
        sig_regions = []
        # Iterate, test and plot each region
        for i, region in enumerate(fsaverage_labels):

            # --------- Cluster permutations test ---------#
            if run_permutations_GA and pick_ori != 'vector':

                # Define data to test
                region_subj_data = all_subj_source_data[:, i, :, :]
                region_subj_data = np.expand_dims(region_subj_data, axis=-1)  # Need shape (n_subjects, n_freqs, n_times, n_channels,)

                # Run clusters permutations test
                clusters_mask, clusters_mask_plot, significant_pvalues = functions_analysis.run_time_frequency_test(data=region_subj_data, pval_threshold=p_threshold,
                                                                                               t_thresh=t_thresh, n_permutations=n_permutations)

                if isinstance(t_thresh, dict):
                    fname = f'GA_{region.name}_{l_freq}_{h_freq}_tTFCE_pval{p_threshold}'
                else:
                    fname = f'GA_{region.name}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{p_threshold}'

                # Define image title
                title = fname + f'_{significant_pvalues}'

            else:
                clusters_mask_plot = None
                clusters_mask = None
                if isinstance(t_thresh, dict):
                    fname = f'GA_{region.name}_{l_freq}_{h_freq}'
                else:
                    fname = f'GA_{region.name}_{l_freq}_{h_freq}'

                # Define title
                title = fname

            # --------- Plot --------- #
            ga_tf_region = GA_stc.pick(GA_stc.ch_names[0])
            ga_tf_region.data = np.expand_dims(GA_stc_data[i], axis=0)
            if 'fix' in run_params['epoch_id'] or 'sac' in run_params['epoch_id']:
                hist_data = pd.concat(fixations_duration[param][param_value])

            fig = plot_general.source_tf(tf=ga_tf_region, clusters_mask_plot=clusters_mask_plot, hist_data=hist_data, display_figs=display_figs,
                                         save_fig=save_fig, fig_path=fig_path, fname=fname, title=title)

            # Free up memory
            plt.close(fig)

            # Save significant regions to plot brain
            if isinstance(clusters_mask_plot, np.ndarray):
                sig_regions.append(region)

        # Plot brain with marked regions
        if len(sig_regions):
            try:
                mne.viz.close_all_3d_figures()
            except:
                pass

            Brain = mne.viz.get_brain_class()
            brain = Brain("fsaverage", hemi="split", surf="pial", views=['lat', 'med'], subjects_dir=subjects_dir, size=(1080, 720))
            for label in sig_regions:
                brain.add_label(label, borders=False)

            # Save
            if save_fig:
                brain.save_image(filename=fig_path + 'sig/' + 'brain_regions.png')
                brain.save_image(filename=fig_path + 'sig/svg/' + 'brain_regions.pdf')


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
                # Crop to active times
                if active_times:
                    subject_data_0 = stcs_default_dict[param][comparison[0]][i].crop(tmin=active_times[0], tmax=active_times[1])
                    subject_data_1 = stcs_default_dict[param][comparison[1]][i].crop(tmin=active_times[0], tmax=active_times[1])
                else:
                    subject_data_0 = stcs_default_dict[param][comparison[0]][i]
                    subject_data_1 = stcs_default_dict[param][comparison[1]][i]
                stcs_diff.append(subject_data_0 - subject_data_1)

            # Average evoked stcs
            all_subj_diff_data = np.zeros(tuple([len(stcs_diff)]+[size for size in stcs_diff[0].data.shape]))
            for i, stc in enumerate(stcs_diff):
                all_subj_diff_data[i] = stcs_diff[i].data

            GA_stc_diff_data = all_subj_diff_data.mean(0)

            # Copy Source Time Course from default subject morph to define GA STC
            GA_stc_diff = GA_stc.copy()

            # Reeplace data
            GA_stc_diff.data = GA_stc_diff_data
            GA_stc_diff.subject = 'fsaverage'

            # Delete previous significance plots
            if os.path.exists(fig_path_diff + f'sig/'):
                shutil.rmtree(fig_path_diff + f'sig/')

            # Define TF quadrants
            quadrants_regions = {'early_low': [], 'late_low': [], 'early_high': [], 'late_high': []}

            sig_regions = []
            sig_data = []
            sig_tfr = []
            clusters_masks = []

            # Iterate, test and plot each region
            for i, region in enumerate(fsaverage_labels):

                #--------- Cluster permutations test ---------#
                if run_permutations_diff and pick_ori != 'vector':

                    # Define data to test
                    region_subj_diff_data = all_subj_diff_data[:, i, :, :]
                    region_subj_diff_data = np.expand_dims(region_subj_diff_data, axis=-1)  # Need shape (n_subjects, n_freqs, n_times, n_channels,)

                    # Run clusters permutations test
                    clusters_mask, clusters_mask_plot, significant_pvalues = functions_analysis.run_time_frequency_test(data=region_subj_diff_data, pval_threshold=p_threshold, t_thresh=t_thresh, n_permutations=n_permutations)

                    if isinstance(t_thresh, dict):
                        fname = f'GA_{region.name}_{l_freq}_{h_freq}_tTFCE_pval{p_threshold}'
                    else:
                        fname = f'GA_{region.name}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{p_threshold}'

                    # Define image title
                    title = fname + f'_{significant_pvalues}'

                else:
                    clusters_mask_plot = None
                    clusters_mask = None
                    significant_pvalues = None
                    if isinstance(t_thresh, dict):
                        fname = f'GA_{region.name}_{l_freq}_{h_freq}'
                    else:
                        fname = f'GA_{region.name}_{l_freq}_{h_freq}'

                    title = fname

                if active_times:
                    fname += f"_{active_times[0]}_{active_times[1]}"

                # --------- Plot --------- #
                ga_tf_diff_region = GA_stc_diff.pick(GA_stc_diff.ch_names[0])
                ga_tf_diff_region.data = np.expand_dims(GA_stc_diff_data[i], axis=0)
                hist_data = pd.concat(fixations_duration[param][comparison[0]] + fixations_duration[param][comparison[1]])

                fig = plot_general.source_tf(tf=ga_tf_diff_region, clusters_mask_plot=clusters_mask_plot, hist_data=hist_data, display_figs=display_figs,
                                             save_fig=save_fig, fig_path=fig_path_diff, fname=fname, title=title)

                # Close figure
                if not display_figs:
                    plt.close(fig)

                # Save significant regions to plot brain and run clustering / quadrants
                if isinstance(clusters_mask_plot, np.ndarray):
                    sig_regions.append(region)
                    sig_data.append(GA_stc_diff_data[i].flatten())
                    sig_tfr.append(ga_tf_diff_region.copy())
                    clusters_masks.append(clusters_mask_plot)

                    # Find quadrant
                    time_limit_idx, _ = functions_general.find_nearest(ga_tf_diff_region.times, time_limit)
                    freq_limit_idx, _ = functions_general.find_nearest(ga_tf_diff_region.freqs, freq_limit)

                    if clusters_mask_plot[:freq_limit_idx, :time_limit_idx].any():
                        quadrants_regions['early_low'].append(region)
                    if clusters_mask_plot[freq_limit_idx:, :time_limit_idx:].any():
                        quadrants_regions['early_high'].append(region)
                    if clusters_mask_plot[:freq_limit_idx, time_limit_idx:].any():
                        quadrants_regions['late_low'].append(region)
                    if clusters_mask_plot[freq_limit_idx:, time_limit_idx:].any():
                        quadrants_regions['late_high'].append(region)

            # Plot brain with marked regions
            if len(sig_regions):

                if run_clustering:
                    functions_analysis.cluster_regions(n_clusters=n_clusters, sig_data=sig_data, sig_regions=sig_regions, sig_tfr=sig_tfr, clusters_masks=clusters_masks,
                                                       l_freq=l_freq, h_freq=h_freq, active_times=active_times, subjects_dir=subjects_dir,
                                                       display_figs=display_figs, save_fig=save_fig, fig_path_diff=fig_path_diff)
                if run_quadrants:
                    functions_analysis.quadrant_regions(quadrants_regions=quadrants_regions, sig_regions=sig_regions, sig_tfr=sig_tfr, clusters_masks=clusters_masks,
                                                        sig_chs_percent=sig_chs_percent, l_freq=l_freq, h_freq=h_freq, active_times=active_times,
                                                        hist_data=hist_data, subjects_dir=subjects_dir,
                                                        display_figs=display_figs, save_fig=save_fig, fig_path_diff=fig_path_diff)


# Final print
print(trial_params)
print(meg_params)

for param in param_values.keys():
    print(param)
    for param_value in param_values[param]:
        print(param_value)
        print(f"Average epochs: {np.mean(epochs_num[param][param_value])} +/- {np.std(epochs_num[param][param_value])}")

for quadrants in quadrants_regions.keys():
    print(f"{quadrants}: {len(quadrants_regions[quadrants])}")