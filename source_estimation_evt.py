import os
import functions_analysis
import functions_general
import mne
import mne.beamformer as beamformer
from paths import paths
import load
import setup
import numpy as np
import plot_general
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import itertools

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

# ICA
use_ica_data = True

# Frequency band
filter_sensors = True
band_id = None
filter_method = 'iir'

# Trial selection
trial_params = {'epoch_id': ['tgt_fix_vs', 'it_fix_vs_subsampled'],  # use'+' to mix conditions (red+blue)
                'corrans': True,
                'tgtpres': True,
                'mss': None,
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evtdur': None}

# Compare features
run_comparison = True

# Baseline
bline_mode_subj = 'mean'
bline_mode_ga = False
plot_edge = 0.15

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'volume'
ico = 5
spacing = 5.  # Only for volume source estimation
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
source_power = False
estimate_epochs = False  # epochs and covariance cannot be both true (but they can be both false and estimate sources from evoked)
estimate_covariance = False
visualize_alignment = False

# Plot
initial_time = 0.1
positive_cbar = None  # None for free determination, False to include negative values
plot_individuals = False
plot_ga = True

# Permutations test
run_permutations_GA = True
run_permutations_diff = True
desired_tval = 0.01
p_threshold = 0.05
mask_negatives = False


#--------- Setup ---------#

# Load subject
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Source computation method
if estimate_covariance:
    source_computation = 'cov'
elif estimate_epochs:
    source_computation = 'epo'
else:
    source_computation = 'evk'

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()


# --------- Freesurfer Path ---------#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

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
            trial_dur = vs_dur[run_params['mss']]
        else:
            trial_dur = None  # Change to trial_dur = None to use all trials for no 'vs' epochs

        # Get time windows from epoch_id name
        tmin, tmax, _ = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'], plot_edge=plot_edge)

        # Get baseline duration for epoch_id
        baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'], tmin=tmin, tmax=tmax,
                                                                          cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                          cross2_dur=cross2_dur)

        # Paths
        run_path = f"/Band_{band_id}/{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}_trialdur{trial_dur}" \
                   f"_evtdur{run_params['evtdur']}_{tmin}_{tmax}_bline{baseline}/"

        # Data paths
        epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path
        evoked_save_path = paths().save_path() + f'Evoked_{data_type}/' + run_path
        cov_save_path = paths().save_path() + f'Cov_Epochs_{data_type}/' + run_path

        # Source plots paths
        if source_power or estimate_covariance:
            run_path = run_path.replace(f"{run_params['epoch_id']}_", f"{run_params['epoch_id']}_power_")
        run_path = run_path.replace('Band_None', f'Band_{band_id}')

        # Define path
        if surf_vol == 'volume' or surf_vol == 'mixed':
            fig_path = paths().plots_path() + f'Source_Space_{data_type}/' + run_path + f'{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}_{bline_mode_subj}_{source_computation}/'
        else:
            fig_path = paths().plots_path() + f'Source_Space_{data_type}/' + run_path + f'{model_name}_{surf_vol}_ico{ico}_{pick_ori}_{bline_mode_subj}_{source_computation}/'

        # Save source estimates time courses on default's subject source space
        stcs_default_dict[param][param_value] = []

        # Iterate over participants
        for subject_code in exp_info.subjects_ids:
            # Load subject
            if use_ica_data:
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            else:
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
                if estimate_epochs:
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                    # Pick meg channels for source modeling
                    epochs.pick('meg')
                evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]
                # Pick meg channels for source modeling
                evoked.pick('meg')
            except:
                # Get epochs
                try:
                    # load epochs
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                except:
                    # Compute epochs
                    if use_ica_data:
                        if band_id and filter_sensors:
                            meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=save_data,
                                                          method=filter_method)
                        else:
                            meg_data = load.ica_data(subject=subject)
                    else:
                        if band_id and filter_sensors:
                            meg_data = load.filtered_data(subject=subject, band_id=band_id, use_ica_data=False,
                                                          save_data=save_data, method=filter_method)
                        else:
                            meg_data = subject.load_preproc_meg_data()
                    # Epoch data
                    epochs, events = functions_analysis.epoch_data(subject=subject, mss=run_params['mss'], corr_ans=run_params['corrans'], tgt_pres=run_params['tgtpres'],
                                                                   epoch_id=run_params['epoch_id'], meg_data=meg_data, tmin=tmin, trial_dur=trial_dur,
                                                                   tmax=tmax, reject=run_params['reject'], baseline=baseline,
                                                                   save_data=save_data, epochs_save_path=epochs_save_path,
                                                                   epochs_data_fname=epochs_data_fname)

                # Define evoked from epochs
                evoked = epochs.average()

                # Save evoked data
                if save_data:
                    os.makedirs(evoked_save_path, exist_ok=True)
                    evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

                # Pick meg channels for source modeling
                evoked.pick('meg')
                epochs.pick('meg')


            # --------- Source estimation ---------#

            # Estimate sources from covariance matrix
            if estimate_covariance:
                # Covariance method
                cov_method = 'shrunk'

                # Covariance matrix rank
                rank = sum([ch_type == 'mag' for ch_type in evoked.get_channel_types()]) - len(evoked.info['bads'])
                if use_ica_data:
                    rank -= len(subject.ex_components)

                # Define active times
                active_times = [0, tmax]

                # Covariance fnames
                cov_baseline_fname = f'Subject_{subject.subject_id}_times{baseline}_{cov_method}_{rank}-cov.fif'
                cov_act_fname = f'Subject_{subject.subject_id}_times{active_times}_{cov_method}_{rank}-cov.fif'

                stc = functions_analysis.estimate_sources_cov(subject=subject, baseline=baseline, band_id=band_id, filter_sensors=filter_sensors,
                                                              filter_method=filter_method, use_ica_data=use_ica_data, epoch_id=run_params['epoch_id'], mss=run_params['mss'],
                                                              corr_ans=run_params['corrans'], tgt_pres=run_params['tgtpres'], trial_dur=trial_dur, reject=run_params['tgtpres'],
                                                              tmin=tmin, tmax=tmax, filters=filters, active_times=active_times, rank=rank, bline_mode_subj=bline_mode_subj,
                                                              save_data=save_data, cov_save_path=cov_save_path, cov_act_fname=cov_act_fname,
                                                              cov_baseline_fname=cov_baseline_fname, epochs_save_path=epochs_save_path, epochs_data_fname=epochs_data_fname)

            # Estimate sources from epochs
            elif estimate_epochs:
                # Define sources estimated on epochs
                stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

                # Define stc object
                stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

                # Set data as zero to average epochs
                stc.data = np.zeros(shape=(stc.data.shape))
                for stc_epoch in stc_epochs:
                    data = stc_epoch.data
                    if source_power:
                        # Compute source power on epochs and average
                        if band_id and not filter_sensors:
                            # Filter source data
                            data = functions_general.butter_bandpass_filter(data, band_id=band_id, sfreq=epochs.info['sfreq'], order=3)
                        # Compute envelope
                        analytic_signal = hilbert(data, axis=-1)
                        signal_envelope = np.abs(analytic_signal)
                        # Sum data of every epoch
                        stc.data += signal_envelope

                    else:
                        stc.data += data
                    # Divide by epochs number
                    stc.data /= len(epochs)

                if source_power:
                    # Drop edges due to artifacts from power computation
                    stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.times.max() - plot_edge)

            # Estimate sources from evoked
            else:
                # Apply filter and get source estimates
                stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

                if source_power:
                    # Compute envelope in source space
                    data = stc.data
                    if band_id and not filter_sensors:
                        # Filter source data
                        data = functions_general.butter_bandpass_filter(data, band_id=band_id, sfreq=evoked.info['sfreq'], order=3)
                    # Compute envelope
                    analytic_signal = hilbert(data, axis=-1)
                    signal_envelope = np.abs(analytic_signal)
                    # Save envelope as data
                    stc.data = signal_envelope

                    # Drop edges due to artifacts from power computation
                    stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.tmax - plot_edge)

            if bline_mode_subj and not estimate_covariance:
                # Apply baseline correction
                print(f'Applying baseline correction: {bline_mode_subj} from {baseline[0]} to {baseline[1]}')
                # stc.apply_baseline(baseline=baseline)  # mean
                if bline_mode_subj == 'db':
                    stc.data = 10 * np.log10(stc.data / np.expand_dims(stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=-1), axis=-1))
                elif bline_mode_subj == 'ratio':
                    stc.data = stc.data / np.expand_dims(stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=-1), axis=-1)
                elif bline_mode_subj == 'mean':
                    stc.data = stc.data - np.expand_dims(stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=-1), axis=-1)

            if band_id and source_power and not estimate_covariance:
                # Filter higher frequencies than corresponding to nyquist of bandpass filter higher freq
                l_freq, h_freq = functions_general.get_freq_band(band_id=band_id)
                stc.data = functions_general.butter_lowpass_filter(data=stc.data, h_freq=h_freq/2, sfreq=evoked.info['sfreq'], order=3)

            # Morph to default subject
            if subject_code != 'fsaverage':
                # Get Source space for default subject
                if surf_vol == 'volume':
                    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_volume_ico{ico}_{int(spacing)}-src.fif'
                elif surf_vol == 'surface':
                    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_surface_ico{ico}-src.fif'
                elif surf_vol == 'mixed':
                    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_mixed_ico{ico}_{int(spacing)}-src.fif'

                src_default = mne.read_source_spaces(fname_src)

                # Define morph function
                morph = mne.compute_source_morph(src=src, subject_from=subject_code, subject_to='fsaverage', src_to=src_default, subjects_dir=subjects_dir)

                # Apply morph
                stc_default = morph.apply(stc)

                # Append to fs_stcs to make GA
                stcs_default_dict[param][param_value].append(stc_default)

            else:
                src_default = src

                stc_default = stc

                # Append to fs_stcs to make GA
                stcs_default_dict[param][param_value].append(stc)

            # Plot
            if plot_individuals:
                fname = f'{subject.subject_id}'
                plot_general.sources(stc=stc, src=src, subject=subject_code, subjects_dir=subjects_dir, initial_time=initial_time, surf_vol=surf_vol,
                                     force_fsaverage=force_fsaverage, estimate_covariance=estimate_covariance, pick_ori=pick_ori, mask_negatives=mask_negatives,
                                     positive_cbar=positive_cbar, views=['lat', 'med'], save_fig=save_fig, save_vid=False, fig_path=fig_path, fname=fname)

        # Grand Average: Average evoked stcs from this epoch_id
        all_subj_source_data = np.zeros(tuple([len(stcs_default_dict[param][param_value])] + [size for size in stcs_default_dict[param][param_value][0].data.shape]))
        for j, stc in enumerate(stcs_default_dict[param][param_value]):
            all_subj_source_data[j] = stcs_default_dict[param][param_value][j].data
        if mask_negatives:
            all_subj_source_data[all_subj_source_data < 0] = 0

        # Define GA data
        GA_stc_data = all_subj_source_data.mean(0)

        # Copy Source Time Course from default subject morph to define GA STC
        GA_stc = stc_default.copy()

        # Reeplace data
        GA_stc.data = GA_stc_data
        GA_stc.subject = 'fsaverage'

        # Apply baseline on GA data
        if bline_mode_ga and not estimate_covariance:
            print(f'Applying baseline correction: {bline_mode_ga} from {baseline[0]} to {baseline[1]}')
            # GA_stc.apply_baseline(baseline=baseline)
            if bline_mode_ga == 'db':
                GA_stc.data = 10 * np.log10(GA_stc.data / GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None])
            elif bline_mode_ga == 'ratio':
                GA_stc.data = GA_stc.data / GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]
            elif bline_mode_ga == 'mean':
                GA_stc.data = GA_stc.data - GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]

        # Save GA from epoch id
        GA_stcs[param][param_value] = GA_stc

        # --------- Plot GA ---------#
        if plot_ga:
            fname = 'GA'
            brain = plot_general.sources(stc=GA_stc, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=initial_time, surf_vol=surf_vol,
                                         force_fsaverage=force_fsaverage, estimate_covariance=estimate_covariance, pick_ori=pick_ori, mask_negatives=mask_negatives,
                                         positive_cbar=positive_cbar, views=['lat', 'med'], save_fig=save_fig, save_vid=True, fig_path=fig_path, fname=fname)

        # --------- Test significance compared to baseline --------- #
        if run_permutations_GA and pick_ori != 'vector':
            stc_all_cluster_vis, significance_voxels, significance_mask, t_thresh_name, time_label, p_threshold = \
                functions_analysis.run_source_permutations_test(src=src_default, stc=GA_stc, source_data=all_subj_source_data, subject='fsaverage', exp_info=exp_info,
                                                                save_regions=True, fig_path=fig_path, surf_vol=surf_vol, desired_tval=desired_tval, mask_negatives=mask_negatives,
                                                                p_threshold=p_threshold)

            # If covariance estimation, no time variable. Clusters are static
            if significance_mask is not None and estimate_covariance:
                # Mask data
                GA_stc_sig = GA_stc.copy()
                GA_stc_sig.data[significance_mask] = 0

                # --------- Plot GA significant clusters ---------#
                fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                brain = plot_general.sources(stc=GA_stc_sig, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0, surf_vol=surf_vol,
                                     time_label=time_label, force_fsaverage=force_fsaverage, estimate_covariance=estimate_covariance, pick_ori=pick_ori, views=['lat', 'med'],
                                     mask_negatives=mask_negatives, positive_cbar=True, save_vid=False, save_fig=save_fig, fig_path=fig_path, fname=fname)

            # If time variable, visualize clusters using mne's function
            elif significance_mask is not None:
                fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                brain = plot_general.sources(stc=stc_all_cluster_vis, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0,
                                             surf_vol=surf_vol, time_label=time_label, force_fsaverage=force_fsaverage, estimate_covariance=estimate_covariance,
                                             pick_ori=pick_ori, views=['lat', 'med'], mask_negatives=mask_negatives, positive_cbar=True,
                                             save_vid=False, save_fig=save_fig, fig_path=fig_path, fname=fname)


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
            for i in range(len(stcs_default_dict[param][comparison[np.argmax(comparison)]])):
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

            # --------- Plots ---------#
            if plot_ga:
                fname = f'GA'
                brain = plot_general.sources(stc=GA_stc_diff, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0.408, surf_vol=surf_vol,
                                             pick_ori=pick_ori, force_fsaverage=force_fsaverage, estimate_covariance=estimate_covariance, mask_negatives=mask_negatives,
                                             views=['cor'], save_vid=False, save_fig=False, fig_path=fig_path_diff, fname=fname, positive_cbar=True, hemi='both')

            GA_stc_diff_crop = GA_stc_diff.copy().crop(tmin=0.2, tmax=0.4)
            GA_stc_diff_crop_mean = GA_stc_diff_crop.mean()


            #--------- Cluster permutations test ---------#
            if run_permutations_diff and pick_ori != 'vector':
                stc_all_cluster_vis, significance_voxels, significance_mask, t_thresh_name, time_label, p_threshold = \
                    functions_analysis.run_source_permutations_test(src=src_default, stc=GA_stc_diff, source_data=all_subj_diff_data, subject='fsaverage',
                                                                    exp_info=exp_info, save_regions=True, fig_path=fig_path_diff, surf_vol=surf_vol, desired_tval=desired_tval,
                                                                    mask_negatives=mask_negatives, p_threshold=p_threshold)

                if significance_mask is not None and estimate_covariance:
                    # Mask data
                    GA_stc_diff_sig = GA_stc_diff.copy()
                    GA_stc_diff_sig.data[significance_mask] = 0

                    # --------- Plots ---------#
                    fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                    brain = plot_general.sources(stc=GA_stc_diff_sig, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0,
                                                 surf_vol=surf_vol, time_label=time_label, force_fsaverage=force_fsaverage, estimate_covariance=estimate_covariance,
                                                 pick_ori=pick_ori, views=['lat', 'med'], mask_negatives=mask_negatives, positive_cbar=True,
                                                 save_vid=False, save_fig=save_fig, fig_path=fig_path_diff, fname=fname)

                elif significance_mask is not None:
                    fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                    brain = plot_general.sources(stc=stc_all_cluster_vis, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0,
                                                 surf_vol=surf_vol, time_label=time_label, force_fsaverage=force_fsaverage, estimate_covariance=estimate_covariance,
                                                 pick_ori=pick_ori, views=['lat', 'med'], mask_negatives=mask_negatives, positive_cbar=True,
                                                 save_vid=False, save_fig=save_fig, fig_path=fig_path_diff, fname=fname)
