import glob
import os.path

import functions_general
import functions_analysis
import load
import mne
import plot_general
import save
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mne.stats import permutation_cluster_test, f_mway_rm, f_threshold_mway_rm
import itertools
import copy


#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
plot_individuals = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
use_ica_data = True

# Trial selection and filters parameters. A field with 2 values will compute the difference between the conditions specified
trial_params = {'epoch_id': 'cross2',  # 'ms', 'cross2', 'it_sac_ms+tgt_sac_ms', 'it_sac_vs+tgt_sac_vs', 'it_sac_vs', 'tgt_fix_vs', 'sac_emap', 'hl_start'
                'corrans': None,
                'tgtpres': None,
                'mss': [1, 2, 4],
                'reject': None,
                'evtdur': None,
                'rel_sac': None
                }
run_comparison = True

# Select channels
chs_ids = ['parietal_occipital']  # region_hemisphere

# Power time frequency params
l_freq = 1
h_freq = 40
run_itc = False
plot_edge = 0.15

# Plots parameters
# Colorbar
vmax_power = None
vmin_power = None
vmin_itc, vmax_itc = None, None
# plot_joint max and min topoplots
plot_max, plot_min = True, True
overlay_broadband_power = False

# Baseline method
# logratio: dividing by the mean of baseline values and taking the log
# ratio: dividing by the mean of baseline values
# mean: subtracting the mean of baseline values
bline_mode = 'logratio'
ga_plot_bline_mode = 'mean'

# Time Frequency config
tf_method = 'morlet'  # 'morlet' or 'multitaper'
return_average_tfr = True
output = 'power'

# Permutations cluster test parameters
run_permutations_ga = False
run_permutations_anova = True
run_permutations_dif = False
n_permutations = 5120
degrees_of_freedom = len(exp_info.subjects_ids) - 1
desired_tval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# F-threshold for repeated-measures ANOVA (computed dynamically in the ANOVA section)
desired_fval = 0.05
pval_threshold = 0.05
significant_channels = 0.5  # Percent of total region channels

# Save data of each id
power_data = {}
itc_data = {}
ga_power_data = {}
ga_itc_data = {}

# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}


# --------- Run ---------#
for param in param_values.keys():
    power_data[param] = {}
    itc_data[param] = {}
    ga_power_data[param] = {}
    ga_itc_data[param] = {}
    for param_value in param_values[param]:

        # Get run parameters from trial params
        run_params = copy.copy(trial_params)
        # Set first value of parameters comparisons to avoid having lists in run params
        if len(param_values.keys()) > 1:
            for key in param_values.keys():
                run_params[key] = param_values[key][0]
        # Set comparison key value
        run_params[param] = param_value

        #---------- Setup ----------#
        # Redefine epoch id
        if 'rel_sac' in run_params.keys() and run_params['rel_sac'] != None:
            run_params['epoch_id'] = run_params['epoch_id'].replace('fix', 'sac')

        # Windows durations
        cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
        if 'vs' in run_params['epoch_id'] and 'fix' not in run_params['epoch_id'] and 'sac' not in run_params['epoch_id']:
            trial_dur = vs_dur[run_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
        else:
            trial_dur = None

        # morlet wavelet n_cycles divisor based on epochs duration
        if 'fix' in run_params['epoch_id'] or 'sac' in run_params['epoch_id']:
            n_cycles_div = 4.
        else:
            n_cycles_div = 2.

        # Get time windows from epoch_id name
        map = {'ms': {'tmin': -cross1_dur, 'tmax': mss_duration[run_params['mss']], 'plot_xlim': (-cross1_dur + plot_edge, mss_duration[run_params['mss']] - plot_edge)},
               'cross2': {'tmin': -cross1_dur -mss_duration[run_params['mss']], 'tmax': cross2_dur, 'plot_xlim': (0, cross2_dur - plot_edge)},
               'fix_vs': {'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
               'tgt_fix_ms': {'tmin': -0.4, 'tmax': 0.4, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
               'it_sac_ms+tgt_sac_ms': {'tmin': -0.4, 'tmax': 0.4, 'plot_xlim': (-0.4 + plot_edge, 0.4 - plot_edge)},
               'it_sac_ms': {'tmin': -0.4, 'tmax': 0.4, 'plot_xlim': (-0.4 + plot_edge, 0.4 - plot_edge)},
               'it_sac_vs+tgt_sac_vs': {'tmin': -0.4, 'tmax': 0.4, 'plot_xlim': (-0.4 + plot_edge, 0.4 - plot_edge)},
               'it_sac_vs': {'tmin': -0.4, 'tmax': 0.4, 'plot_xlim': (-0.4 + plot_edge, 0.4 - plot_edge)},
               'tgt_fix_vs': {'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
               'sac_emap': {'tmin': -0.5, 'tmax': 3, 'plot_xlim': (-0.3, 2.5)},
               'hl_start': {'tmin': -3, 'tmax': 35, 'plot_xlim': (-2.5, 33)}}
        run_params['tmin'], run_params['tmax'], plot_xlim = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'],
                                                                                            plot_edge=plot_edge, map=map)

        # Define time-frequency bands to plot in plot_joint
        timefreqs_joint, timefreqs_tfr, vlines_times = functions_general.get_plots_timefreqs(epoch_id=run_params['epoch_id'], mss=run_params['mss'], cross2_dur=cross2_dur,
                                                                                             mss_duration=mss_duration, plot_xlim=plot_xlim,
                                                                                             plot_min=plot_min, plot_max=plot_max)

        # Get baseline duration for epoch_id
        run_params['baseline'], run_params['plot_baseline'] = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'],
                                                                                                      tmin=run_params['tmin'], tmax=run_params['tmax'],
                                                                                                      cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                                                      cross2_dur=cross2_dur, plot_edge=plot_edge)

        # Specific run path for saving data and plots
        if use_ica_data:
            data_type = 'ICA'
        else:
            data_type = 'RAW'

        # Save ids
        save_id = f"{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}_trialdur{trial_dur}_evtdur{run_params['evtdur']}"
        # Redefine save id
        if 'rel_sac' in run_params.keys() and run_params['rel_sac'] != None:
            save_id = run_params['rel_sac'] + '_' + save_id
        plot_id = f"{save_id}_{round(plot_xlim[0],2)}_{round(plot_xlim[1], 2)}_bline{run_params['plot_baseline']}_cyc{int(n_cycles_div)}/"

        # Save data paths
        if return_average_tfr:
            trf_save_path = paths().save_path() + f"Time_Frequency_{data_type}/{tf_method}/{save_id}_{run_params['tmin']}_{run_params['tmax']}_bline{run_params['baseline']}_cyc{int(n_cycles_div)}/"
        else:
            trf_save_path = paths().save_path() + f"Time_Frequency_Epochs_{data_type}/{tf_method}/{save_id}_{run_params['tmin']}_{run_params['tmax']}_bline{run_params['baseline']}_cyc{int(n_cycles_div)}/"
        epochs_save_path = paths().save_path() + f"Epochs_{data_type}/Band_None/{save_id}_{run_params['tmin']}_{run_params['tmax']}_bline{run_params['baseline']}/"
        # Save figures paths
        trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/{tf_method}/' + plot_id

        # Grand average data variable
        grand_avg_power_fname = f'Grand_Average_power_{l_freq}_{h_freq}_tfr.h5'
        if output == 'phase':
            grand_avg_power_fname = grand_avg_power_fname.replace('power', 'phase')
        grand_avg_itc_fname = f'Grand_Average_itc_{l_freq}_{h_freq}_tfr.h5'

        #------------ Run -------------#
        try:
            # Raise error if run_permutations or run_comparison == True to load data from all subjects
            if run_permutations_ga or run_comparison:
                raise ValueError

            # Get files matching name with extended frequency range
            matching_files = glob.glob(trf_save_path + grand_avg_power_fname.replace(f'{l_freq}_{h_freq}', '*'))
            if len(matching_files):
                for file in matching_files:
                    l_freq_file = int(file.split('_')[-3])
                    h_freq_file = int(file.split('_')[-2])

                    if l_freq_file <= l_freq and h_freq_file >= h_freq:
                        grand_avg_power = mne.time_frequency.read_tfrs(file)[0]

                        # Crop to desired frequencies
                        grand_avg_power = grand_avg_power.crop(fmin=l_freq, fmax=h_freq)
                        break

                    else:
                        raise ValueError('No file found with desired frequency range')

            else:
                raise ValueError('No file found with desired frequency range')

            # Save grand avg param data
            ga_power_data[param][param_value] = grand_avg_power

            if run_itc:
                # Get files matching name with extended frequency range
                matching_files = glob.glob(trf_save_path + grand_avg_itc_fname.replace(f'{l_freq}_{h_freq}', '*'))
                if len(matching_files):
                    for file in matching_files:
                        l_freq_file = int(file.split('_')[-3])
                        h_freq_file = int(file.split('_')[-2])
                        if l_freq_file <= l_freq and h_freq_file >= h_freq:
                            grand_avg_itc = mne.time_frequency.read_tfrs(file)[0]
                            # Crop to desired frequencies
                            grand_avg_itc = grand_avg_itc.crop(fmin=l_freq, fmax=h_freq)
                            break

                        else:
                            raise ValueError('No file found with desired frequency range')

                else:
                    raise ValueError('No file found with desired frequency range')

                # Save grand avg param data
                ga_itc_data[param][param_value] = grand_avg_itc

        except:
            power_data[param][param_value] = []
            itc_data[param][param_value] = []

            for subject_code in exp_info.subjects_ids:

                # Define save path and file name for loading and saving epoched, evoked, and GA data
                if use_ica_data:
                    # Load subject object
                    subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
                else:
                    # Load subject object
                    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

                # Data filenames
                power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
                if output == 'phase':
                    power_data_fname = power_data_fname.replace('Power', 'Phase')
                itc_data_fname = f'ITC_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
                epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
                # Subject plots path
                trf_fig_path_subj = trf_fig_path + f'{subject.subject_id}/'

                try:
                    # Get files matching name with extended frequency range
                    matching_files = glob.glob(trf_save_path + power_data_fname.replace(f'{l_freq}_{h_freq}', '*'))
                    if len(matching_files):
                        for file in matching_files:
                            l_freq_file = int(file.split('_')[-3])
                            h_freq_file = int(file.split('_')[-2])

                            # If file contains desired frequencies, Load
                            if l_freq_file <= l_freq and h_freq_file >= h_freq:
                                power = mne.time_frequency.read_tfrs(file)

                                # Crop to desired frequencies
                                power = power.crop(fmin=l_freq, fmax=h_freq)
                                break
                            else:
                                raise ValueError('No file found with desired frequency range')
                    else:
                        raise ValueError('No file found with desired frequency range')

                    if run_itc:
                        # Get files matching name with extended frequency range
                        matching_files = glob.glob(trf_save_path + itc_data_fname.replace(f'{l_freq}_{h_freq}', '*'))
                        if len(matching_files):
                            for file in matching_files:
                                l_freq_file = int(file.split('_')[-3])
                                h_freq_file = int(file.split('_')[-2])

                                # If file contains desired frequencies, Load
                                if l_freq_file <= l_freq and h_freq_file >= h_freq:
                                    itc = mne.time_frequency.read_tfrs(file)

                                    # Crop to desired frequencies
                                    itc = itc.crop(fmin=l_freq, fmax=h_freq)
                                    break
                                else:
                                    raise ValueError('No file found with desired frequency range')
                        else:
                            raise ValueError('No file found with desired frequency range')

                except:
                    if os.path.exists(epochs_save_path + epochs_data_fname):
                        # Load epoched data
                        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                    else:
                        # Load meg data
                        if use_ica_data:
                            meg_data = load.ica_data(subject=subject)
                        else:
                            meg_data = subject.load_preproc_meg_data()

                        # Epoch data
                        epochs, events = functions_analysis.epoch_data(subject=subject, mss=run_params['mss'], corr_ans=run_params['corrans'], trial_dur=trial_dur,
                                                                       tgt_pres=run_params['tgtpres'], baseline=run_params['baseline'], reject=run_params['reject'],
                                                                       rel_sac=run_params['rel_sac'], evt_dur=run_params['evtdur'], epoch_id=run_params['epoch_id'],
                                                                       meg_data=meg_data, tmin=run_params['tmin'], tmax=run_params['tmax'], save_data=save_data,
                                                                       epochs_save_path=epochs_save_path, epochs_data_fname=epochs_data_fname)
                    # Compute power and PLI over frequencies
                    if tf_method == 'morlet':
                        power = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                                  n_cycles_div=n_cycles_div, average=return_average_tfr,
                                                                  return_itc=run_itc, output=output, save_data=save_data,
                                                                  trf_save_path=trf_save_path, power_data_fname=power_data_fname,
                                                                  itc_data_fname=itc_data_fname, n_jobs=4)
                    elif tf_method == 'multitaper':
                        power = functions_analysis.time_frequency_multitaper(epochs=epochs, l_freq=l_freq, h_freq=h_freq, n_cycles_div=n_cycles_div,
                                                                                         average=return_average_tfr, return_itc=run_itc, save_data=save_data,
                                                                                         trf_save_path=trf_save_path, power_data_fname=power_data_fname,
                                                                                         itc_data_fname=itc_data_fname, n_jobs=4)
                    else:
                        raise ValueError('Invalid Time Frequency computation method. Please use "morlet" or "multitaper".')

                    if run_itc:
                        power, itc = power

                    if not return_average_tfr and output == 'power':
                        # Average epochs
                        power = power.average()
                        if run_itc:
                            itc = itc.average()

                # Apply baseline
                power.apply_baseline(baseline=run_params['plot_baseline'], mode=bline_mode)
                if run_itc:
                    itc.apply_baseline(baseline=run_params['plot_baseline'], mode=bline_mode)

                # Append data for GA
                power_data[param][param_value].append(power)
                if run_itc:
                    itc_data[param][param_value].append(itc)

                # Plot power time-frequency
                if plot_individuals:
                    for chs_id in chs_ids:
                        fname = f'Power_plotjoint_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                        plot_general.tfr_plotjoint_picks(tfr=power, plot_baseline=None, bline_mode=bline_mode, vlines_times=vlines_times,
                                                         timefreqs=None, chs_id=chs_id, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min, vmin=None,
                                                         vmax=None, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path_subj, fname=fname)

                        if run_itc:
                            # Plot ITC time-frequency
                            fname = f'ITC_plotjoint_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                            plot_general.tfr_plotjoint_picks(tfr=itc, plot_baseline=None, bline_mode=bline_mode, vlines_times=vlines_times,
                                                             timefreqs=None, chs_id=chs_id, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min, vmin=None,
                                                             vmax=None, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path_subj, fname=fname)

                # Free up memory
                del power
                if run_itc:
                    del itc

            # Compute grand average
            grand_avg_power = mne.grand_average(power_data[param][param_value])
            ga_power_data[param][param_value] = grand_avg_power
            if run_itc:
                grand_avg_itc = mne.grand_average(itc_data[param][param_value])
                ga_itc_data[param][param_value] = grand_avg_itc

            if save_data:
                # Save trf data
                grand_avg_power.save(trf_save_path + grand_avg_power_fname, overwrite=True)
                if run_itc:
                    grand_avg_itc.save(trf_save_path + grand_avg_itc_fname, overwrite=True)


        #--------- Permutation cluster test data (per condition) -----------#
        for chs_id in chs_ids:
            if run_itc:
                ga_permutations_list = [grand_avg_power, grand_avg_itc]
                titles_list = [output, 'ITC']
                if run_permutations_ga:
                    subj_permutations_list = [power_data[param][param_value], itc_data[param][param_value]]
                else:
                    subj_permutations_list = [None] * len(titles_list)
            else:
                ga_permutations_list = [grand_avg_power]
                titles_list = [output]
                if run_permutations_ga:
                    subj_permutations_list = [power_data[param][param_value]]
                else:
                    subj_permutations_list = [None] * len(titles_list)

            # Define selected channels list
            picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power.info)

            for grand_avg, subj_list, title in zip(ga_permutations_list, subj_permutations_list, titles_list):

                if run_permutations_ga:

                    permutations_test_data = [data.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).pick(picks).data for data in subj_list]
                    permutations_test_data_array = np.array([data for data in permutations_test_data])
                    permutations_test_data_array = permutations_test_data_array.swapaxes(1, 2).swapaxes(2, 3)

                    # Get channel adjacency
                    ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg.info, ch_type='mag', picks='mag')

                    # Define minimum significant channels to show on TF plot
                    min_sig_chs = len(picks) * significant_channels

                    # Run clusters permutations test
                    clusters_mask, clusters_mask_plot, significant_pvalues, significant_clusters = functions_analysis.run_permutations_test_tf(data=permutations_test_data_array, pval_threshold=pval_threshold,
                                                                                                   t_thresh=pval_threshold, n_permutations=n_permutations, min_sig_chs=min_sig_chs)

                    # Define image args to plot mask
                    image_args = {'mask': clusters_mask_plot, 'mask_style': 'contour'}

                    if type(t_thresh) == dict:
                        fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_tTFCE_pval{pval_threshold}_chs{significant_channels}'
                    else:
                        fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{pval_threshold}_chs{significant_channels}'

                    fig_title = f'{title}_{bline_mode}_pval_{functions_general.largest_cluster_pval(significant_pvalues, significant_clusters)}'
                else:
                    image_args = None
                    clusters_mask = None
                    fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                    fig_title = f'{title}_{bline_mode}'

                #--------- Plots ---------#
                # Power Plotjoint
                plot_general.tfr_plotjoint_picks(tfr=grand_avg, plot_baseline=run_params['plot_baseline'], bline_mode=ga_plot_bline_mode, vlines_times=vlines_times,
                                                 timefreqs=timefreqs_joint, image_args=image_args, clusters_mask=clusters_mask, chs_id=chs_id, plot_xlim=plot_xlim,
                                                 plot_max=plot_max, plot_min=plot_min, vmin=vmin_power, vmax=vmax_power, display_figs=display_figs, save_fig=save_fig,
                                                 title=fig_title, trf_fig_path=trf_fig_path, fname=fname, fontsize=22, ticksize=22)

                # Plot Power time-frequency in time scaled axes
                fname = f'GA_{title}_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                fig, ax_tf = plot_general.tfr_times(tfr=grand_avg, chs_id=chs_id, timefreqs_tfr=timefreqs_tfr, baseline=run_params['plot_baseline'],
                                                    bline_mode=ga_plot_bline_mode, plot_xlim=plot_xlim, vlines_times=vlines_times, vmin=vmin_power, vmax=vmax_power,
                                                    topo_vmin=vmin_power, topo_vmax=vmax_power, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path,
                                                    fname=fname, fontsize=22, ticksize=22)


# --------- ANOVA permutation cluster test across all conditions ----------- #
# Compare all groups simultaneously instead of pairwise differences
for param in param_values.keys():
    if len(param_values[param]) > 1 and run_comparison:

        # All condition values for this param
        conditions = param_values[param]
        n_cond = len(conditions)
        conditions_str = '_vs_'.join([str(c) for c in conditions])

        print(f'\n{"="*60}')
        print(f'Running ANOVA across all conditions: {param} = {conditions}')
        print(f'{"="*60}\n')

        # ---------- Compute common time window across conditions ---------- #
        # For 'ms' epoch_id, each MSS has a different duration, so we crop to the shortest
        # For 'cross2', plot_xlim covers only the cross2 period (same for all MSS) since
        # baseline (cross1) has already been applied
        cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

        # Get plot_xlim for each condition to find the common (shortest) window
        all_plot_xlims = []
        for cond_val in conditions:
            tmp_params = copy.copy(trial_params)
            if len(param_values.keys()) > 1:
                for key in param_values.keys():
                    tmp_params[key] = param_values[key][0]
            tmp_params[param] = cond_val
            _, _, cond_plot_xlim = functions_general.get_time_lims(epoch_id=tmp_params['epoch_id'], mss=tmp_params['mss'],
                                                                    plot_edge=plot_edge, map=map)
            all_plot_xlims.append(cond_plot_xlim)

        # Common time window: latest start, earliest end
        common_tmin = max([xlim[0] for xlim in all_plot_xlims])
        common_tmax = min([xlim[1] for xlim in all_plot_xlims])
        common_plot_xlim = (common_tmin, common_tmax)

        if any(xlim != all_plot_xlims[0] for xlim in all_plot_xlims):
            print(f'Conditions have different time windows. Cropping to common window: {common_plot_xlim}')
            print(f'  Individual windows: {dict(zip(conditions, all_plot_xlims))}')
        else:
            print(f'All conditions share the same time window: {common_plot_xlim}')

        # Crop all subject-level data to the common time window
        power_data_cropped = {}
        itc_data_cropped = {}
        ga_power_data_cropped = {}
        ga_itc_data_cropped = {}
        for cond_val in conditions:
            power_data_cropped[cond_val] = [data.copy().crop(tmin=common_tmin, tmax=common_tmax)
                                            for data in power_data[param][cond_val]]
            ga_power_data_cropped[cond_val] = mne.grand_average(power_data_cropped[cond_val])
            if run_itc:
                itc_data_cropped[cond_val] = [data.copy().crop(tmin=common_tmin, tmax=common_tmax)
                                              for data in itc_data[param][cond_val]]
                ga_itc_data_cropped[cond_val] = mne.grand_average(itc_data_cropped[cond_val])

        # Use common plot_xlim and the last condition's vlines/timefreqs for the comparison plots
        # (recompute vlines and timefreqs using the smallest mss for ms epoch_id)
        if param == 'mss':
            ref_mss = min(conditions)
        else:
            ref_mss = trial_params.get('mss', None)
            if isinstance(ref_mss, list):
                ref_mss = ref_mss[0]
        common_timefreqs_joint, common_timefreqs_tfr, common_vlines_times = functions_general.get_plots_timefreqs(
            epoch_id=trial_params['epoch_id'], mss=ref_mss, cross2_dur=cross2_dur,
            mss_duration=mss_duration, plot_xlim=common_plot_xlim,
            plot_min=plot_min, plot_max=plot_max)

        # ---------- Pairwise difference plots (like TF_diff) ---------- #
        for comparison in list(itertools.combinations(conditions, 2)):

            if all(type(element) == int for element in comparison):
                comparison = sorted(comparison, reverse=True)

            # Figure difference save path
            if param == 'epoch_id':
                tfr_fig_path_dif = trf_fig_path.replace(f'{param_values[param][-1]}', f'{comparison[0]}-{comparison[1]}')
            else:
                tfr_fig_path_dif = trf_fig_path.replace(f'{param}{param_values[param][-1]}', f'{param}{comparison[0]}-{comparison[1]}')

            print(f'Plotting pairwise difference: {param} {comparison[0]} - {comparison[1]}')

            # Compute subjects difference (using cropped data with matching time axes)
            power_data_dif = []
            for i in range(len(power_data_cropped[comparison[0]])):
                power_data_dif.append(power_data_cropped[comparison[0]][i] - power_data_cropped[comparison[1]][i])
            if run_itc:
                itc_data_dif = []
                for i in range(len(itc_data_cropped[comparison[0]])):
                    itc_data_dif.append(itc_data_cropped[comparison[0]][i] - itc_data_cropped[comparison[1]][i])

            # Compute grand average of difference
            grand_avg_power_dif = mne.grand_average(power_data_dif)
            if run_itc:
                grand_avg_itc_dif = mne.grand_average(itc_data_dif)

            for chs_id in chs_ids:
                if run_itc:
                    ga_dif_list = [grand_avg_power_dif, grand_avg_itc_dif]
                    titles_list = [output, 'ITC']
                    if run_permutations_dif:
                        subj_dif_list = [power_data_dif, itc_data_dif]
                    else:
                        subj_dif_list = [None, None]
                else:
                    ga_dif_list = [grand_avg_power_dif]
                    titles_list = [output]
                    if run_permutations_dif:
                        subj_dif_list = [power_data_dif]
                    else:
                        subj_dif_list = [None]

                for grand_avg_d, subj_list_d, title in zip(ga_dif_list, subj_dif_list, titles_list):

                    if run_permutations_dif:

                        picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_d.info)
                        permutations_test_data = [data.copy().crop(tmin=common_plot_xlim[0], tmax=common_plot_xlim[1]).pick(picks).data for data in subj_list_d]
                        permutations_test_data_array = np.array([data for data in permutations_test_data])
                        permutations_test_data_array = permutations_test_data_array.swapaxes(1, 2).swapaxes(2, 3)

                        # Get channel adjacency
                        ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg_d.info, ch_type='mag', picks='mag')

                        # Define minimum significant channels to show on TF plot
                        min_sig_chs = len(picks) * significant_channels

                        # Run clusters permutations test
                        clusters_mask, clusters_mask_plot, significant_pvalues, significant_clusters = functions_analysis.run_permutations_test_tf(data=permutations_test_data_array,
                                                                                                                             pval_threshold=pval_threshold,
                                                                                                                             t_thresh=pval_threshold,
                                                                                                                             n_permutations=n_permutations,
                                                                                                                             min_sig_chs=min_sig_chs)
                        # Define image args to plot mask
                        image_args = {'mask': clusters_mask_plot, 'mask_style': 'contour'}

                        if isinstance(t_thresh, dict):
                            fname = f'GA_{title}_diff_{comparison[0]}-{comparison[1]}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_tTFCE_pval{pval_threshold}_chs{significant_channels}'
                        else:
                            fname = f'GA_{title}_diff_{comparison[0]}-{comparison[1]}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{pval_threshold}_chs{significant_channels}'
                    else:
                        image_args = None
                        clusters_mask = None
                        fname = f'GA_{title}_diff_{comparison[0]}-{comparison[1]}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'

                    # --------- Plots ---------#
                    # Build title with max p-value from pairwise permutation test
                    if run_permutations_dif and significant_pvalues:
                        # Get p value of largest cluster
                        plot_title = f'{title}_{bline_mode}_pval_{functions_general.largest_cluster_pval(significant_pvalues, significant_clusters)}'
                    else:
                        plot_title = f'{title}_{bline_mode}'

                    # Power Plotjoint
                    plot_general.tfr_plotjoint_picks(tfr=grand_avg_d, plot_baseline=None, bline_mode=bline_mode, vlines_times=common_vlines_times, timefreqs=common_timefreqs_joint,
                                                     image_args=image_args, clusters_mask=clusters_mask, chs_id=chs_id, plot_xlim=common_plot_xlim, plot_max=plot_max, plot_min=plot_min,
                                                     title=plot_title, vmin=vmin_power, vmax=vmax_power, display_figs=display_figs, save_fig=save_fig, trf_fig_path=tfr_fig_path_dif, fname=fname)


        # ---------- ANOVA: Permutation cluster F-test across all conditions ---------- #
        if run_permutations_anova:

            print(f'\nRunning permutation cluster ANOVA F-test across {n_cond} conditions: {conditions}')

            # Figure ANOVA save path
            if param == 'epoch_id':
                tfr_fig_path_anova = trf_fig_path.replace(f'{param_values[param][-1]}', f'ANOVA_{conditions_str}')
            else:
                tfr_fig_path_anova = trf_fig_path.replace(f'{param}{param_values[param][-1]}', f'{param}_ANOVA_{conditions_str}')

            for chs_id in chs_ids:

                picks = functions_general.pick_chs(chs_id=chs_id, info=ga_power_data[param][conditions[0]].info)

                if run_itc:
                    data_types_list = ['power', 'itc']
                    subj_data_dicts = [power_data, itc_data]
                    ga_data_dicts = [ga_power_data, ga_itc_data]
                    titles_list = [output, 'ITC']
                else:
                    data_types_list = ['power']
                    subj_data_dicts = [power_data]
                    ga_data_dicts = [ga_power_data]
                    titles_list = [output]

                for data_type_label, subj_dict, ga_dict, title in zip(data_types_list, subj_data_dicts, ga_data_dicts, titles_list):

                    # Build data arrays for each condition: list of arrays, one per condition
                    # Using cropped data so all conditions share the same time axis
                    # Each array shape: (n_subjects, n_channels, n_freqs, n_times)
                    condition_arrays = []
                    for cond_val in conditions:
                        if data_type_label == 'power':
                            subj_list = power_data_cropped[cond_val]
                        else:
                            subj_list = itc_data_cropped[cond_val]
                        cond_data = [data.copy().pick(picks).data for data in subj_list]
                        cond_array = np.array(cond_data)
                        condition_arrays.append(cond_array)

                    # Rearrange each condition array:
                    # (n_subjects, n_channels, n_freqs, n_times) -> (n_subjects, n_times, n_freqs, n_channels)
                    n_subjects = condition_arrays[0].shape[0]
                    condition_arrays_swapped = []
                    for cond_array in condition_arrays:
                        condition_arrays_swapped.append(cond_array.swapaxes(1, 2).swapaxes(2, 3))

                    # Define repeated-measures ANOVA stat function
                    factor_levels = [n_cond]  # one factor with n_cond levels

                    def stat_fun_rm(*args):
                        # args: n_cond arrays, each of shape (n_subjects, n_tests)
                        # f_mway_rm expects 3D: (n_subjects, n_conditions, n_tests)
                        # Stack conditions along axis 1
                        data = np.stack(args, axis=1)  # (n_subjects, n_cond, n_tests)
                        return f_mway_rm(data, factor_levels=factor_levels,
                                         effects='A', return_pvals=False)[0]

                    # Define F-threshold for the repeated-measures ANOVA
                    f_threshold = f_threshold_mway_rm(n_subjects=n_subjects, factor_levels=factor_levels,
                                                      effects='A', pvalue=desired_fval)

                    # Get channel adjacency
                    ch_adjacency_sparse = functions_general.get_channel_adjacency(info=ga_dict[param][conditions[0]].info, ch_type='mag', picks='mag')

                    # Define minimum significant channels to show on TF plot
                    min_sig_chs = len(picks) * significant_channels

                    # Run permutation cluster test with repeated-measures F-statistic
                    F_obs, clusters, p_values, H0 = permutation_cluster_test(
                        condition_arrays_swapped,
                        threshold=f_threshold,
                        stat_fun=stat_fun_rm,
                        n_permutations=n_permutations,
                        out_type='mask',
                        n_jobs=4
                    )

                    # Build clusters mask from significant clusters
                    good_clusters_idx = np.where(p_values < pval_threshold)[0]
                    significant_clusters_anova = [clusters[idx] for idx in good_clusters_idx]
                    significant_pvalues_anova = [p_values[idx] for idx in good_clusters_idx]

                    # Reshape to data's shape by adding all clusters into one bool array
                    # Shape matches (n_times, n_freqs, n_channels)
                    clusters_mask_anova = np.zeros(condition_arrays_swapped[0][0].shape)
                    if len(significant_clusters_anova):
                        for significant_cluster in significant_clusters_anova:
                            clusters_mask_anova += significant_cluster

                        if min_sig_chs:
                            clusters_mask_plot_anova = clusters_mask_anova.sum(axis=-1) > min_sig_chs
                        else:
                            clusters_mask_plot_anova = clusters_mask_anova.sum(axis=-1)
                        clusters_mask_plot_anova = clusters_mask_plot_anova.astype(bool)

                    else:
                        clusters_mask_plot_anova = None

                    # Print results
                    if len(significant_pvalues_anova):
                        print(f'  {title} - {chs_id}: Found {len(significant_pvalues_anova)} significant cluster(s), '
                              f'p-values: {[round(p, 4) for p in significant_pvalues_anova]}')
                    else:
                        print(f'  {title} - {chs_id}: No significant clusters found')

                    # --------- ANOVA Plots --------- #
                    # 1) Plot each condition's grand average with the ANOVA significance mask overlaid
                    for cond_val in conditions:
                        if data_type_label == 'power':
                            ga_cond = ga_power_data_cropped[cond_val]
                        else:
                            ga_cond = ga_itc_data_cropped[cond_val]

                        if clusters_mask_plot_anova is not None:
                            image_args_anova = {'mask': clusters_mask_plot_anova, 'mask_style': 'contour'}
                        else:
                            image_args_anova = None

                        fname = f'GA_{title}_ANOVA_{conditions_str}_{param}{cond_val}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_f{round(f_threshold, 2)}_pval{pval_threshold}_chs{significant_channels}'

                        # Build title with max p-value from ANOVA F-test
                        if significant_pvalues_anova:
                            cond_title = f'{title}_{bline_mode}_ANOVA_largest_cluster_pval_{functions_general.largest_cluster_pval(significant_pvalues_anova, significant_clusters_anova):.4f}'
                        else:
                            cond_title = f'{title}_{bline_mode}_ANOVA_no_sig_clusters'

                        plot_general.tfr_plotjoint_picks(tfr=ga_cond, plot_baseline=None, bline_mode=ga_plot_bline_mode, vlines_times=common_vlines_times,
                                                         timefreqs=common_timefreqs_joint, image_args=image_args_anova, clusters_mask=clusters_mask_anova, chs_id=chs_id,
                                                         plot_xlim=common_plot_xlim, plot_max=plot_max, plot_min=plot_min, vmin=vmin_power, vmax=vmax_power,
                                                         display_figs=display_figs, save_fig=save_fig, trf_fig_path=tfr_fig_path_anova, fname=fname,
                                                         title=cond_title, fontsize=22, ticksize=22)

                    # 2) Plot F-statistic map
                    # Use one cropped GA as template to get times and freqs
                    ga_template = ga_power_data_cropped[conditions[0]].copy().pick(picks)

                    # F_obs shape is (n_times, n_freqs, n_channels) — average over channels for 2D TF map
                    F_obs_avg = F_obs.mean(axis=-1)  # (n_times, n_freqs)

                    fig_f, ax_f = plt.subplots(1, 1, figsize=(12, 6))
                    extent = [ga_template.times[0], ga_template.times[-1], ga_template.freqs[0], ga_template.freqs[-1]]
                    im = ax_f.imshow(F_obs_avg, aspect='auto', origin='lower', extent=extent, cmap='hot')
                    plt.colorbar(im, ax=ax_f, label='F-statistic')

                    # Overlay significance contour
                    if clusters_mask_plot_anova is not None:
                        ax_f.contour(clusters_mask_plot_anova, levels=[0.5], colors='green', linewidths=2,
                                     extent=extent, origin='lower')

                    ax_f.set_xlabel('Time (s)', fontsize=18)
                    ax_f.set_ylabel('Frequency (Hz)', fontsize=18)
                    if significant_pvalues_anova:
                        ax_f.set_title(f'ANOVA F-statistic: {param} {conditions_str}\n{title} - {chs_id} - largest cluster pval: {functions_general.largest_cluster_pval(significant_pvalues_anova, significant_clusters_anova):.4f}', fontsize=20)
                    else:
                        ax_f.set_title(f'ANOVA F-statistic: {param} {conditions_str}\n{title} - {chs_id} - no sig. clusters', fontsize=20)
                    ax_f.tick_params(labelsize=16)

                    # Add vlines
                    if common_vlines_times is not None:
                        for vt in common_vlines_times:
                            ax_f.axvline(x=vt, color='gray', linestyle='--', linewidth=1.5)

                    fig_f.tight_layout()

                    if save_fig:
                        fname_f = f'GA_{title}_ANOVA_Fstat_{conditions_str}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_f{round(f_threshold, 2)}_pval{pval_threshold}_chs{significant_channels}'
                        save.fig(fig=fig_f, path=tfr_fig_path_anova, fname=fname_f)

                    plt.close(fig_f)

                    # 3) Post-hoc pairwise difference plots with ANOVA significance mask overlaid
                    for comp in list(itertools.combinations(conditions, 2)):
                        if all(type(element) == int for element in comp):
                            comp = sorted(comp, reverse=True)

                        # Compute subjects difference for this pair (using cropped data)
                        data_dif_posthoc = []
                        if data_type_label == 'power':
                            subj_a = power_data_cropped[comp[0]]
                            subj_b = power_data_cropped[comp[1]]
                        else:
                            subj_a = itc_data_cropped[comp[0]]
                            subj_b = itc_data_cropped[comp[1]]
                        for i in range(len(subj_a)):
                            data_dif_posthoc.append(subj_a[i] - subj_b[i])

                        grand_avg_dif_posthoc = mne.grand_average(data_dif_posthoc)

                        if clusters_mask_plot_anova is not None:
                            image_args_posthoc = {'mask': clusters_mask_plot_anova, 'mask_style': 'contour'}
                        else:
                            image_args_posthoc = None

                        fname = f'GA_{title}_ANOVA_{conditions_str}_diff_{comp[0]}-{comp[1]}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_f{round(f_threshold, 2)}_pval{pval_threshold}_chs{significant_channels}'

                        # Build title with p-value from ANOVA F-test
                        if significant_pvalues_anova:
                            posthoc_title = f'{title}_{bline_mode}_ANOVA_largest_cluster_pval_{functions_general.largest_cluster_pval(significant_pvalues_anova, significant_clusters_anova):.4f}'
                        else:
                            posthoc_title = f'{title}_{bline_mode}_ANOVA_no_sig_clusters'

                        plot_general.tfr_plotjoint_picks(tfr=grand_avg_dif_posthoc, plot_baseline=None, bline_mode=bline_mode, vlines_times=common_vlines_times,
                                                         timefreqs=common_timefreqs_joint, image_args=image_args_posthoc, clusters_mask=clusters_mask_anova,
                                                         chs_id=chs_id, plot_xlim=common_plot_xlim, plot_max=plot_max, plot_min=plot_min,
                                                         title=posthoc_title, vmin=vmin_power, vmax=vmax_power, display_figs=display_figs, save_fig=save_fig,
                                                         trf_fig_path=tfr_fig_path_anova, fname=fname, fontsize=22, ticksize=22)
