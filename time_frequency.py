import functions_general
import functions_analysis
import load
import mne
import plot_general
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mne.stats import permutation_cluster_1samp_test
import itertools


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

# Trial selection and filters parameters. A field with 2 values will compute the difference between the conditions specified
trial_params = {'epoch_id': 'tgt_fix_ms',
                'corrans': [True, False],
                'tgtpres': True,
                'mss': None,
                'reject': None,
                'evtdur': None,
                }
run_comparison = True

# Select channels
chs_ids = ['occipital']  # region_hemisphere

use_ica_data = True

# Power time frequency params
n_cycles_div = 2.
l_freq = 1
h_freq = 40
run_itc = True
plot_edge = 0.15

# Plots parameters
# Colorbar
vmax_power = 0.2
vmin_power = -0.2
vmin_itc, vmax_itc = None, None
# plot_joint max and min topoplots
plot_max, plot_min = True, True

# Baseline method
# logratio: dividing by the mean of baseline values and taking the log
# ratio: dividing by the mean of baseline values
# mean: subtracting the mean of baseline values
bline_mode = 'logratio'

# Topoplot bands
topo_bands = ['Alpha', 'Alpha', 'Theta', 'Alpha']

# Time Frequency config
tf_method = 'morlet'  # 'morlet' or 'multitaper'
return_average_tfr = True
output = 'power'

# Permutations cluster test parameters
run_permutations_ga = False
run_permutations_dif = True
n_permutations = 1024
degrees_of_freedom = len(exp_info.subjects_ids) - 1
desired_tval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# t_thresh = dict(start=0, step=0.2)
pval_threshold = 0.05
significant_channels = 0.5  # Percent of total region channels

# Save data of each id
power_data = {}
itc_data = {}

# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}

# --------- Run ---------#
for param in param_values.keys():
    power_data[param] = {}
    itc_data[param] = {}
    for param_value in param_values[param]:
        # Get run parameters from
        run_params = trial_params
        # Set first value of parameters comparisons to avoid having lists in run params
        if len(param_values.keys()) > 1:
            for key in param_values.keys():
                run_params[key] = param_values[key][0]
        # Set comparison key value
        run_params[param] = param_value

        #---------- Setup ----------#
        # Windows durations
        cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
        if 'vs' in run_params['epoch_id']:
            trial_dur = vs_dur[run_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
        else:
            trial_dur = None

        # Get time windows from epoch_id name
        map = dict(ms={'tmin': -cross1_dur, 'tmax': mss_duration[run_params['mss']], 'plot_xlim': (-cross1_dur + plot_edge, mss_duration[run_params['mss']] - plot_edge)},
                   fix_vs={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
                   tgt_fix_ms={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
                   tgt_fix_vs={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
                   sac_emap={'tmin': -0.5, 'tmax': 3, 'plot_xlim': (-0.3, 2.5)},
                   hl_start={'tmin': -3, 'tmax': 35, 'plot_xlim': (-2.5, 33)})
        tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'], plot_edge=plot_edge, map=map)

        # Define time-frequency bands to plot in plot_joint
        timefreqs_joint, timefreqs_tfr, vlines_times = functions_general.get_plots_timefreqs(epoch_id=run_params['epoch_id'], mss=run_params['mss'], cross2_dur=cross2_dur,
                                                                                             mss_duration=mss_duration, topo_bands=topo_bands, plot_xlim=plot_xlim,
                                                                                             plot_min=plot_min, plot_max=plot_max)

        # Get baseline duration for epoch_id
        baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'], tmin=tmin, tmax=tmax,
                                                                          cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                          cross2_dur=cross2_dur, plot_edge=plot_edge)

        # Specific run path for saving data and plots
        if use_ica_data:
            data_type = 'ICA'
        else:
            data_type = 'RAW'

        # Save ids
        save_id = f"{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}_trialdur{trial_dur}_evtdur{run_params['evtdur']}"
        plot_id = f"{save_id}_{round(plot_xlim[0],2)}_{round(plot_xlim[1], 2)}_bline{baseline}_cyc{int(n_cycles_div)}/"

        # Save data paths
        if return_average_tfr:
            trf_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{tf_method}/{save_id}_{tmin}_{tmax}_bline{baseline}_cyc{int(n_cycles_div)}/'
        else:
            trf_save_path = paths().save_path() + f'Time_Frequency_Epochs_{data_type}/{tf_method}/{save_id}_{tmin}_{tmax}_bline{baseline}_cyc{int(n_cycles_div)}/'
        epochs_save_path = paths().save_path() + f'Epochs_{data_type}/Band_None/{save_id}_{tmin}_{tmax}_bline{baseline}/'
        # Save figures paths
        trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/{tf_method}/' + plot_id

        # Grand average data variable
        grand_avg_power_fname = f'Grand_Average_power_{l_freq}_{h_freq}_tfr.h5'
        grand_avg_itc_fname = f'Grand_Average_itc_{l_freq}_{h_freq}_tfr.h5'

        #------------ Run -------------#
        try:
            # Raise error if run_permutations == True to load data from all subjects
            if run_permutations_ga or run_permutations_dif:
                raise ValueError

            # Load previous power data
            grand_avg_power = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_power_fname)[0]
            if run_itc:
                # Load previous itc data
                grand_avg_itc = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_itc_fname)[0]

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
                itc_data_fname = f'ITC_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
                epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
                # Subject plots path
                trf_fig_path_subj = trf_fig_path + f'{subject.subject_id}/'

                try:
                    # Load previous data
                    power = mne.time_frequency.read_tfrs(trf_save_path + power_data_fname)[0]
                    if run_itc:
                        itc = mne.time_frequency.read_tfrs(trf_save_path + itc_data_fname)[0]
                except:
                    try:
                        # Load epoched data
                        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                    except:
                        # Load meg data
                        if use_ica_data:
                            meg_data = load.ica_data(subject=subject)
                        else:
                            meg_data = subject.load_preproc_meg_data()

                        # Epoch data
                        epochs, events = functions_analysis.epoch_data(subject=subject, mss=run_params['mss'], corr_ans=run_params['corrans'], trial_dur=trial_dur,
                                                                       tgt_pres=run_params['tgtpres'], baseline=baseline, reject=run_params['reject'],
                                                                       epoch_id=run_params['epoch_id'], meg_data=meg_data, tmin=tmin, tmax=tmax,
                                                                       save_data=save_data, epochs_save_path=epochs_save_path,
                                                                       epochs_data_fname=epochs_data_fname)

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
                power.apply_baseline(baseline=plot_baseline, mode=bline_mode)
                if run_itc:
                    itc.apply_baseline(baseline=plot_baseline, mode=bline_mode)

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

            # Compute grand average
            grand_avg_power = mne.grand_average(power_data[param][param_value])
            if run_itc:
                grand_avg_itc = mne.grand_average(itc_data[param][param_value])

            if save_data:
                # Save trf data
                grand_avg_power.save(trf_save_path + grand_avg_power_fname, overwrite=True)
                if run_itc:
                    grand_avg_itc.save(trf_save_path + grand_avg_itc_fname, overwrite=True)

        #--------- Permutation cluster test data -----------#
        # for chs_id, timefreqs_joint in zip(chs_ids, [[(1.25, 12)], [(0.6, 10)]]):
        for chs_id in chs_ids:
            if run_itc:
                ga_permutations_list = [grand_avg_power, grand_avg_itc]
                titles_list = ['Power', 'ITC']
                if run_permutations_ga:
                    subj_permutations_list = [power_data[param][param_value], itc_data[param][param_value]]
                else:
                    subj_permutations_list = []
            else:
                ga_permutations_list = [grand_avg_power]
                titles_list = ['Power']
                if run_permutations_ga:
                    subj_permutations_list = [power_data[param][param_value]]
                else:
                    subj_permutations_list = []

            for grand_avg, subj_list, title in zip(ga_permutations_list, subj_permutations_list, titles_list):

                if run_permutations_ga:

                    picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg.info)
                    permutations_test_data = [data.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).pick(picks).data for data in subj_list]
                    # permutations_test_data_array = np.array([data.mean(0) for data in permutations_test_data])
                    permutations_test_data_array = np.array([data for data in permutations_test_data])
                    permutations_test_data_array = permutations_test_data_array.swapaxes(1, 2).swapaxes(2, 3)

                    # Get channel adjacency
                    ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg.info, ch_type='mag', picks='mag')
                    # Clusters out type
                    if type(t_thresh) == dict:
                        out_type = 'indices'
                    else:
                        out_type = 'mask'

                    # Permutations cluster test (TFCE if t_thresh as dict)
                    t_tfce, clusters, p_tfce, H0 = permutation_cluster_1samp_test(X=permutations_test_data_array, threshold=t_thresh, n_permutations=n_permutations, out_type=out_type, n_jobs=4)

                    # Make clusters mask
                    if type(t_thresh) == dict:
                        # If TFCE use p-vaues of voxels directly
                        p_tfce = p_tfce.reshape(permutations_test_data_array.shape[-2:])  # Reshape to data's shape
                        clusters_mask_plot = p_tfce < pval_threshold
                        clusters_mask = None
                    else:
                        # Get significant clusters
                        good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
                        significant_clusters = [clusters[idx] for idx in good_clusters_idx]

                        # Reshape to data's shape by adding all clusters into one bool array
                        clusters_mask = np.zeros(permutations_test_data_array[0].shape)
                        if len(significant_clusters):
                            for significant_cluster in significant_clusters:
                                clusters_mask += significant_cluster
                                clusters_mask_plot = clusters_mask.sum(axis=-1) > len(picks) * significant_channels
                                clusters_mask_plot = clusters_mask_plot.astype(bool)

                            # Cluster contour
                            image_args = {'mask': clusters_mask_plot, 'mask_style': 'contour'}
                        else:
                            image_args = None

                    if type(t_thresh) == dict:
                        fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_tTFCE_pval{pval_threshold}_chs{significant_channels}'
                    else:
                        fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{pval_threshold}_chs{significant_channels}'
                else:
                    image_args = None
                    clusters_mask = None
                    if type(t_thresh) == dict:
                        fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                    else:
                        fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'

                #--------- Plots ---------#
                # Power Plotjoint
                plot_general.tfr_plotjoint_picks(tfr=grand_avg, plot_baseline=None, bline_mode=bline_mode, vlines_times=vlines_times, timefreqs=timefreqs_joint,
                                                 image_args=image_args, clusters_mask=clusters_mask, chs_id=chs_id, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min,
                                                 vmin=vmin_power, vmax=vmax_power, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

                # Plot Power time-frequency in time scalde axes
                fname = f'GA_{title}_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                plot_general.tfr_times(tfr=grand_avg, chs_id=chs_id, timefreqs_tfr=timefreqs_tfr, baseline=None, bline_mode=bline_mode,
                                       plot_xlim=plot_xlim, vlines_times=vlines_times, topo_vmin=vmin_power, topo_vmax=vmax_power, subject=None, display_figs=display_figs,
                                       save_fig=save_fig, fig_path=trf_fig_path, fname=fname, vmin=vmin_power, vmax=vmax_power, fontsize=16, ticksize=18)


# --------- Permutation cluster test on difference between conditions -----------#
# Take difference of conditions if applies
for param in param_values.keys():
    if len(param_values[param]) > 1 and run_comparison:
        for comparison in list(itertools.combinations(param_values[param], 2)):

            if all(type(element) == int for element in comparison):
                comparison = sorted(comparison, reverse=True)

            # Figure difference save path
            if param == 'epoch_id':
                tfr_fig_path_dif = trf_fig_path.replace(f'{param_values[param][-1]}', f'{comparison[0]}-{comparison[1]}')
            else:
                tfr_fig_path_dif = trf_fig_path.replace(f'{param}{param_values[param][-1]}', f'{param}{comparison[0]}-{comparison[1]}')

            print(f'Taking difference between conditions: {param} {comparison[0]} - {comparison[1]}')

            # Compute subjects difference
            power_data_dif = []
            for i in range(len(power_data[param][comparison[np.argmax(comparison)]])):
                power_data_dif.append(power_data[param][comparison[0]][i] - power_data[param][comparison[1]][i])
            if run_itc:
                itc_data_dif = []
                for i in range(len(itc_data[param][comparison[np.argmax(comparison)]])):
                    itc_data_dif.append(itc_data[param][comparison[0]][i] - itc_data[param][comparison[1]][i])

            # To use subjects difference
            grand_avg_power_dif = mne.grand_average(power_data_dif)
            if run_itc:
                grand_avg_itc_dif = mne.grand_average(itc_data_dif)

            for chs_id in chs_ids:
                if run_itc:
                    ga_permutations_list = [grand_avg_power_dif, grand_avg_itc_dif]
                    titles_list = ['Power', 'ITC']
                    if run_permutations_dif:
                        subj_permutations_list = [power_data_dif, itc_data_dif]
                    else:
                        subj_permutations_list = []
                else:
                    ga_permutations_list = [grand_avg_power_dif]
                    titles_list = ['Power']
                    if run_permutations_dif:
                        subj_permutations_list = [power_data_dif]
                    else:
                        subj_permutations_list = []

                for grand_avg, subj_list, title in zip(ga_permutations_list, subj_permutations_list, titles_list):

                    if run_permutations_dif:

                        picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg.info)
                        permutations_test_data = [data.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).pick(picks).data for data in subj_list]
                        # permutations_test_data_array = np.array([data.mean(0) for data in permutations_test_data])
                        permutations_test_data_array = np.array([data for data in permutations_test_data])
                        permutations_test_data_array = permutations_test_data_array.swapaxes(1, 2).swapaxes(2, 3)

                        # Get channel adjacency
                        ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg.info, ch_type='mag', picks='mag')
                        # Clusters out type
                        if type(t_thresh) == dict:
                            out_type = 'indices'
                        else:
                            out_type = 'mask'

                        # Permutations cluster test (TFCE if t_thresh as dict)
                        t_tfce, clusters, p_tfce, H0 = permutation_cluster_1samp_test(X=permutations_test_data_array, threshold=t_thresh, n_permutations=n_permutations,
                                                                                      out_type=out_type, n_jobs=4)

                        # Make clusters mask
                        if type(t_thresh) == dict:
                            # If TFCE use p-vaues of voxels directly
                            p_tfce = p_tfce.reshape(permutations_test_data_array.shape[-2:])  # Reshape to data's shape
                            clusters_mask_plot = p_tfce < pval_threshold
                            clusters_mask = None

                            # Cluster contour
                            image_args = {'mask': clusters_mask_plot, 'mask_style': 'contour'}
                        else:
                            # Get significant clusters
                            good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
                            significant_clusters = [clusters[idx] for idx in good_clusters_idx]

                            # Reshape to data's shape by adding all clusters into one bool array
                            clusters_mask = np.zeros(permutations_test_data_array[0].shape)
                            if len(significant_clusters):
                                for significant_cluster in significant_clusters:
                                    clusters_mask += significant_cluster
                                    clusters_mask_plot = clusters_mask.sum(axis=-1) > len(picks) * significant_channels
                                    clusters_mask_plot = clusters_mask_plot.astype(bool)

                                # Cluster contour
                                image_args = {'mask': clusters_mask_plot, 'mask_style': 'contour'}
                            else:
                                image_args = None

                        if type(t_thresh) == dict:
                            fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_tTFCE_pval{pval_threshold}_chs{significant_channels}'
                        else:
                            fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{pval_threshold}_chs{significant_channels}'
                    else:
                        image_args = None
                        clusters_mask = None
                        if type(t_thresh) == dict:
                            fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                        else:
                            fname = f'GA_{title}_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'

                    # --------- Plots ---------#
                    # Power Plotjoint
                    plot_general.tfr_plotjoint_picks(tfr=grand_avg, plot_baseline=None, bline_mode=bline_mode, vlines_times=vlines_times, timefreqs=timefreqs_joint,
                                                     image_args=image_args, clusters_mask=clusters_mask, chs_id=chs_id, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min,
                                                     vmin=vmin_power, vmax=vmax_power, display_figs=display_figs, save_fig=save_fig, trf_fig_path=tfr_fig_path_dif, fname=fname)
