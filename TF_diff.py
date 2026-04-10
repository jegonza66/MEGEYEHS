import functions_general
import functions_analysis
import load
import mne
import os
import plot_general
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from mne.stats import permutation_cluster_1samp_test

#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
plot_individuals = False
use_saved_data = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# Select channels
chs_ids = ['parietal_occipital']
# ICA / RAW
use_ica_data = True
corr_ans = None
tgt_pres = None
epoch_id = 'ms'
trial_dur = None
evt_dur = None

# Time frequency params
n_cycles_div = 2.
l_freq = 1
h_freq = 40
run_itc = False
return_average_tfr = True
output = 'power'

# Baseline method
# logratio: dividing by the mean of baseline values and taking the log
# ratio: dividing by the mean of baseline values
# mean: subtracting the mean of baseline values
bline_mode = 'logratio'
#----------#

# Duration
# Windows durations
cross1_dur, cross2_dur, mss_duration, _ = functions_general.get_duration()
vs_dur = 4

# Define time windows for each epoch id
cross2_diff_onset = 0
cross2_diff_offset = 0

if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

for mssh, mssl in [(2, 1), (4, 1), (4, 2)]:

    tmin_mssl = -cross1_dur
    tmax_mssl = mss_duration[mssl] + cross2_dur + vs_dur
    tmin_mssh = -cross1_dur
    tmax_mssh = mss_duration[mssh] + cross2_dur + vs_dur
    baseline = (tmin_mssl, 0)

    mssl_ms_window_times = (-cross1_dur, mss_duration[mssl])
    mssl_cross2_window_times = (mss_duration[mssl] + cross2_diff_onset, mss_duration[mssl] + cross2_dur + cross2_diff_offset)
    mssh_cross2_window_times = (mss_duration[mssh] + cross2_diff_onset, mss_duration[mssh] + cross2_dur + cross2_diff_offset)
    cross2_filename_times = f'{mssl_cross2_window_times[0]-mssl_ms_window_times[1]}_{mssl_cross2_window_times[1]-mssl_ms_window_times[1]}'

    # Specific run path for loading and saving data
    mss_diff_name = f'{mssh}-{mssl}'
    run_path_mssl = f'/{epoch_id}_mss{mssl}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}_evtdur{evt_dur}_{tmin_mssl}_{tmax_mssl}_bline{baseline}'
    run_path_mssh = f'/{epoch_id}_mss{mssh}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}_evtdur{evt_dur}_{tmin_mssh}_{tmax_mssh}_bline{baseline}'
    run_path_diff = f'/{epoch_id}_mss{mss_diff_name}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}_evtdur{evt_dur}_{tmin_mssl}_{tmax_mssl}_bline{baseline}'

    # Data paths for TFR
    if return_average_tfr:
        main_path = paths().save_path() + f'Time_Frequency_{data_type}/morlet/'
    else:
        main_path = paths().save_path() + f'Time_Frequency_Epochs_{data_type}/morlet/'

    trf_path_mssl = main_path + f'' + run_path_mssl + f'_cyc{int(n_cycles_div)}/'
    trf_path_mssh = main_path + f'' + run_path_mssh + f'_cyc{int(n_cycles_div)}/'
    trf_diff_save_path = main_path + f'' + run_path_diff + f'_cyc{int(n_cycles_div)}/'

    # Data paths for epochs
    epochs_path_mssl = paths().save_path() + f'Epochs_{data_type}/Band_None/' + run_path_mssl + '/'
    epochs_path_mssh = paths().save_path() + f'Epochs_{data_type}/Band_None/' + run_path_mssh + '/'

    # Save figures paths
    trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/morlet/' + run_path_diff + f'_cyc{int(n_cycles_div)}/'

    # Grand average data variable
    grand_avg_power_ms_fname = f'Grand_Average_power_ms_{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_itc_ms_fname = f'Grand_Average_itc_ms_{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_power_cross2_fname = f'Grand_Average_power_cross2_{cross2_filename_times}_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_itc_cross2_fname = f'Grand_Average_itc_cross2_{cross2_filename_times}_{l_freq}_{h_freq}_tfr.h5'

    averages_power_ms_diff = []
    averages_itc_ms_diff = []
    averages_power_cross2_diff = []
    averages_itc_cross2_diff = []

    # Variables to store data as array for TFCE
    data_power_ms_diff = []
    data_power_cross2_diff = []

    for subject_code in exp_info.subjects_ids:
        # Define save path and file name for loading and saving epoched, evoked, and GA data
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

        # Load data filenames
        power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
        itc_data_fname = f'ITC_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

        # Subject plots path
        trf_fig_path_subj = trf_fig_path + f'{subject.subject_id}/'

        try:
            if not use_saved_data:
                raise ValueError('Not using saved data')
            # MS difference
            file_path = trf_diff_save_path + f'ms_{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_' + power_data_fname.replace(f'{l_freq}_{h_freq}', '*')
            power_diff_ms = load.time_frequency_range(file_path=file_path, l_freq=l_freq, h_freq=h_freq)
            # Cross 2 difference
            file_path = trf_diff_save_path + f'cross2_{cross2_filename_times}_' + power_data_fname.replace(f'{l_freq}_{h_freq}', '*')
            power_diff_cross2 = load.time_frequency_range(file_path=file_path, l_freq=l_freq, h_freq=h_freq)

            if run_itc:
                # MS difference
                file_path = trf_diff_save_path + f'ms_{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_' + itc_data_fname.replace(f'{l_freq}_{h_freq}', '*')
                itc_diff_ms = load.time_frequency_range(file_path=file_path, l_freq=l_freq, h_freq=h_freq)

                # MS difference
                file_path = trf_diff_save_path + f'cross2_{cross2_filename_times}_' + itc_data_fname.replace(f'{l_freq}_{h_freq}', '*')
                itc_diff_cross2 = load.time_frequency_range(file_path=file_path, l_freq=l_freq, h_freq=h_freq)

        except:
            # Compute difference
            try:
                if not use_saved_data:
                    raise ValueError('Not using saved data')
                # MSSL
                power_mssl = load.time_frequency_range(file_path=trf_path_mssl + power_data_fname, l_freq=l_freq, h_freq=h_freq)
                # MSSSH
                power_mssh = load.time_frequency_range(file_path=trf_path_mssh + power_data_fname, l_freq=l_freq, h_freq=h_freq)

                if run_itc:
                    # MSSL
                    itc_diff_ms = load.time_frequency_range(file_path=trf_path_mssl + itc_data_fname, l_freq=l_freq, h_freq=h_freq)
                    # MSSH
                    itc_diff_cross2 = load.time_frequency_range(file_path=trf_path_mssh + itc_data_fname, l_freq=l_freq, h_freq=h_freq)

            except:
                # Compute power using from epoched data
                for mss, tmin, tmax, epochs_path, trf_save_path in zip((mssl, mssh), (tmin_mssl, tmin_mssh), (tmax_mssl, tmax_mssh),
                                                                                     (epochs_path_mssl, epochs_path_mssh), (trf_path_mssl, trf_path_mssh)):
                    try:
                        if not use_saved_data:
                            raise ValueError('Not using saved data')
                        # Load epoched data
                        epochs = mne.read_epochs(epochs_path + epochs_data_fname)
                    except:
                        # Compute epochs
                        if use_ica_data:
                            # Load meg data
                            meg_data = load.ica_data(subject=subject)
                        else:
                            # Load meg data
                            meg_data = subject.load_preproc_meg_data()

                        # Epoch data
                        epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans, tgt_pres=tgt_pres, epoch_id=epoch_id,
                                                                       meg_data=meg_data, tmin=tmin, tmax=tmax, save_data=save_data, epochs_save_path=epochs_path,
                                                                       epochs_data_fname=epochs_data_fname)

                    # Compute power and PLI over frequencies and save
                    power = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq, n_cycles_div=n_cycles_div, average=return_average_tfr,
                                                              return_itc=run_itc, output=output, save_data=save_data, trf_save_path=trf_save_path,
                                                              power_data_fname=power_data_fname, itc_data_fname=itc_data_fname, n_jobs=4)

                    if run_itc:
                        power, itc = power

                # Load previous data
                power_mssl = mne.time_frequency.read_tfrs(trf_path_mssl + power_data_fname, condition=0)
                power_mssh = mne.time_frequency.read_tfrs(trf_path_mssh + power_data_fname, condition=0)
                if run_itc:
                    itc_mssl = mne.time_frequency.read_tfrs(trf_path_mssl + itc_data_fname, condition=0)
                    itc_mssh = mne.time_frequency.read_tfrs(trf_path_mssh + itc_data_fname, condition=0)

                # Average epochs
                if not return_average_tfr:
                    power_mssl = power_mssl.average()
                    power_mssh = power_mssh.average()
                    if run_itc:
                        itc_mssl = itc_mssl.average()
                        itc_mssh = itc_mssh.average()

            # Apply baseline to power and itc
            power_mssl.apply_baseline(baseline=baseline, mode=bline_mode)
            power_mssh.apply_baseline(baseline=baseline, mode=bline_mode)
            if run_itc:
                itc_mssl.apply_baseline(baseline=baseline, mode=bline_mode)
                itc_mssh.apply_baseline(baseline=baseline, mode=bline_mode)

            # Get time windows to compare
            # mssl
            power_mssl_ms = power_mssl.copy().crop(tmin=mssl_ms_window_times[0], tmax=mssl_ms_window_times[1])
            power_mssl_cross2 = power_mssl.copy().crop(tmin=mssl_cross2_window_times[0], tmax=mssl_cross2_window_times[1])
            # mssh
            power_mssh_ms = power_mssh.copy().crop(tmin=mssl_ms_window_times[0], tmax=mssl_ms_window_times[1])

            # Force matching times by copying variable and changing data
            power_mssh_cross2 = power_mssl_cross2.copy()
            power_mssh_cross2.data = power_mssh.copy().crop(tmin=mssh_cross2_window_times[0], tmax=mssh_cross2_window_times[1]).data

            if run_itc:
                # mssl
                itc_mssl_ms = itc_mssl.copy().crop(tmin=mssl_ms_window_times[0], tmax=mssl_ms_window_times[1])
                itc_mssl_cross2 = itc_mssl.copy().crop(tmin=mssl_cross2_window_times[0], tmax=mssl_cross2_window_times[1])
                # mssh
                itc_mssh_ms = itc_mssh.copy().crop(tmin=mssl_ms_window_times[0], tmax=mssl_ms_window_times[1])
                # Force matching times by copying variable and changin data
                itc_mssh_cross2 = itc_mssl_cross2.copy()
                itc_mssh_cross2.data = itc_mssh.copy().crop(tmin=mssh_cross2_window_times[0], tmax=mssh_cross2_window_times[1]).data

            # Condition difference
            # MS
            power_diff_ms = power_mssh_ms - power_mssl_ms
            # Cross2
            power_diff_cross2 = power_mssh_cross2 - power_mssl_cross2
            if run_itc:
                # MS
                itc_diff_ms = itc_mssh_ms - itc_mssl_ms
                # Cross2
                itc_diff_cross2 = itc_mssh_cross2 - itc_mssl_cross2

            # Save data
            if save_data:
                # Save epoched data
                os.makedirs(trf_diff_save_path, exist_ok=True)
                power_diff_ms.save(trf_diff_save_path + f'ms_{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_' + power_data_fname, overwrite=True)
                power_diff_cross2.save(trf_diff_save_path + f'cross2_{cross2_filename_times}_' + power_data_fname, overwrite=True)
                if run_itc:
                    itc_diff_ms.save(trf_diff_save_path + f'ms_{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_' + itc_data_fname, overwrite=True)
                    itc_diff_cross2.save(trf_diff_save_path + f'cross2_{cross2_filename_times}_' + itc_data_fname, overwrite=True)

        # Append data for GA
        # MS
        averages_power_ms_diff.append(power_diff_ms)
        # Cross2
        averages_power_cross2_diff.append(power_diff_cross2)
        if run_itc:
            # MS
            averages_itc_ms_diff.append(itc_diff_ms)
            # Cross2
            averages_itc_cross2_diff.append(itc_diff_cross2)

        # Append subjects trf data to list as dataframe
        data_power_ms_diff.append(power_diff_ms.to_data_frame())
        data_power_cross2_diff.append(power_diff_cross2.to_data_frame())
        if plot_individuals:
            for chs_id in chs_ids:
                # Plot power time-frequency
                # Power Plotjoint MS
                fname = f'Power_ms_plotjoint_{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                plot_general.tfr_plotjoint_picks(tfr=power_diff_ms, plot_baseline=None, bline_mode=bline_mode,
                                                 chs_id=chs_id, plot_max=True, plot_min=True,
                                                 display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path_subj,
                                                 fname=fname)
                # Power Plotjoint Cross2
                fname = f'Power_cross2_plotjoint_{cross2_filename_times}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                plot_general.tfr_plotjoint_picks(tfr=power_diff_cross2, plot_baseline=None, bline_mode=bline_mode,
                                                 chs_id=chs_id, plot_max=True, plot_min=True,
                                                 display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path_subj,
                                                 fname=fname)

                if run_itc:
                    # Plot ITC time-frequency
                    # ITC Plotjoint MS
                    fname = f'ITC_ms_plotjoint_{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                    plot_general.tfr_plotjoint_picks(tfr=itc_diff_ms, plot_baseline=None, bline_mode=bline_mode,
                                                     chs_id=chs_id, plot_max=True, plot_min=True,
                                                     display_figs=display_figs, save_fig=save_fig,
                                                     trf_fig_path=trf_fig_path_subj, fname=fname)
                    # ITC Plot joint
                    fname = f'ITC_cross2_plotjoint_{cross2_filename_times}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                    plot_general.tfr_plotjoint_picks(tfr=itc_diff_cross2, plot_baseline=None,
                                                     bline_mode=bline_mode,
                                                     chs_id=chs_id, plot_max=True, plot_min=True,
                                                     display_figs=display_figs, save_fig=save_fig,
                                                     trf_fig_path=trf_fig_path_subj, fname=fname)

    # Compute grand average
    grand_avg_power_ms_diff = mne.grand_average(averages_power_ms_diff)
    grand_avg_power_cross2_diff = mne.grand_average(averages_power_cross2_diff)
    if run_itc:
        grand_avg_itc_ms_diff = mne.grand_average(averages_itc_ms_diff)
        grand_avg_itc_cross2_diff = mne.grand_average(averages_itc_cross2_diff)

    # Set time relative to VS screen
    grand_avg_power_cross2_diff.shift_time(tshift=-2, relative=False)
    if run_itc:
        grand_avg_itc_cross2_diff.shift_time(tshift=-2, relative=False)

    if save_data:
        # Save trf data
        grand_avg_power_ms_diff.save(trf_diff_save_path + grand_avg_power_ms_fname, overwrite=True)
        grand_avg_power_cross2_diff.save(trf_diff_save_path + grand_avg_power_cross2_fname, overwrite=True)
        if run_itc:
            grand_avg_itc_ms_diff.save(trf_diff_save_path + grand_avg_itc_ms_fname, overwrite=True)
            grand_avg_itc_cross2_diff.save(trf_diff_save_path + grand_avg_itc_cross2_fname, overwrite=True)


    #------ TFCE -----#
    for chs_id in chs_ids:
        picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power_ms_diff.info)

        data_power_ms_diff_array = []
        data_power_cross2_diff_array = []

        # print('Deletting bad channels columns from data...')
        for i in range(len(data_power_ms_diff)):

            # Keep selected channels
            data_power_ms_diff[i] = data_power_ms_diff[i][data_power_ms_diff[i].columns & picks]
            data_power_cross2_diff[i] = data_power_cross2_diff[i][data_power_cross2_diff[i].columns & picks]

            # Convert to array and drop times and frequencies
            data_power_ms_diff_array.append(data_power_ms_diff[i].to_numpy().reshape(len(power_diff_ms.freqs), len(power_diff_ms.times), data_power_ms_diff[i].to_numpy().shape[-1]))
            data_power_cross2_diff_array.append(data_power_cross2_diff[i].to_numpy().reshape(len(power_diff_cross2.freqs), len(power_diff_cross2.times), data_power_cross2_diff[i].to_numpy().shape[-1]))

        # Make array of all subjects
        data_power_ms_diff_array = np.stack(data_power_ms_diff_array, axis=-1)
        data_power_cross2_diff_array = np.stack(data_power_cross2_diff_array, axis=-1)

        data_power_ms_diff_array = np.moveaxis(data_power_ms_diff_array, [-1], [0])
        data_power_cross2_diff_array = np.moveaxis(data_power_cross2_diff_array, [-1], [0])

        for data, ga, title, fname_times in zip([data_power_ms_diff_array, data_power_cross2_diff_array],
                                   [grand_avg_power_ms_diff, grand_avg_power_cross2_diff],
                                   ['ms', 'cross2'], [f'{mssl_ms_window_times[0]}_{mssl_ms_window_times[1]}_', f'{cross2_filename_times}_']):

            # Permutation cluster test parameters
            n_permutations = 5120
            degrees_of_freedom = len(exp_info.subjects_ids) - 1
            desired_tval = 0.01
            t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
            # t_thresh = dict(start=0, step=0.2)
            significant_channels = 0.5
            pval_threshold = 0.05

            # Get channel adjacency
            ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg_power_ms_diff.info, ch_type='mag', picks=picks)
            # Clusters out type
            if type(t_thresh) == dict:
                out_type = 'indices'
            else:
                out_type = 'mask'

            # Permutations cluster test (TFCE if t_thresh as dict)
            t_tfce, clusters, p_tfce, H0 = permutation_cluster_1samp_test(X=data, threshold=t_thresh, n_permutations=n_permutations, out_type=out_type, seed=42)

            # Make clusters mask
            if type(t_thresh) == dict:
                # Reshape to data's shape
                p_tfce = p_tfce.reshape(data.shape[-2:])

                clusters_mask_plot = p_tfce < pval_threshold
                clusters_mask = None

                # Cluster contour
                image_args = {'mask': clusters_mask_plot, 'mask_style': 'contour'}

            else:
                # Get significant clusters
                good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
                significant_clusters = [clusters[idx] for idx in good_clusters_idx]
                significant_pvalues = [p_tfce[idx] for idx in good_clusters_idx]

                # Rehsape to data's shape by adding all clusters into one bool array
                clusters_mask = np.zeros(data[0].shape)
                if len(significant_clusters):
                    for significant_cluster in significant_clusters:
                        clusters_mask += significant_cluster
                        clusters_mask_plot = clusters_mask.sum(axis=-1) > len(picks) * significant_channels
                        clusters_mask_plot = clusters_mask_plot.astype(bool)
                    # Cluster contour
                    image_args = {'mask': clusters_mask_plot, 'mask_style': 'contour'}
                else:
                    image_args = None

            # Power Plotjoint
            if type(t_thresh) == dict:
                fname = f'GA_Power_{title}_plotjoint_{fname_times}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_tTFCE_pval{pval_threshold}_chs{significant_channels}'
            else:
                fname = f'GA_Power_{title}_plotjoint_{fname_times}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{pval_threshold}_chs{significant_channels}'

            if len(significant_pvalues):
                title = f'{title}_{bline_mode}_pval_{functions_general.largest_cluster_pval(significant_pvalues, significant_clusters)}'
            else:
                title = f'{title}_{bline_mode}'
            plot_general.tfr_plotjoint_picks(tfr=ga, plot_baseline=None, bline_mode=bline_mode, image_args=image_args, chs_id=chs_id, plot_max=True, plot_min=True, vmin=-0.2,
                                             vmax=0.2, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, clusters_mask=clusters_mask,
                                             title=title, fname=fname, fontsize=20, ticksize=20)

        if run_itc:
            # Plot ITC time-frequency
            # ITC Plotjoint MS
            fname = f'GA_ITC_ms_plotjoint_{fname_times}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            plot_general.tfr_plotjoint_picks(tfr=grand_avg_itc_ms_diff, plot_baseline=None, bline_mode=bline_mode,
                                             chs_id=chs_id, plot_max=True, plot_min=True, vmin=-0.1, vmax=0.1,
                                             display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)
            # ITC Plot joint
            fname = f'GA_ITC_cross2_plotjoint_{fname_times}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            plot_general.tfr_plotjoint_picks(tfr=grand_avg_itc_cross2_diff, plot_baseline=None, bline_mode=bline_mode,
                                             chs_id=chs_id, plot_max=True, plot_min=True, vmin=-0.1, vmax=0.1,
                                             display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)