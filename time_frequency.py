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
from mne.stats import permutation_cluster_1samp_test


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
# Select channels
chs_ids = ['frontal_central', 'parietal_occipital']  # region_hemisphere
# ICA / RAW
use_ica_data = True
# Epochs
epoch_id = 'ms'
corr_ans = None
tgt_pres = None
mss = 2
reject = None  # 'subject' for subject's default. False for no rejection, dict for specific values. None for default dict(mag=5e-12) for magnetometers
evt_dur = None

# Power time frequency params
n_cycles_div = 2.
l_freq = 1
h_freq = 40
run_itc = False
plot_edge = 0.15

# Plots parameters
# Colorbar
vmax_power = 0.2
vmin_power = -0.2
vmin_itc, vmax_itc = None, None
# plot_joint max and min topoplots
plot_max, plot_min = False, False

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
run_permutations = True
n_permutations = 1024
degrees_of_freedom = len(exp_info.subjects_ids) - 1
desired_tval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# t_thresh = dict(start=0, step=0.2)
pval_threshold = 0.05
significant_channels = 0.5  # Percent of total region channels

#---------- Setup ----------#

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
if 'vs' in epoch_id:
    trial_dur = vs_dur[mss]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_dur = None

# Get time windows from epoch_id name
map = dict(ms={'tmin': -cross1_dur, 'tmax': mss_duration[mss], 'plot_xlim': (-cross1_dur + plot_edge, mss_duration[mss] - plot_edge)},
           fix_vs={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
           fix_ms={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
           tgt_fix={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3 + plot_edge, 0.6 - plot_edge)},
           sac_emap={'tmin': -0.5, 'tmax': 3, 'plot_xlim': (-0.3, 2.5)},
           hl_start={'tmin': -3, 'tmax': 35, 'plot_xlim': (-2.5, 33)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, mss=mss, plot_edge=plot_edge, map=map)

# Define time-frequency bands to plot in plot_joint
timefreqs_joint, timefreqs_tfr, vlines_times = functions_general.get_plots_timefreqs(epoch_id=epoch_id, mss=mss, cross2_dur=cross2_dur, mss_duration=mss_duration,
                                                                                     topo_bands=topo_bands, plot_xlim=plot_xlim)
timefreqs_joint = [(1.25, 12)]
timefreqs_tfr = timefreqs_joint

if (plot_max or plot_min):
    timefreqs_joint = None
    timefreqs_tfr = None

# Get baseline duration for epoch_id
baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=epoch_id, mss=mss, tmin=tmin, tmax=tmax,
                                                                  cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                  cross2_dur=cross2_dur, plot_edge=plot_edge)

# Specific run path for saving data and plots
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Save ids
save_id = f'{epoch_id}_mss{mss}_Corr{corr_ans}_tgt{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}'
plot_id = f'{save_id}_{round(plot_xlim[0],2)}_{round(plot_xlim[1], 2)}_bline{baseline}_cyc{int(n_cycles_div)}/'

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
    if run_permutations:
        raise ValueError

    # Load previous power data
    grand_avg_power = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_power_fname)[0]
    if run_itc:
        # Load previous itc data
        grand_avg_itc = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_itc_fname)[0]

except:

    averages_power = []
    averages_itc = []
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
                epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans, trial_dur=trial_dur,
                                                               tgt_pres=tgt_pres, baseline=baseline, reject=reject,
                                                               epoch_id=epoch_id, meg_data=meg_data, tmin=tmin, tmax=tmax,
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
        averages_power.append(power)

        # Plot power time-frequency
        if plot_individuals:
            for chs_id in chs_ids:
                fname = f'Power_plotjoint_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                plot_general.tfr_plotjoint_picks(tfr=power, plot_baseline=None, bline_mode=bline_mode, vlines_times=vlines_times,
                                                 timefreqs=None, chs_id=chs_id, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min, vmin=None,
                                                 vmax=None, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path_subj, fname=fname)

                if run_itc:
                    averages_itc.append(itc)

                    # Plot ITC time-frequency
                    fname = f'ITC_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                    plot_general.tfr_bands(subject=subject, tfr=itc, chs_id=chs_id, plot_xlim=plot_xlim,
                                     baseline=plot_baseline, bline_mode=bline_mode,
                                     display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname,
                                     fontsize=16, ticksize=18)

                    # ITC topoplot
                    fig = itc.plot_topo(baseline=plot_baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
                                        cmap='jet', show=display_figs, title='Inter-Trial coherence')
                    if save_fig:
                        fname = f'ITC_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                        save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)

                    # Free up memory
                    del itc
        # Free up memory
        del power

    # Compute grand average
    grand_avg_power = mne.grand_average(averages_power)
    if run_itc:
        grand_avg_itc = mne.grand_average(averages_itc)

    if save_data:
        # Save trf data
        grand_avg_power.save(trf_save_path + grand_avg_power_fname, overwrite=True)
        if run_itc:
            grand_avg_itc.save(trf_save_path + grand_avg_itc_fname, overwrite=True)

#--------- Permutation cluster test data -----------#
for chs_id, timefreqs_joint in zip(chs_ids, [[(1.25, 12)], [(0.6, 10)]]):
# for chs_id in chs_ids:
    if run_permutations:

        picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power.info)
        permutations_test_data = [power.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).pick(picks).data for power in averages_power]
        # permutations_test_data_array = np.array([data.mean(0) for data in permutations_test_data])
        permutations_test_data_array = np.array([data for data in permutations_test_data])
        permutations_test_data_array = permutations_test_data_array.swapaxes(1, 2).swapaxes(2, 3)

        # Get channel adjacency
        ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg_power.info, ch_type='mag', picks='mag')
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
        else:
            # Get significant clusters
            good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
            significant_clusters = [clusters[idx] for idx in good_clusters_idx]

            # Rehsape to data's shape by adding all clusters into one bool array
            clusters_mask = np.zeros(permutations_test_data_array[0].shape)
            if len(significant_clusters):
                for significant_cluster in significant_clusters:
                    clusters_mask += significant_cluster
                    clusters_mask_plot = clusters_mask.sum(axis=-1) > len(picks) * significant_channels
                    clusters_mask_plot = clusters_mask_plot.astype(bool)

        # Cluster contour
        image_args = {'mask': clusters_mask_plot, 'mask_style': 'contour'}

        if type(t_thresh) == dict:
            fname = f'GA_Power_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_tTFCE_pval{pval_threshold}_chs{significant_channels}'
        else:
            fname = f'GA_Power_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}_t{round(t_thresh, 2)}_pval{pval_threshold}_chs{significant_channels}'
    else:
        image_args = None
        clusters_mask = None
        if type(t_thresh) == dict:
            fname = f'GA_Power_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        else:
            fname = f'GA_Power_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'

    #--------- Plots ---------#
    # Power Plotjoint
    plot_general.tfr_plotjoint_picks(tfr=grand_avg_power, plot_baseline=None, bline_mode=bline_mode, vlines_times=vlines_times, timefreqs=timefreqs_joint,
                                     image_args=image_args, clusters_mask=clusters_mask, chs_id=chs_id, plot_xlim=plot_xlim, plot_max=plot_max, plot_min=plot_min,
                                     vmin=vmin_power, vmax=vmax_power, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

    # Plot Power time-frequency in time scalde axes
    fname = f'Power_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_times(tfr=grand_avg_power, chs_id=chs_id, timefreqs_tfr=timefreqs_tfr, baseline=None, bline_mode=bline_mode,
                           plot_xlim=plot_xlim, vlines_times=vlines_times, topo_vmin=vmin_power, topo_vmax=vmax_power, subject=None, display_figs=display_figs,
                           save_fig=save_fig, fig_path=trf_fig_path, fname=fname, vmin=vmin_power, vmax=vmax_power, fontsize=16, ticksize=18)

    if run_itc:
        # ITC Plot joint
        fname = f'GA_ITC_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        plot_general.tfr_plotjoint_picks(tfr=grand_avg_itc, plot_baseline=None, bline_mode=None,
                                         vlines_times=vlines_times, timefreqs=timefreqs_joint, plot_xlim=plot_xlim,
                                         chs_id=chs_id, vmin=vmin_itc, vmax=vmax_itc, plot_max=plot_max, plot_min=plot_min,
                                         display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)
        # Plot ITC time-frequency
        fname = f'ITC_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        plot_general.tfr_times(tfr=grand_avg_itc, chs_id=chs_id, timefreqs_tfr=timefreqs_tfr, baseline=None, bline_mode=bline_mode,
                               plot_xlim=plot_xlim, vlines_times=vlines_times, topo_vmin=vmin_itc, topo_vmax=vmax_itc, subject=None, display_figs=display_figs,
                               save_fig=save_fig, fig_path=trf_fig_path, fname=fname, vmin=vmin_itc, vmax=vmax_itc, fontsize=16, ticksize=18)

# Power Plot joint
fname = f'GA_Power_plotjoint_mag_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint(tfr=grand_avg_power, plot_baseline=None, bline_mode=bline_mode, plot_xlim=plot_xlim,
                           vmin=vmin_power, vmax=vmax_power, timefreqs=timefreqs_joint, plot_max=plot_max, plot_min=plot_min,
                           vlines_times=vlines_times, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

if run_itc:
    # ITC Plot joint
    fname = f'GA_ITC_plotjoint_mag_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_plotjoint(tfr=grand_avg_itc, plot_baseline=None, bline_mode=bline_mode, plot_xlim=plot_xlim,
                               vmin=vmin_itc, vmax=vmax_itc, timefreqs=timefreqs_joint, plot_max=plot_max, plot_min=plot_min,
                               vlines_times=vlines_times, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)
