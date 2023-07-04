import functions_general
import functions_analysis
import load
import mne
import plot_general
import save
import setup
from paths import paths
import matplotlib.pyplot as plt

#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# Select channels
chs_id = 'parietal_occipital'  # region_hemisphere
# ICA / RAW
use_ica_data = True
epoch_id = 'vs'
corr_ans = None
tgt_pres = None
mss = 1
reject = None  # 'subject' for subject's default. False for no rejection, dict for specific values. None for default 5e-12 for magnetometers
n_cycles_div = 2.
# Power frequency range
l_freq = 40
h_freq = 100
log_bands = False

# Trial durations
vs_dur = {1: (2, 9.8), 2: (3, 9.8), 4: (3.5, 9.8), None: (2, 9.8)}
plot_edge = 0.15
trial_dur = vs_dur[mss]  # Edit this to determine the minimum visual search duration for the trial selection (this will also affect ms trials)

# Plots parameters
# Colorbar
vmin_power, vmax_power = None, None
vmin_itc, vmax_itc = None, None
topo_vmin, topo_vmax = None, None
# plot_joint max and min topoplots
plot_max, plot_min = True, True
# Baseline method
bline_mode = 'logratio'
# Topoplot bands
topo_bands = ['Alpha', 'Alpha', 'Theta', 'Alpha']

#----------#

# Windows durations
dur, cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration(epoch_id=epoch_id, vs_dur=vs_dur, mss=mss)

# Get time windows from epoch_id name
map = dict(tgt_fix={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3, 0.6)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, mss=mss, plot_edge=plot_edge, map=map)

# Define time-frequency bands to plot in plot_joint
if (plot_max or plot_min):
    timefreqs_joint, timefreqs_tfr, vlines_times = functions_general.get_plots_timefreqs(epoch_id=epoch_id, mss=mss,
                                                                                         cross2_dur=cross2_dur,
                                                                                         mss_duration=mss_duration,
                                                                                         topo_bands=topo_bands, plot_xlim=plot_xlim)
    timefreqs_joint = None
else:
    timefreqs_joint, timefreqs_tfr, vlines_times = functions_general.get_plots_timefreqs(epoch_id=epoch_id, mss=mss,
                                                                                         cross2_dur=cross2_dur,
                                                                                         mss_duration=mss_duration,
                                                                                         topo_bands=topo_bands, plot_xlim=plot_xlim)

# Get baseline duration for epoch_id
baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=epoch_id, mss=mss, tmin=tmin, tmax=tmax, plot_xlim=plot_xlim,
                                                                  cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                  cross2_dur=cross2_dur)
plot_baseline = None
# freqs type
if log_bands:
    freqs_type = 'log'
else:
    freqs_type = 'lin'

# Specific run path for saving data and plots
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Save ids
save_id = f'{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}'
if (epoch_id == 'ms' or epoch_id == 'vs') and trial_dur:
    save_id += f'_tdur{trial_dur}'
plot_id = f'{save_id}_{plot_xlim[0]}_{plot_xlim[1]}_bline{baseline}_cycles{int(n_cycles_div)}/'

# Save data paths
trf_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/{save_id}_{tmin}_{tmax}_bline{baseline}_cycles{int(n_cycles_div)}/'
epochs_save_path = paths().save_path() + f'Epochs_{data_type}/Band_None/{save_id}_{tmin}_{tmax}_bline{baseline}/'
# Save figures paths
trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + plot_id + f'{chs_id}/'

# Grand average data variable
grand_avg_power_fname = f'Grand_Average_power_{l_freq}_{h_freq}_tfr.h5'
grand_avg_itc_fname = f'Grand_Average_itc_{l_freq}_{h_freq}_tfr.h5'

try:
    # Load previous power data
    grand_avg_power = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_power_fname)[0]
    # Load previous itc data
    grand_avg_itc = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_itc_fname)[0]
    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power.info)

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
            power = mne.time_frequency.read_tfrs(trf_save_path + power_data_fname, condition=0)
            itc = mne.time_frequency.read_tfrs(trf_save_path + itc_data_fname, condition=0)
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
            power, itc = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq, freqs_type=freqs_type,
                                                           n_cycles_div=n_cycles_div, save_data=save_data, trf_save_path=trf_save_path,
                                                           power_data_fname=power_data_fname, itc_data_fname=itc_data_fname)

        # Append data for GA
        averages_power.append(power)
        averages_itc.append(itc)

        # Plot power time-frequency
        fname = f'Power_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        plot_general.tfr_bands(subject=subject, tfr=power, chs_id=chs_id, plot_xlim=plot_xlim,
                         baseline=plot_baseline, bline_mode=bline_mode,
                         display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname,
                         fontsize=16, ticksize=18)

        # Plot ITC time-frequency
        fname = f'ITC_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        plot_general.tfr_bands(subject=subject, tfr=itc, chs_id=chs_id, plot_xlim=plot_xlim,
                         baseline=plot_baseline, bline_mode=bline_mode,
                         display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname,
                         fontsize=16, ticksize=18)

        # Power topoplot
        fig = power.plot_topo(baseline=plot_baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap='jet',
                              show=display_figs, title='Power')
        if save_fig:
            fname = f'Power_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)

        # ITC topoplot
        fig = itc.plot_topo(baseline=plot_baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap='jet', show=display_figs, title='Inter-Trial coherence')
        if save_fig:
            fname = f'ITC_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)

        # Free up memory
        plt.close('all')
        del power, itc

    # Compute grand average
    grand_avg_power = mne.grand_average(averages_power)
    grand_avg_itc = mne.grand_average(averages_itc)

    if save_data:
        # Save trf data
        grand_avg_power.save(trf_save_path + grand_avg_power_fname, overwrite=True)
        grand_avg_itc.save(trf_save_path + grand_avg_itc_fname, overwrite=True)


# Plot Power time-frequency
fname = f'Power_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_times(tfr=grand_avg_power, chs_id=chs_id, timefreqs_tfr=None, baseline=plot_baseline, bline_mode=bline_mode,
                       plot_xlim=plot_xlim, vlines_times=vlines_times, topo_vmin=topo_vmin, topo_vmax=topo_vmax, subject=None, display_figs=display_figs,
                       save_fig=save_fig, fig_path=trf_fig_path, fname=fname, vmin=vmin_power, vmax=vmax_power, fontsize=16, ticksize=18)

# Plot ITC time-frequency
fname = f'ITC_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_times(tfr=grand_avg_itc, chs_id=chs_id, timefreqs_tfr=None, baseline=plot_baseline, bline_mode=bline_mode,
                       plot_xlim=plot_xlim, vlines_times=vlines_times, topo_vmin=topo_vmin, topo_vmax=topo_vmax, subject=None, display_figs=display_figs,
                       save_fig=save_fig, fig_path=trf_fig_path, fname=fname, vmin=vmin_itc, vmax=vmax_itc, fontsize=16, ticksize=18)

# Power topoplot
fig = grand_avg_power.plot_topo(baseline=plot_baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
                                cmap='jet', show=display_figs, title='Power')
if save_fig:
    fname = f'GA_Power_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    save.fig(fig=fig, path=trf_fig_path, fname=fname)

# ITC topoplot
fig = grand_avg_itc.plot_topo(baseline=plot_baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap='jet',
                              show=display_figs, title='Inter-Trial coherence')
if save_fig:
    fname = f'GA_ITC_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    save.fig(fig=fig, path=trf_fig_path, fname=fname)

# Power Plot joint
fname = f'GA_Power_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint_picks(tfr=grand_avg_power, plot_baseline=plot_baseline, bline_mode=bline_mode, vlines_times=vlines_times,
                                 timefreqs=timefreqs_joint, plot_xlim=plot_xlim, chs_id=chs_id, vmin=vmin_power, vmax=vmin_power,
                                 plot_max=plot_max, plot_min=plot_min, display_figs=display_figs, save_fig=save_fig,
                                 trf_fig_path=trf_fig_path, fname=fname)

# ITC Plot joint
fname = f'GA_ITC_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint_picks(tfr=grand_avg_itc, plot_baseline=plot_baseline, bline_mode=bline_mode, vlines_times=vlines_times,
                                 timefreqs=timefreqs_joint, plot_xlim=plot_xlim, chs_id=chs_id, vmin=vmin_itc, vmax=vmax_itc,
                                 plot_max=plot_max, plot_min=plot_min, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

# Power Plot joint
fname = f'GA_Power_plotjoint_mag_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint(tfr=grand_avg_power, plot_baseline=plot_baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                           vmin=vmin_power, vmax=vmax_power, timefreqs=timefreqs_joint, plot_max=plot_max, plot_min=plot_min,
                           vlines_times=vlines_times, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

# ITC Plot joint
fname = f'GA_ITC_plotjoint_mag_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint(tfr=grand_avg_itc, plot_baseline=plot_baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                           vmin=vmin_itc, vmax=vmax_itc, timefreqs=timefreqs_joint, plot_max=plot_max, plot_min=plot_min,
                           vlines_times=vlines_times, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

