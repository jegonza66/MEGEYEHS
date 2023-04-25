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
chs_id = 'occipital'  # region_hemisphere
# ICA / RAW
use_ica_data = True
corr_ans = None
tgt_pres = None
mss = 4
epoch_id = 'fix_ms'
# epoch_id = 'sac_ms'
reject = False  # None for subject's default. False for no rejection, dict for specific values
n_cycles_div = 4.
# Power frequency range
l_freq = 1
h_freq = 40
log_bands = False

# Baseline method
bline_mode = 'logratio'
#----------#

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 0}
cross1_dur = 0.75
cross2_dur = 1
vs_dur = 4
plot_edge = 0.15

# Duration
if 'ms' in epoch_id:
    dur = mss_duration[mss] + cross2_dur + vs_dur
elif 'cross2' in epoch_id:
    dur = cross2_dur + vs_dur  # seconds
else:
    dur = 0

# Get time windows from epoch_id name
map_times = dict(ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
                 cross2={'tmin': -cross1_dur - mss_duration[mss], 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                 sac={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)},
                 fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.2)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

# Baseline duration
if 'sac' in epoch_id:
    baseline = (tmin, 0)
    plot_baseline = (plot_xlim[0], 0)
    # baseline = None
elif 'fix' in epoch_id or 'fix' in epoch_id:
    baseline = (tmin, -0.05)
    plot_baseline = (plot_xlim[0], 0)
elif 'ms' in epoch_id or 'cross2' in epoch_id and mss:
    baseline = (tmin, 0)
    plot_baseline = (plot_xlim[0], 0)
else:
    baseline = (tmin, 0)
    plot_baseline = (plot_xlim[0], 0)

# Vlines times
vlines_times = [0, mss_duration[mss], mss_duration[mss] + 1]

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
plot_id = f'{save_id}_{plot_xlim[0]}_{plot_xlim[1]}_bline{baseline}/'

# Save data paths
trf_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/{save_id}_{tmin}_{tmax}_bline{baseline}/'
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
                    meg_data = subjec.load_preproc_meg_data()

                # Epoch data
                epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans,
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
        plot_general.tfr(subject=subject, tfr=power, chs_id=chs_id, plot_xlim=plot_xlim, epoch_id=epoch_id, mss=mss,
                         cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                         baseline=plot_baseline, bline_mode=bline_mode,
                         display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname,
                         fontsize=16, ticksize=18)

        # Plot ITC time-frequency
        fname = f'ITC_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        plot_general.tfr(subject=subject, tfr=itc, chs_id=chs_id, plot_xlim=plot_xlim, epoch_id=epoch_id, mss=mss,
                         cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
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
plot_general.tfr(tfr=grand_avg_power, chs_id=chs_id, baseline=plot_baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                 epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                 subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname,
                 vmin=None, vmax=None, fontsize=16, ticksize=18)

# Plot ITC time-frequency
fname = f'ITC_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr(tfr=grand_avg_itc, chs_id=chs_id, baseline=plot_baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                 epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                 subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname,
                 vmin=None, vmax=None, fontsize=16, ticksize=18)



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
                                 plot_xlim=plot_xlim, chs_id=chs_id, vmin=None, vmax=None, plot_max=True, plot_min=True,
                                 display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

# ITC Plot joint
fname = f'GA_ITC_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint_picks(tfr=grand_avg_itc, plot_baseline=plot_baseline, bline_mode=bline_mode, vlines_times=vlines_times,
                                 plot_xlim=plot_xlim, chs_id=chs_id, vmin=None, vmax=None, plot_max=True, plot_min=True,
                                 display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)


# Power Plot joint
fname = f'GA_Power_plotjoint_mag_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint(tfr=grand_avg_power, plot_baseline=plot_baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                           vmin=None, vmax=None, plot_max=True, plot_min=True, vlines_times=vlines_times,
                           display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

# ITC Plot joint
fname = f'GA_ITC_plotjoint_mag_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint(tfr=grand_avg_itc, plot_baseline=plot_baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                           vmin=None, vmax=None, plot_max=True, plot_min=True, vlines_times=vlines_times,
                           display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

