import functions_general
import functions_analysis
import load
import mne
import os
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
chs_id = 'parietal_occipital_temporal'
# ICA / RAW
use_ica_data = True
corr_ans = None
tgt_pres = None
mss = None
epoch_id = 'ms'
# epoch_id = 'fix_vs'
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

# Duration
if 'ms' in epoch_id:
    dur = mss_duration[mss] + cross2_dur + vs_dur
elif 'cross2' in epoch_id:
    dur = cross2_dur + vs_dur  # seconds
else:
    dur = 0

# freqs type
if log_bands:
    freqs_type = 'log'
else:
    freqs_type = 'lin'

if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

for mssh, mssl in [(4, 1), (4, 2), (2, 1)]:

    tmin_mssl = -cross1_dur
    tmax_mssl = mss_duration[mssl] + cross2_dur + vs_dur
    tmin_mssh = -cross1_dur
    tmax_mssh = mss_duration[mssh] + cross2_dur + vs_dur
    baseline = (tmin_mssl, 0)

    # Specific run path for loading data
    load_id_mssl = f'{epoch_id}_mss{mssl}_Corr_{corr_ans}_tgt_{tgt_pres}'
    load_id_mssh = f'{epoch_id}_mss{mssh}_Corr_{corr_ans}_tgt_{tgt_pres}'
    load_path_mssl = f'/{load_id_mssl}_{tmin_mssl}_{tmax_mssl}_bline{baseline}/'
    load_path_mssh = f'/{load_id_mssh}_{tmin_mssh}_{tmax_mssh}_bline{baseline}/'

    # Load data paths
    trf_load_path_mssl = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + load_path_mssl
    trf_load_path_mssh = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + load_path_mssh
    epochs_load_path_mssl = paths().save_path() + f'Epochs_{data_type}/Band_None/' + load_path_mssl
    epochs_load_path_mssh = paths().save_path() + f'Epochs_{data_type}/Band_None/' + load_path_mssh

    # Save data paths
    save_id = f'{epoch_id}_mss{mssh}-{mssl}_Corr_{corr_ans}_tgt_{tgt_pres}'
    save_path = f'/{save_id}_{tmin_mssl}_{tmax_mssl}_bline{baseline}/'
    trf_diff_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + save_path

    # Save figures paths
    plot_path = f'/{save_id}_bline{baseline}/'
    trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + plot_path + f'{chs_id}/'

    # Grand average data variable
    grand_avg_power_ms_fname = f'Grand_Average_power_ms_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_itc_ms_fname = f'Grand_Average_itc_ms_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_power_cross2_fname = f'Grand_Average_power_cross2_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_itc_cross2_fname = f'Grand_Average_itc_cross2_{l_freq}_{h_freq}_tfr.h5'

    # Grand Average
    try:
        # Load previous power data
        grand_avg_power_ms_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_power_ms_fname)[0]
        grand_avg_power_cross2_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_power_cross2_fname)[0]
        # Load previous itc data
        grand_avg_itc_ms_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_itc_ms_fname)[0]
        grand_avg_itc_cross2_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_itc_cross2_fname)[0]

        # Pick plot channels
        picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power_ms_diff.info)

    except:

        averages_power_ms_diff = []
        averages_itc_ms_diff = []
        averages_power_cross2_diff = []
        averages_itc_cross2_diff = []

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
                # Load difference data
                power_diff_ms = mne.time_frequency.read_tfrs(trf_diff_save_path + 'ms_' + power_data_fname, condition=0)
                itc_diff_ms = mne.time_frequency.read_tfrs(trf_diff_save_path + 'ms_' + itc_data_fname, condition=0)
                power_diff_cross2 = mne.time_frequency.read_tfrs(trf_diff_save_path + 'cross2_' + power_data_fname, condition=0)
                itc_diff_cross2 = mne.time_frequency.read_tfrs(trf_diff_save_path + 'cross2_' + itc_data_fname, condition=0)
            except:
                # Compute difference
                try:
                    # Load previous data
                    power_mssl = mne.time_frequency.read_tfrs(trf_load_path_mssl + power_data_fname, condition=0)
                    power_mssh = mne.time_frequency.read_tfrs(trf_load_path_mssh + power_data_fname, condition=0)
                    itc_mssl = mne.time_frequency.read_tfrs(trf_load_path_mssl + itc_data_fname, condition=0)
                    itc_mssh = mne.time_frequency.read_tfrs(trf_load_path_mssh + itc_data_fname, condition=0)
                except:
                    try:
                        # Compute power using saved epoched data
                        for mss, tmin, tmax, epochs_load_path, trf_save_path in zip((mssl, mssh), (tmin_mssl, tmin_mssh),
                                                                                    (tmax_mssl, tmax_mssh),
                                                                                    (epochs_load_path_mssl,
                                                                                     epochs_load_path_mssh),
                                                                                    (trf_load_path_mssl,
                                                                                     trf_load_path_mssh)):
                            # Load epoched data
                            epochs = mne.read_epochs(epochs_load_path + epochs_data_fname)
                            # Compute power and PLI over frequencies
                            power, itc = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                                           freqs_type=freqs_type,
                                                                           n_cycles_div=2., save_data=save_data,
                                                                           trf_save_path=trf_save_path,
                                                                           power_data_fname=power_data_fname,
                                                                           itc_data_fname=itc_data_fname)
                    except:
                        if use_ica_data:
                            # Load meg data
                            meg_data = load.ica_data(subject=subject)
                        else:
                            # Load meg data
                            meg_data = subject.load_preproc_meg_data()
                        for mss, tmin, tmax, epochs_save_path, trf_save_path in zip((mssl, mssh), (tmin_mssl, tmin_mssh),
                                                                                    (tmax_mssl, tmax_mssh),
                                                                                    (epochs_load_path_mssl,
                                                                                     epochs_load_path_mssh),
                                                                                    (trf_load_path_mssl,
                                                                                     trf_load_path_mssh)):
                            # Epoch data
                            epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans,
                                                                           tgt_pres=tgt_pres, epoch_id=epoch_id, meg_data=meg_data,
                                                                           tmin=tmin, tmax=tmax, reject=dict(mag=1),
                                                                           save_data=save_data, epochs_save_path=epochs_save_path,
                                                                           epochs_data_fname=epochs_data_fname)

                            # Compute power and PLI over frequencies
                            power, itc = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                                           freqs_type=freqs_type,
                                                                           n_cycles_div=2., save_data=save_data,
                                                                           trf_save_path=trf_save_path,
                                                                           power_data_fname=power_data_fname,
                                                                           itc_data_fname=itc_data_fname)

                    # Load previous data
                    power_mssl = mne.time_frequency.read_tfrs(trf_load_path_mssl + power_data_fname, condition=0)
                    power_mssh = mne.time_frequency.read_tfrs(trf_load_path_mssh + power_data_fname, condition=0)
                    itc_mssl = mne.time_frequency.read_tfrs(trf_load_path_mssl + itc_data_fname, condition=0)
                    itc_mssh = mne.time_frequency.read_tfrs(trf_load_path_mssh + itc_data_fname, condition=0)

                # Apply baseline to power and itc
                power_mssl.apply_baseline(baseline=baseline, mode=bline_mode)
                power_mssh.apply_baseline(baseline=baseline, mode=bline_mode)
                itc_mssl.apply_baseline(baseline=baseline, mode=bline_mode)
                itc_mssh.apply_baseline(baseline=baseline, mode=bline_mode)

                # Get time windows to compare
                # mssl
                mssl_ms_window_times = (-cross1_dur, mss_duration[mssl])
                mssl_cross2_window_times = (mss_duration[mssl], mss_duration[mssl] + cross2_dur)
                power_mssl_ms = power_mssl.copy().crop(tmin=mssl_ms_window_times[0], tmax=mssl_ms_window_times[1])
                itc_mssl_ms = itc_mssl.copy().crop(tmin=mssl_ms_window_times[0], tmax=mssl_ms_window_times[1])
                power_mssl_cross2 = power_mssl.copy().crop(tmin=mssl_cross2_window_times[0], tmax=mssl_cross2_window_times[1])
                itc_mssl_cross2 = itc_mssl.copy().crop(tmin=mssl_cross2_window_times[0], tmax=mssl_cross2_window_times[1])
                # mssh
                mssh_cross2_window_times = (mss_duration[mssh], mss_duration[mssh] + cross2_dur)
                power_mssh_ms = power_mssh.copy().crop(tmin=mssl_ms_window_times[0], tmax=mssl_ms_window_times[1])
                itc_mssh_ms = itc_mssh.copy().crop(tmin=mssl_ms_window_times[0], tmax=mssl_ms_window_times[1])
                # Force matching times by copying variable and changin data
                power_mssh_cross2 = power_mssl_cross2.copy()
                power_mssh_cross2.data = power_mssh.copy().crop(tmin=mssh_cross2_window_times[0], tmax=mssh_cross2_window_times[1]).data
                itc_mssh_cross2 = itc_mssl_cross2.copy()
                itc_mssh_cross2.data = itc_mssh.copy().crop(tmin=mssh_cross2_window_times[0], tmax=mssh_cross2_window_times[1]).data

                # Condition difference
                # MS
                power_diff_ms = power_mssh_ms - power_mssl_ms
                itc_diff_ms = itc_mssh_ms - itc_mssl_ms
                # Cross2
                power_diff_cross2 = power_mssh_cross2 - power_mssl_cross2
                itc_diff_cross2 = itc_mssh_cross2 - itc_mssl_cross2

                # Save data
                if save_data:
                    # Save epoched data
                    os.makedirs(trf_diff_save_path, exist_ok=True)
                    power_diff_ms.save(trf_diff_save_path + 'ms_' + power_data_fname, overwrite=True)
                    itc_diff_ms.save(trf_diff_save_path + 'ms_' + itc_data_fname, overwrite=True)
                    power_diff_cross2.save(trf_diff_save_path + 'cross2_' + power_data_fname, overwrite=True)
                    itc_diff_cross2.save(trf_diff_save_path + 'cross2_' + itc_data_fname, overwrite=True)

            # Append data for GA
            # MS
            averages_power_ms_diff.append(power_diff_ms)
            averages_itc_ms_diff.append(itc_diff_ms)
            # Cross2
            averages_power_cross2_diff.append(power_diff_cross2)
            averages_itc_cross2_diff.append(itc_diff_cross2)

            # Plot power time-frequency
            # MS
            fname = f'Power_ms_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            plot_general.tfr_bands(subject=subject, tfr=power_diff_ms, chs_id=chs_id, epoch_id=epoch_id, mss=mss,
                             cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                             display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname)
            # Cross2
            fname = f'Power_cross2_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            plot_general.tfr_bands(subject=subject, tfr=power_diff_cross2, chs_id=chs_id, epoch_id=epoch_id, mss=mss,
                             cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                             display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname)

            # Plot ITC time-frequency
            # MS
            fname = f'ITC_ms_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            plot_general.tfr_bands(subject=subject, tfr=itc_diff_ms, chs_id=chs_id, epoch_id=epoch_id, mss=mss,
                             cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                             display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname)
            # Cross2
            fname = f'ITC_cross2_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            plot_general.tfr_bands(subject=subject, tfr=itc_diff_cross2, chs_id=chs_id, epoch_id=epoch_id, mss=mss,
                             cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                             display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname)

            # Power topoplot
            # MS
            fig = power_diff_ms.plot_topo(cmap='jet', show=display_figs, title='Power', baseline=baseline, mode=bline_mode)
            if save_fig:
                fname = f'Power_ms_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)
            # Cross2
            fig = power_diff_ms.plot_topo(cmap='jet', show=display_figs, title='Power')
            if save_fig:
                fname = f'Power_itc_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)

            # ITC topoplot
            # MS
            fig = itc_diff_ms.plot_topo(cmap='jet', show=display_figs, title='Inter-Trial coherence', baseline=baseline, mode=bline_mode)
            if save_fig:
                fname = f'ITC_ms_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)
            # Cross2
            fig = itc_diff_cross2.plot_topo(cmap='jet', show=display_figs, title='Inter-Trial coherence')
            if save_fig:
                fname = f'ITC_cross2_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
                save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)

        # Compute grand average
        grand_avg_power_ms_diff = mne.grand_average(averages_power_ms_diff)
        grand_avg_itc_ms_diff = mne.grand_average(averages_itc_ms_diff)
        grand_avg_power_cross2_diff = mne.grand_average(averages_power_cross2_diff)
        grand_avg_itc_cross2_diff = mne.grand_average(averages_itc_cross2_diff)

        if save_data:
            # Save trf data
            grand_avg_power_ms_diff.save(trf_diff_save_path + grand_avg_power_ms_fname, overwrite=True)
            grand_avg_power_cross2_diff.save(trf_diff_save_path + grand_avg_power_cross2_fname, overwrite=True)
            grand_avg_itc_ms_diff.save(trf_diff_save_path + grand_avg_itc_ms_fname, overwrite=True)
            grand_avg_itc_cross2_diff.save(trf_diff_save_path + grand_avg_itc_cross2_fname, overwrite=True)

    # Plot Power time-frequency
    # MS
    fname = f'Power_ms_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_bands(tfr=grand_avg_power_ms_diff, chs_id=chs_id,
                     epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                     subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname,
                     vmin=-0.15, vmax=0.15, fontsize=16, ticksize=18)
    # Cross2
    fname = f'Power_cross2_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_bands(tfr=grand_avg_power_cross2_diff, chs_id=chs_id,
                     epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                     subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname,
                     vmin=-0.15, vmax=0.15, fontsize=16, ticksize=18)

    # Plot ITC time-frequency
    # MS
    fname = f'ITC_ms_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_bands(tfr=grand_avg_itc_ms_diff, chs_id=chs_id, epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur,
                     mss_duration=mss_duration, cross2_dur=cross2_dur, subject=None, display_figs=display_figs,
                     save_fig=save_fig, fig_path=trf_fig_path, fname=fname,
                     vmin=-0.15, vmax=0.15, fontsize=16, ticksize=18)
    # Cross2
    fname = f'ITC_cross2_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_bands(tfr=grand_avg_itc_cross2_diff, chs_id=chs_id,  epoch_id=epoch_id, mss=mss,
                     cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                     subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname,
                     vmin=-0.15, vmax=0.15, fontsize=16, ticksize=18)

    # Power topoplot
    # MS
    fig = grand_avg_power_ms_diff.plot_topo(cmap='jet', show=display_figs)
    if save_fig:
        fname = f'GA_Power_ms_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)
    # Cross2
    fig = grand_avg_power_cross2_diff.plot_topo(cmap='jet', show=display_figs)
    if save_fig:
        fname = f'GA_Power_cross2_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)

    # ITC topoplot
    # MS
    fig = grand_avg_itc_ms_diff.plot_topo(cmap='jet', show=display_figs, title='Inter-Trial coherence')
    if save_fig:
        fname = f'GA_ITC_ms_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)
    # Cross2
    fig = grand_avg_itc_cross2_diff.plot_topo(cmap='jet', show=display_figs, title='Inter-Trial coherence')
    if save_fig:
        fname = f'GA_ITC_cross2_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)


    # Power Plotjoint MS
    fname = f'GA_Power_ms_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_plotjoint_picks(tfr=grand_avg_power_ms_diff, plot_baseline=None, bline_mode=bline_mode,
                                     chs_id=chs_id, plot_max=False, plot_min=True, vmin=-0.1, vmax=0.1,
                                     display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)
    # Power Plotjoint Cross2
    fname = f'GA_Power_cross2_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_plotjoint_picks(tfr=grand_avg_power_cross2_diff, plot_baseline=None, bline_mode=bline_mode,
                                     chs_id=chs_id, plot_max=False, plot_min=True,vmin=-0.1, vmax=0.1,
                                     display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)
    # ITC Plotjoint MS
    fname = f'GA_ITC_ms_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_plotjoint_picks(tfr=grand_avg_itc_ms_diff, plot_baseline=None, bline_mode=bline_mode,
                                     chs_id=chs_id, plot_max=False, plot_min=True, vmin=-0.1, vmax=0.1,
                                     display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)
    # ITC Plot joint
    fname = f'GA_ITC_cross2_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_plotjoint_picks(tfr=grand_avg_itc_cross2_diff, plot_baseline=None, bline_mode=bline_mode,
                                     chs_id=chs_id, plot_max=False, plot_min=True, vmin=-0.1, vmax=0.1,
                                     display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

## Correct vs incorrect

import functions_general
import functions_analysis
import load
import mne
import os
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
chs_id = 'parietal'
# ICA / RAW
use_ica_data = True
corr_ans = None
tgt_pres = None
mss = None
epoch_id = 'ms'
# epoch_id = 'fix_vs'
# Power frequency range
l_freq = 1
h_freq = 100
log_bands = True

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
if 'cross1' in epoch_id and mss:
    dur = cross1_dur + mss_duration[mss] + cross2_dur + vs_dur  # seconds
elif 'ms' in epoch_id:
    dur = mss_duration[1]
elif 'cross2' in epoch_id:
    dur = cross2_dur + vs_dur  # seconds
else:
    dur = 0

# Get time windows from epoch_id name
map_times = dict(cross1={'tmin': 0, 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                 ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
                 cross2={'tmin': -cross1_dur - mss_duration[mss], 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                 sac={'tmin': -0.2, 'tmax': 0.2, 'plot_xlim': (-0.2, 0.2)},
                 fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

baseline = (tmin, 0)

# freqs type
if log_bands:
    freqs_type = 'log'
else:
    freqs_type = 'lin'

if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

averages_power_diff = []
averages_itc_diff = []

for subject_code in exp_info.subjects_ids:
    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if use_ica_data:
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Specific run path for loading data
    load_id_corr = f'{epoch_id}_mss{mss}_Corr_True_tgt_{tgt_pres}'
    load_id_inc = f'{epoch_id}_mss{mss}_Corr_False_tgt_{tgt_pres}'
    load_path_corr = f'/{load_id_corr}_{tmin}_{tmax}_bline{baseline}/'
    load_path_inc = f'/{load_id_inc}_{tmin}_{tmax}_bline{baseline}/'

    # Load data paths
    trf_load_path_corr = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + load_path_corr
    trf_load_path_inc = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + load_path_inc
    epochs_load_path_corr = paths().save_path() + f'Epochs_{data_type}/Band_None/' + load_path_corr
    epochs_load_path_itc = paths().save_path() + f'Epochs_{data_type}/Band_None/' + load_path_inc

    # Load data filenames
    power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
    itc_data_fname = f'ITC_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

    # Save data paths
    save_id = f'{epoch_id}_mss{mss}_Corr_True-False_tgt_{tgt_pres}'
    save_path = f'/{save_id}_{tmin}_{tmax}_bline{baseline}/'
    trf_diff_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + save_path

    # Save figures paths
    plot_path = f'/{save_id}_bline{baseline}/'
    trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + plot_path + f'{chs_id}/'
    # Subject plots path
    trf_fig_path_subj = trf_fig_path + f'{subject.subject_id}/'

    # Grand average data variable
    grand_avg_power_fname = f'Grand_Average_power_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_itc_fname = f'Grand_Average_itc_{l_freq}_{h_freq}_tfr.h5'

    try:
        raise ValueError()
        # Load difference data
        power_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + power_data_fname, condition=0)
        itc_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + itc_data_fname, condition=0)
    except:
        # Compute difference
        try:
            # Load previous power and itc data
            power_corr = mne.time_frequency.read_tfrs(trf_load_path_corr + power_data_fname, condition=0)
            power_inc = mne.time_frequency.read_tfrs(trf_load_path_inc + power_data_fname, condition=0)
            itc_corr = mne.time_frequency.read_tfrs(trf_load_path_corr + itc_data_fname, condition=0)
            itc_inc = mne.time_frequency.read_tfrs(trf_load_path_inc + itc_data_fname, condition=0)
        except:
            try:
                # Compute power and itc data from previous epoched data
                for epochs_load_path, trf_save_path in zip((epochs_load_path_corr, epochs_load_path_itc),
                                                           (trf_load_path_corr, trf_load_path_inc)):
                    # Load epoched data
                    epochs = mne.read_epochs(epochs_load_path + epochs_data_fname)
                    # Compute power and PLI over frequencies
                    power, itc = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                                   freqs_type=freqs_type,
                                                                   n_cycles_div=2., save_data=save_data,
                                                                   trf_save_path=trf_save_path,
                                                                   power_data_fname=power_data_fname,
                                                                   itc_data_fname=itc_data_fname)
            except:
                # Get Epochs from Raw data
                if use_ica_data:
                    # Load meg data
                    meg_data = load.ica_data(subject=subject)
                else:
                    # Load meg data
                    meg_data = subject.load_preproc_meg_data()
                for corr_ans, epochs_save_path, trf_save_path in zip((True, False), (epochs_load_path_corr, epochs_load_path_itc),
                                                                     (trf_load_path_corr, trf_load_path_inc)):
                    # Epoch data
                    epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans,
                                                                   tgt_pres=tgt_pres, epoch_id=epoch_id, meg_data=meg_data,
                                                                   tmin=tmin, tmax=tmax, reject=dict(mag=1),
                                                                   save_data=save_data, epochs_save_path=epochs_save_path,
                                                                   epochs_data_fname=epochs_data_fname)
                    # Compute power and PLI over frequencies
                    power, itc = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                                   freqs_type=freqs_type,
                                                                   n_cycles_div=4., save_data=save_data,
                                                                   trf_save_path=trf_save_path,
                                                                   power_data_fname=power_data_fname,
                                                                   itc_data_fname=itc_data_fname)
            # Load previous data
            power_corr = mne.time_frequency.read_tfrs(trf_load_path_corr + power_data_fname, condition=0)
            power_inc = mne.time_frequency.read_tfrs(trf_load_path_inc + power_data_fname, condition=0)
            itc_corr = mne.time_frequency.read_tfrs(trf_load_path_corr + itc_data_fname, condition=0)
            itc_inc = mne.time_frequency.read_tfrs(trf_load_path_inc + itc_data_fname, condition=0)

        # Apply baseline to power and itc
        power_corr.apply_baseline(baseline=baseline, mode=bline_mode)
        power_inc.apply_baseline(baseline=baseline, mode=bline_mode)
        itc_corr.apply_baseline(baseline=baseline, mode=bline_mode)
        itc_inc.apply_baseline(baseline=baseline, mode=bline_mode)

        # # Get time windows to compare
        # # corr
        # power_corr = power_corr.copy().crop(tmin=tmin, tmax=tmax)
        # itc_corr = itc_corr.copy().crop(tmin=tmin, tmax=tmax)
        # # inc
        # power_inc = power_inc.copy().crop(tmin=tmin, tmax=tmax)
        # itc_inc = itc_inc.copy().crop(tmin=tmin, tmax=tmax)
        #
        # # Force matching times by copying variable and changin data
        # power_mssh_cross2 = power_mssl_cross2.copy()
        # power_mssh_cross2.data = power_mssh.copy().crop(tmin=mssh_cross2_window_times[0], tmax=mssh_cross2_window_times[1]).data
        # itc_mssh_cross2 = itc_mssl_cross2.copy()
        # itc_mssh_cross2.data = itc_mssh.copy().crop(tmin=mssh_cross2_window_times[0], tmax=mssh_cross2_window_times[1]).data

        # Condition difference
        power_diff = power_corr - power_inc
        itc_diff = itc_corr - itc_inc

        # Save data
        if save_data:
            # Save epoched data
            os.makedirs(trf_diff_save_path, exist_ok=True)
            power_diff.save(trf_diff_save_path + power_data_fname, overwrite=True)
            itc_diff.save(trf_diff_save_path + itc_data_fname, overwrite=True)

    # Append data for GA
    averages_power_diff.append(power_diff)
    averages_itc_diff.append(itc_diff)

    # Plot power time-frequency
    fname = f'Power_{epoch_id}_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_bands(subject=subject, tfr=power_diff, chs_id=chs_id, epoch_id=epoch_id, mss=mss,
                     cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                     display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname)
    # Plot ITC time-frequency
    fname = f'ITC_{epoch_id}_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_bands(subject=subject, tfr=itc_diff, chs_id=chs_id, epoch_id=epoch_id, mss=mss,
                     cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                     display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname)

    # Power topoplot
    fig = power_diff.plot_topo(cmap='jet', show=display_figs, title='Power')
    if save_fig:
        fname = f'Power_{epoch_id}_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)
    # ITC topoplot
    fig = itc_diff.plot_topo(cmap='jet', show=display_figs, title='Inter-Trial coherence')
    if save_fig:
        fname = f'ITC_{epoch_id}_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)

# Grand Average
try:
    # Load previous power data
    grand_avg_power_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_power_fname)[0]
    # Load previous itc data
    grand_avg_itc_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_itc_fname)[0]

    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power_diff.info)
except:
    # Compute grand average
    grand_avg_power_diff = mne.grand_average(averages_power_diff)
    grand_avg_itc_diff = mne.grand_average(averages_itc_diff)

    if save_data:
        # Save trf data
        grand_avg_power_diff.save(trf_diff_save_path + grand_avg_power_fname, overwrite=True)
        grand_avg_itc_diff.save(trf_diff_save_path + grand_avg_itc_fname, overwrite=True)

# Plot Power time-frequency
fname = f'Power_{epoch_id}_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_bands(tfr=grand_avg_power_diff, chs_id=chs_id,
                 epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                 subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname)
# Plot ITC time-frequency
fname = f'ITC_{epoch_id}_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_bands(tfr=grand_avg_itc_diff, chs_id=chs_id, epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration,
                 cross2_dur=cross2_dur,
                 subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname)

# Power topoplot
fig = grand_avg_power_diff.plot_topo(cmap='jet', show=display_figs, title='Power')
if save_fig:
    fname = f'GA_Power_{epoch_id}_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    save.fig(fig=fig, path=trf_fig_path, fname=fname)

# ITC topoplot
fig = grand_avg_itc_diff.plot_topo(cmap='jet', show=display_figs, title='Inter-Trial coherence')
if save_fig:
    fname = f'GA_ITC_{epoch_id}_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    save.fig(fig=fig, path=trf_fig_path, fname=fname)


## Correct vs incorrect per MSS

import functions_general
import functions_analysis
import load
import mne
import os
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
chs_id = 'frontal'
# ICA / RAW
use_ica_data = True
corr_ans = None
tgt_pres = None
# mss = None
epoch_id = 'ms'
# epoch_id = 'fix_vs'
# Power frequency range
l_freq = 1
h_freq = 100
log_bands = True

# Baseline method
bline_mode = 'logratio'
#----------#

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 0}
cross1_dur = 0.75
cross2_dur = 1
vs_dur = 4
plot_edge = 0.15

# freqs type
if log_bands:
    freqs_type = 'log'
else:
    freqs_type = 'lin'

if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

for mss in [1, 2, 4]:

    # Duration
    if 'cross1' in epoch_id and mss:
        dur = cross1_dur + mss_duration[mss] + cross2_dur + vs_dur  # seconds
    elif 'ms' in epoch_id:
        dur = mss_duration[mss] + cross2_dur + vs_dur  # seconds
    elif 'cross2' in epoch_id:
        dur = cross2_dur + vs_dur  # seconds
    else:
        dur = 0

    # Get time windows from epoch_id name
    map_times = dict(cross1={'tmin': 0, 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                     ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
                     cross2={'tmin': -cross1_dur - mss_duration[mss], 'tmax': dur,
                             'plot_xlim': (-cross1_dur - mss_duration[mss] + plot_edge, dur - plot_edge)},
                     sac={'tmin': -0.2, 'tmax': 0.2, 'plot_xlim': (-0.2, 0.2)},
                     fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)})
    tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

    # Baseline duration
    if 'sac' in epoch_id:
        baseline = (tmin, 0)
        # baseline = None
    elif 'fix' in epoch_id or 'fix' in epoch_id:
        baseline = (tmin, -0.05)
    elif 'cross1' in epoch_id or 'ms' in epoch_id or 'cross2' in epoch_id and mss:
        baseline = (tmin, 0)
    else:
        baseline = (tmin, 0)

    averages_power_diff = []
    averages_itc_diff = []

    for subject_code in exp_info.subjects_ids:
        # Define save path and file name for loading and saving epoched, evoked, and GA data
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

        # Specific run path for loading data
        load_id_corr = f'{epoch_id}_mss{mss}_Corr_True_tgt_{tgt_pres}'
        load_id_inc = f'{epoch_id}_mss{mss}_Corr_False_tgt_{tgt_pres}'
        load_path_corr = f'/{load_id_corr}_{tmin}_{tmax}_bline{baseline}/'
        load_path_inc = f'/{load_id_inc}_{tmin}_{tmax}_bline{baseline}/'

        # Load data paths
        trf_load_path_corr = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + load_path_corr
        trf_load_path_inc = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + load_path_inc
        epochs_load_path_corr = paths().save_path() + f'Epochs_{data_type}/Band_None/' + load_path_corr
        epochs_load_path_itc = paths().save_path() + f'Epochs_{data_type}/Band_None/' + load_path_inc

        # Load data filenames
        power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
        itc_data_fname = f'ITC_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

        # Save data paths
        save_id = f'{epoch_id}_mss{mss}_Corr_True-False_tgt_{tgt_pres}'
        save_path = f'/{save_id}_{tmin}_{tmax}_bline{baseline}/'
        trf_diff_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + save_path

        # Save figures paths
        plot_path = f'/{save_id}_bline{baseline}/'
        trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + plot_path + f'{chs_id}/'
        # Subject plots path
        trf_fig_path_subj = trf_fig_path + f'{subject.subject_id}/'

        # Grand average data variable
        grand_avg_power_fname = f'Grand_Average_power_{l_freq}_{h_freq}_tfr.h5'
        grand_avg_itc_fname = f'Grand_Average_itc_{l_freq}_{h_freq}_tfr.h5'

        try:
            # Load difference data
            power_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + power_data_fname, condition=0)
            itc_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + itc_data_fname, condition=0)
        except:
            # Compute difference
            try:
                # Load previous power and itc data
                power_corr = mne.time_frequency.read_tfrs(trf_load_path_corr + power_data_fname, condition=0)
                power_inc = mne.time_frequency.read_tfrs(trf_load_path_inc + power_data_fname, condition=0)
                itc_corr = mne.time_frequency.read_tfrs(trf_load_path_corr + itc_data_fname, condition=0)
                itc_inc = mne.time_frequency.read_tfrs(trf_load_path_inc + itc_data_fname, condition=0)
            except:
                try:
                    # Compute power and itc data from previous epoched data
                    for epochs_load_path, trf_save_path in zip((epochs_load_path_corr, epochs_load_path_itc),
                                                               (trf_load_path_corr, trf_load_path_inc)):
                        # Load epoched data
                        epochs = mne.read_epochs(epochs_load_path + epochs_data_fname)
                        # Compute power and PLI over frequencies
                        power, itc = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                                       freqs_type=freqs_type,
                                                                       n_cycles_div=2., save_data=save_data,
                                                                       trf_save_path=trf_save_path,
                                                                       power_data_fname=power_data_fname,
                                                                       itc_data_fname=itc_data_fname)
                except:
                    # Get Epochs from Raw data
                    if use_ica_data:
                        # Load meg data
                        meg_data = load.ica_data(subject=subject)
                    else:
                        # Load meg data
                        meg_data = subject.load_preproc_meg_data()
                    for corr_ans, epochs_save_path, trf_save_path in zip((True, False), (epochs_load_path_corr, epochs_load_path_itc),
                                                                         (trf_load_path_corr, trf_load_path_inc)):
                        # Epoch data
                        epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans,
                                                                       tgt_pres=tgt_pres, epoch_id=epoch_id, meg_data=meg_data,
                                                                       tmin=tmin, tmax=tmax, reject=dict(mag=1),
                                                                       save_data=save_data, epochs_save_path=epochs_save_path,
                                                                       epochs_data_fname=epochs_data_fname)
                        # Compute power and PLI over frequencies
                        power, itc = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                                       freqs_type=freqs_type,
                                                                       n_cycles_div=4., save_data=save_data,
                                                                       trf_save_path=trf_save_path,
                                                                       power_data_fname=power_data_fname,
                                                                       itc_data_fname=itc_data_fname)
                    # Free memory
                    del meg_data
                # Free memory
                del  epochs, power, itc

                # Load previous data
                power_corr = mne.time_frequency.read_tfrs(trf_load_path_corr + power_data_fname, condition=0)
                power_inc = mne.time_frequency.read_tfrs(trf_load_path_inc + power_data_fname, condition=0)
                itc_corr = mne.time_frequency.read_tfrs(trf_load_path_corr + itc_data_fname, condition=0)
                itc_inc = mne.time_frequency.read_tfrs(trf_load_path_inc + itc_data_fname, condition=0)

            # Apply baseline to power and itc
            power_corr.apply_baseline(baseline=baseline, mode=bline_mode)
            power_inc.apply_baseline(baseline=baseline, mode=bline_mode)
            itc_corr.apply_baseline(baseline=baseline, mode=bline_mode)
            itc_inc.apply_baseline(baseline=baseline, mode=bline_mode)

            # Condition difference
            power_diff = power_corr - power_inc
            itc_diff = itc_corr - itc_inc

            # Save data
            if save_data:
                # Save epoched data
                os.makedirs(trf_diff_save_path, exist_ok=True)
                power_diff.save(trf_diff_save_path + power_data_fname, overwrite=True)
                itc_diff.save(trf_diff_save_path + itc_data_fname, overwrite=True)

            # Free memory
            del power_corr, power_inc, itc_corr, itc_inc

        # Append data for GA
        averages_power_diff.append(power_diff)
        averages_itc_diff.append(itc_diff)

        # Plot power time-frequency
        fname = f'Power_{epoch_id}_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        plot_general.tfr_bands(subject=subject, tfr=power_diff, chs_id=chs_id, epoch_id=epoch_id, mss=mss,
                         cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                         display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname)
        # Plot ITC time-frequency
        fname = f'ITC_{epoch_id}_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        plot_general.tfr_bands(subject=subject, tfr=itc_diff, chs_id=chs_id, epoch_id=epoch_id, mss=mss,
                         cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                         display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname)

        # Power topoplot
        fig = power_diff.plot_topo(cmap='jet', show=display_figs, title='Power')
        if save_fig:
            fname = f'Power_{epoch_id}_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)
        # ITC topoplot
        fig = itc_diff.plot_topo(cmap='jet', show=display_figs, title='Inter-Trial coherence')
        if save_fig:
            fname = f'ITC_{epoch_id}_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)

        del power_diff, itc_diff

    # Grand Average
    try:
        # Load previous power data
        grand_avg_power_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_power_fname)[0]
        # Load previous itc data
        grand_avg_itc_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_itc_fname)[0]

        # Pick plot channels
        picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power_diff.info)
    except:
        # Compute grand average
        grand_avg_power_diff = mne.grand_average(averages_power_diff)
        grand_avg_itc_diff = mne.grand_average(averages_itc_diff)

        if save_data:
            # Save trf data
            grand_avg_power_diff.save(trf_diff_save_path + grand_avg_power_fname, overwrite=True)
            grand_avg_itc_diff.save(trf_diff_save_path + grand_avg_itc_fname, overwrite=True)

    # Plot Power time-frequency
    fname = f'Power_{epoch_id}_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_bands(tfr=grand_avg_power_diff, chs_id=chs_id,
                     epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur,
                     subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname)
    # Plot ITC time-frequency
    fname = f'ITC_{epoch_id}_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_bands(tfr=grand_avg_itc_diff, chs_id=chs_id, epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur, mss_duration=mss_duration,
                     cross2_dur=cross2_dur,
                     subject=None, display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path, fname=fname)

    # Power topoplot
    fig = grand_avg_power_diff.plot_topo(cmap='jet', show=display_figs, title='Power')
    if save_fig:
        fname = f'GA_Power_{epoch_id}_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)

    # ITC topoplot
    fig = grand_avg_itc_diff.plot_topo(cmap='jet', show=display_figs, title='Inter-Trial coherence')
    if save_fig:
        fname = f'GA_ITC_{epoch_id}_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        save.fig(fig=fig, path=trf_fig_path, fname=fname)

    del averages_power_diff, averages_itc_diff, grand_avg_power_diff, grand_avg_itc_diff

