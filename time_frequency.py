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
import os


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
chs_id = 'central'  # region_hemisphere
# ICA / RAW
use_ica_data = True
# Epochs
epoch_id = 'blue'
corr_ans = None
tgt_pres = None
mss = None
reject = None  # 'subject' for subject's default. False for no rejection, dict for specific values. None for default 5e-12 for magnetometers
# Trial durations
vs_dur = {1: (2, 9.8), 2: (3, 9.8), 4: (3.5, 9.8), None: (2, 9.8)}
plot_edge = 0.15
trial_dur = vs_dur[mss]  # Edit this to determine the minimum visual search duration for the trial selection (this will also affect ms trials)
evt_dur = None

# Power time frequency params
n_cycles_div = 2.
l_freq = 1
h_freq = 40
log_bands = False
run_itc = False

# Sources params
estimate_sources = False
sources_from_tfr = False
if estimate_sources:
    default_subject = exp_info.subjects_ids[0]
    # Source model config
    surf_vol = 'surface'
    ico = 4
    spacing = 10.

# Plots parameters
# Colorbar
vmin_power, vmax_power = None, None
vmin_itc, vmax_itc = None, None
topo_vmin, topo_vmax = None, None
# plot_joint max and min topoplots
plot_max, plot_min = True, True

# Baseline method
# logratio: dividing by the mean of baseline values and taking the log
# ratio: dividing by the mean of baseline values
# mean: subtracting the mean of baseline values
bline_mode = 'logratio'

# Topoplot bands
topo_bands = ['Alpha', 'Alpha', 'Theta', 'Alpha']
#----------#

# Time Frequency config
if estimate_sources:
    return_average_tfr = False
    output = 'complex'
    run_itc = False

else:
    return_average_tfr = True
    output = 'power'

# Windows durations
dur, cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration(epoch_id=epoch_id, vs_dur=vs_dur, mss=mss)

# Get time windows from epoch_id name
map = dict(tgt_fix={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.3, 0.6)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, mss=mss, plot_edge=plot_edge, map=map)

# Define time-frequency bands to plot in plot_joint
timefreqs_joint, timefreqs_tfr, vlines_times = functions_general.get_plots_timefreqs(epoch_id=epoch_id, mss=mss,
                                                                                         cross2_dur=cross2_dur,
                                                                                         mss_duration=mss_duration,
                                                                                         topo_bands=topo_bands, plot_xlim=plot_xlim)
if (plot_max or plot_min):
    timefreqs_joint = None

# Get baseline duration for epoch_id
baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=epoch_id, mss=mss, tmin=tmin, tmax=tmax, plot_xlim=plot_xlim,
                                                                  cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                  cross2_dur=cross2_dur)
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
save_id = f'{epoch_id}_mss{mss}_Corr{corr_ans}_tgt{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}'
plot_id = f'{save_id}_{plot_xlim[0]}_{plot_xlim[1]}_bline{baseline}_cyc{int(n_cycles_div)}/'

# Save data paths
if return_average_tfr:
    trf_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{save_id}_{tmin}_{tmax}_bline{baseline}_cyc{int(n_cycles_div)}/'
else:
    trf_save_path = paths().save_path() + f'Time_Frequency_Epochs_{data_type}/{save_id}_{tmin}_{tmax}_bline{baseline}_cyc{int(n_cycles_div)}/'
epochs_save_path = paths().save_path() + f'Epochs_{data_type}/Band_None/{save_id}_{tmin}_{tmax}_bline{baseline}/'
# Save figures paths
trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/' + plot_id + f'{chs_id}/'

# Grand average data variable
grand_avg_power_fname = f'Grand_Average_power_{l_freq}_{h_freq}_tfr.h5'
grand_avg_itc_fname = f'Grand_Average_itc_{l_freq}_{h_freq}_tfr.h5'

# Freesurfer directory
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')

# Source estimates grand average
stcs_fs = []

try:
    raise ValueError
    # Load previous power data
    grand_avg_power = mne.time_frequency.read_tfrs(trf_save_path + grand_avg_power_fname)[0]
    if run_itc:
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
            power = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq, freqs_type=freqs_type,
                                                      n_cycles_div=n_cycles_div, average=return_average_tfr,
                                                      return_itc=run_itc, output=output, save_data=save_data,
                                                      trf_save_path=trf_save_path, power_data_fname=power_data_fname,
                                                      itc_data_fname=itc_data_fname, n_jobs=4)
            if run_itc:
                power, itc = power


            # -------------- Extra Sources ---------------#
            if estimate_sources:
                sources_path_subject = paths().sources_path() + subject.subject_id

                # Check if subject has MRI data
                try:
                    fs_subj_path = os.path.join(subjects_dir, subject.subject_id)
                    os.listdir(fs_subj_path)
                except:
                    subject_code = 'fsaverage'

                # Load forward model
                if surf_vol == 'volume':
                    fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
                elif surf_vol == 'surface':
                    fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
                elif surf_vol == 'mixed':
                    fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
                fwd = mne.read_forward_solution(fname_fwd)
                src = fwd['src']

                if sources_from_tfr:
                    # From TFR
                    # rank
                    rank = sum([ch_type == 'mag' for ch_type in power.get_channel_types()]) - len(power.info['bads'])
                    if use_ica_data:
                        rank -= len(subject.ex_components)

                    power.apply_baseline(baseline=baseline, mode=bline_mode)
                    csd = mne.time_frequency.csd_tfr(power, tmin=0, tmax=None)
                    csd_baseline = mne.time_frequency.csd_tfr(power, tmin=None, tmax=0)

                    # Compute scalar DICS beamfomer
                    filters = mne.beamformer.make_dics(
                        info=power.info,
                        forward=fwd,
                        csd=csd,
                        noise_csd=csd_baseline,
                        pick_ori=None,
                        rank=dict(mag=rank),
                        real_filter=True
                    )

                    # project the TFR for each epoch to source space
                    epochs_stcs = mne.beamformer.apply_dics_tfr_epochs(power, filters, return_generator=True)

                    # average across frequencies and epochs
                    data = np.zeros((fwd["nsource"], power.times.size))
                    for epoch_stcs in epochs_stcs:
                        for stc in epoch_stcs:
                            data += (stc.data * np.conj(stc.data)).real

                    stc.data = data / len(power) / len(power.freqs)

                    # apply a baseline correction
                    stc.apply_baseline(baseline=baseline)

                else:
                    # From epochs
                    # rank
                    rank = sum([ch_type == 'mag' for ch_type in epochs.get_channel_types()]) - len(epochs.info['bads'])
                    if use_ica_data:
                        rank -= len(subject.ex_components)
                    h_freq = 12
                    l_freq = 8
                    freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)
                    csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=0, n_jobs=4)
                    csd_baseline = mne.time_frequency.csd_morlet(epochs, freqs, tmax=0, n_jobs=4)

                    # compute scalar DICS beamfomer
                    filters = mne.beamformer.make_dics(
                        info=epochs.info,
                        forward=fwd,
                        csd=csd,
                        noise_csd=csd_baseline,
                        pick_ori=None,
                        rank=dict(mag=rank),
                        real_filter=True,
                    )

                    # Project the TFR for each epoch to source space
                    stc, _ = mne.beamformer.apply_dics_csd(csd, filters)
                    stc_base, _ = mne.beamformer.apply_dics_csd(csd_baseline, filters)

                    # Apply baseline
                    stc /= stc_base

                # Morph to default subject
                if subject_code != default_subject:
                    # Get Source space for default subject
                    if surf_vol == 'volume':
                        fname_src = paths().sources_path() + default_subject + f'/{default_subject}_volume_ico{ico}_{int(spacing)}-src.fif'
                    elif surf_vol == 'surface':
                        fname_src = paths().sources_path() + default_subject + f'/{default_subject}_surface_ico{ico}-src.fif'
                    elif surf_vol == 'mixed':
                        fname_src = paths().sources_path() + default_subject + f'/{default_subject}_mixed_ico{ico}_{int(spacing)}-src.fif'

                    src_fs = mne.read_source_spaces(fname_src)

                    # Define morph function
                    morph = mne.compute_source_morph(src=src, subject_from=subject_code, subject_to=default_subject,
                                                     src_to=src_fs, subjects_dir=subjects_dir)
                    # Morph
                    stc_fs = morph.apply(stc)

                    # Append to fs_stcs to make GA
                    stcs_fs.append(stc_fs)

                else:
                    # Append to fs_stcs to make GA
                    stcs_fs.append(stc)

                # Plot
                if surf_vol == 'surface':

                    message = f"DICS source power in the {l_freq}-{h_freq} Hz frequency band"
                    # 3D plot
                    brain = stc.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, hemi='both',
                                     spacing=f'ico{ico}', time_unit='s', smoothing_steps=7, time_label=message)

                elif surf_vol == 'volume':
                    # 3D plot
                    brain = stc.plot_3d(src=fwd['src'], subject=subject_code, subjects_dir=subjects_dir)
                    # Nutmeg plot
                    stc.plot(src=fwd['src'], subject=subject_code, subjects_dir=subjects_dir)

            # --------------- End extra DICS sources ------------------#

            if not return_average_tfr and output == 'power':
                # Average epochs
                power = power.average()
                if run_itc:
                    itc = itc.average()

        # Append data for GA
        averages_power.append(power)

        # Plot power time-frequency
        fname = f'Power_tf_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
        plot_general.tfr_bands(subject=subject, tfr=power, chs_id=chs_id, plot_xlim=plot_xlim,
                         baseline=plot_baseline, bline_mode=bline_mode,
                         display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path_subj, fname=fname,
                         fontsize=16, ticksize=18)

        # Power topoplot
        fig = power.plot_topo(baseline=plot_baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1], cmap='jet',
                              show=display_figs, title='Power')
        if save_fig:
            fname = f'Power_topoch_{subject.subject_id}_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
            save.fig(fig=fig, path=trf_fig_path_subj, fname=fname)

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
        plt.close('all')
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


# Plot Power time-frequency
fname = f'Power_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_times(tfr=grand_avg_power, chs_id=chs_id, timefreqs_tfr=timefreqs_tfr, baseline=plot_baseline, bline_mode=bline_mode,
                       plot_xlim=plot_xlim, vlines_times=vlines_times, topo_vmin=topo_vmin, topo_vmax=topo_vmax, subject=None, display_figs=display_figs,
                       save_fig=save_fig, fig_path=trf_fig_path, fname=fname, vmin=vmin_power, vmax=vmax_power, fontsize=16, ticksize=18)

# Power Plot joint
fname = f'GA_Power_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint_picks(tfr=grand_avg_power, plot_baseline=plot_baseline, bline_mode=bline_mode, vlines_times=vlines_times,
                                 timefreqs=timefreqs_joint, plot_xlim=plot_xlim, chs_id=chs_id, vmin=vmin_power, vmax=vmax_power,
                                 plot_max=plot_max, plot_min=plot_min, display_figs=display_figs, save_fig=save_fig,
                                 trf_fig_path=trf_fig_path, fname=fname)

# Power Plot joint
fname = f'GA_Power_plotjoint_mag_{bline_mode}_{l_freq}_{h_freq}'
plot_general.tfr_plotjoint(tfr=grand_avg_power, plot_baseline=plot_baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                           vmin=vmin_power, vmax=vmax_power, timefreqs=timefreqs_joint, plot_max=plot_max, plot_min=plot_min,
                           vlines_times=vlines_times, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

# # Power topoplot
# fig = grand_avg_power.plot_topo(baseline=plot_baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
#                                 cmap='jet', show=display_figs, title='Power')
# if save_fig:
#     fname = f'GA_Power_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
#     save.fig(fig=fig, path=trf_fig_path, fname=fname)

if run_itc:
    # Plot ITC time-frequency
    fname = f'ITC_tf_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_times(tfr=grand_avg_itc, chs_id=chs_id, timefreqs_tfr=None, baseline=plot_baseline, bline_mode=bline_mode,
                           plot_xlim=plot_xlim, vlines_times=vlines_times, topo_vmin=topo_vmin, topo_vmax=topo_vmax, subject=None, display_figs=display_figs,
                           save_fig=save_fig, fig_path=trf_fig_path, fname=fname, vmin=vmin_itc, vmax=vmax_itc, fontsize=16, ticksize=18)

    # ITC Plot joint
    fname = f'GA_ITC_plotjoint_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_plotjoint_picks(tfr=grand_avg_itc, plot_baseline=plot_baseline, bline_mode=bline_mode,
                                     vlines_times=vlines_times, timefreqs=timefreqs_joint, plot_xlim=plot_xlim,
                                     chs_id=chs_id, vmin=vmin_itc, vmax=vmax_itc, plot_max=plot_max, plot_min=plot_min,
                                     display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

    # ITC Plot joint
    fname = f'GA_ITC_plotjoint_mag_{bline_mode}_{l_freq}_{h_freq}'
    plot_general.tfr_plotjoint(tfr=grand_avg_itc, plot_baseline=plot_baseline, bline_mode=bline_mode, plot_xlim=plot_xlim,
                               vmin=vmin_itc, vmax=vmax_itc, timefreqs=timefreqs_joint, plot_max=plot_max, plot_min=plot_min,
                               vlines_times=vlines_times, display_figs=display_figs, save_fig=save_fig, trf_fig_path=trf_fig_path, fname=fname)

    # # ITC topoplot
    # fig = grand_avg_itc.plot_topo(baseline=plot_baseline, mode=bline_mode, tmin=plot_xlim[0], tmax=plot_xlim[1],
    #                               cmap='jet',
    #                               show=display_figs, title='Inter-Trial coherence')
    # if save_fig:
    #     fname = f'GA_ITC_topoch_{chs_id}_{bline_mode}_{l_freq}_{h_freq}'
    #     save.fig(fig=fig, path=trf_fig_path, fname=fname)

