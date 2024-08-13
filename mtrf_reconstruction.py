import functions_analysis
import functions_general
import load
import save
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
import mne
import plot_general


#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()


#-----  Parameters -----#
# Select features (events)
trial_params = {'input_features': ['sac_cross1', 'fix_cross1', 'sac_ms', 'fix_ms', 'sac_cross2', 'fix_cross2', 'sac_vs', 'fix_vs'],
                'corrans': None,
                'tgtpres': None,
                'mss': None,
                'evtdur': None}

# MEG parameters
meg_params = {'chs_id': 'parietal_occipital',
              'band_id': 'HGamma',
              'data_type': 'ICA'}

# TRF parameters
trf_params = {'epoch_id': 'vs',
              'standarize': False,
              'fit_power': True,
              'alpha': None,
              'tmin': -0.05,
              'tmax': 0.15}

trf_params['baseline'] = (trf_params['tmin'], -0.05)

# Get frequencies from band id
l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])

# Window durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

# Time Frequency params
plot_edge = 0.15

bline_mode = 'logratio'
ga_plot_bline_mode = 'mean'

if 'vs' in trial_params['input_features']:
    trial_params['trialdur'] = vs_dur[trial_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_params['trialdur'] = None

# Region channels to run trf separately
all_chs_regions = ['frontal', 'temporal', 'central', 'parietal', 'occipital']

# Define Grand average variables
ga = {}
for feature in trial_params['input_features']:
    ga[feature] = []

# Figure path
fig_path = paths().plots_path() + (f"TRF_SIM_{meg_params['data_type']}/Band_{meg_params['band_id']}/{trial_params['input_features']}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_"
                                   f"tgtpres{trial_params['tgtpres']}_trialdur{trial_params['trialdur']}_evtdur{trial_params['evtdur']}_{trf_params['tmin']}_{trf_params['tmax']}_"
                                   f"bline{trf_params['baseline']}_alpha{trf_params['alpha']}_std{trf_params['standarize']}/{meg_params['chs_id']}/")

# Change path to include envelope power
if trf_params['fit_power']:
    fig_path = fig_path.replace(f"{meg_params['data_type']}", f"{meg_params['data_type']}_ENV")

# Save path
save_path = fig_path.replace(paths().plots_path(), paths().save_path())
save_path = save_path.replace('_SIM', '')

# Define Grand average variables to plot TRF evoked responses
feature_evokeds = {}
for feature in trial_params['input_features']:
    feature_evokeds[feature] = []

# MSS whole trial evokeds
trial_evokeds = {1: [], 2: [], 4: []}

# Iterate over subjects
for subject_code in exp_info.subjects_ids:

    # Load subject and MEG
    if meg_params['data_type'] == 'ICA':
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        # Load MEG
        if meg_params['band_id']:
            meg_data = load.filtered_data(subject=subject, band_id=meg_params['band_id'], save_data=save_data)
        else:
            meg_data = load.ica_data(subject=subject)
    elif meg_params['data_type'] == 'RAW':
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        # Load MEG
        if meg_params['band_id']:
            meg_data = load.filtered_data(subject=subject, band_id=meg_params['band_id'], use_ica_data=False, save_data=save_data)
        else:
            meg_data = subject.load_preproc_meg_data()

    # TRF paths
    trf_path = save_path
    trf_fname = f'TRF_{subject_code}.pkl'

    try:
        # Load TRF
        rf = load.var(trf_path + trf_fname)
        print('Loaded Receptive Field')

    except:
        # Compute TRF for defined features
        rf = functions_analysis.compute_trf(subject=subject, meg_data=meg_data, trial_params=trial_params, trf_params=trf_params, meg_params=meg_params,
                                            save_data=save_data, trf_path=trf_path, trf_fname=trf_fname)

    # Get model coeficients as separate responses to each feature
    trf_fig_path = fig_path.replace('_SIM', '')
    subj_evoked, feature_evokeds = functions_analysis.make_trf_evoked(subject=subject, rf=rf, meg_data=meg_data, evokeds=feature_evokeds,
                                                                      trf_params=trf_params, trial_params=trial_params, meg_params=meg_params,
                                                                      display_figs=display_figs, plot_individuals=plot_individuals, save_fig=save_fig, fig_path=trf_fig_path)

    # Select channels and pick in data
    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)

    # Reconstruct ntire signal from selected features TRFs
    reconstructed_meg = functions_analysis.reconstruct_meg_from_trf(subject=subject, rf=rf, meg_data=meg_data,  picks=picks,
                                                                    trial_params=trial_params, meg_params=meg_params)

    for mss_subj in [1, 2, 4]:

        recon_tmin, recon_tmax, plot_xlim = functions_general.get_time_lims(epoch_id=trf_params['epoch_id'], mss=mss_subj, plot_edge=plot_edge)

        print(f'Epoching data for MSS: {mss_subj}')
        # Epoch data
        epochs, events = functions_analysis.epoch_data(subject=subject, meg_data=reconstructed_meg, epoch_id=trf_params['epoch_id'], mss=mss_subj,
                                                       tgt_pres=trial_params['tgtpres'], corr_ans=trial_params['corrans'], trial_dur=trial_params['trialdur'],
                                                       baseline=trf_params['baseline'], reject=False, tmin=recon_tmin, tmax=recon_tmax,
                                                       save_data=False, epochs_save_path=None, epochs_data_fname=None)

        if trf_params['fit_power']:
            evoked = epochs.average()
            trial_evokeds[mss_subj].append(evoked)

        else:
            _, plot_baseline = functions_general.get_baseline_duration(epoch_id=trf_params['epoch_id'], mss=trial_params['mss'], tmin=recon_tmin, tmax=recon_tmax,
                                                                       cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur, plot_edge=plot_edge)

            power = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq, n_cycles_div=2., average=True, return_itc=False,
                                                      output='power', save_data=False, trf_save_path=None, power_data_fname=None, itc_data_fname=None, n_jobs=4)

            power.apply_baseline(baseline=plot_baseline, mode=bline_mode)

            # Append data for GA
            trial_evokeds[mss_subj].append(power)

# Plot features Grand average evoked
_ = functions_analysis.trf_grand_average(feature_evokeds=feature_evokeds, trf_params=trf_params, trial_params=trial_params, meg_params=meg_params,
                                         display_figs=display_figs, save_fig=save_fig, fig_path=trf_fig_path)

grand_avg = {}
for mss in [1, 2, 4]:
    if trf_params['fit_power']:
        grand_avg[mss] = mne.grand_average(trial_evokeds[mss])

        # Plot
        fig = grand_avg[mss].plot(window_title=f'MSS: {mss}')
        fig.suptitle(f'MSS: {mss}')
        fig.tight_layout()
        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=f'GA_mss{mss}')

    else:
        # Compute grand average
        grand_avg[mss] = mne.grand_average(trial_evokeds[mss])

        # Define time-frequency bands to plot in plot_joint
        recon_tmin, recon_tmax, plot_xlim = functions_general.get_time_lims(epoch_id=trf_params['epoch_id'], mss=mss, plot_edge=plot_edge)
        timefreqs_joint, timefreqs_tfr, vlines_times = functions_general.get_plots_timefreqs(epoch_id=trf_params['epoch_id'], mss=mss, cross2_dur=cross2_dur,
                                                                                             mss_duration=mss_duration, topo_bands=None, plot_xlim=plot_xlim,
                                                                                             plot_min=True, plot_max=True)

        _, plot_baseline = functions_general.get_baseline_duration(epoch_id=trf_params['epoch_id'], mss=mss, tmin=recon_tmin, tmax=recon_tmax,
                                                                   cross1_dur=cross1_dur, mss_duration=mss_duration, cross2_dur=cross2_dur, plot_edge=plot_edge)

        # Plot Power time-frequency in time scalde axes
        fname = f"GA_Power_mss{mss}_tf_{meg_params['chs_id']}_{bline_mode}_{l_freq}_{h_freq}"
        fig, ax_tf = plot_general.tfr_times(tfr=grand_avg[mss], chs_id=meg_params['chs_id'], timefreqs_tfr=timefreqs_tfr, baseline=plot_baseline, bline_mode=ga_plot_bline_mode,
                                            plot_xlim=plot_xlim, vlines_times=vlines_times, vmin=None, vmax=None, topo_vmin=None, topo_vmax=None,
                                            display_figs=display_figs, save_fig=False, fig_path=fig_path, fname=fname, fontsize=18, ticksize=18)

        # Crop to plot times and selected channels
        broadband_power = grand_avg[mss].copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1]).pick(picks)

        # Select time index of baseline period end
        idx0, value0 = functions_general.find_nearest(broadband_power.times, values=grand_avg[mss].times[0] + cross1_dur)

        # Apply mean baseline
        baseline_power = broadband_power.data.mean(0).mean(0) - np.mean(broadband_power.data.mean(0).mean(0)[:idx0])

        # Define axes on right side of TF plot
        ax_tf_r = ax_tf.twinx()

        # Plot
        ax_tf_r.plot(broadband_power.times, baseline_power, color=f'k', linewidth=3)
        ax_tf_r.set_ylabel('Average Power (dB)')

        fig.tight_layout()

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)