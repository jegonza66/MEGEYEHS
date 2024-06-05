import functions_analysis
import functions_general
import load
import save
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
import mne

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

# Select features (events)
input_features = ['sac_ms', 'sac_vs']

# Select channels and frequency band
chs_id = 'parietal_occipital'  # region_hemisphere (frontal_L)
band_id = 'HGamma'

# Select trials
corr_ans = None
tgt_pres = None
mss = None
evt_dur = None

# TRF parameters
standarize = True  # Standarize MEG signal
fit_power = False  # Compute MEG power with apply_hilbert
alpha = None  # Regularization parameter
tmin = -0.3
tmax = 0.6
baseline = (tmin, -0.05)


# Window durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
if 'vs' in input_features:
    trial_dur = vs_dur[mss]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_dur = None

# ICA / RAW
use_ica_data = True

# Specific run path for saving data and plots
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Region channels to run trf separately
all_chs_regions = ['frontal', 'temporal', 'central', 'parietal', 'occipital']

# Define Grand average variables
ga = {}
for feature in input_features:
    ga[feature] = []

# Figure path
fig_path = paths().plots_path() + f'TRF_{data_type}/Band_{band_id}/{input_features}_mss{mss}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}' \
                                  f'_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/{chs_id}/'

# Change path to include envelope power
if fit_power:
    fig_path = fig_path.replace(f'TRF_{data_type}', f'TRF_{data_type}_ENV')

# Save path
save_path = fig_path.replace(paths().plots_path(), paths().save_path())

# Iterate over subjects
for subject_code in exp_info.subjects_ids:
    trf_path = save_path
    trf_fname = f'TRF_{subject_code}.pkl'

    try:
        # Load TRF
        rf = load.var(trf_path + trf_fname)
        # Load MEG sub
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            # Load MEG
            if band_id:
                meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=save_data)
            else:
                meg_data = load.ica_data(subject=subject)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
            # Load MEG
            if band_id:
                meg_data = load.filtered_data(subject=subject, band_id=band_id, use_ica_data=False, save_data=save_data)
            else:
                meg_data = subject.load_preproc_meg_data()
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
        meg_sub = meg_data.copy().pick(picks)
        print('Loaded Receptive Field')

    except:

        print(f'Computing TRF for {input_features}')

        # Define save path and file name for loading and saving epoched, evoked, and GA data
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            # Load MEG
            if band_id:
                meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=save_data)
            else:
                meg_data = load.ica_data(subject=subject)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
            # Load MEG
            if band_id:
                meg_data = load.filtered_data(subject=subject, band_id=band_id, use_ica_data=False, save_data=save_data)
            else:
                meg_data = subject.load_preproc_meg_data()

        # Get condition trials
        cond_trials, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=mss, trial_dur=trial_dur, corr_ans=corr_ans, tgt_pres=tgt_pres)

        # Bad annotations filepath
        subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
        fname_bad_annot = f'bad_annot_array.pkl'

        try:
            bad_annotations_array = load.var(subj_path + fname_bad_annot)
            print(f'Loaded bad annotations array')
        except:
            print(f'Computing bad annotations array...')
            bad_annotations_array = functions_analysis.get_bad_annot_array(meg_data=meg_data, subj_path=subj_path, fname=fname_bad_annot)

        # Iterate over input features
        input_arrays = {}
        for feature in input_features:

            subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
            fname_var = f'{feature}_mss{mss}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}_evtdur{evt_dur}_array.pkl'

            try:
                input_arrays[feature] = load.var(file_path=subj_path + fname_var)
                print(f'Loaded input array for {feature}')

            except:
                print(f'Computing input array for {feature}...')
                # Exception for subsampled distractor fixations
                if 'subsampled' in feature:
                    # Subsampled epochs path
                    epochs_save_id = f'{feature}_mss{mss}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}_evtdur{evt_dur}'
                    epochs_save_path = paths().save_path() + f'Epochs_{data_type}' + f'/Band_{band_id}/{epochs_save_id}_{tmin}_{tmax}_bline{baseline}/'
                    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

                    # Load epoched data
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

                    # Get epochs id from metadata
                    epoch_keys = epochs.metadata['id'].to_list()

                else:
                    epoch_keys = None

                input_arrays = functions_analysis.make_mtrf_input(input_arrays=input_arrays, var_name=feature,
                                                                  subject=subject, meg_data=meg_data, evt_dur=evt_dur,
                                                                  cond_trials=cond_trials, epoch_keys=epoch_keys,
                                                                  bad_annotations_array=bad_annotations_array,
                                                                  subj_path=subj_path, fname=fname_var)

        # Concatenate input arrays as one
        model_input = np.array([input_arrays[key] for key in input_arrays.keys()]).T

        # All regions or selected (multiple) regions
        if chs_id == 'mag' or '_' in chs_id:
            # rf as a dictionary containing the rf of each region
            rf = {}
            # iterate over regions
            for chs_subset in all_chs_regions:
                # Use only regions in channels id, or all in case of chs_id == 'mag'
                if chs_subset in chs_id or chs_id == 'mag':
                    print(f'Fitting mTRF for region {chs_subset}')
                    rf[chs_subset] = functions_analysis.fit_mtrf(meg_data=meg_data, tmin=tmin, tmax=tmax, alpha=alpha, fit_power=fit_power,
                                                                 model_input=model_input, chs_id=chs_subset, standarize=standarize,
                                                                 n_jobs=4)
        # One region
        else:
            rf = functions_analysis.fit_mtrf(meg_data=meg_data, tmin=tmin, tmax=tmax, alpha=alpha, fit_power=fit_power,
                                             model_input=model_input, chs_id=chs_id, standarize=standarize, n_jobs=4)
        # Save TRF
        if save_data:
            save.var(var=rf, path=trf_path, fname=trf_fname)

    # Get model coeficients as separate responses to target and items
    trf = {}
    evoked = {}
    evoked_list = {}
    for i, feature in enumerate(input_features):

        # All or multiple regions
        if chs_id == 'mag' or '_' in chs_id:

            # Define evoked from TRF list to concatenate all
            evoked_list[feature] = []

            # iterate over regions
            for chs_idx, chs_subset in enumerate(rf.keys()):
                # Get channels subset info
                picks = functions_general.pick_chs(chs_id=chs_subset, info=meg_data.info)
                meg_sub = meg_data.copy().pick(picks)

                # Get TRF coeficients from chs subset
                trf[feature] = rf[chs_subset].coef_[:, i, :]

                if chs_idx == 0:
                    # Define evoked object from arrays of TRF
                    evoked[feature] = mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=tmin, baseline=baseline)
                else:
                    # Append evoked object from arrays of TRF to list, to concatenate all
                    evoked_list[feature].append(mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=tmin, baseline=baseline))
            # Concatenate evoked from al regions
            evoked[feature] = evoked[feature].add_channels(evoked_list[feature])
        else:
            trf[feature] = rf.coef_[:, i, :]
            # Define evoked objects from arrays of TRF
            evoked[feature] = mne.EvokedArray(data=trf[feature], info=meg_sub.info, tmin=tmin, baseline=baseline)

        # Append for Grand average
        ga[feature].append(evoked[feature])
        # Plot
        fig = evoked[feature].plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin, tmax), titles=feature)
        fig.suptitle(f'{feature}')

        if save_fig:
            # Save
            fig_path_subj = fig_path + f'{subject.subject_id}/'
            fname = f'{feature}_{chs_id}'
            save.fig(fig=fig, fname=fname, path=fig_path_subj)


grand_avg = {}
for feature in input_features:
    # Compute grand average
    grand_avg[feature] = mne.grand_average(ga[feature], interpolate_bads=True)
    plot_times_idx = np.where((grand_avg[feature].times > tmin) & (grand_avg[feature].times < tmax))[0]
    data = grand_avg[feature].get_data()[:, plot_times_idx]

    # Plot
    fig = grand_avg[feature].plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin, tmax), titles=feature)

    if save_fig:
        # Save
        fname = f'{feature}_GA_{chs_id}'
        save.fig(fig=fig, fname=fname, path=fig_path)