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
plot_individuals = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
trial_params = {'corrans': None,
                'tgtpres': None,
                'mss': None,
                'evtdur': None,
                }

meg_params = {'chs_id': 'parietal_occipital',
              'band_id': 'HGamma',
              'data_type': 'ICA'
              }

# TRF parameters
trf_params = {'input_features': ['sac_cross1', 'fix_cross1', 'sac_ms', 'it_fix_ms+tgt_fix_ms', 'sac_cross2', 'fix_cross2', 'sac_vs', 'it_fix_vs+tgt_fix_vs'],   # Select features (events)
              'standarize': True,
              'fit_power': False,
              'alpha': None,
              'tmin': -0.3,
              'tmax': 0.6,
              }
trf_params['baseline'] = (trf_params['tmin'], -0.05)

# Window durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
if 'vs' in trf_params['input_features']:
    trial_params['trialdur'] = vs_dur[trial_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_params['trialdur'] = None

# Figure path
fig_path = paths().plots_path() + (f"TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{trf_params['input_features']}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_"
                                   f"tgtpres{trial_params['tgtpres']}_trialdur{trial_params['trialdur']}_evtdur{trial_params['evtdur']}_{trf_params['tmin']}_{trf_params['tmax']}_"
                                   f"bline{trf_params['baseline']}_alpha{trf_params['alpha']}_std{trf_params['standarize']}/{meg_params['chs_id']}/")

# Change path to include envelope power
if trf_params['fit_power']:
    fig_path = fig_path.replace(f"TRF_{meg_params['data_type']}", f"TRF_{meg_params['data_type']}_ENV")

# Save path
save_path = fig_path.replace(paths().plots_path(), paths().save_path())

# Define Grand average variables
feature_evokeds = {}
for feature in trf_params['input_features']:
    feature_evokeds[feature] = []

# Iterate over subjects
for subject_code in exp_info.subjects_ids:
    trf_path = save_path
    trf_fname = f'TRF_{subject_code}.pkl'

    if meg_params['data_type'] == 'ICA':
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        # Load MEG
        if meg_params['band_id']:
            meg_data = load.filtered_data(subject=subject, band_id=meg_params['band_id'], save_data=save_data)
        else:
            meg_data = load.ica_data(subject=subject)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        # Load MEG
        if meg_params['band_id']:
            meg_data = load.filtered_data(subject=subject, band_id=meg_params['band_id'], use_ica_data=False, save_data=save_data)
        else:
            meg_data = subject.load_preproc_meg_data()

    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
    meg_sub = meg_data.copy().pick(picks)

    try:
        # Load TRF
        rf = load.var(trf_path + trf_fname)
        print('Loaded Receptive Field')

    except:
        # Compute TRF for defined features
        rf = functions_analysis.compute_trf(subject=subject, meg_data=meg_data, trial_params=trial_params, trf_params=trf_params,
                                            save_data=save_data, trf_path=trf_path, trf_fname=trf_fname)

    # Get model coeficients as separate responses to each feature
    subj_evoked, feature_evokeds = functions_analysis.make_trf_evoked(subject=subject, rf=rf, meg_data=meg_data, evokeds=feature_evokeds,
                                                                      trf_params=trf_params, trial_params=trial_params, meg_params=meg_params,
                                                                      plot_individuals=plot_individuals, save_fig=save_fig, fig_path=fig_path)

fname = f"{feature}_GA_{meg_params['chs_id']}"
grand_avg = functions_analysis.trf_grand_average(feature_evokeds=feature_evokeds, trf_params=trf_params, trial_params=trial_params, meg_params=meg_params,
                                                 display_figs=display_figs, save_fig=save_fig, fig_path=fig_path)
