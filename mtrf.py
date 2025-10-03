import os.path
import functions_analysis
import functions_general
import plot_general
import load
import setup
from paths import paths
import matplotlib.pyplot as plt
import scipy
import numpy as np


#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
use_saved_data = True
save_data = True
save_fig = True
display_figs = True
plot_individuals = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
trial_params = {'corrans': None,
                'tgtpres': True,
                'mss': None,
                'evtdur': None,
                }

meg_params = {'chs_id': 'mag',
              'band_id': (0.1, 40),
              'data_type': 'ICA',
              'downsample': 300
              }

# TRF parameters
trf_params = {'input_features': {'it_fix_vs+tgt_fix_vs': ['correct'],
                                 'tgt_fix_vs': ['correct'], # 'n_fix', 'duration', {'fix_vs': ['item', 'fix_target']} 'pupil'? 'distance'?, 'mss'?,
                                 'sac_vs': None, # duration , avg_vel
                                 'blue': None,
                                 'red': None  # Select features (events)
                                 },
              'add_features': ['tgt_fix_vs', 'tgt_fix_vs-correct'],
              'standarize': True,
              'fit_power': False,
              'alpha': 1000,
              'tmin': -0.2,
              'tmax': 0.5,
              }
trf_params['baseline'] = (trf_params['tmin'], -0.05)

# Permutations cluster test parameters
run_permutations = True
n_permutations = 1024
degrees_of_freedom = len(exp_info.subjects_ids) - 1
desired_tval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# t_thresh = dict(start=0, step=0.2)
pval_threshold = 0.05

# Window durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
if 'vs' in trf_params['input_features']:
    trial_params['trialdur'] = vs_dur[trial_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_params['trialdur'] = None

# Figure path
fig_path = paths().plots_path() + (f"TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{trf_params['input_features']}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_"
                                   f"tgtpres{trial_params['tgtpres']}_trialdur{trial_params['trialdur']}_evtdur{trial_params['evtdur']}_{trf_params['tmin']}_{trf_params['tmax']}_"
                                   f"bline{trf_params['baseline']}_alpha{trf_params['alpha']}_std{trf_params['standarize']}/{meg_params['chs_id']}/").replace(":", "")

# Change path to include downsampled data
if meg_params['downsample'] is not None:
    fig_path = fig_path.replace(f"Band_{meg_params['band_id']}", f"Band_{meg_params['band_id']}_downsample_{meg_params['downsample']}")

# Change path to include envelope power
if trf_params['fit_power']:
    fig_path = fig_path.replace(f"TRF_{meg_params['data_type']}", f"TRF_{meg_params['data_type']}_ENV")

# if trf_params['add_features']:
#     fig_path = fig_path.replace(f"{trf_params['input_features']}".replace(":", ""), f"{trf_params['input_features']}_add{trf_params['add_features']}").replace(":", "")

# Save path
save_path = fig_path.replace(paths().plots_path(), paths().save_path())

# Define Grand average variables
feature_evokeds = {}
if isinstance(trf_params['input_features'], dict):
    elements = trf_params['input_features'].keys()
elif isinstance(trf_params['input_features'], list):
    elements = trf_params['input_features']
for feature in elements:
    feature_evokeds[feature] = []
    if isinstance(trf_params['input_features'], dict):
        try:
            for value in trf_params['input_features'][feature]:
                feature_value = f'{feature}-{value}'
                feature_evokeds[feature_value] = []
        except:
            pass

# Iterate over subjects
for sub_idx, subject_code in enumerate(exp_info.subjects_ids):
    trf_path = save_path
    trf_fname = f'TRF_{subject_code}.pkl'

    if meg_params['data_type'] == 'ICA':
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    # Load MEG data
    meg_data = load.meg(subject=subject, meg_params=meg_params)

    picks = functions_general.pick_chs(chs_id=meg_params['chs_id'], info=meg_data.info)
    meg_sub = meg_data.copy().pick(picks)

    if os.path.exists(trf_path + trf_fname) and use_saved_data:
        # Load TRF
        rf = load.var(trf_path + trf_fname)
        print('Loaded Receptive Field')

    else:
        # Compute TRF for defined features
        rf = functions_analysis.compute_trf(subject=subject, meg_data=meg_data, trial_params=trial_params, trf_params=trf_params, meg_params=meg_params,
                                            use_saved_data=use_saved_data, save_data=save_data, trf_path=trf_path, trf_fname=trf_fname)

    # Get model coeficients as separate responses to each feature
    feature_evokeds = functions_analysis.parse_trf_to_evoked(subject=subject, rf=rf, meg_data=meg_data, feature_evokeds=feature_evokeds,
                                                             trf_params=trf_params, meg_params=meg_params, sub_idx=sub_idx,
                                                             plot_individuals=plot_individuals, save_fig=save_fig, fig_path=fig_path)

# Grand average
grand_avg = functions_analysis.trf_grand_average(feature_evokeds=feature_evokeds, trf_params=trf_params, meg_params=meg_params,
                                                 display_figs=display_figs, save_fig=save_fig, fig_path=fig_path)

# Run permutations
if run_permutations:
    clusters_mask = {}
    clusters_pvalues = {}
    channel_sets = []

    for ev_list in feature_evokeds.values():
        for ev in ev_list:
            channel_sets.append(set(ev.ch_names))
    # Intersection
    common = set.intersection(*channel_sets)
    ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg[list(grand_avg.keys())[0]].info, ch_type='mag')

    for feature in grand_avg.keys():
        print('Running permutations test for feature:', feature)
        data = np.array([ev.copy().pick(list(common)).data.T for ev in feature_evokeds[feature]])
        clusters_mask_transp, clusters_pvalues[feature] = functions_analysis.run_permutations_test(data=data, pval_threshold=pval_threshold, t_thresh=t_thresh, adj_matrix=ch_adjacency_sparse, n_permutations=n_permutations)
        clusters_mask[feature] = clusters_mask_transp.T
else:
    clusters_mask = None

joint_ylims = [dict(mag=[-6e13, 6e13]), dict(mag=[-6e13, 6e13]), dict(mag=[-6e13, 6e13]), dict(mag=[-6e13, 6e13]), dict(mag=[-1.5e14, 1.5e14]), dict(mag=[-1.5e14, 1.5e14]), dict(mag=[-1.5e14, 1.5e14])]

# Plot features figure
if isinstance(t_thresh, dict):
    fname = f'GA_features_TFCE_{pval_threshold}_{n_permutations}'
else:
    fname = f'GA_features_{round(t_thresh, 2)}_{pval_threshold}_{n_permutations}'
fig = plot_general.plot_trf_features(grand_avg=grand_avg, clusters_mask=clusters_mask, joint_ylims=joint_ylims, save_fig=save_fig, fig_path=fig_path, fname=fname)