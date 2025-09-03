## Surface Time-Frequency
import os
import shutil
import functions_analysis
import functions_general
import mne
import mne.beamformer as beamformer
import save
from paths import paths
import load
import setup
import numpy as np
import plot_general
import matplotlib.pyplot as plt
import itertools
import scipy
import pandas as pd


# Load experiment info
exp_info = setup.exp_info()

#--------- Define Parameters ---------#

# Save
save_fig = True
save_data = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#----- Parameters -----#
# Trial selection
trial_params = {'corrans': None,
                'tgtpres': None,
                'mss': None,
                'evtdur': None}

meg_params = {'regions_id': 'all',
              'band_id': None,
              'filter_sensors': None,
              'filter_method': 'iir',
              'data_type': 'ICA'
              }

# TRF parameters
trf_params = {'input_features': {'it_fix_vs+tgt_fix_vs': ['fix_target', 'n_fix'], 'sac': ['amp']},   # Select features (events)
              'standarize': False,
              'fit_power': True,
              'alpha': None,
              'tmin': -0.2,
              'tmax': 0.5,
              'baseline': (-0.3, -0.05)
              }


# Get TF frequency limits
if meg_params['band_id'] is None:
    l_freq, h_freq = (1, 40)
else:
    l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'surface'
labels_mode = 'pca_flip'
ico = 4
spacing = 5.  # Only for volume source estimation
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
parcelation='aparc.a2009s'


# Plot
plot_individuals = False
plot_ga = False
fontsize = 22
params = {'font.size': fontsize}
plt.rcParams.update(params)

#--------- Setup ---------#

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
if 'vs' in trf_params['input_features']:
    trial_params['trialdur'] = vs_dur[trial_params['mss']]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_params['trialdur'] = None

# Adapt to hfreq model
if meg_params['band_id'] == 'HGamma' or ((isinstance(meg_params['band_id'], list) or isinstance(meg_params['band_id'], tuple)) and meg_params['band_id'][0] > 40):
    model_name = 'hfreq-' + model_name

# --------- Freesurfer Path ---------#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Surface labels id by region
aparc_region_labels = {'occipital': ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine'],
                       'parietal': ['postcentral', 'superiorparietal', 'supramarginal', 'inferiorparietal', 'precuneus'],
                       'temporal': ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'transversetemporal', 'fusiform', 'entorhinal', 'parahippocampal', 'temporalpole'],
                       'frontal': ['precentral', 'caudalmiddlefrontal', 'superiorfrontal', 'rostralmiddlefrontal', 'lateralorbitofrontal', 'parstriangularis', 'parsorbitalis', 'parsopercularis', 'medialorbitofrontal', 'paracentral', 'frontalpole'],
                       'insula': ['insula'],
                       'cingulate': ['isthmuscingulate', 'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate']}

aparc_region_labels['all'] = [value for key in aparc_region_labels.keys() for value in aparc_region_labels[key]]

# Extract region labels data
region_labels = [element for region in meg_params['regions_id'].split('_') for element in aparc_region_labels[region]]

# Get parcelation labels
fsaverage_labels = functions_analysis.get_labels(subject_code='fsaverage', parcelation=parcelation, subjects_dir=subjects_dir, surf_vol=surf_vol)
if parcelation == 'aparc':
    fsaverage_labels = [label for label in fsaverage_labels for label_id in region_labels if label.name.startswith(label_id + '-')]


# Paths
# Figure path
fig_path = paths().plots_path() + (f"Source_TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{trf_params['input_features']}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_"
                                   f"tgtpres{trial_params['tgtpres']}_trialdur{trial_params['trialdur']}_evtdur{trial_params['evtdur']}_{trf_params['tmin']}_{trf_params['tmax']}_"
                                   f"bline{trf_params['baseline']}_alpha{trf_params['alpha']}_std{trf_params['standarize']}/{meg_params['regions_id']}/")

# Change path to include envelope power
if trf_params['fit_power']:
    fig_path = fig_path.replace(f"TRF_{meg_params['data_type']}", f"TRF_{meg_params['data_type']}_ENV")

# Save path
save_path = fig_path.replace(paths().plots_path(), paths().save_path())
# Source paths
source_model_path = f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}"
labels_model_path = source_model_path + f"_{parcelation}_{labels_mode}/"

# Iterate over participants
for subject_code in exp_info.subjects_ids:

    # Load subject
    if meg_params['data_type'] == 'ICA':
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    elif meg_params['data_type'] == 'RAW':
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    print(f'Computing labels time series for participant {subject_code}')
    # --------- Coord systems alignment ---------#
    if force_fsaverage:
        subject_code = 'fsaverage'
        dig = False
    else:
        # Check if subject has MRI data
        try:
            fs_subj_path = os.path.join(subjects_dir, subject.subject_id)
            os.listdir(fs_subj_path)
            dig = True
        except:
            subject_code = 'fsaverage'
            dig = False

    # Load forward model
    fname_fwd = paths().fwd_path(subject=subject, subject_code=subject_code, ico=ico, spacing=spacing, surf_vol=surf_vol)
    fwd = mne.read_forward_solution(fname_fwd)
    src = fwd['src']

    # Load filter
    fname_filter = paths().filter_path(subject=subject, subject_code=subject_code, ico=ico, spacing=spacing, surf_vol=surf_vol, pick_ori=pick_ori,
                                       model_name=model_name)
    filters = mne.beamformer.read_beamformer(fname_filter)

    # Load MEG
    meg_data = load.meg(subject=subject, meg_params=meg_params, save_data=save_data)

    # --------- Source estimation ---------#
    # Get source estimates for candidate voxels only using script from OPM

    stc_raw = beamformer.apply_lcmv_raw(raw=meg_data, filters=filters)

    # Extract region labels data
    region_labels = [element for region in meg_params['regions_id'].split('_') for element in aparc_region_labels[region]]

    # Get labels for FreeSurfer cortical parcellation or segmentation
    used_labels = functions_analysis.get_labels(subject_code=subject_code, parcelation=parcelation, subjects_dir=subjects_dir, surf_vol=surf_vol)

    if surf_vol == 'volume':
        labels_path = subjects_dir + f'/{subject_code}/mri/{parcelation}+aseg.mgz'
        if parcelation == 'aparc':
            used_labels = [label for label in used_labels for label_id in region_labels if label_id in label]
        used_labels = [labels_path, used_labels]

    else:
        if parcelation == 'aparc':
            # Filter labels by region
            used_labels = [label for label in used_labels for label_id in region_labels if label.name.startswith(label_id + '-')]

    # Average the source estimates within each label using sign-flips to reduce signal cancellations
    label_ts = mne.extract_label_time_course(stcs=stc_raw, labels=used_labels, src=src, mode=labels_mode, return_generator=False)

    # Make raw object from stc data using regions as ch names
    regions_raw = mne.io.RawArray(label_ts.T, mne.create_info(ch_names=[f"{label.name}_{label.hemi}" for label in fsaverage_labels], sfreq=meg_data.info['sfreq'], ch_types='misc'))

    try:
        # Load TRF
        rf = load.var(trf_path + trf_fname)
        print('Loaded TRF')

    except:
        print('Computing TRF')
        # Compute TRF for defined features
        rf = functions_analysis.compute_trf(subject=subject, meg_data=regions_raw, trial_params=trial_params, trf_params=trf_params, meg_params=meg_params,
                                            save_data=save_data, trf_path=trf_path, trf_fname=trf_fname)

    # Get model coeficients as separate responses to each feature
    subj_evoked, feature_evokeds = functions_analysis.make_trf_evoked(subject=subject, rf=rf, meg_data=regions_raw, evokeds=feature_evokeds,
                                                                      trf_params=trf_params, meg_params=meg_params,
                                                                      plot_individuals=plot_individuals, save_fig=save_fig, fig_path=fig_path)


# Grand Average: Average evoked stcs from this epoch_id and param value
fname = f"{feature}_GA_{meg_params['chs_id']}"
grand_avg = functions_analysis.trf_grand_average(feature_evokeds=feature_evokeds, trf_params=trf_params, trial_params=trial_params, meg_params=meg_params,
                                                 display_figs=display_figs, save_fig=save_fig, fig_path=fig_path)

