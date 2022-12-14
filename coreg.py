import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from paths import paths
import load
import setup

# Load experiment info
exp_info = setup.exp_info()

# Load run configuration
config = load.config(path=paths().config_path(), fname='config.pkl')

# Define Subjects_dir as Freesurfer output folder
mri_path = paths().mri_path()
subjects_dir = os.path.join(mri_path, 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Load one subject
subject = load.preproc_subject(exp_info=exp_info, subject_code=0)

# PATH TO MRI <-> HEAD TRANSFORMATION (Saved from coreg)
trans_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-trans.fif'.format(subject.subject_id))
fids_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-fiducials.fif'.format(subject.subject_id))

## ALIGNMENT
mne.gui.coregistration(subject=subject.subject_id, subjects_dir=subjects_dir)

## VISUALIZE ALIGNMENT FOR COREGISTRATION
# https://mne.tools/stable/auto_tutorials/inverse/35_dipole_orientations.html#sphx-glr-auto-tutorials-inverse-35-dipole-orientations-py
raw = subject.load_raw_meg_data()
# fid_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-fiducials.fif'.format(subject.subject_id))

surfaces = dict(brain=0.6, outer_skull=0.5, head=0.4)
fig = mne.viz.plot_alignment(raw.info, trans=trans_path, subject=subject.subject_id,
                             subjects_dir=subjects_dir, surfaces=surfaces,
                             show_axes=True, dig=True, eeg=[], meg='sensors',
                             coord_frame='meg', mri_fiducials=fids_path)
mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))

## SOURCE RECONSTRUCTION
# https://mne.tools/stable/auto_tutorials/io/60_ctf_bst_auditory.html#sphx-glr-auto-tutorials-io-60-ctf-bst-auditory-py
# https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py

# Load Preprocesed
raw = subject.load_preproc_meg()

# PICK MEG AND STIM CHS
raw.pick(['meg', 'misc'])
# Exclude bad channels
bads = subject.bad_channels
raw.info['bads'].extend(bads)

# Get events from annotations
all_events, all_event_id = mne.events_from_annotations(raw, verbose=False)
# Reject based on channel amplitude
reject = dict(mag=subject.config.general.reject_amp)
# Select epochs
epoch_ids = ['fix_vs']
epoch_keys = [key for epoch_id in epoch_ids for key in all_event_id.keys() if epoch_id in key]
# Get events and ids matchig selection
metadata, events, event_id = mne.epochs.make_metadata(events=all_events, event_id=all_event_id,
                                                      row_events=epoch_keys, tmin=0, tmax=0,
                                                      sfreq=raw.info['sfreq'])
# Epoch data
epochs = mne.Epochs(raw, events, event_id=event_id, reject=reject, event_repeated='drop', metadata=metadata, preload=True)

# epochs_standard = mne.concatenate_epochs([epochs[key] for key in epoch_keys])
# AVERAGE EPOCHS TO GET EVOKED
evoked = epochs.average()
# GET MEG CHS ONLY
evoked.pick('meg')
# FILTER
evoked.filter(l_freq=0.5, h_freq=80., fir_design='firwin')

# LOAD BACKGROUND NOISE
noise = setup.noise(exp_info=exp_info, id='BACK_NOISE')
raw_noise = noise.load_raw_meg_data()
raw_noise.pick('meg')
# COMPUTE COVARIANCE TO WITHDRAW FROM MEG DATA
cov = mne.compute_raw_covariance(raw_noise, reject=reject)
# cov.plot(raw_noise.info)

# SETUP SOURCE SPACE
src = mne.setup_source_space(subject=subject.subject_id, spacing='oct4', subjects_dir=subjects_dir)  # change oct4 to oct 6 for real analysis

# SET UP BEM MODEL
model = mne.make_bem_model(subject=subject.subject_id, ico=4, conductivity=[0.3], subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
# COMPUTE SOURCE RECONSTRUCTION
fwd = mne.make_forward_solution(evoked.info, trans=trans_path, src=src, bem=bem)

# MAKE INVERSE OPERATOR FROM EVOKED
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov)
# PARAMETROS STANDAR A CHEQUEAR
snr = 3.0
lambda2 = 1.0 / snr ** 2

# COMPUTE INVERSE SOLUTION TO GET SOURCES TIME COURSES
stc_standard = mne.minimum_norm.apply_inverse(evoked, inv, lambda2, 'dSPM')
brain = stc_standard.plot(subjects_dir=subjects_dir, subject=subject.subject_id,
                          surface='inflated', time_viewer=True, hemi='both',
                          initial_time=0.1, time_unit='s')



## PLOT SOURCE SPACE
# plot_bem_kwargs = dict(
#     subject=subject.subject_id, subjects_dir=subjects_dir,
#     brain_surfaces='white', orientation='coronal',
#     slices=[50, 100, 150, 200])
# mne.viz.plot_bem(src=src, **plot_bem_kwargs)
# # Volume source
# sphere = (0.0, 0.0, 0.04, 0.09)
# vol_src = mne.setup_volume_source_space(
#     subject=subject.subject_id, subjects_dir=subjects_dir, sphere=sphere, sphere_units='m',
#     add_interpolator=False)  # just for speed!
# print(vol_src)
# mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)
# # 3D
# fig = mne.viz.plot_alignment(subject=subject.subject_id, subjects_dir=subjects_dir,
#                              surfaces='white', coord_frame='mri',
#                              src=src)
# mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
#                     distance=0.30, focalpoint=(-0.03, -0.01, 0.03))