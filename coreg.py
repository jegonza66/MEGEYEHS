import os
import functions_analysis
import functions_general
import matplotlib.pyplot as plt
plt.figure()
plt.close('all')
import mne
from paths import paths
import load
import setup

# Load experiment info
exp_info = setup.exp_info()

# Define Subjects_dir as Freesurfer output folder
mri_path = paths().mri_path()
subjects_dir = os.path.join(mri_path, 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Load one subject
subject = load.preproc_subject(exp_info=exp_info, subject_code=0)

# PATH TO MRI <-> HEAD TRANSFORMATION (Saved from coreg)
trans_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-trans.fif'.format(subject.subject_id))
fids_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-fiducials.fif'.format(subject.subject_id))

# Load clean data
# meg_data = load.ica_data(subject=subject)
meg_data = subject.load_preproc_meg()

# try:
#     # Visualize MEG/MRI alignment
#     surfaces = dict(brain=0.6, outer_skull=0.5, head=0.4)
#     fig = mne.viz.plot_alignment(meg_data.info, trans=trans_path, subject=subject.subject_id,
#                                  subjects_dir=subjects_dir, surfaces=surfaces,
#                                  show_axes=True, dig=True, eeg=[], meg='sensors',
#                                  coord_frame='meg', mri_fiducials=fids_path)
#     mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))
# except:
#     # Align and save fiducials and transformation files to FreeSurfer/subject/bem folder
#     mne.gui.coregistration(subject=subject.subject_id, subjects_dir=subjects_dir)


# Exclude bad channels
# bads = subject.bad_channels
# meg_data.info['bads'].extend(bads)

# Get events from annotations
# Select epochs
epoch_id = 'fix_vs'

# Duration
dur = None  # seconds
# Screen
screen = functions_general.get_screen(epoch_id=epoch_id)
# MSS
mss = functions_general.get_mss(epoch_id=epoch_id)
# Item
tgt = functions_general.get_item(epoch_id=epoch_id)
# Saccades direction
dir = functions_general.get_dir(epoch_id=epoch_id)
# Get time windows from epoch_id name
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)


metadata, events, events_id, metadata_sup = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                             screen=screen, mss=mss, dur=dur, tgt=tgt,
                                                                             dir=dir, meg_data=meg_data)

# Reject based on channel amplitude
reject = dict(mag=subject.config.general.reject_amp)
reject = dict(mag=2.5e-12)

# Epoch data
epochs = mne.Epochs(meg_data, events, event_id=events_id, reject=reject, tmin=tmin, tmax=tmax,
                    event_repeated='drop', metadata=metadata, preload=True)

# Define evoked from epochs
evoked = epochs.average()
# GET MEG CHS ONLY
evoked.pick('meg')

# FILTER
evoked.filter(l_freq=0.5, h_freq=80., fir_design='firwin')

# LOAD BACKGROUND NOISE
noise = setup.noise(exp_info=exp_info, id='BACK_NOISE')

raw_noise = noise.load_preproc_data()

raw_noise.pick('meg')
# COMPUTE COVARIANCE TO WITHDRAW FROM MEG DATA
cov = mne.compute_raw_covariance(raw_noise, reject=reject)
# cov.plot(raw_noise.info)

# --------- Source reconstruction ---------#
# SETUP SOURCE SPACE
src = mne.setup_source_space(subject=subject.subject_id, spacing='oct4', subjects_dir=subjects_dir)  # change oct4 to oct 6 for real analysis (better quality)

# SET UP BEM MODEL
model = mne.make_bem_model(subject=subject.subject_id, ico=4, conductivity=[0.3], subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# Plot bem
# mne.viz.plot_bem(subject=subject.subject_id,
#                      subjects_dir=subjects_dir,
#                      brain_surfaces='white',
#                      src=src,
#                      orientation='coronal')


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
                          initial_time=0., time_unit='s')



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