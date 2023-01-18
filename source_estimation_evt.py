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


# --------- Setup ---------#
# Load experiment info
exp_info = setup.exp_info()

# Define Subjects_dir as Freesurfer output folder
mri_path = paths().mri_path()
subjects_dir = os.path.join(mri_path, 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# --------- Load data ---------#
# Load subject and meg clean data
subject = load.preproc_subject(exp_info=exp_info, subject_code=0)
meg_data = load.ica_data(subject=subject)
# meg_data = subject.load_preproc_meg()

# Exclude bad channels
bads = subject.bad_channels
meg_data.info['bads'].extend(bads)

# --------- Coord systems alignment ---------#
# Path to MRI <-> HEAD Transformation (Saved from coreg)
trans_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-trans2.fif'.format(subject.subject_id))
fids_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-fiducials.fif'.format(subject.subject_id))
dig_info_path = paths().opt_path() + subject.subject_id + '/info_raw.fif'

# Load raw meg data with dig info
info_raw = mne.io.read_raw_fif(dig_info_path)

# Visualize MEG/MRI alignment
surfaces = dict(brain=0.7, outer_skull=0.5, head=0.4)
fig = mne.viz.plot_alignment(info_raw.info, trans=trans_path, subject=subject.subject_id,
                             subjects_dir=subjects_dir, surfaces=surfaces,
                             show_axes=True, dig=True, eeg=[], meg='sensors',
                             coord_frame='meg', mri_fiducials=fids_path)
mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))

# --------- Epoch data ---------#
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

# Get events based on conditions
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
# evoked.plot(spatial_colors=True)

# Pick meg channels for source modeling
evoked.pick('meg')

# Filter
# evoked.filter(l_freq=0.5, h_freq=80., fir_design='firwin')

# --------- Inverse operation computation ---------#
# Volume vs Surface estimation
surf_vol = 'surface'
# Setup
sources_path = paths().sources_path()
sources_path_subject = sources_path + subject.subject_id
fname_inv = sources_path_subject + f'/{subject.subject_id}_{surf_vol}-inv.fif'

# Load
inv = mne.minimum_norm.read_inverse_operator(fname_inv)
# Plot sources with bem
# mne.viz.plot_bem(src=inv['src'], subject=subject.subject_id, subjects_dir=subjects_dir,
#                  brain_surfaces='white', orientation='coronal')

# Inverse solution parameters (standard from mne)
snr = 3.0
lambda2 = 1.0 / snr ** 2
pick_ori = None  # 'vector' For dipoles
initial_time = 0.1

# Compute inverse solution to get sources time series
stc = mne.minimum_norm.apply_inverse(evoked=evoked, inverse_operator=inv, lambda2=lambda2, method='dSPM',
                                     pick_ori=pick_ori)

# Plot
if surf_vol == 'surface':
    if pick_ori == 'vector':
        brain = stc.plot(subjects_dir=subjects_dir, subject=subject.subject_id,
                         time_viewer=False, hemi='both',
                         initial_time=initial_time, time_unit='s',
                         brain_kwargs=dict(silhouette=True), smoothing_steps=7)
    else:
        brain = stc.plot(subjects_dir=subjects_dir, subject=subject.subject_id,
                         surface='inflated', time_viewer=True, hemi='both',
                         initial_time=initial_time, time_unit='s')

elif surf_vol == 'volume':
    stc.plot(inv['src'], subject=subject.subject_id, subjects_dir=subjects_dir)


# --------- Morph to fsaverage ---------#
morph = mne.compute_source_morph(src=inv['src'], subject_from=subject.subject_id, subject_to='fsaverage', subjects_dir=subjects_dir)
stc_fs = morph.apply(stc)

# Plot in fsaverage space
if surf_vol == 'surface':
    if pick_ori == 'vector':
        brain = stc_fs.plot(subjects_dir=subjects_dir, subject='fsaverage',
                            time_viewer=False, hemi='both',
                            initial_time=initial_time, time_unit='s')
    else:
        brain = stc_fs.plot(subjects_dir=subjects_dir, subject='fsaverage',
                         surface='flat', time_viewer=False, hemi='both',
                         initial_time=initial_time, time_unit='s')
        brain.add_annotation('HCPMMP1_combined', borders=2)

elif surf_vol == 'volume':
    stc.plot(inv['src'], subject=subject.subject_id, subjects_dir=subjects_dir)

