import os
import functions_analysis
import functions_general
import mne

import save
from paths import paths
import load
import setup


# --------- Define Parameters ---------#
# Subject
subject_code = 0
# Select epochs
epoch_id = 'l_sac'
# Get time windows from epoch_id name
map_times = dict(sac={'tmin': -0.05, 'tmax': 0.07, 'plot_xlim': (-0.05, 0.07)})
# Duration
dur = None  # seconds
# ICA
use_ica_data = True
# Frequency band
band_id = None
# Volume vs Surface estimation
surf_vol = 'volume'
pick_ori = None  # 'vector' For dipoles

# --------- Setup ---------#
# Load experiment info
exp_info = setup.exp_info()
# Load subject and meg clean data
subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

# Define Subjects_dir as Freesurfer output folder
mri_path = paths().mri_path()
subjects_dir = os.path.join(mri_path, 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir


# --------- Paths ---------#
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)
run_path = f'/{band_id}/{epoch_id}_{tmin}_{tmax}/'
if use_ica_data:
    fig_path = paths().plots_path() + 'Source_Space_ICA/' + run_path + f'{surf_vol}_{pick_ori}/'
else:
    fig_path = paths().plots_path() + 'Source_Space_RAW/' + run_path + f'{surf_vol}_{pick_ori}/'
os.makedirs(fig_path, exist_ok=True)

# Evoked data paths
if use_ica_data:
    evoked_save_path = paths().save_path() + f'Evoked_ICA/' + run_path
else:
    evoked_save_path = paths().save_path() + f'Evoked_RAW/' + run_path
os.makedirs(evoked_save_path, exist_ok=True)

evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

# Source data path
sources_path = paths().sources_path()
sources_path_subject = sources_path + subject.subject_id
if use_ica_data:
    fname_inv = sources_path_subject + f'/{subject.subject_id}_{surf_vol}-inv_ica.fif'
else:
    fname_inv = sources_path_subject + f'/{subject.subject_id}_{surf_vol}-inv.fif'


# Path to MRI <-> HEAD Transformation (Saved from coreg)
trans_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-trans.fif'.format(subject.subject_id))
fids_path = os.path.join(subjects_dir, subject.subject_id, 'bem', '{}-fiducials.fif'.format(subject.subject_id))
dig_info_path = paths().opt_path() + subject.subject_id + '/info_raw.fif'


# --------- Coord systems alignment ---------#
# Load raw meg data with dig info
info_raw = mne.io.read_raw_fif(dig_info_path)

# Visualize MEG/MRI alignment
# surfaces = dict(brain=0.7, outer_skull=0.5, head=0.4)
# fig = mne.viz.plot_alignment(info_raw.info, trans=trans_path, subject=subject.subject_id,
#                              subjects_dir=subjects_dir, surfaces=surfaces,
#                              show_axes=True, dig=True, eeg=[], meg='sensors',
#                              coord_frame='meg', mri_fiducials=fids_path)
# mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))


try:
    # Load evoked data
    evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]
except:
    if use_ica_data:
        meg_data = load.ica_data(subject=subject)
    else:
        meg_data = subject.load_preproc_meg()

    # --------- Epoch data ---------#
    # Screen
    screen = functions_general.get_screen(epoch_id=epoch_id)
    # MSS
    mss = functions_general.get_mss(epoch_id=epoch_id)
    # Item
    tgt = functions_general.get_item(epoch_id=epoch_id)
    # Saccades direction
    dir = functions_general.get_dir(epoch_id=epoch_id)

    # Get events based on conditions
    metadata, events, events_id, metadata_sup = functions_analysis.define_events(subject=subject, epoch_id=epoch_id,
                                                                                 screen=screen, mss=mss, dur=dur,
                                                                                 tgt=tgt, dir=dir, meg_data=meg_data)
    # Reject based on channel amplitude
    reject = dict(mag=subject.config.general.reject_amp)
    # reject = dict(mag=2.5e-12)

    # Epoch data
    epochs = mne.Epochs(meg_data, events, event_id=events_id, reject=reject, tmin=tmin, tmax=tmax,
                        event_repeated='drop', metadata=metadata, preload=True)

    # Define evoked from epochs
    evoked = epochs.average()

    # Save evoked data
    evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

# Pick meg channels for source modeling
evoked.pick('meg')

# --------- Inverse operation computation ---------#

# Load
inv = mne.minimum_norm.read_inverse_operator(fname_inv)
# Plot sources with bem
# mne.viz.plot_bem(src=inv['src'], subject=subject.subject_id, subjects_dir=subjects_dir,
#                  brain_surfaces='white', orientation='coronal')

# Inverse solution parameters (standard from mne)
snr = 3.0
lambda2 = 1.0 / snr ** 2
initial_time = 0.

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
    fig = stc.plot(inv['src'], subject=subject.subject_id, subjects_dir=subjects_dir)#, clim=dict(kind='value', lims=(48,55,89)))
    fig.tight_layout()
    fname = f'{subject.subject_id}_scaled'
    save.fig(fig=fig, path=fig_path, fname=fname)






## --------- Morph to fsaverage ---------#
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


