import os
import functions_analysis
import functions_general
import mne
from mne.beamformer import make_lcmv, apply_lcmv
import save
from paths import paths
import load
import setup
import numpy as np
import plot_general

foo = ['15909001', '15910001', '15950001', '15911001', '16191001', '16263002']

subjects_ids = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001', '16200001',
                '16201001', '10925091', '16263002', '16269001']

# --------- Define Parameters ---------#
# Subject and Epochs
save_fig = True
# Subject
subject_code = '15950001'
# Select epochs
epoch_id = 'fix_ms'
# ICA
use_ica_data = True

# Trials
corr_ans = None
tgt_pres = None
mss = None

# Source estimation parameters
# Source data
force_fsaverage = True
# Souce model
use_beamformer = True
surf_vol = 'volume'
pick_ori = None  # 'vector' For dipoles, 'max_power' for

# Plot time
initial_time = None
# Frequency band
band_id = None

visualize_alignment = False
# Get time windows from epoch_id name
map_times = dict(sac={'tmin': -0.05, 'tmax': 0.07, 'plot_xlim': (-0.05, 0.07)},
                 fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.05, 0.2)})


# --------- Setup ---------#
# Load experiment info
exp_info = setup.exp_info()
# Load subject
if use_ica_data:
    subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    data_type = 'ICA'
else:
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    data_type = 'RAW'
if use_beamformer:
    model_name = 'Beamformer'
else:
    model_name = 'MNE'

# Get times
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

# Baseline duration
if 'sac' in epoch_id:
    baseline = (tmin, 0)
elif 'fix' in epoch_id or 'fix' in epoch_id:
    baseline = (tmin, -0.05)


# --------- Paths ---------#
run_path = f'/Band_{band_id}/{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_{tmin}_{tmax}_bline{baseline}/'

# Source plots paths
fig_path = paths().plots_path() + f'Source_Space_{data_type}/' + run_path + f'{model_name}_{surf_vol}_{pick_ori}/'

# Data paths
epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path
evoked_save_path = paths().save_path() + f'Evoked_{data_type}/' + run_path

# Data filenames
epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

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

# Source data path
sources_path_subject = paths().sources_path() + subject.subject_id
fname_fwd = sources_path_subject + f'/{subject_code}_volume-fwd.fif'
fname_inv = sources_path_subject + f'/{subject_code}_{surf_vol}-inv_{data_type}.fif'

if visualize_alignment:
    plot_general.mri_meg_alignment(subject=subject, subject_code=subject_code, dig=dig, subjects_dir=subjects_dir)

try:
    # Load data
    if use_beamformer:
        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
    evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]
except:
    if use_ica_data:
        meg_data = load.ica_data(subject=subject)
    else:
        meg_data = subject.load_preproc_meg_data()

    try:
        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
    except:

        # Epoch data
        epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans, tgt_pres=tgt_pres,
                                                       epoch_id=epoch_id, meg_data=meg_data, tmin=tmin, tmax=tmax,
                                                       save_data=True, epochs_save_path=epochs_save_path,
                                                       epochs_data_fname=epochs_data_fname)

    # Define evoked from epochs
    evoked = epochs.average()

    # Save evoked data
    os.makedirs(evoked_save_path, exist_ok=True)
    evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

# Pick meg channels for source modeling
evoked.pick('meg')
epochs.pick('meg')

# --------- Source estimation ---------#
if use_beamformer:
    # Load forward model
    fwd = mne.read_forward_solution(fname_fwd)

    # Compute covariance matrices from epochs for data and from raw for noise
    noise_cov = functions_analysis.noise_cov(exp_info=exp_info, subject=subject, bads=epochs.info['bads'],
                                             use_ica_data=use_ica_data)
    data_cov = mne.compute_covariance(epochs)

    # Define covariance matrices minimum rank as mag channels - excluded components
    # rank = min(np.linalg.matrix_rank(noise_cov.data), np.linalg.matrix_rank(data_cov.data))
    rank = sum([ch_type == 'mag' for ch_type in evoked.get_channel_types()]) - len(evoked.info['bads']) - len(subject.ex_components)

    # Define linearly constrained minimum variance spatial filter
    filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori=pick_ori,
                        rank=dict(mag=rank))  # reg parameter is for regularization on rank deficient matrices (rank < channels)

    # Apply filter and get source estimates
    stc = apply_lcmv(evoked, filters)

    # Plot
    clims = ((stc.data.max() - stc.data.max()/3), (stc.data.max() - stc.data.max()/4), stc.data.max())
    fig = stc.plot(fwd['src'], subject=subject_code, subjects_dir=subjects_dir, initial_time=initial_time,
                   clim=dict(kind='value', lims=clims))

    if save_fig:
        fname = f'{subject.subject_id}'
        if subject_code == 'fsaverage':
            fname += '_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)

    # 3D Plot
    # stc.plot_3d(src=fwd['src'], subject=subject_code, subjects_dir=subjects_dir, hemi='both', surface='white',
    #             initial_time=initial_time, time_unit='s', smoothing_steps=7)


if not use_beamformer:
    # Load
    inv = mne.minimum_norm.read_inverse_operator(fname_inv)

    # Inverse solution parameters (standard from mne)
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
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
        fig = stc.plot(inv['src'], subject=subject.subject_id, subjects_dir=subjects_dir, initial_time=initial_time)#, clim=dict(kind='value', lims=(48,55,95)))
        fig.tight_layout()

        if save_fig:
            fname = f'{subject.subject_id}'
            if subject_code == 'fsaverage':
                fname += '_fsaverage'
            save.fig(fig=fig, path=fig_path, fname=fname)

        # stc.plot_3d(src=inv['src'], subject=subject.subject_id, subjects_dir=subjects_dir, hemi='both', surface='white',
        #             initial_time=initial_time, time_unit='s', smoothing_steps=7)






## --------- Morph to fsaverage ---------#
if use_beamformer:
    src = fwd['src']
else:
    src = inv['src']
morph = mne.compute_source_morph(src=src, subject_from=subject.subject_id, subject_to='fsaverage', subjects_dir=subjects_dir)
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
    fig = stc.plot(src, subject=subject.subject_id, subjects_dir=subjects_dir, initial_time=initial_time)  # , clim=dict(kind='value', lims=(48,55,95)))
    if save_fig:
        fname = f'{subject.subject_id}_morph_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)

    stc.plot_3d(src=src, subject=subject.subject_id, subjects_dir=subjects_dir, hemi='both', surface='white',
                initial_time=initial_time, time_unit='s', smoothing_steps=7)


