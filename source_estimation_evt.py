import os
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

foo = ['15909001', '15910001', '15950001', '15911001', '16191001', '16263002']

subjects_ids = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001', '16200001',
                '16201001', '10925091', '16263002', '16269001']

# --------- Define Parameters ---------#
# Subject and Epochs
save_fig = True
# Select epochs
epoch_id = 'fix_ms'
# ICA
use_ica_data = True

# Trials
corr_ans = None
tgt_pres = None
mss = None

# Epochs parameters
reject = None

# Source estimation parameters
force_fsaverage = False
surf_vol = 'volume'
ico = 5
spacing = 5.
pick_ori = None  # 'vector' For dipoles, 'max_power' for

# Plot time
initial_time = None

# Frequency band
band_id = None

visualize_alignment = False


# --------- Setup ---------#
# Load experiment info
exp_info = setup.exp_info()

# Load subject
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Model
model_name = 'lcmv'  # ('lcmv', 'dics')

# Get time windows from epoch_id name
map_times = dict(sac={'tmin': -0.05, 'tmax': 0.07, 'plot_xlim': (-0.05, 0.07)},
                 fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.05, 0.2)})

# Get times
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

# Baseline duration
if 'sac' in epoch_id:
    baseline = (tmin, 0)
elif 'fix' in epoch_id or 'fix' in epoch_id:
    baseline = (tmin, -0.05)
else:
    baseline = (tmin, 0)

# --------- Paths ---------#
run_path = f'/Band_{band_id}/{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_{tmin}_{tmax}_bline{baseline}/'

# Source plots paths
fig_path = paths().plots_path() + f'Source_Space_{data_type}/' + run_path + f'{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}/'

# Data paths
epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path
evoked_save_path = paths().save_path() + f'Evoked_{data_type}/' + run_path

# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

stcs_fs = []

# --------- Run ---------#
for subject_code in exp_info.subjects_ids:
    # Load subject
    if use_ica_data:
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

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

    if visualize_alignment:
        plot_general.mri_meg_alignment(subject=subject, subject_code=subject_code, dig=dig, subjects_dir=subjects_dir)

    try:
        # Load data
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
                                                           epoch_id=epoch_id, meg_data=meg_data, tmin=tmin, tmax=tmax, reject=reject,
                                                           save_data=True, epochs_save_path=epochs_save_path, baseline=baseline,
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
    # Source data path
    sources_path_subject = paths().sources_path() + subject.subject_id

    # Load forward model
    if surf_vol == 'volume':
        fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
    elif surf_vol == 'surface':
        fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
    fwd = mne.read_forward_solution(fname_fwd)
    src = fwd['src']

    # Load filter
    if surf_vol == 'volume':
        fname_filter = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
    elif surf_vol == 'surface':
        fname_filter = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}-{model_name}.fif'
    filters = mne.beamformer.read_beamformer(fname_filter)

    # Apply filter and get source estimates
    stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

    # Morph to fsaverage
    if subject_code != 'fsaverage':
        # Get Source space for fsaverage
        if surf_vol == 'volume':
            fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_volume_ico{ico}_{int(spacing)}-src.fif'
        else:
            fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_surface_ico{ico}-src.fif'

        src_fs = mne.read_source_spaces(fname_src)

        # Define morph function
        morph = mne.compute_source_morph(src=src, subject_from=subject.subject_id, subject_to='fsaverage',
                                         src_to=src_fs, subjects_dir=subjects_dir)
        # Morph
        stc_fs = morph.apply(stc)

        # Append to fs_stcs to make GA
        stcs_fs.append(stc_fs)

    else:
        # Append to fs_stcs to make GA
        stcs_fs.append(stc)

    # Plot
    clims = (stc.data.max()*0.66, stc.data.max()*0.75, stc.data.max())
    fig = stc.plot(src, subject=subject_code, subjects_dir=subjects_dir, initial_time=initial_time,
                   clim=dict(kind='value', lims=clims))

    if save_fig:
        fname = f'{subject.subject_id}'
        if subject_code == 'fsaverage':
            fname += '_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)

    # 3D Plot
    # stc.plot_3d(src=fwd['src'], subject=subject_code, subjects_dir=subjects_dir, hemi='both', surface='white',
    #             initial_time=initial_time, time_unit='s', smoothing_steps=7)


# Average evoked stcs
source_data_fs = np.zeros((len(stcs_fs), stcs_fs[0].data.shape[0], stcs_fs[0].data.shape[1]))
for i, stc in enumerate(stcs_fs):
    source_data_fs[i] = stcs_fs[i].data
GA_stc_data = source_data_fs.mean(0)

# Copy Source Time Course from las fsaverage morph to define GA STC
try:
    GA_stc = stc_fs.copy()
except:
    GA_stc = stc.copy()

# Reeplace data
GA_stc.data = GA_stc_data
GA_stc.subject = 'fsaverage'

# Read fsaverage surface from any subject
if surf_vol == 'volume':
    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_volume_ico{ico}_{int(spacing)}-src.fif'
else:
    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_surface_ico{ico}-src.fif'
src_fs = mne.read_source_spaces(fname_src)

# Plot
clims = (GA_stc.data.max()*0.66, GA_stc.data.max()*0.75, GA_stc.data.max())
fig = GA_stc.plot(src_fs, subject='fsaverage', subjects_dir=subjects_dir, initial_time=initial_time,
                  clim=dict(kind='value', lims=clims))

if save_fig:
    fname = 'GA'
    if force_fsaverage:
        fname += '_fsaverage'
    save.fig(fig=fig, path=fig_path, fname=fname)


## --------- Morph to fsaverage ---------#

src = fwd['src']

if surf_vol == 'volume':
    # Load fsaverage volume Source Space
    fname_fsaverage_src = subjects_dir + '/fsaverage/bem/fsaverage-vol-5-src.fif'
    src_fs = mne.read_source_spaces(fname_fsaverage_src)
else:
    src_fs = None

morph = mne.compute_source_morph(src=src, subject_from=subject.subject_id, subject_to='fsaverage', src_to=src_fs, subjects_dir=subjects_dir)
stc_fs = morph.apply(stc)

# Plot in fsaverage space
if surf_vol == 'surface':
    if pick_ori == 'vector':
        brain = stc_fs.plot(subjects_dir=subjects_dir, subject='fsaverage', time_viewer=False, hemi='both',
                            initial_time=initial_time, time_unit='s')
    else:
        brain = stc_fs.plot(subjects_dir=subjects_dir, subject='fsaverage', surface='flat', time_viewer=False,
                            hemi='both', initial_time=initial_time, time_unit='s')
        brain.add_annotation('HCPMMP1_combined', borders=2)

elif surf_vol == 'volume':
    clims = (stc_fs.data.max()*0.66, stc_fs.data.max()*0.75, stc_fs.data.max())
    fig = stc_fs.plot(src_fs, subject='fsaverage', subjects_dir=subjects_dir, initial_time=initial_time, clim=dict(kind='value', lims=clims))
    if save_fig:
        fname = f'{subject.subject_id}_morph_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)

    stc_fs.plot_3d(src=src, subject=subject.subject_id, subjects_dir=subjects_dir, hemi='both', surface='white',
                initial_time=initial_time, time_unit='s', smoothing_steps=7)


