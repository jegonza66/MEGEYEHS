import os
import functions_analysis
import functions_general
import mne
import mne.beamformer as beamformer
import save
from paths import paths
import load
import setup
import mne_connectivity
import plot_general
import matplotlib.pyplot as plt
import numpy as np

foo = ['15909001', '15910001', '15950001', '15911001', '16191001', '16263002']

subjects_ids = ['15909001', '15912001', '15910001', '15950001', '15911001', '11535009', '16191001', '16200001',
                '16201001', '10925091', '16263002', '16269001']

# --------- Define Parameters ---------#
# Subject and Epochs
save_fig = True
# Subject
subject_code = '15909001'
# Select epochs
epoch_id = 'ms'
# ICA
use_ica_data = True

# Trials
corr_ans = None
tgt_pres = None
mss = 4

# Source estimation parameters
# Source data
force_fsaverage = False
# Souce model ('volume'/'surface')
use_beamformer = True
surf_vol = 'surface'
pick_ori = None  # 'vector' For dipoles, 'max_power' for

# Plot time
initial_time = None
# Frequency band
band_id = 'Theta'
visualize_alignment = False

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 0}
cross1_dur = 0.75
cross2_dur = 1
vs_dur = 4
plot_edge = 0.15

if 'ms' in epoch_id:
    dur = mss_duration[mss] + cross2_dur + vs_dur
elif 'cross2' in epoch_id:
    dur = cross2_dur + vs_dur  # seconds
else:
    dur = 0

# Get time windows from epoch_id name
map_times = dict(ms={'tmin': -0.1, 'tmax': mss_duration[mss], 'plot_xlim': (0, mss_duration[mss])},
                 cross2={'tmin': 0, 'tmax': cross2_dur, 'plot_xlim': (0, cross2_dur)},
                 vs={'tmin': 0, 'tmax': vs_dur, 'plot_xlim': (0, vs_dur)},
                 sac={'tmin': -0.05, 'tmax': 0.07, 'plot_xlim': (-0.05, 0.07)},
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
elif 'ms' in epoch_id or 'cross2' in epoch_id and mss:
    baseline = (tmin, 0)
    plot_baseline = (plot_xlim[0], 0)
else:
    baseline = (tmin, 0)
    plot_baseline = (plot_xlim[0], 0)


# --------- Paths ---------#
run_path = f'/Band_{band_id}/{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_{tmin}_{tmax}_bline{baseline}/'

# Source plots paths
fig_path = paths().plots_path() + f'Connectivity_{data_type}/' + run_path + f'{model_name}_{surf_vol}_{pick_ori}/'

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
    fs_subj_path = os.path.join(subjects_dir, subject_code)
    dig = False
else:
    # Check if subject has MRI data
    try:
        fs_subj_path = os.path.join(subjects_dir, subject_code)
        os.listdir(fs_subj_path)
        dig = True
    except:
        subject_code = 'fsaverage'
        fs_subj_path = os.path.join(subjects_dir, subject_code)
        dig = False

# Source data path
sources_path_subject = paths().sources_path() + subject.subject_id
fname_fwd = sources_path_subject + f'/{subject_code}_{surf_vol}-fwd.fif'
fname_inv = sources_path_subject + f'/{subject_code}_{surf_vol}-inv_{data_type}.fif'

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
                                                       epoch_id=epoch_id, meg_data=meg_data, tmin=tmin, tmax=tmax,
                                                       reject=dict(mag=4e-12), save_data=True, epochs_save_path=epochs_save_path,
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
filters = beamformer.make_lcmv(info=epochs.info, forward=fwd, data_cov=data_cov, reg=0.05, noise_cov=noise_cov,
                               pick_ori=pick_ori, rank=dict(mag=rank))  # reg parameter is for regularization on rank deficient matrices (rank < channels)

# Apply filter and get source estimates
# stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)
stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
parcelation = 'aparc'
labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

# Get sources from forward model
src = fwd['src']
# Average the source estimates within each label using sign-flips to reduce signal cancellations, also here we return a generator
label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=labels, src=src, mode='mean_flip', return_generator=True)


fmin, fmax = functions_general.get_freq_band(band_id=band_id)
tmin = 0
tmax = mss_duration[mss]
sfreq = epochs.info['sfreq']  # the sampling frequency
method = 'pli'
con = mne_connectivity.spectral_connectivity_epochs(label_ts, method=method, mode='multitaper', sfreq=sfreq,
                                                    fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, faverage=True,  mt_adaptive=True)


# Plot in circle
# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]
lh_labels = [name for name in label_names if name.endswith('lh')]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
mne_connectivity.viz.plot_connectivity_circle(con.get_data(output='dense')[:, :, 0], label_names, n_lines=300,
                         node_angles=node_angles, node_colors=label_colors,
                         title='All-to-All Connectivity left-Auditory '
                               'Condition (PLI)', ax=ax)
fig.tight_layout()

if save_fig:
    fname = f'{subject.subject_id}'
    if subject_code == 'fsaverage':
        fname += '_fsaverage'
    save.fig(fig=fig, path=fig_path, fname=fname)