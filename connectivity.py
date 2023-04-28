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



# --------- Define Parameters ---------#
save_fig = True
display_figs = False
# Select epochs by id
epoch_id = 'vs'
# Data
use_ica_data = True
# Frequency band
band_id = 'Theta'

# Trials
corr_ans = None
tgt_pres = None
mss = None

# Source estimation parameters
force_fsaverage = False
use_beamformer = True
# Souce model ('volume'/'surface')
surf_vol = 'surface'
pick_ori = None  # 'vector' For dipoles, 'max_power' for
# Parcelation (aparc / aparc.a2009s)
parcelation = 'aparc'
connectivity_method = 'pli'


# --------- Setup ---------#
# Frequencies from band
fmin, fmax = functions_general.get_freq_band(band_id=band_id)

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 2}
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

# Load experiment info
exp_info = setup.exp_info()

# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

run_path_data = f'/Band_None/{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_{tmin}_{tmax}_bline{baseline}/'
# Data paths
epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path_data
evoked_save_path = paths().save_path() + f'Evoked_{data_type}/' + run_path_data

# Model
if use_beamformer:
    model_name = 'Beamformer'
else:
    model_name = 'MNE'

# Source plots paths
run_path_plot = run_path_data.replace('Band_None', f'Band_{band_id}')
fig_path = paths().plots_path() + f'Connectivity_{data_type}/' + run_path_plot + \
           f'{model_name}_{surf_vol}_{pick_ori}_{parcelation}_{connectivity_method}/'  # Replace band id for None because Epochs are the same on all bands

# Connectivity matrix
# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
fsaverage_labels = mne.read_labels_from_annot(subject='fsaverage', parc=parcelation, subjects_dir=subjects_dir)
# Remove 'unknown' label for fsaverage aparc labels
if parcelation == 'aparc':
    print("Dropping extra 'unkown' label from lh.")
    drop_idxs = [i for i, label in enumerate(fsaverage_labels) if 'unknown' in label.name]
    for drop_idx in drop_idxs:
        fsaverage_labels.pop(drop_idx)

con_matrix = np.zeros((len(exp_info.subjects_ids), len(fsaverage_labels), len(fsaverage_labels)))

# Turn on/off show figures
if display_figs:
    plt.ion()
else:
    plt.ioff()

# --------- Run ---------#
for subj_num, subject_code in enumerate(exp_info.subjects_ids):
    # Load subject
    if use_ica_data:
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
    else:
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)


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

    # --------- Paths ---------#
    # Data filenames
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

    # Source data path
    sources_path_subject = paths().sources_path() + subject.subject_id
    fname_fwd = sources_path_subject + f'/{subject_code}_{surf_vol}-fwd.fif'
    fname_inv = sources_path_subject + f'/{subject_code}_{surf_vol}-inv_{data_type}.fif'

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
                                                           baseline=baseline, reject=None, save_data=True,
                                                           epochs_save_path=epochs_save_path,
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
    stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

    # --------- Connectivity ---------#
    if subject_code != 'fsaverage':
        # Get labels for FreeSurfer cortical parcellation
        labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
    else:
        labels = fsaverage_labels

    # Get sources from forward model
    src = fwd['src']
    # Average the source estimates within each label using sign-flips to reduce signal cancellations, also here we return a generator
    label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=labels, src=src, mode='mean_flip', return_generator=True)

    # Compute connectivity
    con_tmin = 0
    con = mne_connectivity.spectral_connectivity_epochs(label_ts, method=connectivity_method, mode='multitaper', sfreq=epochs.info['sfreq'],
                                                        fmin=fmin, fmax=fmax, tmin=con_tmin, tmax=tmax, faverage=True,  mt_adaptive=True)

    # Get connectivity matrix
    con_matrix[subj_num] = con.get_data(output='dense')[:, :, 0]

    # Plot circle
    plot_general.connectivity_circle(subject=subject, labels=labels, con=con_matrix[subj_num], connectivity_method='pli',
                                     subject_code=subject_code, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=None)

    # Plot connectome
    plot_general.connectome(subject=subject, labels=labels, adjacency_matrix=con_matrix[subj_num], subject_code=subject_code,
                            save_fig=True, fig_path=fig_path, fname=None)


# --------- Grand Average ---------#
# Get connectivity matrix for GA
ga_con_matrix = con_matrix.mean(0)


# Plot circle
plot_general.connectivity_circle(subject='GA', labels=labels, con=ga_con_matrix, connectivity_method='pli', subject_code='fsaverage',
                                 display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname='GA_circle')


# Plot connectome
plot_general.connectome(subject='GA', labels=labels, adjacency_matrix=ga_con_matrix, subject_code='fsaverage',
                        save_fig=True, fig_path=fig_path, fname='GA_connectome')



# # Plot markers positions
# plotting.plot_markers(np.arange(len(labels)), nodes_pos*1000)
#
# # Plot connectome in 3d interactive
# view = plotting.view_connectome(ga_con_matrix, nodes_pos, edge_threshold=ga_con_matrix.max()*0.9).open_in_browser()


## Epochs and evokeds in source space

stc_epochs[0].plot(hemi='both')

epochs_data = np.zeros((len(stc_epochs), stc_epochs[0].data.shape[0], stc_epochs[0].data.shape[1]))

print('Getting epoched data')
for epoch_num, epoch in enumerate(stc_epochs):
    epochs_data[epoch_num] = epoch.data
    print(f'\rProgress: {int((epoch_num + 1) / len(stc_epochs) * 100)}%', end='')

print('Averaging evoked data')
evoked_data = epochs_data.mean(0)

evoked_stc = epoch.copy()
evoked_stc.data = evoked_data

evoked_stc.plot(hemi='both')