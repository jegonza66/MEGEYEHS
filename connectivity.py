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
epoch_id = 'ms'
run_id = 'ms'
# Data
use_ica_data = True
# Frequency band
band_id = 'Beta'

# Trials
corr_ans = None
tgt_pres = None
mss = None

# Source estimation parameters
force_fsaverage = False
# Model
model_name = 'lcmv'
ico = 4
spacing = 10.
# Souce model ('volume'/'surface'/'mixed')
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
plot_edge = 0.15
vs_dur = {1: (2, 9.8), 2: (3, 9.8), 4: (3.5, 9.8), None: (2, 9.8)}
trial_dur = vs_dur[mss]

# Get time windows from epoch_id name
map = dict(ms={'tmin': -cross1_dur, 'tmax': mss_duration[mss], 'plot_xlim': (None, None)},
           vs={'tmin': - cross2_dur, 'tmax': vs_dur[mss][0], 'plot_xlim': (None, None)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, mss=mss, cross1_dur=cross1_dur,
                                                        mss_duration=mss_duration, cross2_dur=cross2_dur, dur=None,
                                                        plot_edge=plot_edge, map=map)
con_tmin = 0
if run_id == 'cross1':
    con_tmin = -cross1_dur
    con_tmax = 0
elif run_id == 'cross2':
    con_tmin = -cross2_dur
    con_tmax = 0
else:
    con_tmax = tmax

# Get baseline duration for epoch_id
baseline = None

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

# Data paths
run_path_data = f'/Band_None/{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}'
if (epoch_id == 'ms' or epoch_id == 'vs') and trial_dur:
    run_path_data += f'_tdur{trial_dur}'

epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path_data + f'_{tmin}_{tmax}_bline{baseline}/'
evoked_save_path = paths().save_path() + f'Evoked_{data_type}/' + run_path_data + f'_{tmin}_{tmax}_bline{baseline}/'

# Source plots paths
run_path_plot = run_path_data.replace('Band_None', f'Band_{band_id}')
run_path_plot = run_path_plot.replace(epoch_id, run_id)
fig_path = paths().plots_path() + f'Connectivity_{data_type}/' + run_path_plot + \
           f'{model_name}_{surf_vol}_ico{ico}_{spacing}_{pick_ori}_{parcelation}_{connectivity_method}/'  # Replace band id for None because Epochs are the same on all bands

# Set up connectivity matrix
if surf_vol == 'surface':  # or surf_vol == 'mixed':
    # Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
    fsaverage_labels = mne.read_labels_from_annot(subject='fsaverage', parc=parcelation, subjects_dir=subjects_dir)
    # Remove 'unknown' label for fsaverage aparc labels
    if parcelation == 'aparc':
        print("Dropping extra 'unkown' label from lh.")
        drop_idxs = [i for i, label in enumerate(fsaverage_labels) if 'unknown' in label.name]
        for drop_idx in drop_idxs:
            fsaverage_labels.pop(drop_idx)
    con_matrix = np.zeros((len(exp_info.subjects_ids), len(fsaverage_labels), len(fsaverage_labels)))
    # if surf_vol == 'mixed':
    #     fsaverage_labels + [1]
if surf_vol == 'volume':
    labels_fname = subjects_dir + f'/fsaverage/mri/aparc+aseg.mgz'
    fsaverage_labels = mne.get_volume_labels_from_aseg(labels_fname, return_colors=True)
    # Drop extra labels in fsaverage
    drop_idxs = [i for i, label in enumerate(fsaverage_labels[0]) if (label == 'ctx-lh-corpuscallosum' or
                                                                   label == 'ctx-rh-corpuscallosum')]
    for drop_idx in drop_idxs:
        fsaverage_labels[0].pop(drop_idx)
        fsaverage_labels[1].pop(drop_idx)
    con_matrix = np.zeros((len(exp_info.subjects_ids), len(fsaverage_labels[0]), len(fsaverage_labels[0])))

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
                                                           epoch_id=epoch_id, meg_data=meg_data, trial_dur=trial_dur,
                                                           tmin=tmin, tmax=tmax, baseline=baseline, reject=None,
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
    # Source data path
    sources_path_subject = paths().sources_path() + subject.subject_id

    # Load forward model
    if surf_vol == 'volume':
        fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
    elif surf_vol == 'surface':
        fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
    # elif surf_vol == 'mixed':
    #     fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
    fwd = mne.read_forward_solution(fname_fwd)
    # Get sources from forward model
    src = fwd['src']

    # Load filter
    if surf_vol == 'volume':
        fname_filter = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
    elif surf_vol == 'surface':
        fname_filter = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}-{model_name}.fif'
    # elif surf_vol == 'mixed':
    #     fname_filter = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
    filters = mne.beamformer.read_beamformer(fname_filter)

    # Apply filter and get source estimates
    stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)


    # --------- Connectivity ---------#
    if surf_vol == 'volume':
        labels = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
    elif subject_code != 'fsaverage':
        # Get labels for FreeSurfer cortical parcellation
        labels = mne.read_labels_from_annot(subject=subject_code, parc=parcelation, subjects_dir=subjects_dir)
    else:
        labels = fsaverage_labels

    # Average the source estimates within each label using sign-flips to reduce signal cancellations, also here we return a generator
    label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=labels, src=src, mode='auto', return_generator=True)

    # Compute connectivity
    con = mne_connectivity.spectral_connectivity_epochs(label_ts, method=connectivity_method, mode='multitaper', sfreq=epochs.info['sfreq'],
                                                        fmin=fmin, fmax=fmax, tmin=con_tmin, tmax=con_tmax, faverage=True,  mt_adaptive=True)

    # Get connectivity matrix
    con_matrix[subj_num] = con.get_data(output='dense')[:, :, 0]

    # Plot circle
    plot_general.connectivity_circle(subject=subject, labels=labels, surf_vol=surf_vol, con=con_matrix[subj_num], connectivity_method='pli',
                                     subject_code=subject_code, display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=None)

    # Plot connectome
    plot_general.connectome(subject=subject, labels=labels, adjacency_matrix=con_matrix[subj_num], subject_code=subject_code,
                            save_fig=True, fig_path=fig_path, fname=None)


# --------- Grand Average ---------#
# Get connectivity matrix for GA
ga_con_matrix = con_matrix.mean(0)


# Plot circle
plot_general.connectivity_circle(subject='GA', labels=labels, surf_vol=surf_vol, con=ga_con_matrix, connectivity_method='pli', subject_code='fsaverage',
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