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
import matplotlib.pyplot as plt
from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test, spatio_temporal_cluster_test, summarize_clusters_stc

# Load experiment info
exp_info = setup.exp_info()

#--------- Define Parameters ---------#

# Subject and Epochs
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

# Select epochs
run_id = 'tgt_fix--it_fix_subsampled'  # use '--' to compute difference between 2 conditions
# ICA
use_ica_data = True

# Trials
corr_ans = True
tgt_pres = True
mss = None
t_dur = None
evt_dur = None

# Epochs parameters
reject = None  # None to use defalt {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value

# Source estimation parameters
force_fsaverage = False
# Model
model_name = 'lcmv'  # ('lcmv', 'dics')
surf_vol = 'volume'
ico = 5
spacing = 5.
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximices output power
# Default source subject
default_subject = exp_info.subjects_ids[0]  # Any subject or 'fsaverage'

# Plot
initial_time = None
cbar_percent_lims = (99.9, 99.95, 100)

# Frequency band
band_id = None

visualize_alignment = False


#--------- Setup ---------#

# Load subject
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Define time windows from epoch_id name
map_times = dict(sac={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.05, 0.2)},
                 it_fix_subsampled={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.05, 0.2)},
                 tgt_fix={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.05, 0.2)})

# Get time limits
tmin, tmax, _ = functions_general.get_time_lims(epoch_id=run_id, map=map_times)

# Baseline duration
if 'sac' in run_id:
    baseline = (tmin, 0)
elif 'fix' in run_id or 'fix' in run_id:
    baseline = (tmin, -0.05)
else:
    baseline = (tmin, 0)

# Plot colobar limits
if pick_ori == 'max-power':
    clim = dict(kind='percent', pos_lims=cbar_percent_lims)
else:
    clim = dict(kind='percent', lims=cbar_percent_lims)


# --------- Freesurfer Path ---------#

# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir


# --------- Run ---------#

# Save source estimates time courses on FreeSurfer
stcs_fs_dict = {}

# Run on separate events based on epochs ids to compute difference
epoch_ids = run_id.split('--')

# Iterate over epoch ids (if applies)
for epoch_id in epoch_ids:
    # Data and plots paths
    run_path = f'/Band_{band_id}/{epoch_id}_mss{mss}_Corr{corr_ans}_tgt{tgt_pres}_tdur{t_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}/'

    # Data paths
    epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path
    evoked_save_path = paths().save_path() + f'Evoked_{data_type}/' + run_path

    # Source plots paths
    fig_path = paths().plots_path() + f'Source_Space_{data_type}/' + run_path + f'{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}/'

    # Save source estimates time courses on FreeSurfer
    stcs_fs_dict[epoch_id] = []

    # Iterate over participants
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
        
        # Plot alignment visualization (if True)
        if visualize_alignment:
            plot_general.mri_meg_alignment(subject=subject, subject_code=subject_code, dig=dig, subjects_dir=subjects_dir)

        # Get epochs and evoked
        try:
            # Load data
            epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
            evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]
        except:
            # Compute
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
        elif surf_vol == 'mixed':
            fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
        fwd = mne.read_forward_solution(fname_fwd)
        src = fwd['src']

        # Load filter
        if surf_vol == 'volume':
            fname_filter = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
        elif surf_vol == 'surface':
            fname_filter = sources_path_subject + f'/{subject_code}_surface_ico{ico}_{pick_ori}-{model_name}.fif'
        elif surf_vol == 'mixed':
            fname_filter = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}_{pick_ori}-{model_name}.fif'
        filters = mne.beamformer.read_beamformer(fname_filter)

        # Apply filter and get source estimates
        stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

        # Morph to default subject
        if subject_code != exp_info.subjects_ids[0]:
            # Get Source space for default subject
            if surf_vol == 'volume':
                fname_src = paths().sources_path() + default_subject + f'/{default_subject}_volume_ico{ico}_{int(spacing)}-src.fif'
            elif surf_vol == 'surface':
                fname_src = paths().sources_path() + default_subject + f'/{default_subject}_surface_ico{ico}-src.fif'
            elif surf_vol == 'mixed':
                fname_src = paths().sources_path() + default_subject + f'/{default_subject}_mixed_ico{ico}_{int(spacing)}-src.fif'

            src_fs = mne.read_source_spaces(fname_src)

            # Define morph function
            morph = mne.compute_source_morph(src=src, subject_from=subject_code, subject_to=default_subject,
                                             src_to=src_fs, subjects_dir=subjects_dir)
            # Morph
            stc_fs = morph.apply(stc)

            # Append to fs_stcs to make GA
            stcs_fs_dict[epoch_id].append(stc_fs)

        else:
            # Append to fs_stcs to make GA
            stcs_fs_dict[epoch_id].append(stc)

        # Plot
        # if surf_vol == 'volume':
        #     fig = stc.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, initial_time=initial_time, clim=clim)
        #     # Save figure
        #     if save_fig :
        #         fname = f'{subject.subject_id}'
        #         if subject_code == 'fsaverage':
        #             fname += '_fsaverage'
        #         save.fig(fig=fig, path=fig_path, fname=fname)

    # Grand Average: Average evoked stcs from this epoch_id
    source_data_fs = np.zeros(tuple([len(stcs_fs_dict[epoch_id])] + [size for size in stcs_fs_dict[epoch_id][0].data.shape]))
    for i, stc in enumerate(stcs_fs_dict[epoch_id]):
        source_data_fs[i] = stcs_fs_dict[epoch_id][i].data
    GA_stc_data = source_data_fs.mean(0)

    # Copy Source Time Course from default subject morph to define GA STC
    GA_stc = stc_fs.copy()

    # Reeplace data
    GA_stc.data = GA_stc_data
    GA_stc.subject = default_subject

    # Nutmeg Plot
    fig = GA_stc.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, initial_time=initial_time, clim=clim)
    if save_fig and surf_vol == 'volume':
        fname = 'GA'
        if force_fsaverage:
            fname += '_fsaverage'
        save.fig(fig=fig, path=fig_path, fname=fname)


#----- Difference between conditions -----#

# Take difference of conditions if applies
if len(stcs_fs_dict.keys()) > 1:
    print(f'Taking difference between conditions: {epoch_ids[0]} - {epoch_ids[1]}')
    stcs_fs = []
    for i in range(len(stcs_fs_dict[epoch_ids[0]])):
        stcs_fs.append(stcs_fs_dict[list(stcs_fs_dict.keys())[0]][i] - stcs_fs_dict[list(stcs_fs_dict.keys())[1]][i])
        # stcs_fs.append(stcs_fs_dict[list(stcs_fs_dict.keys())[0]][i])

    # Variable for 2 conditions test
    print(f'Getting data from conditions: {epoch_ids[0]}, {epoch_ids[1]}')
    stcs_2samp = []
    for epoch_id in epoch_ids:
        source_data_fs = np.zeros(tuple([len(stcs_fs_dict[epoch_id])] + [size for size in stcs_fs_dict[epoch_id][0].data.shape[::-1]]))
        for i in range(len(stcs_fs_dict[epoch_id])):
            source_data_fs[i] = stcs_fs_dict[epoch_id][i].data.T
        stcs_2samp.append(source_data_fs)

    # Redefine figure save path
    fig_path = fig_path.replace(epoch_id, run_id)

    # Average evoked stcs
    source_data_fs = np.zeros(tuple([len(stcs_fs)]+[size for size in stcs_fs[0].data.shape]))
    for i, stc in enumerate(stcs_fs):
        source_data_fs[i] = stcs_fs[i].data
    GA_stc_data = source_data_fs.mean(0)

    # Copy Source Time Course from default subject morph to define GA STC
    GA_stc = stc_fs.copy()

    # Reeplace data
    GA_stc.data = GA_stc_data
    GA_stc.subject = default_subject

    # Get vertices from source space
    fsave_vertices = [s["vertno"] for s in src_fs]


    #--------- Cluster permutations test ---------#

    # Compute source space adjacency matrix
    print("Computing adjacency matrix")
    adjacency_matrix = mne.spatial_src_adjacency(src_fs)

    # Transpose source_fs_data from shape (subjects x space x time) to shape (subjects x time x space)
    source_data_fs = source_data_fs.swapaxes(1, 2)

    # Define the t-value threshold for cluster formation
    desired_pval = 0.0005
    df = len(exp_info.subjects_ids) - 1  # degrees of freedom for the test
    t_thresh = stats.distributions.t.ppf(1 - desired_pval / 2, df=df)
    t_thresh = dict(start=0, step=0.2)

    # Run permutations
    T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(X=source_data_fs,
                                                                                     n_permutations=512,
                                                                                     adjacency=adjacency_matrix,
                                                                                     n_jobs=4,
                                                                                     threshold=t_thresh)

    # Select the clusters that are statistically significant at p
    p_threshold = 0.05
    good_clusters_idx = np.where(cluster_p_values < p_threshold)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    # [cluster_p_values[idx] for idx in good_clusters_idx]

    # Select clusters for visualization
    stc_all_cluster_vis = summarize_clusters_stc(clu=clu, p_thresh=p_threshold, tstep=GA_stc.tstep, vertices=fsave_vertices,
                                                 subject=default_subject)


    # --------- Plots ---------#

    # 3D Plot
    if surf_vol == 'volume' or surf_vol == 'mixed':
        # Clusters 3D plot
        stc_all_cluster_vis.plot_3d(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='both', surface='white',
                                    initial_time=initial_time, time_unit='s', smoothing_steps=7)
        # Clusters Nutmeg plot
        fig = stc_all_cluster_vis.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir)
        if save_fig and surf_vol == 'volume':
            if type(t_thresh) == dict:
                fname = f'Clus_t_TFCE_p{p_threshold}'
            else:
                fname = f'Clus_t{round(t_thresh, 2)}_p{p_threshold}'
            if force_fsaverage:
                fname += '_fsaverage'
            save.fig(fig=fig, path=fig_path, fname=fname)

        # Difference Nutmeg plot
        fig = GA_stc.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, initial_time=initial_time, clim=clim)
        if save_fig and surf_vol == 'volume':
            fname = 'GA_diff_evoked'
            if force_fsaverage:
                fname += '_fsaverage'
            save.fig(fig=fig, path=fig_path, fname=fname)

        # Difference 3D plot
        GA_stc.plot_3d(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='both', surface='white',
                       initial_time=0, time_unit='s', smoothing_steps=7)

    else:
        # Clusters 3D plot
        stc_all_cluster_vis.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='both', surface='white',
                                 alpha=0.35, spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', smoothing_steps=7)

        # Difference 3D plot
        GA_stc.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='both', surface='white',
                    alpha=0.35, spacing=f'ico{ico}', initial_time=initial_time, time_unit='s',
                    smoothing_steps=7)
