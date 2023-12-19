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
from scipy.signal import hilbert
from mne.stats import spatio_temporal_cluster_1samp_test, spatio_temporal_cluster_test, summarize_clusters_stc

# Load experiment info
exp_info = setup.exp_info()

#--------- Define Parameters ---------#

# Save
save_fig = True
save_data = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

# Select epochs
run_id = 'cross2_mss4--cross2_mss1'  # use '--' to compute difference between 2 conditions
# ICA
use_ica_data = True

# Trials
corr_ans = None
tgt_pres = None
mss = None
evt_dur = None
t_dur = None

# Get time limits
tmin, tmax, = -0.75, 3
# Baseline
baseline = (tmin, 0)
bline_mode = 'db'

# Epochs parameters
reject = None  # None to use defalt {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'surface'
ico = 4
spacing = 10.
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximices output power
source_power = True

# Default source subject
default_subject = exp_info.subjects_ids[0]  # Any subject or 'fsaverage'
visualize_alignment = False

# Plot
initial_time = 0.75
cbar_percent_lims = (99.9, 99.95, 100)

# Frequency band
band_id = 'Alpha'
filter_method = 'fir'

# Permutations test
run_permutations = False


#--------- Setup ---------#

# Load subject
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Plot colobar limits
clim_3d = dict(kind='percent', pos_lims=cbar_percent_lims)
clim_nutmeg = dict(kind='percent', lims=cbar_percent_lims)

# Screen durations
vs_dur = {1: (2, 9.8), 2: (3, 9.8), 4: (3.5, 9.8), None: (2, 9.8)}

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

    if 'mss' in epoch_id:
        mss = int(epoch_id.split('_mss')[-1][:1])
        screen = epoch_id.split('_mss')[0]
        t_dur = vs_dur[mss]

    # Windows durations
    dur, cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration(epoch_id=screen, vs_dur=vs_dur, mss=mss)

    # Get time windows from epoch_id name
    map = dict(ms={'tmin': -cross1_dur, 'tmax': mss_duration[1], 'plot_xlim': (-cross1_dur - mss_duration[mss], dur)},
               cross2={'tmin': -cross1_dur - mss_duration[mss], 'tmax': 1, 'plot_xlim': (-cross1_dur - mss_duration[mss], 1)})

    tmin, tmax, _ = functions_general.get_time_lims(epoch_id=screen, mss=mss, plot_edge=0, map=map)

    # Data and plots paths
    if 'mss' not in epoch_id:
        epoch_id += f'_mss{mss}'
    run_path = f'/Band_{band_id}/{epoch_id}_Corr{corr_ans}_tgt{tgt_pres}_tdur{t_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}/'

    # Data paths
    epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path
    evoked_save_path = paths().save_path() + f'Evoked_{data_type}/' + run_path

    # Source plots paths
    fig_path = paths().plots_path() + f'Source_Space_{data_type}/' + run_path + f'{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}/'
    # Redefine figure save path
    fig_path_diff = fig_path.replace(epoch_id, run_id)

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
                if band_id:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=save_data,
                                                  method=filter_method)
                else:
                    meg_data = load.ica_data(subject=subject)
            else:
                if band_id:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, use_ica_data=False,
                                                  save_data=save_data, method=filter_method)
                else:
                    meg_data = subject.load_preproc_meg_data()

            try:
                epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
            except:
                # Epoch data
                epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans, tgt_pres=tgt_pres,
                                                               epoch_id=epoch_id, meg_data=meg_data, tmin=tmin,
                                                               tmax=tmax, reject=reject, baseline=baseline,
                                                               save_data=save_data, epochs_save_path=epochs_save_path,
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

        # Compute power in source space
        if source_power:
            data = stc.data
            analytic_signal = hilbert(data, axis=-1)
            amplitude_power = np.abs(analytic_signal)**2
            stc.data = amplitude_power

        # Apply baseline correction
        # stc.apply_baseline(baseline=baseline)
        if bline_mode == 'db':
            stc.data = 10 * np.log10(stc.data / stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None])
        elif bline_mode == 'ratio':
            stc.data = stc.data / stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]
        elif bline_mode == 'mean':
            stc.data = stc.data - stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]

        # Morph to default subject
        if subject_code != default_subject:
            # Get Source space for default subject
            if surf_vol == 'volume':
                fname_src = paths().sources_path() + default_subject + f'/{default_subject}_volume_ico{ico}_{int(spacing)}-src.fif'
            elif surf_vol == 'surface':
                fname_src = paths().sources_path() + default_subject + f'/{default_subject}_surface_ico{ico}-src.fif'
            elif surf_vol == 'mixed':
                fname_src = paths().sources_path() + default_subject + f'/{default_subject}_mixed_ico{ico}_{int(spacing)}-src.fif'

            src_fs = mne.read_source_spaces(fname_src)

            # Morph to default subject
            morph = mne.compute_source_morph(src=src, subject_from=subject_code,
                                                 subject_to=default_subject,
                                                 src_to=src_fs, subjects_dir=subjects_dir)

            # Apply morph
            stc_fs = morph.apply(stc)

            # Append to fs_stcs to make GA
            stcs_fs_dict[epoch_id].append(stc_fs)

        else:
            # Append to fs_stcs to make GA
            stcs_fs_dict[epoch_id].append(stc)

        # # Plot
        # if surf_vol == 'volume':
        #     fname = f'{subject.subject_id}'
        #     if subject_code == 'fsaverage':
        #         fname += '_fsaverage'
        #
        #     # 3D plot
        #     brain = stc.copy().crop(tmin=0).plot_3d(src=src, subject=subject_code, subjects_dir=subjects_dir, hemi='split',
        #                            spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1920, 1072))
        #     if save_fig:
        #         # brain.show_view(azimuth=-90)
        #         os.makedirs(fig_path + '/svg/', exist_ok=True)
        #         brain.save_image(filename=fig_path + fname + '.png')
        #         brain.save_image(filename=fig_path + '/svg/' + fname + '.pdf')
        #
        #     # Nutmeg
        #     fig = stc.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, initial_time=initial_time, clim=clim_nutmeg)
        #     # Save figure
        #     if save_fig:
        #         save.fig(fig=fig, path=fig_path, fname=fname)
        #
        # elif surf_vol == 'surface':
        #     # 3D plot
        #     brain = stc.copy().crop(tmin=0).plot(src=src, subject=subject_code, subjects_dir=subjects_dir, hemi='split',
        #                      spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1920, 1072))
        #     # brain.add_annotation('aparc.DKTatlas40')
        #     if save_fig:
        #         fname = f'{subject.subject_id}'
        #         if subject_code == 'fsaverage':
        #             fname += '_fsaverage'
        #         # brain.show_view(azimuth=-90)
        #         os.makedirs(fig_path + '/svg/', exist_ok=True)
        #         brain.save_image(filename=fig_path + fname + '.png')
        #         brain.save_image(filename=fig_path + '/svg/' + fname + '.pdf')

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

    # Apply baseline on GA data
    GA_stc.apply_baseline(baseline=baseline)
    # if bline_mode == 'db':
    #     GA_stc.data = 10 * np.log10(GA_stc.data / GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None])
    # elif bline_mode == 'ratio':
    #     GA_stc.data = GA_stc.data / GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]
    # elif bline_mode == 'mean':
    #     GA_stc.data = GA_stc.data - GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]

    if surf_vol == 'volume':
        fname = 'GA'
        if force_fsaverage:
            fname += '_fsaverage'

        # 3D plot
        brain = GA_stc.copy().crop(tmin=0).plot_3d(src=src_fs, subject=default_subject, subjects_dir=subjects_dir,  hemi='split',
                               spacing=f'ico{ico}', initial_time=initial_time, size=(1920, 1072))
        if save_fig:
            # brain.show_view(azimuth=-90)
            os.makedirs(fig_path + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path + fname + '.png')
            brain.save_image(filename=fig_path + '/svg/' + fname + '.pdf')
            brain.save_movie(filename=fig_path + fname + '.mp4', tmin=0, time_dilation=12, framerate=30)

        # Nutmeg plot
        fig = GA_stc.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, initial_time=initial_time, clim=clim_nutmeg)
        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    elif surf_vol == 'surface':
        # 3D plot
        brain = GA_stc.copy().crop(tmin=0).plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='split',
                            spacing=f'ico{ico}', time_unit='s', views='lateral', size=(1920, 1072))
        if save_fig:
            fname = 'GA'
            if force_fsaverage:
                fname += '_fsaverage'
            # brain.show_view(azimuth=-90)
            os.makedirs(fig_path + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path + fname + '.png')
            brain.save_image(filename=fig_path + '/svg/' + fname + '.pdf')
            brain.save_movie(filename=fig_path + fname + '.mp4', tmin=0, time_dilation=12, framerate=30)

#----- Difference between conditions -----#

# Take difference of conditions if applies
if len(stcs_fs_dict.keys()) > 1:
    print(f'Taking difference between conditions: {epoch_ids[0]} - {epoch_ids[1]}')

    stcs_fs = []
    # Compute difference for cross2
    if any('cross2' in id for id in epoch_ids):
        stcs_fs_base = []
        for i in range(len(stcs_fs_dict[epoch_ids[0]])):
            # Crop data to apply baseline and compute cross2 difference
            subj_stc_0 = stcs_fs_dict[epoch_ids[0]][i].copy().crop(tmin=0, tmax=cross2_dur)
            subj_stc_1 = stcs_fs_dict[epoch_ids[1]][i].copy().crop(tmin=0, tmax=cross2_dur)
            subj_stc_0_base = stcs_fs_dict[epoch_ids[0]][i].copy().crop(tmin=None, tmax=stcs_fs_dict[epoch_ids[0]][i].tmin + cross1_dur)
            subj_stc_1_base = stcs_fs_dict[epoch_ids[1]][i].copy().crop(tmin=None, tmax=stcs_fs_dict[epoch_ids[1]][i].tmin + cross1_dur)

            # Save baseline difference and cross2 difference
            stcs_fs.append(subj_stc_0 - subj_stc_1)
            stcs_fs_base.append(subj_stc_0_base - subj_stc_1_base)

    # Compute difference for other ids
    else:
        for i in range(len(stcs_fs_dict[epoch_ids[0]])):
            stcs_fs.append(stcs_fs_dict[epoch_ids[0]][i] - stcs_fs_dict[epoch_ids[1]][i])

    # Variable for 2 conditions test
    print(f'Getting data from conditions: {epoch_ids[0]}, {epoch_ids[1]}')
    stcs_2samp = []
    for epoch_id in epoch_ids:
        source_data_fs = np.zeros(tuple([len(stcs_fs_dict[epoch_id])] + [size for size in stcs_fs_dict[epoch_id][0].data.shape[::-1]]))
        for i in range(len(stcs_fs_dict[epoch_id])):
            source_data_fs[i] = stcs_fs_dict[epoch_id][i].data.T
        stcs_2samp.append(source_data_fs)

    # Average evoked stcs
    source_data_fs = np.zeros(tuple([len(stcs_fs)]+[size for size in stcs_fs[0].data.shape]))
    for i, stc in enumerate(stcs_fs):
        source_data_fs[i] = stcs_fs[i].data
    GA_stc_diff_data = source_data_fs.mean(0)

    # Copy Source Time Course from default subject morph to define GA STC
    GA_stc_diff = GA_stc.copy()

    # Reeplace data
    GA_stc_diff.data = GA_stc_diff_data
    GA_stc_diff.subject = default_subject

    # Apply baseline on GA diff
    if any('cross2' in id for id in epoch_ids):
        source_data_fs_base = np.zeros(tuple([len(stcs_fs)] + [size for size in stcs_fs_base[0].data.shape]))
        for i, stc in enumerate(stcs_fs_base):
            source_data_fs_base[i] = stcs_fs_base[i].data
        GA_stc_diff_data_base = source_data_fs_base.mean(0)

        # if bline_mode == 'db':
        #     GA_stc_diff_data = 10 * np.log10(GA_stc_diff_data / GA_stc_diff_data_base.mean(axis=1)[:, None])
        # elif bline_mode == 'ratio':
        #     GA_stc_diff_data = GA_stc_diff_data / GA_stc_diff_data_base.mean(axis=1)[:, None]
        # elif bline_mode == 'mean':
        #     GA_stc_diff_data = GA_stc_diff_data - GA_stc_diff_data_base.mean(axis=1)[:, None]

        # Apply mean baseline
        GA_stc_diff_data = GA_stc_diff_data - GA_stc_diff_data_base.mean(axis=1)[:, None]

        # Overwrite data in stc
        GA_stc_diff.data = GA_stc_diff_data

    else:
        # Apply mean baseline
        GA_stc_diff.apply_baseline(baseline=baseline)
        # if bline_mode == 'db':
        #     GA_stc_diff.data = 10 * np.log10(
        #         GA_stc_diff.data / GA_stc_diff.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None])
        # elif bline_mode == 'ratio':
        #     GA_stc_diff.data = GA_stc_diff.data / GA_stc_diff.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]
        # elif bline_mode == 'mean':
        #     GA_stc_diff.data = GA_stc_diff.data - GA_stc_diff.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]

    # Get vertices from source space
    fsave_vertices = [s["vertno"] for s in src_fs]

    #--------- Cluster permutations test ---------#

    if run_permutations:
        # Compute source space adjacency matrix
        print("Computing adjacency matrix")
        adjacency_matrix = mne.spatial_src_adjacency(src_fs)

        # Transpose source_fs_data from shape (subjects x space x time) to shape (subjects x time x space)
        source_data_fs = source_data_fs.swapaxes(1, 2)

        # Define the t-value threshold for cluster formation
        desired_pval = 0.001
        df = len(exp_info.subjects_ids) - 1  # degrees of freedom for the test
        t_thresh = stats.distributions.t.ppf(1 - desired_pval / 2, df=df)

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
        stc_all_cluster_vis = summarize_clusters_stc(clu=clu, p_thresh=p_threshold, tstep=GA_stc_diff.tstep, vertices=fsave_vertices,
                                                     subject=default_subject)


    # --------- Plots ---------#

    # 3D Plot
    if surf_vol == 'volume' or surf_vol == 'mixed':
        if run_permutations:
            # Clusters 3D plot
            brain = stc_all_cluster_vis.plot_3d(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='split',
                                        spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1920, 1072))

            if save_fig:
                fname = f'Clus_t{round(t_thresh, 2)}_p{p_threshold}'
                if force_fsaverage:
                    fname += '_fsaverage'
                # brain.show_view(azimuth=-90)
                os.makedirs(fig_path_diff + '/svg/', exist_ok=True)
                brain.save_image(filename=fig_path_diff + fname + '.png')
                brain.save_image(filename=fig_path_diff + '/svg/' + fname + '.pdf')
                brain.save_movie(filename=fig_path_diff + fname + '.mp4', tmin=0, time_dilation=12, framerate=30)

            # Clusters Nutmeg plot
            fig = stc_all_cluster_vis.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir)
            if save_fig and surf_vol == 'volume':
                if type(t_thresh) == dict:
                    fname = f'Clus_t_TFCE_p{p_threshold}'
                else:
                    fname = f'Clus_t{round(t_thresh, 2)}_p{p_threshold}'
                if force_fsaverage:
                    fname += '_fsaverage'
                save.fig(fig=fig, path=fig_path_diff, fname=fname)

        # Difference Nutmeg plot
        fig = GA_stc_diff.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, initial_time=initial_time, clim=clim_nutmeg)
        if save_fig and surf_vol == 'volume':
            fname = 'GA'
            if force_fsaverage:
                fname += '_fsaverage'
            save.fig(fig=fig, path=fig_path_diff, fname=fname)

        # Difference 3D plot
        brain = GA_stc_diff.copy().crop(tmin=0).plot_3d(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='split',
                       spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1920, 1072))
        if save_fig:
            fname = 'GA'
            if force_fsaverage:
                fname += '_fsaverage'
            # brain.show_view(azimuth=-90)
            os.makedirs(fig_path_diff + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path_diff + fname + '.png')
            brain.save_image(filename=fig_path_diff + '/svg/' + fname + '.pdf')
            brain.save_movie(filename=fig_path_diff + fname + '.mp4', tmin=0, time_dilation=12, framerate=30)

    elif surf_vol == 'surface':
        if run_permutations:
            # Clusters 3D plot
            fig = stc_all_cluster_vis.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='split',
                                     spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1920, 1072))
            if save_fig:
                if type(t_thresh) == dict:
                    fname = f'Clus_t_TFCE_p{p_threshold}'
                else:
                    fname = f'Clus_t{round(t_thresh, 2)}_p{p_threshold}'
                if force_fsaverage:
                    fname += '_fsaverage'
                # fig.show_view(azimuth=-90)
                os.makedirs(fig_path_diff + '/svg/', exist_ok=True)
                fig.save_image(filename=fig_path_diff + fname + '.png')
                brain.save_image(filename=fig_path_diff + '/svg/' + fname + '.pdf')
                brain.save_movie(filename=fig_path_diff + fname + '.mp4', tmin=0, time_dilation=12, framerate=30)

        # Difference 3D plot
        brain = GA_stc_diff.copy().crop(tmin=0).plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='split',
                    spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1920, 1072))
        if save_fig:
            fname = 'GA'
            if force_fsaverage:
                fname += '_fsaverage'
            # brain.show_view(azimuth=-90)
            os.makedirs(fig_path_diff + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path_diff + fname + '.png')
            brain.save_image(filename=fig_path_diff + '/svg/' + fname + '.pdf')
            brain.save_movie(filename=fig_path_diff + fname + '.mp4', tmin=0, time_dilation=12, framerate=30)