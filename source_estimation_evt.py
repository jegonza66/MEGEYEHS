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
plot_individuals = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

# Select epochs
run_id = 'cross2_mss2--cross2_mss1'  # use '--' to compute difference between 2 conditions
# ICA
use_ica_data = True

# Trials
corr_ans = None
tgt_pres = None
mss = None
# Screen durations
vs_dur = {1: (2, 9.8), 2: (3, 9.8), 4: (3.5, 9.8), None: (2, 9.8)}
evt_dur = None
trial_dur = vs_dur[mss]

# Baseline
bline_mode_subj = 'db'
bline_mode_ga = False
plot_edge = 0.25

# Epochs parameters
reject = None  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'volume'
ico = 5
spacing = 10.
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
source_power = True
estimate_epochs = False  # epochs and covariance cannot be both true (but they can be both false and estimate sources from evoked)
estimate_covariance = True

# Default source subject
default_subject = exp_info.subjects_ids[0]  # Any subject or 'fsaverage'
visualize_alignment = False

# Plot
initial_time = None
clim_3d = dict(kind='percent', pos_lims=(85, 95, 100))
clim_nutmeg = dict(kind='percent', pos_lims=(99.9, 99.95, 100))

# Frequency band
filter_sensors = True
band_id = 'Alpha'
filter_method = 'iir'

# Permutations test
run_permutations = True


#--------- Setup ---------#

# Load subject
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Source computation method
if estimate_covariance:
    source_computation = 'cov'
elif estimate_epochs:
    source_computation = 'epo'
else:
    source_computation = 'evk'


# --------- Freesurfer Path ---------#

# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir


# --------- Run ---------#

# Save source estimates time courses on FreeSurfer
stcs_fs_dict = {}
GA_stcs = {}

# Run on separate events based on epochs ids to compute difference
epoch_ids = run_id.split('--')

# Iterate over epoch ids (if applies)
for i, epoch_id in enumerate(epoch_ids):

    if 'mss' in epoch_id:
        mss = int(epoch_id.split('_mss')[-1][:1])
        epoch_id = epoch_id.split('_mss')[0]
        trial_dur = vs_dur[mss]

    # Windows durations
    dur, cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration(epoch_id=epoch_id, vs_dur=vs_dur, mss=mss)

    # Get time windows from epoch_id name
    map = dict(ms={'tmin': -cross1_dur, 'tmax': mss_duration[1], 'plot_xlim': (-cross1_dur + plot_edge, mss_duration[1] - plot_edge)})
    tmin, tmax, _ = functions_general.get_time_lims(epoch_id=epoch_id, mss=mss, plot_edge=plot_edge, map=map)

    # Get baseline duration for epoch_id
    baseline, plot_baseline = functions_general.get_baseline_duration(epoch_id=epoch_id, mss=mss, tmin=tmin, tmax=tmax,
                                                                      cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                      cross2_dur=cross2_dur)

    # Data and plots paths
    if filter_sensors:
        run_path = f'/Band_{band_id}/{epoch_id}_mss{mss}_Corr{corr_ans}_tgt{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}/'
    else:
        run_path = f'/Band_None/{epoch_id}_mss{mss}_Corr{corr_ans}_tgt{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}/'

    # Data paths
    epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + run_path
    evoked_save_path = paths().save_path() + f'Evoked_{data_type}/' + run_path
    cov_save_path = paths().save_path() + f'Cov_Epochs_{data_type}/' + run_path

    # Source plots paths
    if source_power:
        run_path = run_path.replace(f'{epoch_id}_mss{mss}', f'{epoch_id}_mss{mss}_power')
    run_path = run_path.replace('Band_None', f'Band_{band_id}')

    # Define path
    if surf_vol == 'volume' or surf_vol == 'mixed':
        fig_path = paths().plots_path() + f'Source_Space_{data_type}/' + run_path + f'{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_{pick_ori}_{source_computation}/'
    else:
        fig_path = paths().plots_path() + f'Source_Space_{data_type}/' + run_path + f'{model_name}_{surf_vol}_ico{ico}_{pick_ori}_{source_computation}/'

    # Redefine figure save path
    fig_path_diff = fig_path.replace(f'{epoch_id}_mss{mss}', run_id)

    # Save source estimates time courses on FreeSurfer
    stcs_fs_dict[epoch_ids[i]] = []

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
                if band_id and filter_sensors:
                    meg_data = load.filtered_data(subject=subject, band_id=band_id, save_data=save_data,
                                                  method=filter_method)
                else:
                    meg_data = load.ica_data(subject=subject)
            else:
                if band_id and filter_sensors:
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

        if estimate_covariance:
            # Covariance method
            cov_method = 'shrunk'

            # Spatial filter
            rank = sum([ch_type == 'mag' for ch_type in epochs.get_channel_types()]) - len(epochs.info['bads'])
            if use_ica_data:
                rank -= len(subject.ex_components)

            # Define active times
            if baseline[0] == tmin:
                active_times = [baseline[1], tmax]
            elif baseline[1] == tmax:
                active_times = [tmin, baseline[0]]
            else:
                raise ValueError('Active time window undefined')

            # Covariance fnames
            cov_baseline_fname = f'Subject_{subject.subject_id}_times{baseline}_{cov_method}_{rank}-cov.fif'
            cov_act_fname = f'Subject_{subject.subject_id}_times{active_times}_{cov_method}_{rank}-cov.fif'

            try:
                # Load covariance matrix
                baseline_cov = mne.read_cov(fname=cov_save_path + cov_baseline_fname)
            except:
                # Compute covariance matrices
                baseline_cov = mne.cov.compute_covariance(epochs=epochs, tmin=baseline[0], tmax=baseline[1], method="shrunk", rank=dict(mag=rank))
                # Save
                os.makedirs(cov_save_path, exist_ok=True)
                baseline_cov.save(fname=cov_save_path + cov_baseline_fname, overwrite=True)

            try:
                # Load covariance matrix
                active_cov = mne.read_cov(fname=cov_save_path + cov_act_fname)
            except:
                # Compute covariance matrices
                active_cov = mne.cov.compute_covariance(epochs=epochs, tmin=active_times[0], tmax=active_times[1], method="shrunk", rank=dict(mag=rank))
                # Save
                os.makedirs(cov_save_path, exist_ok=True)
                active_cov.save(fname=cov_save_path + cov_act_fname, overwrite=True)

            # Compute sources and apply baseline
            stc_base = mne.beamformer.apply_lcmv_cov(baseline_cov, filters)
            stc_act = mne.beamformer.apply_lcmv_cov(active_cov, filters)
            stc = stc_act / stc_base
            stc.data = 10 * np.log10(stc.data)

        elif not estimate_epochs:
            # Apply filter and get source estimates
            stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

            if source_power:
                # Compute envelope in source space
                data = stc.data
                if band_id and not filter_sensors:
                    # Filter source data
                    data = functions_analysis.butter_bandpass_filter(data, band_id=band_id, sfreq=evoked.info['sfreq'], order=3)
                # Compute envelope
                analytic_signal = hilbert(data, axis=-1)
                signal_envelope = np.abs(analytic_signal)
                # Save envelope as data
                stc.data = signal_envelope

        elif not estimate_covariance:
            # Define sources estimated on epochs
            stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

            # Define stc object
            stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

            # Set data as zero to average epochs
            stc.data = np.zeros(shape=(stc.data.shape))
            for stc_epoch in stc_epochs:
                data = stc_epoch.data
                if source_power:
                    # Compute source power on epochs and average
                    if band_id and not filter_sensors:
                        # Filter source data
                        data = functions_analysis.butter_bandpass_filter(data, band_id=band_id, sfreq=evoked.info['sfreq'], order=3)
                    # Compute envelope
                    analytic_signal = hilbert(data, axis=-1)
                    signal_envelope = np.abs(analytic_signal)
                    # Sum data of every epoch
                    stc.data += signal_envelope
                else:
                    stc.data += data
                # Divide by epochs number
                stc.data /= len(epochs)

        # Drop edges due to artifacts from power computation
        stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.tmax - plot_edge)

        if bline_mode_subj and not estimate_covariance:
            # Apply baseline correction
            print(f'Applying baseline correction: {bline_mode_subj} from {baseline[0]} to {baseline[1]}')
            # stc.apply_baseline(baseline=baseline)  # mean
            if bline_mode_subj == 'db':
                stc.data = 10 * np.log10(stc.data / stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None])
            elif bline_mode_subj == 'ratio':
                stc.data = stc.data / stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]
            elif bline_mode_subj == 'mean':
                stc.data = stc.data - stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]

        if band_id and source_power and not estimate_covariance:
            # Filter higher frequencies than corresponding to nyquist of bandpass filter higher freq
            l_freq, h_freq = functions_general.get_freq_band(band_id=band_id)
            stc.data = functions_analysis.butter_lowpass_filter(data=stc.data, h_freq=h_freq/2, sfreq=evoked.info['sfreq'], order=3)

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

            # Define morph function
            morph = mne.compute_source_morph(src=src, subject_from=subject_code, subject_to=default_subject, src_to=src_fs, subjects_dir=subjects_dir)

            # Apply morph
            stc_fs = morph.apply(stc)

            # Append to fs_stcs to make GA
            stcs_fs_dict[epoch_ids[i]].append(stc_fs)

        else:
            # Append to fs_stcs to make GA
            stcs_fs_dict[epoch_ids[i]].append(stc)

        # Plot
        if plot_individuals:
            if surf_vol == 'volume':
                fname = f'{subject.subject_id}'
                if subject_code == 'fsaverage':
                    fname += '_fsaverage'

                # 3D plot
                brain = stc.plot_3d(src=src, subject=subject_code, subjects_dir=subjects_dir, hemi='both', surface='white', clim=clim_3d,
                                       spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1000, 500))
                if save_fig:
                    # brain.show_view(azimuth=-90)
                    os.makedirs(fig_path + '/svg/', exist_ok=True)
                    brain.save_image(filename=fig_path + fname + '_3D.png')
                    brain.save_image(filename=fig_path + '/svg/' + fname + '_3D.pdf')

                # Nutmeg
                fig = stc.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, initial_time=initial_time, clim=clim_nutmeg)
                # Save figure
                if save_fig:
                    save.fig(fig=fig, path=fig_path, fname=fname)

            elif surf_vol == 'surface':
                # 3D plot
                brain = stc.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, hemi='split', clim=clim_3d,
                                 spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1000, 500))
                # brain.add_annotation('aparc.DKTatlas40')
                if save_fig:
                    fname = f'{subject.subject_id}'
                    if subject_code == 'fsaverage':
                        fname += '_fsaverage'
                    # brain.show_view(azimuth=-90)
                    os.makedirs(fig_path + '/svg/', exist_ok=True)
                    brain.save_image(filename=fig_path + fname + '_3D.png')
                    brain.save_image(filename=fig_path + '/svg/' + fname + '_3D.pdf')

    # Grand Average: Average evoked stcs from this epoch_id
    source_data_fs = np.zeros(tuple([len(stcs_fs_dict[epoch_ids[i]])] + [size for size in stcs_fs_dict[epoch_ids[i]][0].data.shape]))
    for j, stc in enumerate(stcs_fs_dict[epoch_ids[i]]):
        source_data_fs[j] = stcs_fs_dict[epoch_ids[i]][j].data
    GA_stc_data = source_data_fs.mean(0)

    # Copy Source Time Course from default subject morph to define GA STC
    GA_stc = stc_fs.copy()

    # Reeplace data
    GA_stc.data = GA_stc_data
    GA_stc.subject = default_subject

    # Apply baseline on GA data
    if bline_mode_ga and not estimate_covariance:
        print(f'Applying baseline correction: {bline_mode_ga} from {baseline[0]} to {baseline[1]}')
        # GA_stc.apply_baseline(baseline=baseline)
        if bline_mode_ga == 'db':
            GA_stc.data = 10 * np.log10(GA_stc.data / GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None])
        elif bline_mode_ga == 'ratio':
            GA_stc.data = GA_stc.data / GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]
        elif bline_mode_ga == 'mean':
            GA_stc.data = GA_stc.data - GA_stc.copy().crop(tmin=baseline[0], tmax=baseline[1]).data.mean(axis=1)[:, None]

    # Save GA from epoch id
    GA_stcs[epoch_ids[i]] = GA_stc

    #--------- Plot GA ---------#

    if surf_vol == 'volume':
        # 3D plot
        brain = GA_stc.plot_3d(src=src_fs, subject=default_subject, subjects_dir=subjects_dir,  hemi='both', clim=clim_3d,
                               spacing=f'ico{ico}', initial_time=initial_time, size=(1000, 500))
        # brain.show_view(azimuth=-90)
        if save_fig:
            fname = 'GA_3D'
            if force_fsaverage:
                fname += '_fsaverage'
            os.makedirs(fig_path + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path + fname + '.png')
            brain.save_image(filename=fig_path + '/svg/' + fname + '.pdf')
            if not estimate_covariance:
                brain.save_movie(filename=fig_path + fname + '.mp4', time_dilation=12, framerate=30)

        # Nutmeg plot
        fig = GA_stc.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, initial_time=initial_time)
        if save_fig:
            fname = 'GA'
            if force_fsaverage:
                fname += '_fsaverage'
            save.fig(fig=fig, path=fig_path, fname=fname)

    elif surf_vol == 'surface':
        # 3D plot
        brain = GA_stc.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='split', clim=clim_3d,
                            spacing=f'ico{ico}', time_unit='s', views='lateral', size=(1000, 500))
        if save_fig:
            fname = 'GA_3D'
            if force_fsaverage:
                fname += '_fsaverage'
            # brain.show_view(azimuth=-90)
            os.makedirs(fig_path + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path + fname + '.png')
            brain.save_image(filename=fig_path + '/svg/' + fname + '.pdf')
            if not estimate_covariance:
              brain.save_movie(filename=fig_path + fname + '.mp4', time_dilation=12, framerate=30)


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

    # --------- Plots ---------#
    # 3D Plot
    if surf_vol == 'volume' or surf_vol == 'mixed':
        # Difference Nutmeg plot
        fig = GA_stc_diff.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, initial_time=initial_time)
        if save_fig and surf_vol == 'volume':
            fname = 'GA'
            if force_fsaverage:
                fname += '_fsaverage'
            save.fig(fig=fig, path=fig_path_diff, fname=fname)

        # Difference 3D plot
        brain = GA_stc_diff.copy().plot_3d(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='both', clim=clim_3d,
                                           spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', size=(1000, 500))
        if save_fig:
            fname = 'GA_3D'
            if force_fsaverage:
                fname += '_fsaverage'
            # brain.show_view(azimuth=-90)
            os.makedirs(fig_path_diff + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path_diff + fname + '.png')
            brain.save_image(filename=fig_path_diff + '/svg/' + fname + '.pdf')
            if not estimate_covariance:
                brain.save_movie(filename=fig_path_diff + fname + '.mp4', time_dilation=12, framerate=30)

    elif surf_vol == 'surface':
        # Difference 3D plot
        brain = GA_stc_diff.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='split', clim=clim_3d,
                                 spacing=f'ico{ico}', initial_time=initial_time, time_unit='s', views='lateral', size=(1000, 500))
        if save_fig:
            fname = 'GA_3D'
            if force_fsaverage:
                fname += '_fsaverage'
            # brain.show_view(azimuth=-90)
            os.makedirs(fig_path_diff + '/svg/', exist_ok=True)
            brain.save_image(filename=fig_path_diff + fname + '.png')
            brain.save_image(filename=fig_path_diff + '/svg/' + fname + '.pdf')
            if not estimate_covariance:
                brain.save_movie(filename=fig_path_diff + fname + '.mp4', time_dilation=12, framerate=30)


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
        n_permutations = 1024
        T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(X=source_data_fs,
                                                                                         n_permutations=n_permutations,
                                                                                         adjacency=adjacency_matrix,
                                                                                         n_jobs=4,
                                                                                         threshold=t_thresh)

        # Select the clusters that are statistically significant at p
        p_threshold = 0.01
        good_clusters_idx = np.where(cluster_p_values < p_threshold)[0]
        good_clusters = [clusters[idx] for idx in good_clusters_idx]
        significant_pvalues = [cluster_p_values[idx] for idx in good_clusters_idx]

        if len(good_clusters):
            # Get vertices from source space
            fsave_vertices = [s["vertno"] for s in src_fs]

            # Select clusters for visualization
            stc_all_cluster_vis = summarize_clusters_stc(clu=clu, p_thresh=p_threshold, tstep=GA_stc_diff.tstep, vertices=fsave_vertices,
                                                         subject=default_subject)

            # Get significant clusters
            significance_mask = np.where(stc_all_cluster_vis.data[:, 0] == 0)[0]

            # Mask data
            GA_stc_diff_sig = GA_stc_diff.copy()
            GA_stc_diff_sig.data[significance_mask] = 0

            # --------- Plots ---------#
            # 3D Plot
            if surf_vol == 'volume' or surf_vol == 'mixed':
                # Clusters 3D plot
                brain = GA_stc_diff_sig.plot_3d(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='both', spacing=f'ico{ico}', initial_time=initial_time,
                                                time_unit='s', size=(1000, 500), clim=clim_3d, time_label=str(significant_pvalues))

                if save_fig:
                    if type(t_thresh) == dict:
                        fname = f'Clus_tTFCE_p{p_threshold}_3D'
                    else:
                        fname = f'Clus_t{round(t_thresh, 2)}_p{p_threshold}_3D'
                    if force_fsaverage:
                        fname += '_fsaverage'
                    # brain.show_view(azimuth=-90)
                    os.makedirs(fig_path_diff + '/svg/', exist_ok=True)
                    brain.save_image(filename=fig_path_diff + fname + '.png')
                    brain.save_image(filename=fig_path_diff + '/svg/' + fname + '.pdf')
                    if not estimate_covariance:
                        brain.save_movie(filename=fig_path_diff + fname + '.mp4', time_dilation=12, framerate=30)

                # Clusters Nutmeg plot
                fig = GA_stc_diff_sig.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, initial_time=initial_time)

                if save_fig and surf_vol == 'volume':
                    if type(t_thresh) == dict:
                        fname = f'Clus_t_TFCE_p{p_threshold}'
                    else:
                        fname = f'Clus_t{round(t_thresh, 2)}_p{p_threshold}'
                    if force_fsaverage:
                        fname += '_fsaverage'
                    save.fig(fig=fig, path=fig_path_diff, fname=fname)

            elif surf_vol == 'surface':
                # Clusters 3D plot
                fig = GA_stc_diff_sig.plot(src=src_fs, subject=default_subject, subjects_dir=subjects_dir, hemi='split', spacing=f'ico{ico}', initial_time=initial_time,
                                           time_unit='s', views='lateral', size=(1000, 500), clim=clim_3d, time_label=str(significant_pvalues))
                if save_fig:
                    if type(t_thresh) == dict:
                        fname = f'Clus_t_TFCE_p{p_threshold}_3D'
                    else:
                        fname = f'Clus_t{round(t_thresh, 2)}_p{p_threshold}_3D'
                    if force_fsaverage:
                        fname += '_fsaverage'
                    # fig.show_view(azimuth=-90)
                    os.makedirs(fig_path_diff + '/svg/', exist_ok=True)
                    fig.save_image(filename=fig_path_diff + fname + '.png')
                    fig.save_image(filename=fig_path_diff + '/svg/' + fname + '.pdf')
                    if not estimate_covariance:
                        fig.save_movie(filename=fig_path_diff + fname + '.mp4', time_dilation=12, framerate=30)