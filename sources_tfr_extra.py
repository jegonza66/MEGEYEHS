import functions_analysis
import load
import mne
import os
import save
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np


#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# Select channels
chs_id = 'parietal_occipital'
# ICA / RAW
use_ica_data = True
corr_ans = None
tgt_pres = None
mss = None
epoch_id = 'ms'
trial_dur = None
evt_dur = None

# Time frequency params
l_freq = 8
h_freq = 12
log_bands = False
n_cycles_div = 2.

# Baseline method
# logratio: dividing by the mean of baseline values and taking the log
# ratio: dividing by the mean of baseline values
# mean: subtracting the mean of baseline values
bline_mode = 'logratio'

# Source model config
sources_from_tfr = True
default_subject = exp_info.subjects_ids[0]

surf_vol = 'surface'
ico = 4
spacing = 10.
# Freesurfer directory
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')


#----------#

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 0}
cross1_dur = 0.75
cross2_dur = 1
vs_dur = 4
if 'ms' in epoch_id:
    dur = mss_duration[mss] + cross2_dur + vs_dur
elif 'cross2' in epoch_id:
    dur = cross2_dur + vs_dur  # seconds
else:
    dur = 0

# freqs type
if log_bands:
    freqs_type = 'log'
else:
    freqs_type = 'lin'

if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)

for mssh, mssl in [(4, 1), (4, 2), (2, 1)]:

    tmin_mssl = -cross1_dur
    tmax_mssl = mss_duration[mssl] + cross2_dur + vs_dur
    tmin_mssh = -cross1_dur
    tmax_mssh = mss_duration[mssh] + cross2_dur + vs_dur
    baseline = (tmin_mssl, 0)

    # Specific run path for loading and saving data
    mss_diff_name = f'{mssh}-{mssl}'
    run_path_mssl = f'/{epoch_id}_mss{mssl}_Corr{corr_ans}_tgt{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}_{tmin_mssl}_{tmax_mssl}_bline{baseline}'
    run_path_mssh = f'/{epoch_id}_mss{mssh}_Corr{corr_ans}_tgt{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}_{tmin_mssh}_{tmax_mssh}_bline{baseline}'
    run_path_diff = f'/{epoch_id}_mss{mss_diff_name}_Corr{corr_ans}_tgt{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}_{tmin_mssl}_{tmax_mssl}_bline{baseline}_cyc{round(n_cycles_div, 1)}/'

    # Data paths
    # Sensor space TFR
    sensor_tfr_path_mssl =  paths().save_path() + f'Time_Frequency_Epochs_{data_type}/' + run_path_mssl + f'_cyc{round(n_cycles_div, 1)}/'
    sensor_tfr_path_mssh =  paths().save_path() + f'Time_Frequency_Epochs_{data_type}/' + run_path_mssh + f'_cyc{round(n_cycles_div, 1)}/'
    sensor_tfr_diff_save_path =  paths().save_path() + f'Time_Frequency_Epochs_{data_type}/' + run_path_diff

    # Source space TFR
    source_tfr_path_mssl = paths().save_path() + f'Source_Space_TFR_{data_type}/' + run_path_mssl + f'_cyc{round(n_cycles_div, 1)}/'
    source_tfr_path_mssh = paths().save_path() + f'Source_Space_TFR_{data_type}/' + run_path_mssh + f'_cyc{round(n_cycles_div, 1)}/'
    source_tfr_diff_save_path = paths().save_path() + f'Source_Space_TFR_{data_type}/' + run_path_diff

    # Epochs
    epochs_path_mssl = paths().save_path() + f'Epochs_{data_type}/Band_None/' + run_path_mssl + '/'
    epochs_path_mssh = paths().save_path() + f'Epochs_{data_type}/Band_None/' + run_path_mssh + '/'

    # Save figures paths
    # Sensor space
    trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/' + run_path_diff + f'/{chs_id}/'
    # Source space
    source_fig_path = paths().plots_path() + f'Source_Space_TFR_{data_type}/' + run_path_diff +'/'

    # Grand average data variable
    grand_avg_power_ms_fname = f'Grand_Average_power_ms_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_itc_ms_fname = f'Grand_Average_itc_ms_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_power_cross2_fname = f'Grand_Average_power_cross2_{l_freq}_{h_freq}_tfr.h5'
    grand_avg_itc_cross2_fname = f'Grand_Average_itc_cross2_{l_freq}_{h_freq}_tfr.h5'

    # Grand Average
    try:
        raise(ValueError)
        # Load previous power data
        grand_avg_power_ms_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_power_ms_fname)[0]
        grand_avg_power_cross2_diff = mne.time_frequency.read_tfrs(trf_diff_save_path + grand_avg_power_cross2_fname)[0]

        # Pick plot channels
        picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power_ms_diff.info)

    except:

        # Variables to store stcs for GA computation
        stcs_fs_dict = {}
        stcs_diff = []

        for subject_code in exp_info.subjects_ids:
            # Define save path and file name for loading and saving epoched, evoked, and GA data
            if use_ica_data:
                # Load subject object
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            else:
                # Load subject object
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

            # Load data filenames
            sensor_power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
            source_power_data_fname = f'Sources_Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
            epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

            # Subject sensor plots path
            trf_fig_path_subj = trf_fig_path + f'{subject.subject_id}/'

            try:
                # Load difference source data
                source_power_diff_ms = mne.read_source_estimate(source_tfr_diff_save_path + 'ms_' + source_power_data_fname)
                source_power_diff_cross2 = mne.read_source_estimate(source_tfr_diff_save_path + 'cross2_' + source_power_data_fname)
            except:
                try:
                    # Load previous source data
                    source_power_mssl = mne.read_source_estimate(sources_tfr_path_mssl + source_power_data_fname)
                    source_power_mssh = mne.read_source_estimate(sources_tfr_path_mssh + source_power_data_fname)
                except:

                    try:
                        # Load previous sensor data
                        sensor_power_mssl = mne.time_frequency.read_tfrs(sensor_tfr_path_mssl + sensor_power_data_fname)[0]
                        sensor_power_mssh = mne.time_frequency.read_tfrs(sensor_tfr_path_mssl + sensor_power_data_fname)[0]

                    except:
                        # Compute power using from epoched data
                        for mss, tmin, tmax, epochs_path, trf_save_path in zip((mssl, mssh), (tmin_mssl, tmin_mssh),
                                                                               (tmax_mssl, tmax_mssh),
                                                                               (epochs_path_mssl, epochs_path_mssh),
                                                                               (sensor_tfr_path_mssl, sensor_tfr_path_mssh)):
                            try:
                                # Load epoched data
                                epochs = mne.read_epochs(epochs_path + epochs_data_fname)
                            except:
                                # Compute epochs
                                if use_ica_data:
                                    # Load meg data
                                    meg_data = load.ica_data(subject=subject)
                                else:
                                    # Load meg data
                                    meg_data = subject.load_preproc_meg_data()

                                # Epoch data
                                epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans,
                                                                               tgt_pres=tgt_pres, epoch_id=epoch_id, meg_data=meg_data,
                                                                               tmin=tmin, tmax=tmax, save_data=save_data,
                                                                               epochs_save_path=epochs_path,
                                                                               epochs_data_fname=epochs_data_fname)

                            # Compute power and PLI over frequencies and save
                            sensor_power = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq,
                                                                             h_freq=h_freq,
                                                                             freqs_type=freqs_type,
                                                                             n_cycles_div=n_cycles_div,
                                                                             average=False, return_itc=False,
                                                                             output='complex', save_data=True,
                                                                             trf_save_path=trf_save_path,
                                                                             power_data_fname=sensor_power_data_fname,
                                                                             n_jobs=4)
                            # Free up memory
                            del sensor_power

                        # Crop power
                        # mssl
                        # Load previous data
                        sensor_power_mssl = mne.time_frequency.read_tfrs(sensor_tfr_path_mssl + sensor_power_data_fname)[0]
                        # Crop
                        mssl_ms_window_times = (-cross1_dur, mss_duration[mssl])
                        mss_cross2_window_times = (mss_duration[mss], mss_duration[mss] + cross2_dur)
                        sensor_power_base_mssl = sensor_power_mssl.copy().crop(tmin=None, tmax=cross1_dur)
                        sensor_power_ms_mssl = sensor_power_mssl.copy().crop(tmin=mssl_ms_window_times[0],  tmax=mssl_ms_window_times[1])
                        sensor_power_cross2_mssl = sensor_power_mssl.copy().crop(tmin=mss_cross2_window_times[0],  tmax=mss_cross2_window_times[1])

                        # mssh
                        # Load previous data
                        sensor_power_mssh = mne.time_frequency.read_tfrs(sensor_tfr_path_mssh + sensor_power_data_fname)[0]
                        # Crop
                        mssh_ms_window_times = (-cross1_dur, mss_duration[mssh])
                        sensor_power_base_mssh = sensor_power_mssh.copy().crop(tmin=None, tmax=cross1_dur)
                        sensor_power_ms_mssh = sensor_power_mssh.copy().crop(tmin=mssh_ms_window_times[0], tmax=mssh_ms_window_times[1])
                        sensor_power_cross2_mssh = sensor_power_mssh.copy().crop(tmin=mss_cross2_window_times[0], tmax=mss_cross2_window_times[1])


                        #--------------- Compute Sources ---------------#
                        for power_all, power_base, power_ms, power_cross2 in zip((sensor_power_mssl, sensor_power_mssh),
                                                                                 (sensor_power_base_mssl, sensor_power_base_mssh),
                                                                                 (sensor_power_ms_mssl, sensor_power_ms_mssh),
                                                                                 (sensor_power_cross2_mssl, sensor_power_cross2_mssh)):

                            # Check if subject has MRI data
                            try:
                                fs_subj_path = os.path.join(subjects_dir, subject.subject_id)
                                os.listdir(fs_subj_path)
                            except:
                                subject_code = 'fsaverage'

                            sources_path_subject = paths().sources_path() + subject_code

                            # Load forward model
                            if surf_vol == 'volume':
                                fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
                            elif surf_vol == 'surface':
                                fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
                            elif surf_vol == 'mixed':
                                fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
                            fwd = mne.read_forward_solution(fname_fwd)
                            src = fwd['src']

                            # rank
                            rank = sum([ch_type == 'mag' for ch_type in power_cross2.get_channel_types()]) - len(
                                power_cross2.info['bads'])
                            if use_ica_data:
                                rank -= len(subject.ex_components)

                            # Compute CSD
                            power_all.apply_baseline(baseline=baseline)
                            csd = mne.time_frequency.csd_tfr(power_all)
                            noise_csd = functions_analysis.noise_csd(exp_info=exp_info, subject=subject,
                                                                     bads=power.info['bads'], use_ica_data=use_ica_data,
                                                                     freqs=freqs)

                            # Compute scalar DICS beamfomer
                            filters = mne.beamformer.make_dics(
                                info=power.info,
                                forward=fwd,
                                csd=csd,
                                noise_csd=noise_csd,
                                pick_ori=None,
                                rank=dict(mag=rank),
                                real_filter=True
                            )

                            # Compute cropped csds
                            csd_baseline = mne.time_frequency.csd_tfr(power_base)
                            csd_ms = mne.time_frequency.csd_tfr(power_ms)
                            csd_cross2 = mne.time_frequency.csd_tfr(power_cross2)

                            # project the TFR for each epoch to source space
                            stc_base = mne.beamformer.apply_dics_tfr_epochs(power_base, filters, return_generator=True)
                            stc_ms = mne.beamformer.apply_dics_tfr_epochs(power_ms, filters, return_generator=True)
                            stc_cross2 = mne.beamformer.apply_dics_tfr_epochs(power_cross2, filters, return_generator=True)

                            # average across frequencies and epochs
                            data = np.zeros((fwd["nsource"], power_cross2.times.size))
                            for epoch_stcs in stcs_cross2:
                                for stc_cross2 in epoch_stcs:
                                    data += (stc_cross2.data * np.conj(stc_cross2.data)).real

                            stc_cross2.data = data / len(power_cross2) / len(power_cross2.freqs)

                            # average across frequencies and epochs
                            data_base = np.zeros((fwd["nsource"], power_base.times.size))
                            for epoch_stcs in stcs_base:
                                for stc_base in epoch_stcs:
                                    data_base += (stc_base.data * np.conj(stc_base.data)).real

                            stc_base.data = data_base / len(power_base) / len(power_base.freqs)

                            # Apply baseline (logratio)
                            stc = stc_cross2.copy()
                            if bline_mode == 'logratio':
                                stc.data = np.log(stc_cross2.data / stc_base.data.mean(axis=1)[:, None])
                            elif bline_mode == 'ratio':
                                stc = stc_cross2.data / stc_base.data.mean(axis=1)[:, None]
                            elif bline_mode == 'mean':
                                stc = stc_cross2.data - stc_base.data.mean(axis=1)[:, None]

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

                                # Load morph map
                                morph_save_path = fname_src.replace(default_subject + '/', f'morphs/{subject_code}_to_')[:-8] + '-morph.h5'
                                try:
                                    morph = mne.read_source_morph(morph_save_path)
                                except:
                                    # Define morph function
                                    morph = mne.compute_source_morph(src=src, subject_from=subject_code,
                                                                     subject_to=default_subject,
                                                                     src_to=src_fs, subjects_dir=subjects_dir)
                                    if save_data:
                                        os.makedirs(morph_save_path.split(f'{subject_code}_to_')[0], exist_ok=True)
                                        morph.save(fname=morph_save_path, overwrite=True)

                                # Apply Morph
                                stc_fs = morph.apply(stc)

                                # Append to fs_stcs to make GA
                                stcs_fs[mss].append(stc_fs)

                            else:
                                # Append to fs_stcs to make GA
                                stcs_fs[mss].append(stc)

                            # Plot
                            if surf_vol == 'surface':
                                message = f"DICS source power in the {l_freq}-{h_freq} Hz frequency band"
                                # 3D plot
                                brain = stc.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, hemi='both',
                                                 spacing=f'ico{ico}', time_unit='s', smoothing_steps=7, time_label=message, views='parietal')
                                if save_fig:
                                    fname = f'{subject.subject_id}'
                                    if subject_code == 'fsaverage':
                                        fname += '_fsaverage'
                                    brain.show_view(azimuth=-90)
                                    os.makedirs(source_fig_path, exist_ok=True)
                                    brain.save_image(filename=source_fig_path + fname + '.png')

                            elif surf_vol == 'volume':
                                fname = f'{subject.subject_id}'
                                if subject_code == 'fsaverage':
                                    fname += '_fsaverage'

                                # 3D plot
                                brain = stc.plot_3d(src=fwd['src'], subject=subject_code, subjects_dir=subjects_dir)
                                if save_fig:
                                    brain.show_view(azimuth=-90)
                                    os.makedirs(source_fig_path, exist_ok=True)
                                    brain.save_image(filename=source_fig_path + fname + '.png')

                                # Nutmeg plot
                                fig =  stc.plot(src=fwd['src'], subject=subject_code, subjects_dir=subjects_dir)
                                if save_fig:
                                    save.fig(fig=fig, path=source_fig_path, fname=fname)


                            # --------------- End extra DICS sources ------------------#

                        # Crop times
                        mss_cross2_window_times = (mss_duration[mss], mss_duration[mss] + cross2_dur)
                        mssl_ms_window_times = (-cross1_dur, mss_duration[mssl])
                        mssh_ms_window_times = (-cross1_dur, mss_duration[mssh])

                        # --------------- Compute Sources ---------------#
                        for mss, sensor_tfr_path, ms_window_times in zip([mssl, mssh],
                                                                         [sensor_tfr_path_mssl, sensor_tfr_path_mssh],
                                                                         [mssl_ms_window_times, mssh_ms_window_times]):

                            power = mne.time_frequency.read_tfrs(sensor_tfr_path + sensor_power_data_fname)[0]

                            # Check if subject has MRI data
                            try:
                                fs_subj_path = os.path.join(subjects_dir, subject.subject_id)
                                os.listdir(fs_subj_path)
                            except:
                                subject_code = 'fsaverage'

                            sources_path_subject = paths().sources_path() + subject_code

                            # Load forward model
                            if surf_vol == 'volume':
                                fname_fwd = sources_path_subject + f'/{subject_code}_volume_ico{ico}_{int(spacing)}-fwd.fif'
                            elif surf_vol == 'surface':
                                fname_fwd = sources_path_subject + f'/{subject_code}_surface_ico{ico}-fwd.fif'
                            elif surf_vol == 'mixed':
                                fname_fwd = sources_path_subject + f'/{subject_code}_mixed_ico{ico}_{int(spacing)}-fwd.fif'
                            fwd = mne.read_forward_solution(fname_fwd)
                            src = fwd['src']

                            # Rank of covariance matrix to use in source filter
                            rank = sum([ch_type == 'mag' for ch_type in power.get_channel_types()]) - len(
                                power.info['bads'])
                            if use_ica_data:
                                rank -= len(subject.ex_components)

                            # Compute CSD
                            # power.apply_baseline(baseline=baseline, mode=bline_mode)
                            power.apply_baseline(baseline=baseline)
                            # csd_baseline = mne.time_frequency.csd_tfr(power, tmax=0)
                            csd = mne.time_frequency.csd_tfr(power)
                            noise_csd = functions_analysis.noise_csd(exp_info=exp_info, subject=subject,
                                                                     bads=power.info['bads'], use_ica_data=use_ica_data,
                                                                     freqs=freqs)

                            # Compute scalar DICS beamfomer
                            filters = mne.beamformer.make_dics(
                                info=power.info,
                                forward=fwd,
                                csd=csd,
                                noise_csd=noise_csd,
                                pick_ori=None,
                                rank=dict(mag=rank),
                                real_filter=True
                            )

                            # Project the TFR for each epoch to source space
                            stcs = mne.beamformer.apply_dics_tfr_epochs(power, filters, return_generator=True)

                            # Average across frequencies and epochs
                            data = np.zeros((fwd["nsource"], power.times.size))
                            for epoch_stcs in stcs:
                                for stc in epoch_stcs:
                                    data += (stc.data * np.conj(stc.data)).real

                            stc.data = data / len(power) / len(power.freqs)

                            # Apply baseline correction
                            stc.apply_baseline(baseline=baseline)

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

                                # Load morph map
                                morph_save_path = fname_src.replace(subject_code + '/', 'morphs/')[:-8].replace(
                                    subject_code, subject_code + f'_to_{default_subject}')
                                try:
                                    mne.read_source_morph(morph_save_path)
                                except:
                                    # Define morph function
                                    morph = mne.compute_source_morph(src=src, subject_from=subject_code,
                                                                     subject_to=default_subject,
                                                                     src_to=src_fs, subjects_dir=subjects_dir)
                                    if save_data:
                                        morph.save(fname=morph_save_path, overwrite=True)

                                # Apply Morph
                                stc_fs = morph.apply(stc)

                                # Append to fs_stcs to make GA
                                stcs_fs[mss].append(stc_fs)

                            else:
                                # Append to fs_stcs to make GA
                                stcs_fs[mss].append(stc)

                            # Plot
                            if surf_vol == 'surface':
                                message = f"DICS source power in the {l_freq}-{h_freq} Hz frequency band"
                                # 3D plot
                                brain = stc_ms.plot(src=src, subject=subject_code, subjects_dir=subjects_dir, hemi='both',
                                                 spacing=f'ico{ico}', time_unit='s', smoothing_steps=7,
                                                 time_label=message, views='parietal')
                                if save_fig:
                                    fname = f'{subject.subject_id}'
                                    if subject_code == 'fsaverage':
                                        fname += '_fsaverage'
                                    brain.show_view(azimuth=-90)
                                    os.makedirs(source_fig_path, exist_ok=True)
                                    brain.save_image(filename=source_fig_path + fname + '.png')

                            elif surf_vol == 'volume':
                                fname = f'{subject.subject_id}'
                                if subject_code == 'fsaverage':
                                    fname += '_fsaverage'

                                # 3D plot
                                brain = stc.plot_3d(src=fwd['src'], subject=subject_code, subjects_dir=subjects_dir, hemi='both')
                                if save_fig:
                                    brain.show_view(azimuth=-90)
                                    os.makedirs(source_fig_path, exist_ok=True)
                                    brain.save_image(filename=source_fig_path + fname + '.png')

                                # Nutmeg plot
                                fig = stc.plot(src=fwd['src'], subject=subject_code, subjects_dir=subjects_dir)
                                if save_fig:
                                    save.fig(fig=fig, path=source_fig_path, fname=fname)

## Taken from source estimation

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
from scipy.signal import hilbert
import itertools

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

#----- Parameters -----#
# Trial selection
trial_params = {'epoch_id': 'tgt_fix_vs',  # use'+' to mix conditions (red+blue)
                'corrans': True,
                'tgtpres': True,
                'mss': [1, 2, 4],
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evtdur': None}

meg_params = {'regions_id': 'parietal_occipital',
              'band_id': None,
              'filter_sensors': False,
              'filter_method': 'iir',
              'data_type': 'ICA'
              }

# TRF parameters
trf_params = {'input_features': ['tgt_fix_vs', 'it_fix_vs_subsampled', 'blue', 'red'],   # Select features (events)
              'standarize': False,
              'fit_power': False,
              'alpha': None,
              'tmin': -0.3,
              'tmax': 0.6,
              'baseline': (-0.3, -0.05)
              }

l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])
l_freq, h_freq = (1, 40)

# Compare features
run_comparison = True

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'surface'
ico = 5
spacing = 5.  # Only for volume source estimation
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
source_estimation = 'epo'  # 'epo' / 'evk' / 'cov' / 'trf'
estimate_source_tf = True
visualize_alignment = False

# Baseline
bline_mode_subj = 'db'
bline_mode_ga = 'mean'
plot_edge = 0.15

# Plot
initial_time = 0.1
difference_initial_time = [0.25, 0.35]
positive_cbar = None  # None for free determination, False to include negative values
plot_individuals = True
plot_ga = True

# Permutations test
run_permutations_GA = True
run_permutations_diff = True
desired_tval = 0.01
p_threshold = 0.05
mask_negatives = False


#--------- Setup ---------#

# Windows durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()

# Adapt to hfreq model
if meg_params['band_id'] == 'HGamma' or ((isinstance(meg_params['band_id'], list) or isinstance(meg_params['band_id'], tuple)) and meg_params['band_id'][0] > 40):
    model_name = 'hfreq-' + model_name


# --------- Freesurfer Path ---------#
# Define Subjects_dir as Freesurfer output folder
subjects_dir = os.path.join(paths().mri_path(), 'FreeSurfer_out')
os.environ["SUBJECTS_DIR"] = subjects_dir

# Get Source space for default subject
if surf_vol == 'volume':
    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_volume_ico{ico}_{int(spacing)}-src.fif'
elif surf_vol == 'surface':
    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_surface_ico{ico}-src.fif'
elif surf_vol == 'mixed':
    fname_src = paths().sources_path() + 'fsaverage' + f'/fsaverage_mixed_ico{ico}_{int(spacing)}-src.fif'

src_default = mne.read_source_spaces(fname_src)

# Surface labels id by region
aparc_region_labels = {'occipital': ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine'],
                       'parietal': ['postcentral', 'superiorparietal', 'supramarginal', 'inferiorparietal', 'precuneus'],
                       'temporal': ['inferiortemporal', 'middletemporal', 'superiortemporal', 'bankssts', 'transversetemporal', 'fusiform', 'entorhinal', 'parahippocampal', 'temporalpole'],
                       'frontal': ['precentral', 'caudalmiddlefrontal', 'superiorfrontal', 'rostralmiddlefrontal', 'lateralorbitofrontal', 'parstriangularis', 'parsorbitalis', 'parsopercularis', 'medialorbitofrontal', 'paracentral', 'frontalpole'],
                       'insula': ['insula'],
                       'cingulate': ['isthmuscingulate', 'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate']}


# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}

# Save source estimates time courses on FreeSurfer
stcs_default_dict = {}
GA_stcs = {}

# --------- Run ---------#
for param in param_values.keys():
    stcs_default_dict[param] = {}
    GA_stcs[param] = {}
    for param_value in param_values[param]:

        # Get run parameters from trial params including all comparison between different parameters
        run_params = trial_params
        # Set first value of parameters comparisons to avoid having lists in run params
        if len(param_values.keys()) > 1:
            for key in param_values.keys():
                run_params[key] = param_values[key][0]
        # Set comparison key value
        run_params[param] = param_value

        # Define trial duration for VS screen analysis
        if 'vs' in run_params['epoch_id'] and 'fix' not in run_params['epoch_id'] and 'sac' not in run_params['epoch_id']:
            run_params['trialdur'] = vs_dur[run_params['mss']]
        else:
            run_params['trialdur'] = None  # Change to trial_dur = None to use all trials for no 'vs' epochs

        # Get time windows from epoch_id name
        run_params['tmin'], run_params['tmax'], _ = functions_general.get_time_lims(epoch_id=run_params['epoch_id'], mss=run_params['mss'], plot_edge=plot_edge)

        # Get baseline duration for epoch_id
        # map = dict(sac={'tmin': -0.0, 'tmax': 0.15, 'plot_xlim': (-0.2 + plot_edge, 0.3 - plot_edge)})
        run_params['baseline'], run_params['plot_baseline'] = functions_general.get_baseline_duration(epoch_id=run_params['epoch_id'], mss=run_params['mss'],
                                                                                                      tmin=run_params['tmin'], tmax=run_params['tmax'],
                                                                                                      cross1_dur=cross1_dur, mss_duration=mss_duration,
                                                                                                      cross2_dur=cross2_dur, plot_edge=plot_edge)

        # Paths
        run_path = (f"Band_{meg_params['band_id']}/{run_params['epoch_id']}_mss{run_params['mss']}_corrans{run_params['corrans']}_tgtpres{run_params['tgtpres']}_"
                    f"trialdur{run_params['trialdur']}_evtdur{run_params['evtdur']}_{run_params['tmin']}_{run_params['tmax']}_bline{run_params['baseline']}/")

        # Data paths
        epochs_save_path = paths().save_path() + f"Epochs_{meg_params['data_type']}/" + run_path
        evoked_save_path = paths().save_path() + f"Evoked_{meg_params['data_type']}/" + run_path
        cov_save_path = paths().save_path() + f"Cov_Epochs_{meg_params['data_type']}/" + run_path

        # Source plots paths
        if source_power or source_estimation == 'cov':
            run_path = run_path.replace(f"{run_params['epoch_id']}_", f"{run_params['epoch_id']}_power_")

        # Define path
        if surf_vol == 'volume' or surf_vol == 'mixed':
            fig_path = paths().plots_path() + (f"Source_Space_{meg_params['data_type']}/" + run_path + f"{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_"
                                               f"{pick_ori}_{bline_mode_subj}_{source_estimation}/")
        else:
            fig_path = paths().plots_path() + (f"Source_Space_{meg_params['data_type']}/" + run_path + f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}_"
                                               f"{bline_mode_subj}_{source_estimation}/")

        # Get parcelation labels
        fsaverage_labels = functions_analysis.get_labels(parcelation='aparc', subjects_dir=subjects_dir, surf_vol=surf_vol)

        # Save source estimates time courses on default's subject source space
        stcs_default_dict[param][param_value] = []

        # Iterate over participants
        for subject_code in exp_info.subjects_ids:
            # Load subject
            if meg_params['data_type'] == 'ICA':
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            elif meg_params['data_type'] == 'RAW':
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

            # Plot alignment visualization
            if visualize_alignment:
                plot_general.mri_meg_alignment(subject=subject, subject_code=subject_code, dig=dig, subjects_dir=subjects_dir)

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

            # Get epochs and evoked
            try:
                # Load data
                if source_estimation == 'trf':
                    # Load MEG
                    meg_data = load.meg(subject=subject, meg_params=meg_params, save_data=save_data)

                else:
                    if source_estimation == 'epo' or estimate_source_tf:
                        epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
                        # Pick meg channels for source modeling
                        epochs.pick('mag')
                    evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]
                    # Pick meg channels for source modeling
                    evoked.pick('mag')

                    if source_estimation == 'cov':
                        channel_types = evoked.get_channel_types()
                        bad_channels = evoked.info['bads']

            except:
                # Get epochs
                try:
                    # load epochs
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

                    if source_estimation == 'cov':
                        channel_types = epochs.get_channel_types()
                        bad_channels = epochs.info['bads']
                    else:
                        # Define evoked from epochs
                        evoked = epochs.average()

                        # Save evoked data
                        if save_data:
                            os.makedirs(evoked_save_path, exist_ok=True)
                            evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

                        # Pick meg channels for source modeling
                        evoked.pick('mag')
                        epochs.pick('mag')

                except:
                    # Load MEG
                    meg_data = load.meg(subject=subject, meg_params=meg_params, save_data=save_data)

                    if source_estimation == 'cov':
                        channel_types = meg_data.get_channel_types()
                        bad_channels = meg_data.info['bads']

                    else:
                        # Epoch data
                        epochs, events = functions_analysis.epoch_data(subject=subject, meg_data=meg_data, mss=run_params['mss'], corr_ans=run_params['corrans'], tgt_pres=run_params['tgtpres'],
                                                                       epoch_id=run_params['epoch_id'],  tmin=run_params['tmin'], trial_dur=run_params['trialdur'],
                                                                       tmax=run_params['tmax'], reject=run_params['reject'], baseline=run_params['baseline'],
                                                                       save_data=save_data, epochs_save_path=epochs_save_path,
                                                                       epochs_data_fname=epochs_data_fname)

                        # Define evoked from epochs
                        evoked = epochs.average()

                        # Save evoked data
                        if save_data:
                            os.makedirs(evoked_save_path, exist_ok=True)
                            evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

                        # Pick meg channels for source modeling
                        evoked.pick('mag')
                        epochs.pick('mag')

            # --------- Source estimation ---------#

            # Source TF
            #------------- EXTRACTING LABELS ----------------#
            region_labels = [element for region in meg_params['regions_id'].split('_') for element in aparc_region_labels[region]]

            if surf_vol == 'volume':
                labels_path = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
                labels = mne.get_volume_labels_from_aseg(labels_path, return_colors=False)
                occipital_labels = [label for label in labels for label_id in region_labels if label_id in label]
                used_labels = [labels_path, occipital_labels]

            elif subject_code != 'fsaverage':
                # Get labels for FreeSurfer cortical parcellation
                labels = mne.read_labels_from_annot(subject=subject_code, parc='aparc', subjects_dir=subjects_dir)
                used_labels = [label for label in labels for label_id in region_labels if label.name.startswith(label_id + '-')]

            else:
                labels = fsaverage_labels
                used_labels = [label for label in labels for label_id in region_labels if label.name.startswith(label_id + '-')]

            # Redefine stc epochs to iterate over
            stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

            # Average the source estimates within each label using sign-flips to reduce signal cancellations
            label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=used_labels, src=src, mode='auto', return_generator=False)

            region_source_array = np.array(label_ts)
            freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)
            n_cycles = freqs / 4.
            source_tf_array = mne.time_frequency.tfr_array_morlet(region_source_array, sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles, output='avg_power')
            source_tf = mne.time_frequency.AverageTFR(info=epochs.copy().pick(picks[:region_source_array.shape[1]]).info, data=source_tf_array,
                                                      times=epochs.times, freqs=freqs, nave=len(label_ts))

            # Crop times near edges
            source_tf.crop(tmin=epochs.tmin + plot_edge, tmax=epochs.tmax - plot_edge)

            # Apply baseline
            if bline_mode_subj == 'db':
                source_tf.data = 10 * np.log10(
                    source_tf.data / np.expand_dims(
                        source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1))
            elif bline_mode_subj == 'ratio':
                source_tf.data = source_tf.data / np.expand_dims(
                    source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)
            elif bline_mode_subj == 'mean':
                source_tf.data = source_tf.data - np.expand_dims(
                    source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)




            # Average over channels
            source_tf.plot(combine='mean')






            #------------ MAKING FILTER ON LABELS -----------#

            occipital_labels_id = ['cuneus', 'lateraloccipital', 'lingual', 'pericalcarine']
            if surf_vol == 'volume':
                labels_path = subjects_dir + f'/{subject_code}/mri/aparc+aseg.mgz'
                labels = mne.get_volume_labels_from_aseg(labels_path, return_colors=False)
                occipital_labels = [label for label in labels for occipital_id in occipital_labels_id if occipital_id in label]
                used_labels = [labels_path, occipital_labels]

            elif subject_code != 'fsaverage':
                # Get labels for FreeSurfer cortical parcellation
                labels = mne.read_labels_from_annot(subject=subject_code, parc='aparc', subjects_dir=subjects_dir)
                used_labels = [label for label in labels for occipital_id in occipital_labels_id if label.name.startswith(occipital_id + '-')]

            else:
                labels = fsaverage_labels
                used_labels = [label for label in labels for occipital_id in occipital_labels_id if label.name.startswith(occipital_id + '-')]

            unified_label = used_labels[0]
            for i in range(len(used_labels) - 1):
                unified_label += used_labels[i + 1]

            # Load meg data previously
            if meg_params['data_type'] == 'ICA':
                if meg_params['band_id'] and meg_params['filter_sensors']:
                    meg_data = load.filtered_data(subject=subject, band_id=meg_params['band_id'], save_data=save_data,
                                                  method=meg_params['filter_method'])
                else:
                    meg_data = load.ica_data(subject=subject)
            elif meg_params['data_type'] == 'RAW':
                if meg_params['band_id'] and meg_params['filter_sensors']:
                    meg_data = load.filtered_data(subject=subject, band_id=meg_params['band_id'], use_ica_data=False,
                                                  save_data=save_data, method=meg_params['filter_method'])
                else:
                    meg_data = subject.load_preproc_meg_data()

            # Make lcmv filter
            data_cov = mne.compute_raw_covariance(meg_data, reject=dict(mag=4e-12), rank=None)
            noise_cov = functions_analysis.noise_cov(exp_info=exp_info, subject=subject, bads=meg_data.info['bads'], use_ica_data=True, high_freq=True)
            rank = sum([ch_type == 'mag' for ch_type in meg_data.get_channel_types()]) - len(meg_data.info['bads'])
            if meg_params['data_type'] == 'ICA':
                rank -= len(subject.ex_components)
            filters = mne.beamformer.make_lcmv(meg_data.info, forward=fwd, data_cov=data_cov, reg=0.05, noise_cov=noise_cov, rank=dict(mag=rank), label=unified_label)

            # Redefine stc epochs to iterate over
            stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

            # Average the source estimates within each label using sign-flips to reduce signal cancellations
            label_ts = mne.extract_label_time_course(stcs=stc_epochs, labels=used_labels, src=src, mode='auto', return_generator=False)

            occipital_source_array = np.array(label_ts)
            freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)
            n_cycles = freqs / 4.
            source_tf_array = mne.time_frequency.tfr_array_morlet(occipital_source_array, sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles,
                                                                  output='avg_power')
            source_tf = mne.time_frequency.AverageTFR(info=epochs.copy().pick(epochs.ch_names[:occipital_source_array.shape[1]]).info, data=source_tf_array,
                                                      times=epochs.times, freqs=freqs, nave=len(label_ts))

            source_tf.crop(tmin=epochs.tmin + plot_edge, tmax=epochs.tmax - plot_edge)

            if bline_mode_subj == 'db':
                source_tf.data = 10 * np.log10(
                    source_tf.data / np.expand_dims(
                        source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1))
            elif bline_mode_subj == 'ratio':
                source_tf.data = source_tf.data / np.expand_dims(
                    source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)
            elif bline_mode_subj == 'mean':
                source_tf.data = source_tf.data - np.expand_dims(
                    source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)

            source_tf.plot()


            # ----------- TAKING VOXEL ON SUBJECTS SPACE ------------#
            max_gamma_voxel_pos = np.array([-12, -75, 16])

            # Extract time series from mni coords
            used_voxels_mm = src[0]['rr'][src[0]['inuse'].astype(bool)] * 1000

            voxel_idx = np.argmin(abs(used_voxels_mm - max_gamma_voxel_pos).sum(axis=1))

            max_gamma_voxel = src[0]['vertno'][voxel_idx]

            # Redefine stc epochs to iterate over
            stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

            max_gamma_epochs = []
            for stc_epoch in stc_epochs:
                # epoch_stc = stc_epoch.to_data_frame()
                # voxels_columns = [f'VOL_{max_gamma_voxel}']
                # max_gamma_epoch = epoch_stc.loc[:, epoch_stc.columns.isin(voxels_columns)].values
                # max_gamma_epochs.append(max_gamma_epoch)

                max_gamma_epoch = stc_epoch.data[voxel_idx]
                max_gamma_epochs.append(max_gamma_epoch)

            max_gamma_array = np.array(max_gamma_epochs)
            # max_gamma_array = max_gamma_array.swapaxes(1, 2)
            max_gamma_array = np.expand_dims(max_gamma_array, axis=1)
            # Remove extra dimensions
            max_gamma_array = np.squeeze(max_gamma_array)

            # Compute TF
            freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)
            n_cycles = freqs / 2.
            source_tf_array = mne.time_frequency.tfr_array_morlet(max_gamma_array, sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles, output='avg_power')
            source_tf = mne.time_frequency.AverageTFR(info=epochs.copy().pick(epochs.ch_names[:max_gamma_array.shape[1]]).info, data=source_tf_array, times=epochs.times, freqs=freqs, nave=len(max_gamma_epochs))

            # Drop plot edges
            source_tf.crop(tmin=epochs.tmin + plot_edge, tmax=epochs.tmax - plot_edge)

            # Apply baseline
            if bline_mode_subj == 'db':
                source_tf.data = 10 * np.log10(
                    source_tf.data / np.expand_dims(
                        source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1))
            elif bline_mode_subj == 'ratio':
                source_tf.data = source_tf.data / np.expand_dims(
                    source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)
            elif bline_mode_subj == 'mean':
                source_tf.data = source_tf.data - np.expand_dims(
                    source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)

            # Plot
            fig, ax = plt.subplots(figsize=(15, 5))
            source_tf.plot(axes=ax)
            for t in [-cross2_dur - mss_duration[run_params['mss']], -cross2_dur, 0]:
                ax.vlines(x=t, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--', colors='gray')


            #----------- TAKING VOXEL ON MNI TEMPLATE SPACE (DEPRECATED MORPHING TO MNI TOOK TOO LONG) ------------#
            selected_box = {'x': (20, 40), 'y': (-50, -40), 'z': (5, 25)}
            selected_box = {'x': (-10, 10), 'y': (-90, -70), 'z': (0, 20)}

            # Extract time series from mni coords
            used_voxels_mm = src_default[0]['rr'][src_default[0]['inuse'].astype(bool)] * 1000

            # Get voxels in box
            voxel_idx = np.where((used_voxels_mm[:, 0] > selected_box['x'][0]) & (used_voxels_mm[:, 0] < selected_box['x'][1]) &
                                 (used_voxels_mm[:, 1] > selected_box['y'][0]) & (used_voxels_mm[:, 1] < selected_box['y'][1]) &
                                 (used_voxels_mm[:, 2] > selected_box['z'][0]) & (used_voxels_mm[:, 2] < selected_box['z'][1]))[0]

            selected_voxels = src_default[0]['vertno'][voxel_idx]

            # Redefine stc epochs to iterate over
            stc_epochs = beamformer.apply_lcmv_epochs(epochs=epochs, filters=filters, return_generator=True)

            # Morph to default subject
            if subject_code != 'fsaverage':
                # Define morph function
                morph = mne.compute_source_morph(src=src, subject_from=subject_code, subject_to='fsaverage', src_to=src_default, subjects_dir=subjects_dir)

            selected_voxels_epochs = []
            for stc_epoch in stc_epochs:
                if subject_code != 'fsaverage':
                    # Apply morph
                    stc_default = morph.apply(stc_epoch)
                epoch_stc = stc_default.to_data_frame()
                voxels_columns = [f'VOL_{selected_voxel}' for selected_voxel in selected_voxels]
                selected_voxels_epoch = epoch_stc.loc[:, epoch_stc.columns.isin(voxels_columns)].values
                selected_voxels_epochs.append(selected_voxels_epoch)

            selected_voxels_array = np.array(selected_voxels_epochs)
            selected_voxels_array = selected_voxels_array.swapaxes(1, 2)
            # selected_voxels_array = np.expand_dims(selected_voxels_array, axis=1)
            freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)
            n_cycles = freqs / 4.
            source_tf_array = mne.time_frequency.tfr_array_morlet(selected_voxels_array, sfreq=epochs.info['sfreq'], freqs=freqs, n_cycles=n_cycles, output='avg_power')
            source_tf_array = source_tf_array.mean(0)
            source_tf_array = np.expand_dims(source_tf_array, axis=0)
            source_tf = mne.time_frequency.AverageTFR(info=epochs.copy().pick(epochs.ch_names[:1]).info, data=source_tf_array, times=epochs.times, freqs=freqs, nave=len(selected_voxels_epochs))

            source_tf.crop(tmin=epochs.tmin + plot_edge, tmax=epochs.tmax - plot_edge)


            if bline_mode_subj == 'db':
                source_tf.data = 10 * np.log10(
                    source_tf.data / np.expand_dims(source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1))
            elif bline_mode_subj == 'ratio':
                source_tf.data = source_tf.data / np.expand_dims(source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)
            elif bline_mode_subj == 'mean':
                source_tf.data = source_tf.data - np.expand_dims(source_tf.copy().crop(tmin=run_params['plot_baseline'][0], tmax=run_params['plot_baseline'][1]).data.mean(axis=-1), axis=-1)

            # Save to variable
            # sources_tf[param][param_value].append(source_tf)

            if plot_individuals:
                # Plot
                fig, ax = plt.subplots(figsize=(15, 5))
                source_tf.plot(axes=ax)
                for t in [-cross2_dur - mss_duration[run_params['mss']], -cross2_dur, 0]:
                    ax.vlines(x=t, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='--', colors='gray')

                if save_fig:
                    fname = f'{subject.subject_id}_tf_source_{[coords[1] for coords in list(selected_box.items())]}'
                    save.fig(fig=fig, path=fig_path, fname=fname)



                # Plot on sensor space to compare
                power = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                          n_cycles_div=n_cycles_div, average=True,
                                                          return_itc=False, output='power', save_data=False)

                chs_id = 'occipital'
                fname = f'{subject.subject_id}_tf_sensor'
                fig = plot_general.tfr_plotjoint_picks(tfr=power, chs_id=chs_id, bline_mode='logratio', display_figs=True, plot_xlim=(-0.15, 0.45),
                                                       save_fig=True, trf_fig_path=fig_path, fname=fname)