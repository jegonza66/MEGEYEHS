import os
import functions_analysis
import functions_general
import load
import save
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import ReceptiveField
import mne

#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# Select channels
chs_id = 'temporal'  # region_hemisphere (frontal_L)
mss = None
evt_dur = 0.4
trial_dur = None
# ICA / RAW
use_ica_data = True
epoch_ids = ['it_fix', 'tgt_fix', 'blue', 'red']
standarize = True

# Specific run path for saving data and plots
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# TRF parameters
tmin = -0.2
tmax = 0.6
alpha = None
baseline = (None, -0.05)

# Define Grand average variables
for var_name in epoch_ids:
    exec(f'{var_name}_ga = []')

plot_edge = 0.1
fig_path = paths().plots_path() + f'TRF_{data_type}/{epoch_ids}_mss{mss}_tdur{trial_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/{chs_id}/'
save_path = paths().save_path() + f'TRF_{data_type}/{epoch_ids}_mss{mss}_tdur{trial_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/{chs_id}/'

for subject_code in exp_info.subjects_ids:
    trf_path = save_path
    trf_fname = f'TRF_{subject_code}.pkl'
    try:
        # Load TRF
        rf = load.var(trf_path+trf_fname)
        # Load MEG sub
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            meg_data = load.ica_data(subject=subject)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
            meg_data = subject.load_preproc_meg_data()
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
        meg_sub = meg_data.copy().pick(picks)
        print('Loaded Receptive Field')
    except:
        print(f'Computing TRF for {epoch_ids}')
        # Define save path and file name for loading and saving epoched, evoked, and GA data
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            meg_data = load.ica_data(subject=subject)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
            meg_data = subject.load_preproc_meg_data()

        # Get condition trials
        cond_trials, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=mss, trial_dur=trial_dur,
                                                                          corr_ans=None, tgt_pres=None)

        # Bad annotations filepath
        subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
        fname = f'bad_annot_array.pkl'
        try:
            bad_annotations_array = load.var(subj_path + fname)
            print(f'Loaded bad annotations array')
        except:
            print(f'Computing bad annotations array...')
            # Get bad annotations times
            bad_annotations_idx = [i for i, annot in enumerate(meg_data.annotations.description) if
                                   ('bad' in annot or 'BAD' in annot)]
            bad_annotations_time = meg_data.annotations.onset[bad_annotations_idx]
            bad_annotations_duration = meg_data.annotations.duration[bad_annotations_idx]
            bad_annotations_endtime = bad_annotations_time + bad_annotations_duration

            bad_indexes = []
            for i in range(len(bad_annotations_time)):
                bad_annotation_span_idx = np.where(np.logical_and((meg_data.times > bad_annotations_time[i]), (meg_data.times < bad_annotations_endtime[i])))[0]
                bad_indexes.append(bad_annotation_span_idx)

            # Flatten all indexes and convert to array
            bad_indexes = functions_general.flatten_list(bad_indexes)
            bad_indexes = np.array(bad_indexes)

            # make bad annotations binary array
            bad_annotations_array = np.ones(len(meg_data.times))
            bad_annotations_array[bad_indexes] = 0

            # Save arrays
            save.var(var=bad_annotations_array, path=subj_path, fname=fname)

        input_arrays = {}
        for var_name in epoch_ids:
            subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
            fname = f'{var_name}_mss{mss}_tdur{trial_dur}_evtdur{evt_dur}_array.pkl'
            try:
                input_arrays[var_name] = load.var(file_path=subj_path + fname)
                print(f'Loaded input array for {var_name}')
            except:
                print(f'Computing input array for {var_name}...')

                # Define events
                metadata, events, _, _ = functions_analysis.define_events(subject=subject, epoch_id=var_name,  evt_dur=evt_dur,
                                                                       trials=cond_trials, meg_data=meg_data)
                # Make input arrays as 0
                input_array = np.zeros(len(meg_data.times))
                # Get events samples index
                evt_idxs = events[:, 0]
                # evt_times = [meg_data.annotations.onset[i] for i, annotation in enumerate(meg_data.annotations.description) if var_name in annotation]
                # Get target fixations indexes in time array
                # evt_idxs, meg_times = functions_general.find_nearest(meg_data.times, evt_times)
                # Set those indexes as 1
                input_array[evt_idxs] = 1
                # Exclude bad annotations
                input_array = input_array * bad_annotations_array
                # Save to all input arrays dictionary
                input_arrays[var_name] = input_array

                # Save arrays
                save.var(var=input_array, path=subj_path, fname=fname)

        # Concatenate input arrays as one
        model_input = np.array([input_arrays[key] for key in input_arrays.keys()]).T

        # Define mTRF model
        rf = ReceptiveField(tmin, tmax, meg_data.info['sfreq'], estimator=alpha, scoring='corrcoef', verbose=False)

        # Get subset channels data as array
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
        meg_sub = meg_data.copy().pick(picks)
        meg_data_array = meg_sub.get_data()
        if standarize:
            # Standarize data
            print('Computing z-score...')
            meg_data_array = np.expand_dims(meg_data_array, axis=0)  # Need shape (n_epochs, n_channels n_times)
            meg_data_array = mne.decoding.Scaler(info=meg_sub.info, scalings='mean').fit_transform(meg_data_array)
            meg_data_array = meg_data_array.squeeze()
        # Transpose to input the model
        meg_data_array = meg_data_array.T

        # Fit TRF
        rf.fit(model_input, meg_data_array)

        # Save TRF
        if save_data:
            save.var(var=rf, path=trf_path, fname=trf_fname)

    # Get model coeficients as separate responses to target and items
    for i, var_name in enumerate(epoch_ids):
        exec(f'{var_name}_trf = rf.coef_[:, i, :]')
        # Define evoked objects from arrays of TRF
        exec(f'{var_name}_evoked = mne.EvokedArray(data={var_name}_trf, info=meg_sub.info, tmin=tmin, baseline=baseline)')
        # Append for Grand average
        exec(f'{var_name}_ga.append({var_name}_evoked)')
        # Plot
        exec(f'fig = {var_name}_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, '
             f'xlim=(tmin+plot_edge, tmax-plot_edge))')

        if save_fig:
            # Save
            fig_path_subj = fig_path + f'{subject.subject_id}/'
            fname = f'{var_name}_{chs_id}'
            exec('save.fig(fig=fig, fname=fname, path=fig_path_subj)')


bads = []

for var_name in epoch_ids:
    # Compute grand average
    exec(f'{var_name}_grand_avg = mne.grand_average({var_name}_ga, interpolate_bads=False)')
    # Append every subject bad channels
    exec(f'{var_name}_grand_avg.info["bads"] = bads')

    # Calculate max and min plot lims excluding bad channels
    exec(f'bad_ch_idx = np.where(np.array({var_name}_grand_avg.info["ch_names"]) == {var_name}_grand_avg.info["bads"])[0]')
    exec(f'plot_times_idx = np.where(({var_name}_grand_avg.times > tmin + plot_edge) & ({var_name}_grand_avg.times < tmax - plot_edge))[0]')
    exec(f'data = {var_name}_grand_avg.get_data()[:, plot_times_idx]')
    exec(f'ylims = [(np.delete(data, bad_ch_idx, axis=0).min()*1.2)*1e15, (np.delete(data, bad_ch_idx, axis=0).max()*1.2)*1e15]')

    # plot
    exec(f'fig = {var_name}_grand_avg.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge),'
         f' ylim=dict(mag=ylims))')

    if save_fig:
        # Save
        fname = f'{var_name}_GA_{chs_id}'
        exec('save.fig(fig=fig, fname=fname, path=fig_path)')

## mTRF to band power
import functions_analysis
import os
import functions_general
import load
import save
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import ReceptiveField
import mne

#----- Path -----#
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_data = True
save_fig = True
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# Select channels
chs_id = 'parietal'  # region_hemisphere (frontal_L)
mss = None
evt_dur = None
trial_dur = None
# ICA / RAW
use_ica_data = True
band_id = 'Alpha'
epoch_ids = ['vs', 'fix_vs']
standarize = True
# TRF parameters
tmin = -0.3
tmax = 2
alpha = None
baseline = (None, -0.05)
fmin, fmax = functions_general.get_freq_band(band_id=band_id)
# Plot
plot_edge = 0.1


# Data type to use in paths
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Specific run path for saving data and plots
fig_path = paths().plots_path() + f'TRF_{data_type}_ENV/Band_{band_id}/{epoch_ids}_mss{mss}_tdur{trial_dur}_evtdur{evt_dur}' \
                                  f'_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/{chs_id}/'
trf_save_path = paths().save_path() + f'TRF_{data_type}_ENV/Band_{band_id}/{epoch_ids}_mss{mss}_tdur{trial_dur}_evtdur{evt_dur}' \
                                      f'_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/{chs_id}/'
env_save_path = paths().save_path() + f'ENV_{data_type}/Band_{band_id}/'

# Define Grand average variables
for var_name in epoch_ids:
    exec(f'{var_name}_ga = []')

for subject_code in exp_info.subjects_ids:
    trf_path = trf_save_path
    trf_fname = f'TRF_{subject_code}.pkl'
    try:
        # Load TRF
        rf = load.var(trf_path+trf_fname)
        # Load MEG sub
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            power_data = load.ica_data(subject=subject)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
            power_data = subject.load_preproc_meg_data()
        picks = functions_general.pick_chs(chs_id=chs_id, info=power_data.info)
        meg_sub = meg_data.copy().pick(picks)
        print('Loaded Receptive Field')
    except:
        print(f'Computing TRF for {epoch_ids}')
        env_subj_path = env_save_path + subject_code + '/'
        env_fname = f'{subject_code}.fif'
        try:
            # Load Envelope in frequency band
            meg_env = mne.io.read_raw_fif(env_subj_path + env_fname, preload=False)
            if use_ica_data:
                # Load subject object
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            else:
                # Load subject object
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        except:
            if use_ica_data:
                # Load subject object
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
                meg_data = load.ica_data(subject=subject, preload=True)
            else:
                # Load subject object
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
                meg_data = subject.load_preproc_meg_data(preload=True)

            # Band-pass filter
            meg_filt = meg_data.filter(fmin, fmax, n_jobs=None, l_trans_bandwidth=1, h_trans_bandwidth=1)

            # Apply hilbert and extract envelope
            meg_env = meg_data.apply_hilbert(envelope=True)

            # Save
            os.makedirs(env_subj_path, exist_ok=True)
            meg_env.save(env_subj_path + env_fname, overwrite=True)

        # Pick channels
        picks = functions_general.pick_chs(chs_id=chs_id, info=meg_env.info)
        meg_sub = meg_env.copy().pick(picks)

        # Extract data as array
        meg_data_array = meg_sub.get_data()

        if standarize:
            # Standarize data
            print('Computing z-score...')
            meg_data_array = np.expand_dims(meg_data_array, axis=0)  # Need shape (n_epochs, n_channels n_times)
            meg_data_array = mne.decoding.Scaler(info=meg_sub.info, scalings='mean').fit_transform(meg_data_array)
            meg_data_array = meg_data_array.squeeze()
        # Transpose to input the model
        meg_data_array = meg_data_array.T

        # Get condition trials
        cond_trials, bh_data_sub = functions_general.get_condition_trials(subject=subject, mss=mss, trial_dur=trial_dur,
                                                                          corr_ans=None, tgt_pres=None)

        # Bad annotations filepath
        subj_path = paths().save_path() + f'TRF/{subject_code}/'
        fname = f'bad_annot_array.pkl'
        try:
            bad_annotations_array = load.var(subj_path + fname)
            print(f'Loaded bad annotations array')
        except:
            print(f'Computing bad annotations array...')
            # Get bad annotations times
            bad_annotations_idx = [i for i, annot in enumerate(meg_env.annotations.description) if
                                   ('bad' in annot or 'BAD' in annot)]
            bad_annotations_time = meg_env.annotations.onset[bad_annotations_idx]
            bad_annotations_duration = meg_env.annotations.duration[bad_annotations_idx]
            bad_annotations_endtime = bad_annotations_time + bad_annotations_duration

            bad_indexes = []
            for i in range(len(bad_annotations_time)):
                bad_annotation_span_idx = np.where(np.logical_and((meg_env.times > bad_annotations_time[i]),
                                                                  (meg_env.times < bad_annotations_endtime[i])))[0]
                bad_indexes.append(bad_annotation_span_idx)

            # Flatten all indexes and convert to array
            bad_indexes = functions_general.flatten_list(bad_indexes)
            bad_indexes = np.array(bad_indexes)

            # make bad annotations binary array
            bad_annotations_array = np.ones(len(meg_env.times))
            bad_annotations_array[bad_indexes] = 0

            # Save arrays
            save.var(var=bad_annotations_array, path=subj_path, fname=fname)

        input_arrays = {}
        for var_name in epoch_ids:
            subj_path = paths().save_path() + f'TRF/{subject_code}/'
            fname = f'{var_name}_mss{mss}_tdur{trial_dur}_evtdur{evt_dur}_array.pkl'
            try:
                input_arrays[var_name] = load.var(file_path=subj_path + fname)
                print(f'Loaded input array for {var_name}')
            except:
                print(f'Computing input array for {var_name}...')

                # Define events
                metadata, events, _, _ = functions_analysis.define_events(subject=subject, epoch_id=var_name,
                                                                          evt_dur=evt_dur,
                                                                          trials=cond_trials, meg_data=meg_env)
                # Make input arrays as 0
                input_array = np.zeros(len(meg_env.times))
                # Get events samples index
                evt_idxs = events[:, 0]
                # Set those indexes as 1
                input_array[evt_idxs] = 1
                # Exclude bad annotations
                input_array = input_array * bad_annotations_array
                # Save to all input arrays dictionary
                input_arrays[var_name] = input_array

                # Save arrays
                save.var(var=input_array, path=subj_path, fname=fname)

        # Concatenate input arrays as one
        model_input = np.array([input_arrays[key] for key in input_arrays.keys()]).T

        # Define mTRF model
        rf = ReceptiveField(tmin, tmax, meg_sub.info['sfreq'], estimator=alpha, scoring='corrcoef', verbose=False)

        # Fit TRF
        rf.fit(model_input, meg_data_array)

        # Save TRF
        if save_data:
            save.var(var=rf, path=trf_path, fname=trf_fname)

        # Get model coeficients as separate responses to target and items
    for i, var_name in enumerate(epoch_ids):
        exec(f'{var_name}_trf = rf.coef_[:, i, :]')
        # Define evoked objects from arrays of TRF
        exec(
            f'{var_name}_evoked = mne.EvokedArray(data={var_name}_trf, info=meg_sub.info, tmin=tmin, baseline=baseline)')
        # Append for Grand average
        exec(f'{var_name}_ga.append({var_name}_evoked)')
        # Plot
        exec(f'fig = {var_name}_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, '
             f'xlim=(tmin+plot_edge, tmax-plot_edge))')

        if save_fig:
            # Save
            fig_path_subj = fig_path + f'{subject_code}/'
            fname = f'{var_name}_{chs_id}'
            exec('save.fig(fig=fig, fname=fname, path=fig_path_subj)')

bads = []

for var_name in epoch_ids:
    # Compute grand average
    exec(f'{var_name}_grand_avg = mne.grand_average({var_name}_ga, interpolate_bads=False)')
    # Append every subject bad channels
    exec(f'{var_name}_grand_avg.info["bads"] = bads')

    # Calculate max and min plot lims excluding bad channels
    exec(
        f'bad_ch_idx = np.where(np.array({var_name}_grand_avg.info["ch_names"]) == {var_name}_grand_avg.info["bads"])[0]')
    exec(
        f'plot_times_idx = np.where(({var_name}_grand_avg.times > tmin + plot_edge) & ({var_name}_grand_avg.times < tmax - plot_edge))[0]')
    exec(f'data = {var_name}_grand_avg.get_data()[:, plot_times_idx]')
    exec(
        f'ylims = [(np.delete(data, bad_ch_idx, axis=0).min()*1.2)*1e15, (np.delete(data, bad_ch_idx, axis=0).max()*1.2)*1e15]')

    # plot
    exec(
        f'fig = {var_name}_grand_avg.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge),'
        f' ylim=dict(mag=ylims))')

    if save_fig:
        # Save
        fname = f'{var_name}_GA_{chs_id}'
        exec('save.fig(fig=fig, fname=fname, path=fig_path)')