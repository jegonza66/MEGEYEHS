import functions_analysis
import functions_general
import load
import save
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
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
chs_id = 'mag'  # region_hemisphere (frontal_L)
all_chs_regions = ['frontal', 'temporal', 'central', 'parietal', 'occipital']

mss = None
evt_dur = None
trial_dur = None
corr_ans = True
tgt_pres = True
band_id = None

# ICA / RAW
use_ica_data = True
epoch_ids = ['it_fix_subsampled', 'tgt_fix', 'blue', 'red']
standarize = True

# Specific run path for saving data and plots
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# TRF parameters
tmin = -0.3
tmax = 0.6
alpha = None
baseline = (None, -0.05)

# Define Grand average variables
for var_name in epoch_ids:
    exec(f'{var_name}_ga = []')

plot_edge = 0.1
fig_path = paths().plots_path() + f'TRF_{data_type}/{epoch_ids}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_tdur{trial_dur}' \
                                  f'_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/{chs_id}/'
save_path = fig_path.replace(paths().plots_path(), paths().save_path())

for subject_code in exp_info.subjects_ids[:12]:
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
                                                                          corr_ans=corr_ans, tgt_pres=tgt_pres)

        # Bad annotations filepath
        subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
        fname_bad_annot = f'bad_annot_array.pkl'
        try:
            bad_annotations_array = load.var(subj_path + fname_bad_annot)
            print(f'Loaded bad annotations array')
        except:
            print(f'Computing bad annotations array...')
            bad_annotations_array = functions_analysis.get_bad_annot_array(meg_data=meg_data, subj_path=subj_path,
                                                                           fname=fname_bad_annot)

        input_arrays = {}
        for var_name in epoch_ids:
            subj_path = paths().save_path() + f'TRF/{subject.subject_id}/'
            fname_var = f'{var_name}_mss{mss}_tdur{trial_dur}_evtdur{evt_dur}_array.pkl'
            try:
                input_arrays[var_name] = load.var(file_path=subj_path + fname_var)
                print(f'Loaded input array for {var_name}')
            except:
                print(f'Computing input array for {var_name}...')
                # Exception for subsampled distractor fixations
                if 'subsampled' in var_name:
                    # Subsampled epochs path
                    epochs_save_id = f'{var_name}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}'
                    epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + f'/Band_{band_id}/{epochs_save_id}_{tmin}_{tmax}_bline{baseline}/'
                    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

                    # Load epoched data
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

                    # Get epochs id from metadata
                    epoch_keys = epochs.metadata['id'].to_list()

                else:
                    epoch_keys = None

                input_arrays = functions_analysis.make_mtrf_input(input_arrays=input_arrays, var_name=var_name,
                                                                  subject=subject, meg_data=meg_data, evt_dur=evt_dur,
                                                                  cond_trials=cond_trials, epoch_keys=epoch_keys,
                                                                  bad_annotations_array=bad_annotations_array,
                                                                  subj_path=subj_path, fname=fname_var)

        # Concatenate input arrays as one
        model_input = np.array([input_arrays[key] for key in input_arrays.keys()]).T

        # All regions or selected (multiple) regions
        if chs_id == 'mag' or '_' in chs_id:
            # rf as a dictionary containing the rf of each region
            rf = {}
            # iterate over regions
            for chs_subset in all_chs_regions:
                # Use only regions in channels id, or all in case of chs_id == 'mag'
                if chs_subset in chs_id or chs_id == 'mag':
                    rf[chs_subset] = functions_analysis.fit_mtrf(meg_data=meg_data, tmin=tmin, tmax=tmax, alpha=alpha,
                                                                 model_input=model_input, chs_id=chs_subset, n_jobs=4)
        # One region
        else:
            rf = functions_analysis.fit_mtrf(meg_data=meg_data, tmin=tmin, tmax=tmax, alpha=alpha,
                                             model_input=model_input, chs_id=chs_id, n_jobs=4)
        # Save TRF
        if save_data:
            save.var(var=rf, path=trf_path, fname=trf_fname)

    # Get model coeficients as separate responses to target and items
    for i, var_name in enumerate(epoch_ids):

        # All or multiple regions
        if chs_id == 'mag' or '_' in chs_id:

            # Define evoked from TRF list to concatenate all
            exec(f'{var_name}_evoked_list = []')

            # iterate over regions
            for chs_idx, chs_subset in enumerate(rf.keys()):
                # Get channels subset info
                picks = functions_general.pick_chs(chs_id=chs_subset, info=meg_data.info)
                meg_sub = meg_data.copy().pick(picks)

                # Get TRF coeficients from chs subset
                exec(f'{var_name}_trf = rf[chs_subset].coef_[:, i, :]')

                if chs_idx == 0:
                    # Define evoked object from arrays of TRF
                    exec(f'{var_name}_evoked = mne.EvokedArray(data={var_name}_trf, info=meg_sub.info, tmin=tmin, baseline=baseline)')
                else:
                    # Append evoked object from arrays of TRF to list, to concatenate all
                    exec(f'{var_name}_evoked_list.append(mne.EvokedArray(data={var_name}_trf, info=meg_sub.info, tmin=tmin, baseline=baseline))')

            # Concatenate evoked from al regions
            exec(f'{var_name}_evoked = {var_name}_evoked.add_channels({var_name}_evoked_list)')

        else:
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
corr_ans = None
tgt_pres = None

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
        fname_bad_annot = f'bad_annot_array.pkl'
        try:
            bad_annotations_array = load.var(subj_path + fname_bad_annot)
            print(f'Loaded bad annotations array')
        except:
            print(f'Computing bad annotations array...')

            bad_annotations_array = functions_analysis.get_bad_annot_array(meg_data=meg_data, subj_path=subj_path,
                                                                           fname=fname_bad_annot)

        input_arrays = {}
        for var_name in epoch_ids:
            subj_path = paths().save_path() + f'TRF/{subject_code}/'
            fname_var = f'{var_name}_mss{mss}_tdur{trial_dur}_evtdur{evt_dur}_array.pkl'
            try:
                input_arrays[var_name] = load.var(file_path=subj_path + fname_var)
                print(f'Loaded input array for {var_name}')
            except:
                print(f'Computing input array for {var_name}...')

                # Exception for subsampled distractor fixations
                if 'subsampled' in var_name:
                    # Subsampled epochs path
                    epochs_save_id = f'{var_name}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}'
                    epochs_save_path = paths().save_path() + f'Epochs_{data_type}/' + f'/Band_{band_id}/{epochs_save_id}_{tmin}_{tmax}_bline{baseline}/'
                    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

                    # Load epoched data
                    epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)

                    # Get epochs id from metadata
                    epoch_keys = epochs.metadata['id'].to_list()

                else:
                    epoch_keys = None

                input_arrays = functions_analysis.make_mtrf_input(input_arrays=input_arrays, var_name=var_name,
                                                                  subject=subject, meg_data=meg_data, evt_dur=evt_dur,
                                                                  cond_trials=cond_trials, epoch_keys=epoch_keys,
                                                                  bad_annotations_array=bad_annotations_array,
                                                                  subj_path=subj_path, fname=fname_var)

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