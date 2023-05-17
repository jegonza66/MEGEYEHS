import functions_general
import functions_analysis
import load
import mne
import plot_general
import save
import setup
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
from mne.decoding import ReceptiveField

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
chs_id = 'occipital'  # region_hemisphere
# ICA / RAW
use_ica_data = True
epoch_ids = ['tgt_fix', 'it_fix', 'fix_vs', 'blue', 'red']

# Specific run path for saving data and plots
if use_ica_data:
    data_tyype = 'ICA'
else:
    data_type = 'RAW'

# TRF parameters
tmin = -0.2
tmax = 0.6
alpha = None
baseline = (None, -0.05)

tgt_ga = []
it_ga = []
fix_vs_ga = []
blue_ga = []
red_ga = []

fig_path = paths().plots_path() + f'TRF/{epoch_ids}_{tmin}_{tmax}_bline{baseline}_alpha{alpha}/'
plot_edge = 0.1

for subject_code in exp_info.subjects_ids:

    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if use_ica_data:
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        meg_data = load.ica_data(subject=subject)
    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        meg_data = subject.load_preproc_meg_data()

    # Append bad channel from first subject
    meg_data.info['bads'].append('MLO31')

    input_arrays = {}
    for key in epoch_ids:
        path = paths().save_path() + f'TRF/{subject.subject_id}/'
        fname = f'{key}_array.pkl'
        try:
            input_arrays[key] = load.var(file_path=path + fname)
            print(f'Loaded input array for {key}')
        except:
            print(f'Computing input array for {key}...')
            # Make input arrays as 0
            input_array = np.zeros(len(meg_data.times))
            # Get target fixations times
            evt_times = [meg_data.annotations.onset[i] for i, annotation in enumerate(meg_data.annotations.description) if key in annotation]
            # Get target fixations indexes in time array
            evt_idxs, meg_times = functions_general.find_nearest(meg_data.times, evt_times)
            # Set those indexes as 1
            input_array[evt_idxs] = 1
            # Save to all input arrays dictionary
            input_arrays[key] = input_array

            # Save arrays
            save.var(var=input_array, path=path, fname=fname)


    # Concatenate input arrays as one
    model_input = np.array([input_arrays[key] for key in input_arrays.keys()]).T

    # Define mTRF model
    rf = ReceptiveField(tmin, tmax, meg_data.info['sfreq'], estimator=alpha, scoring='corrcoef', verbose=False)

    # Get occipital channels data as array
    picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
    meg_occipital = meg_data.copy().pick(picks)

    meg_data_array = meg_occipital.get_data()
    meg_data_array = meg_data_array.T

    # Fit TRF
    rf.fit(model_input, meg_data_array)

    # get model coeficients as separate responses to target and items
    tgt_trf = rf.coef_[:, 0, :]
    it_trf = rf.coef_[:, 1, :]
    fix_vs_trf = rf.coef_[:, 2, :]
    blue_trf = rf.coef_[:, 3, :]
    red_trf = rf.coef_[:, 4, :]

    # Define evoked objects from arrays of TRF
    tgt_evoked = mne.EvokedArray(data=tgt_trf, info=meg_occipital.info, tmin=tmin, baseline=baseline)
    it_evoked = mne.EvokedArray(data=it_trf, info=meg_occipital.info, tmin=tmin, baseline=baseline)
    fix_vs_evoked = mne.EvokedArray(data=fix_vs_trf, info=meg_occipital.info, tmin=tmin, baseline=baseline)
    blue_evoked = mne.EvokedArray(data=blue_trf, info=meg_occipital.info, tmin=tmin, baseline=baseline)
    red_evoked = mne.EvokedArray(data=red_trf, info=meg_occipital.info, tmin=tmin, baseline=baseline)

    # Append for Grand average
    tgt_ga.append(tgt_evoked)
    it_ga.append(it_evoked)
    fix_vs_ga.append(fix_vs_evoked)
    blue_ga.append(blue_evoked)
    red_ga.append(red_evoked)

    # Plot
    fig = tgt_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge))
    if save_fig:
        # Save
        fig_path_subj = fig_path + f'{subject.subject_id}/'
        fname = f'tgt_{chs_id}'
        save.fig(fig=fig, fname=fname, path=fig_path_subj)

    # Plot
    fig = it_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge))
    if save_fig:
        # Save
        fig_path_subj = fig_path + f'{subject.subject_id}/'
        fname = f'it_{chs_id}'
        save.fig(fig=fig, fname=fname, path=fig_path_subj)

    # Plot
    fig = fix_vs_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge))
    if save_fig:
        # Save
        fig_path_subj = fig_path + f'{subject.subject_id}/'
        fname = f'fix_vs_{chs_id}'
        save.fig(fig=fig, fname=fname, path=fig_path_subj)

    # Plot
    fig = blue_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge))
    if save_fig:
        # Save
        fig_path_subj = fig_path + f'{subject.subject_id}/'
        fname = f'blue_{chs_id}'
        save.fig(fig=fig, fname=fname, path=fig_path_subj)

    # Plot
    fig = red_evoked.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge))
    if save_fig:
        # Save
        fig_path_subj = fig_path + f'{subject.subject_id}/'
        fname = f'red_{chs_id}'
        save.fig(fig=fig, fname=fname, path=fig_path_subj)

# Compute grand average
tgt_grand_avg = mne.grand_average(tgt_ga, interpolate_bads=False)
it_grand_avg = mne.grand_average(it_ga, interpolate_bads=False)
fix_vs_grand_avg = mne.grand_average(fix_vs_ga, interpolate_bads=False)
blue_grand_avg = mne.grand_average(blue_ga, interpolate_bads=False)
red_grand_avg = mne.grand_average(red_ga, interpolate_bads=False)

# Plot
bad_ch_idx = np.where(np.array(tgt_grand_avg.info['ch_names']) == tgt_grand_avg.info['bads'])[0]
plot_times_idx = np.where((tgt_grand_avg.times > tmin + plot_edge) & (tgt_grand_avg.times < tmax - plot_edge))[0]

data = tgt_grand_avg.get_data()[:, plot_times_idx]
ylims = [(np.delete(data, bad_ch_idx, axis=0).min()*1.2)*1e15, (np.delete(data, bad_ch_idx, axis=0).max()*1.2)*1e15]

fig = tgt_grand_avg.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge),
                         ylim=dict(mag=ylims))

if save_fig:
    # Save
    fname = f'tgt_GA_{chs_id}'
    save.fig(fig=fig, fname=fname, path=fig_path)

# Plot
data = it_grand_avg.get_data()[:, plot_times_idx]
ylims = [(np.delete(data, bad_ch_idx, axis=0).min()*1.2)*1e15, (np.delete(data, bad_ch_idx, axis=0).max()*1.2)*1e15]
fig = it_grand_avg.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge),
                         ylim=dict(mag=ylims))

if save_fig:
    # Save
    fname = f'it_GA_{chs_id}'
    save.fig(fig=fig, fname=fname, path=fig_path)

# Plot
data = fix_vs_grand_avg.get_data()[:, plot_times_idx]
ylims = [(np.delete(data, bad_ch_idx, axis=0).min()*1.2)*1e15, (np.delete(data, bad_ch_idx, axis=0).max()*1.2)*1e15]
fig = fix_vs_grand_avg.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge),
                         ylim=dict(mag=ylims))
fig.tight_layout()
if save_fig:
    # Save
    fname = f'fix_vs_GA_{chs_id}'
    save.fig(fig=fig, fname=fname, path=fig_path)

# Plot
data = blue_grand_avg.get_data()[:, plot_times_idx]
ylims = [(np.delete(data, bad_ch_idx, axis=0).min()*1.2)*1e15, (np.delete(data, bad_ch_idx, axis=0).max()*1.2)*1e15]
fig = blue_grand_avg.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge),
                         ylim=dict(mag=ylims))

if save_fig:
    # Save
    fname = f'blue_GA_{chs_id}'
    save.fig(fig=fig, fname=fname, path=fig_path)

# Plot
data = red_grand_avg.get_data()[:, plot_times_idx]
ylims = [(np.delete(data, bad_ch_idx, axis=0).min()*1.2)*1e15, (np.delete(data, bad_ch_idx, axis=0).max()*1.2)*1e15]
fig = red_grand_avg.plot(spatial_colors=True, gfp=True, show=display_figs, xlim=(tmin+plot_edge, tmax-plot_edge),
                         ylim=dict(mag=ylims))

if save_fig:
    # Save
    fname = f'red_GA_{chs_id}'
    save.fig(fig=fig, fname=fname, path=fig_path)