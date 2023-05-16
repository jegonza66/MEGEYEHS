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
chs_id = 'parietal'  # region_hemisphere
# ICA / RAW
use_ica_data = True
epoch_id = 'vs'
corr_ans = None
tgt_pres = None
mss = 4
reject = None  # 'subject' for subject's default. False for no rejection, dict for specific values. None for default 4e-12 for magnetometers
n_cycles_div = 4.
# Power frequency range
l_freq = 1
h_freq = 40
log_bands = False


# Specific run path for saving data and plots
if use_ica_data:
    data_tyype = 'ICA'
else:
    data_type = 'RAW'

# TRF parameters
tmin = -0.2
tmax = 0.5
alpha = None

tgt_ga = []
it_ga = []

fig_path = paths().plots_path() + f'TRF/tgt_it_fix_{tmin}_{tmax}/'

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

    # Make input arrays as 0
    tgt_fix_array = np.zeros(len(meg_data.times))
    it_fix_array = np.zeros(len(meg_data.times))
    # Get target fixations times
    tgt_times = [meg_data.annotations.onset[i] for i, annotation in enumerate(meg_data.annotations.description) if 'tgt_fix' in annotation]
    it_times = [meg_data.annotations.onset[i] for i, annotation in enumerate(meg_data.annotations.description) if 'it_fix' in annotation]
    # Get target fixations indexes in time array
    tgt_idxs, meg_times = functions_general.find_nearest(meg_data.times, tgt_times)
    it_idxs, meg_times = functions_general.find_nearest(meg_data.times, it_times)
    # Set those indexes as 1
    tgt_fix_array[tgt_idxs] = 1
    it_fix_array[it_idxs] = 1

    # Concatenate input arrays as one
    input_array = np.array([tgt_fix_array, it_fix_array]).T

    # Define mTRF model
    rf = ReceptiveField(tmin, tmax, meg_data.info['sfreq'], estimator=alpha, scoring='corrcoef', verbose=False)

    # Get occipital channels data as array
    chs_id = 'occipital'
    picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
    meg_occipital = meg_data.copy().pick(picks)

    meg_data_array = meg_occipital.get_data()
    meg_data_array = meg_data_array.T

    # Fit TRF
    rf.fit(input_array, meg_data_array)

    # get model coeficients as separate responses to target and items
    tgt_trf = rf.coef_[:,0,:]
    it_trf = rf.coef_[:,1,:]

    # Define evoked objects from arrays of TRF
    tgt_evoked = mne.EvokedArray(data=tgt_trf, info=meg_occipital.info, tmin=-0.2, baseline=(None, 0))
    it_evoked = mne.EvokedArray(data=it_trf, info=meg_occipital.info, tmin=-0.2, baseline=(None, 0))
    # tgt_evoked.info['bads']=['MLO31']
    # it_evoked.info['bads']=['MLO31']

    # Append for Grand average
    tgt_ga.append(tgt_evoked)
    it_ga.append(it_evoked)

    # Plot
    fig = tgt_evoked.plot(spatial_colors=True, gfp=True)
    # Save
    fig_path_subj = fig_path + f'{subject.subject_id}/'
    fname = f'tgt_{chs_id}'
    save.fig(fig=fig, fname=fname, path=fig_path_subj)

    # Plot
    fig = it_evoked.plot(spatial_colors=True, gfp=True)
    # Save
    fig_path_subj = fig_path + f'{subject.subject_id}/'
    fname = f'it_{chs_id}'
    save.fig(fig=fig, fname=fname, path=fig_path_subj)


# Compute grand average
tgt_grand_avg = mne.grand_average(tgt_ga, interpolate_bads=False)
it_grand_avg = mne.grand_average(it_ga, interpolate_bads=False)

# Plot
fig = tgt_grand_avg.plot(spatial_colors=True, gfp=True)

# Save
fname = f'tgt_GA_{chs_id}'
save.fig(fig=fig, fname=fname, path=fig_path)

# Plot
fig = it_grand_avg.plot(spatial_colors=True, gfp=True)
# Save
fname = f'it_GA_{chs_id}'
save.fig(fig=fig, fname=fname, path=fig_path)
