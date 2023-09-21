# it vs tgt FRF
import load
import mne
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
import save
import setup
from paths import paths
import functions_general
import numpy as np
import scipy.stats

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_fig = True
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Select frequency band -----#
# ICA vs raw data
use_ica_data = True
band_id = None
# Id
epoch_ids = ['it_fix_subsampled', 'tgt_fix']
# Pick MEG chs (Select channels or set picks = 'mag')
chs_id = 'temporal'
chs_ids = [f'{chs_id}_L', f'{chs_id}_R']
labels = ['FRF Distractor', 'FRF Target']
# Trials
corr_ans = True
tgt_pres = True
mss = None
reject = None
trial_dur = None
evt_dur = None

# Get time windows from epoch_id name
tmin, tmax, plot_xlim = -0.3, 0.6, (-0.1, 0.5)
# Baseline
baseline = (None, -0.05)

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Variable to store data and plot
evokeds = {}
means = {}
stds = {}
bads = []
maxs = {}
mins = {}

# Figure
fig, axs = plt.subplots(ncols=len(chs_ids), figsize=(17, 5))

for i, epoch_id in enumerate(epoch_ids):
    evokeds[epoch_id] = {}
    means[epoch_id] = {}
    stds[epoch_id] = {}
    maxs[epoch_id] = {}
    mins[epoch_id] = {}
    for j, lat_chs in enumerate(chs_ids):
        evokeds[epoch_id][lat_chs] = []

        # Specific run path for saving data and plots
        save_id = f'{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}'
        run_path = f'/Band_{band_id}/{save_id}/'

        # Save data paths
        epochs_save_path = save_path + f'Epochs_{data_type}/' + run_path
        evoked_save_path = save_path + f'Evoked_{data_type}/' + run_path
        grand_avg_data_fname = f'Grand_average_ave.fif'
        # Save figures paths
        epochs_fig_path = plot_path + f'Epochs_{data_type}/' + run_path
        evoked_fig_path = plot_path + f'Evoked_{data_type}/' + run_path

        # Iterate over subjects
        for subject_code in exp_info.subjects_ids[:12]:

            if use_ica_data:
                # Load subject object
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
            else:
                # Load subject object
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

            # Data filenames
            evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

            # Load evoked data
            evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]

            # Subsample to selected region
            picks = functions_general.pick_chs(chs_id=lat_chs, info=evoked.info)
            evoked_sub = evoked.pick(picks)
            evoked_data = evoked_sub.get_data()

            # Exclude subjects bad channels
            bad_ch_idx = [i for i, ch_name in enumerate(evoked.info['ch_names']) if ch_name in evoked.info['bads']]
            bads.append(evoked.info['bads'])
            evoked_data = np.delete(evoked_data, bad_ch_idx, axis=0)

            # Append evoked averaged over all channels
            evokeds[epoch_id][lat_chs].append(evoked_data.mean(0))

        # Convert to array
        evokeds[epoch_id][lat_chs] = np.array(evokeds[epoch_id][lat_chs])

        # Take mean and sem over subjects
        means[epoch_id][lat_chs] = evokeds[epoch_id][lat_chs].mean(0)*1e15
        stds[epoch_id][lat_chs] = evokeds[epoch_id][lat_chs].std(0)/np.sqrt(len(evokeds[epoch_id][lat_chs]))*1e15

        # Get min and max values
        maxs[epoch_id][lat_chs] = means[epoch_id][lat_chs].max()
        mins[epoch_id][lat_chs] = means[epoch_id][lat_chs].min()

        # Plot
        axs[j].plot(evoked.times, means[epoch_id][lat_chs], color=f'C{i}', label=labels[i])
        axs[j].fill_between(evoked.times, y1=means[epoch_id][lat_chs]-stds[epoch_id][lat_chs],
                            y2=means[epoch_id][lat_chs]+stds[epoch_id][lat_chs], alpha=0.5, color=f'C{i}', edgecolor=None)
        axs[j].set_xlim(evoked.times[0], evoked.times[-1])


# Get subjects trf from both clases split in left and right
obs_left = [evokeds[epoch_ids[0]][lat_chs[0]], evokeds[epoch_ids[1]][lat_chs[0]]]
obs_right = [evokeds[epoch_ids[0]][lat_chs[1]], evokeds[epoch_ids[1]][lat_chs[1]]]

# Permutation cluster test
n_permutations = 2048
degrees_of_freedom = len(exp_info.subjects_ids[:12]) - 1
desired_pval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_pval / 2, df=degrees_of_freedom)
threshold = dict(start=0, step=0.2)

# Left
t_tfce_left, clusters_left, p_tfce_left, H0_left = permutation_cluster_test(X=obs_left, threshold=t_thresh,
                                                                            n_permutations=n_permutations)

good_clusters_left_idx = np.where(p_tfce_left < 0.05)[0]
significant_clusters_left = np.take(clusters_left, good_clusters_left_idx).tolist()
# Plot significant clusters
if len(significant_clusters_left):
    for cluster in significant_clusters_left:
        axs[0].axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], color='green', alpha=0.5, label='Signif.')

# Right
t_tfce_right, clusters_right, p_tfce_right, H0_right = permutation_cluster_test(X=obs_right, threshold=t_thresh,
                                                                                n_permutations=n_permutations)
good_clusters_right_idx = np.where(p_tfce_right < 0.05)[0]
significant_clusters_right = np.take(clusters_right, good_clusters_right_idx).tolist()
# Plot significant clusters
if len(significant_clusters_right):
    for cluster in significant_clusters_right:
        axs[1].axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], color='green', alpha=0.5, label='Signif.')


# Put legend to 2nd ax
axs[0].legend(loc='lower left')
axs[1].legend(loc='upper left')
# Plot vertical lines
axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='gray')
axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='gray')
axs[0].set_xlabel('time (s)')
axs[1].set_xlabel('time (s)')
axs[0].set_ylabel('FRF (fT)')
axs[1].set_ylabel('FRF (fT)')
fig.suptitle(f'Target vs. Distractor FRF {chs_id} (t_threshold: {t_thresh} - Significance threshold: {0.05})')

axs[0].xaxis.label.set_fontsize(18)
axs[0].yaxis.label.set_fontsize(18)
axs[1].xaxis.label.set_fontsize(18)
axs[1].yaxis.label.set_fontsize(18)
axs[0].xaxis.set_tick_params(labelsize=18)
axs[0].yaxis.set_tick_params(labelsize=18)
axs[1].xaxis.set_tick_params(labelsize=18)
axs[1].yaxis.set_tick_params(labelsize=18)

axs[0].set_xlim(left=-0.2, right=0.5)
axs[1].set_xlim(left=-0.2, right=0.5)

fig.tight_layout()

if save_fig:
    fig_path = paths().plots_path() + f'Evoked_{data_type}/it_vs_tgt/'
    fname = (save_id + f'_{chs_id}').replace(epoch_id, 'Evoked')
    save.fig(fig=fig, path=fig_path, fname=fname)

## it_vs_fix TRF
import load
import mne
import matplotlib.pyplot as plt
import setup
from paths import paths
import functions_general
import numpy as np
import matplotlib
import scipy.stats

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_fig = True
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# ICA vs raw data
use_ica_data = True
standarize = True
band_id = None
# Id
epoch_ids = ['it_fix_subsampled', 'tgt_fix', 'blue', 'red']
# Pick MEG chs (Select channels or set picks = 'mag')
chs_id = 'temporal'
chs_ids = [f'{chs_id}_L', f'{chs_id}_R']
labels = ['TRF Distractor', 'TRF Target']
# Trials
corr_ans = True
tgt_pres = True
mss = None
trial_dur = None
evt_dur = None
# Get time windows from epoch_id name
tmin, tmax, plot_xlim = -0.3, 0.6, (-0.1, 0.5)
# Baseline
baseline = (None, -0.05)
# TRF hiper-parameter
alpha = None
# Reescale and plot with FRF
FRF_TRF = True

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Variable to store data and plot
evokeds = {}
means = {}
stds = {}
bads = []

# Save path
save_path = paths().save_path() + f'TRF_{data_type}/{epoch_ids}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_tdur{trial_dur}' \
                                  f'_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/{chs_id}/'
fig_path = save_path.replace(paths().save_path(), paths().plots_path())

# Figure
matplotlib.rc({'font.size': 20})
matplotlib.rc({'xtick.labelsize': 20})
matplotlib.rc({'ytick.labelsize': 20})

if FRF_TRF:
    axs_0_lims = axs[0].get_ylim()
    axs_1_lims = axs[1].get_ylim()

    axs[0] = axs[0].twinx()
    axs[0].set_ylim(axs_0_lims)

    axs[1] = axs[1].twinx()
    axs[1].set_ylim(axs_1_lims)
else:
    fig, axs = plt.subplots(ncols=2, figsize=(17, 5))

for i, epoch_id in enumerate(epoch_ids[:2]):
    evokeds[epoch_id] = {}
    means[epoch_id] = {}
    stds[epoch_id] = {}
    for j, lat_chs in enumerate(chs_ids):
        evokeds[epoch_id][lat_chs] = []

        # Iterate over subjects
        for subject_code in exp_info.subjects_ids[:12]:

            if use_ica_data:
                # Load subject object
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
                meg_data = load.ica_data(subject=subject)
            else:
                # Load subject object
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
                meg_data = subject.load_preproc_meg_data()

            # Data filenames
            trf_path = save_path
            trf_fname = f'TRF_{subject_code}.pkl'

            # Load rf data
            rf = load.var(trf_path + trf_fname)
            picks = functions_general.pick_chs(chs_id=chs_id, info=meg_data.info)
            meg_sub = meg_data.copy().pick(picks)
            print('Loaded Receptive Field')

            # Get correspoding trf
            trf = rf.coef_[:, i, :]
            # Define evoked objects from arrays of TRF
            evoked = mne.EvokedArray(data=trf, info=meg_sub.info, tmin=tmin, baseline=baseline)
            # Subsample channels
            picks = functions_general.pick_chs(chs_id=lat_chs, info=evoked.info)
            evoked_sub = evoked.pick(picks)
            evoked_data = evoked_sub.get_data()

            # Exclude subjects bad channels
            bad_ch_idx = [i for i, ch_name in enumerate(evoked.info['ch_names']) if ch_name in evoked.info['bads']]
            bads.append(evoked.info['bads'])
            evoked_data = np.delete(evoked_data, bad_ch_idx, axis=0)

            # Append evoked averaged over all channels
            evokeds[epoch_id][lat_chs].append(evoked_data.mean(0))

        # Convert to array
        evokeds[epoch_id][lat_chs] = np.array(evokeds[epoch_id][lat_chs])

        # Take mean and sem over subjects
        means[epoch_id][lat_chs] = evokeds[epoch_id][lat_chs].mean(0)
        stds[epoch_id][lat_chs] = evokeds[epoch_id][lat_chs].std(0)/np.sqrt(len(evokeds[epoch_id][lat_chs]))

        if FRF_TRF:
            # Plot with FRF
            means[epoch_id][lat_chs] -= means[epoch_id][lat_chs].min()
            means[epoch_id][lat_chs] /= means[epoch_id][lat_chs].max()

            means[epoch_id][lat_chs] *= (maxs[epoch_id][lat_chs] - mins[epoch_id][lat_chs])
            means[epoch_id][lat_chs] += mins[epoch_id][lat_chs]

        # Plot
        colors = ['slateblue', 'tomato']
        if FRF_TRF:
            axs[j].plot(evoked.times, means[epoch_id][lat_chs], color=colors[i], label=labels[i])
        else:
            axs[j].plot(evoked.times, means[epoch_id][lat_chs], color=f'C{i}', label=labels[i])
            axs[j].fill_between(evoked.times, y1=means[epoch_id][lat_chs]-stds[epoch_id][lat_chs],
                                y2=means[epoch_id][lat_chs]+stds[epoch_id][lat_chs], alpha=0.5, color=f'C{i}', edgecolor=None)
            axs[j].set_xlim(evoked.times[0], evoked.times[-1])


# Get subjects trf from both classes split in left and right
obs_left = [evokeds[epoch_ids[0]][chs_ids[0]], evokeds[epoch_ids[1]][chs_ids[0]]]
obs_right = [evokeds[epoch_ids[0]][chs_ids[1]], evokeds[epoch_ids[1]][chs_ids[1]]]

# TFCE test
n_permutations = 2048
degrees_of_freedom = len(exp_info.subjects_ids[:12]) - 1
desired_pval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_pval / 2, df=degrees_of_freedom)
threshold = dict(start=0, step=0.2)
threshold = 2

# Left
t_tfce_left, clusters_left, p_tfce_left, H0_left = permutation_cluster_test(X=obs_left, threshold=threshold, n_permutations=n_permutations)
good_clusters_left_idx = np.where(p_tfce_left < 0.05)[0]
significant_clusters_left = np.take(clusters_left, good_clusters_left_idx).tolist()
# Plot significant clusters
if len(significant_clusters_left):
    for cluster in significant_clusters_left:
        axs[0].axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], color='c', alpha=0.5, label='Signif.')

# Right
t_tfce_right, clusters_right, p_tfce_right, H0_right = permutation_cluster_test(X=obs_right, threshold=threshold, n_permutations=n_permutations)
good_clusters_right_idx = np.where(p_tfce_right < 0.05)[0]
significant_clusters_right = np.take(clusters_right, good_clusters_right_idx).tolist()
# Plot significant clusters
if len(significant_clusters_right):
    for cluster in significant_clusters_right:
        axs[1].axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], color='green', alpha=0.5, label='Signif.')

# Put legend to 2nd ax
axs[0].legend(loc='lower right')
axs[1].legend(loc='upper right')
# Plot vertical lines
if not FRF_TRF:
    axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='gray')
    axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='gray')
    axs[0].set_xlabel('time (s)')
    axs[1].set_xlabel('time (s)')
    axs[0].set_ylabel('TRF (a.u.)')
    axs[1].set_ylabel('TRF (a.u.)')
    fig.suptitle(f'Target vs. Distractor TRF {chs_id}')
else:
    axs[0].set_ylabel('TRF (a.u.)')
    axs[1].set_ylabel('TRF (a.u.)')
    fig.suptitle(f'Target vs. Distractor FRF-TRF {chs_id} (t_threshold: {t_thresh} - Significance threshold: {0.05})')

axs[0].xaxis.label.set_fontsize(18)
axs[0].yaxis.label.set_fontsize(18)
axs[1].xaxis.label.set_fontsize(18)
axs[1].yaxis.label.set_fontsize(18)
axs[0].xaxis.set_tick_params(labelsize=18)
axs[0].yaxis.set_tick_params(labelsize=18)
axs[1].xaxis.set_tick_params(labelsize=18)
axs[1].yaxis.set_tick_params(labelsize=18)

axs[0].set_xlim(left=-0.2, right=0.5)
axs[1].set_xlim(left=-0.2, right=0.5)

fig.tight_layout()

if save_fig:
    fig_path += 'it_vs_tgt/'
    if FRF_TRF:
        fname = f'FRF_TRF_{chs_id}'
    else:
        fname = f'TRF_{chs_id}'
    save.fig(fig=fig, path=fig_path, fname=fname)


## it vs tgt all channels
import load
import mne
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
import save
import setup
from paths import paths
import functions_general
import numpy as np
import scipy.stats

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

#----- Save data and display figures -----#
save_fig = True
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Select frequency band -----#
# ICA vs raw data
use_ica_data = True
band_id = None
# Id
epoch_ids = ['it_fix_subsampled', 'tgt_fix']
# Pick MEG chs (Select channels or set picks = 'mag')
chs_id = 'mag'
# Trials
corr_ans = True
tgt_pres = True
mss = None
reject = None
trial_dur = None
evt_dur = None

# Get time windows from epoch_id name
tmin, tmax, plot_xlim = -0.3, 0.6, (-0.1, 0.5)
# Baseline
baseline = (None, -0.05)

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Variable to store data and plot
evokeds_ga = {}
evokeds_data = {}

# Define observations array to pass to TFCE
observations = []

for i, epoch_id in enumerate(epoch_ids):
    # list of evoked objects to make GA
    evokeds_ga[epoch_id] = []
    #" List of evoked data as array to compute TFCE
    evokeds_data[epoch_id] = []

    # Specific run path for saving data and plots
    save_id = f'{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}_tdur{trial_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}'
    run_path = f'/Band_{band_id}/{save_id}/'

    # Save data paths
    epochs_save_path = save_path + f'Epochs_{data_type}/' + run_path
    evoked_save_path = save_path + f'Evoked_{data_type}/' + run_path
    grand_avg_data_fname = f'Grand_average_ave.fif'
    # Save figures paths
    epochs_fig_path = plot_path + f'Epochs_{data_type}/' + run_path
    evoked_fig_path = plot_path + f'Evoked_{data_type}/' + run_path

    # Iterate over subjects
    for subject_code in exp_info.subjects_ids[:12]:

        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

        # Data filenames
        evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

        # Load evoked data
        evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]

        # Subsample to selected region
        picks = functions_general.pick_chs(chs_id=chs_id, info=evoked.info)
        evoked_sub = evoked.pick(picks)
        evoked_data = evoked_sub.get_data()

        # Append subject's evoked to list of evokeds of actual condition to compute Ga
        evokeds_ga[epoch_id].append(evoked)

        # Append subject's evoked to list of evokeds of actual condition to test
        evokeds_data[epoch_id].append(evoked_data.T)

    # Convert to array
    evokeds_epoch_id = np.array(evokeds_data[epoch_id])

    # Append to observations list
    observations.append(evokeds_epoch_id)

# Compute grand average of target fixations
grand_avg = mne.grand_average(evokeds_ga['tgt_fix'], interpolate_bads=False)

# Permutation cluster test parameters
n_permutations = 256
degrees_of_freedom = len(exp_info.subjects_ids[:12]) - 1
desired_pval = 0.05
# t_thresh = scipy.stats.t.ppf(1 - desired_pval / 2, df=degrees_of_freedom)
t_thresh = dict(start=0, step=0.2)
# Get channel adjacency
ch_adjacency_sparse = functions_general.get_channel_adjacency(info=evoked.info)
# Clusters out type
if type(t_thresh) == dict:
    out_type = 'indices'
else:
     out_type = 'mask'

# Permutations cluster test (TFCE if t_thresh as dict)
t_tfce, clusters, p_tfce, H0 = permutation_cluster_test(X=observations, threshold=t_thresh, adjacency=ch_adjacency_sparse,
                                                        n_permutations=n_permutations, out_type=out_type, n_jobs=6)

pval_threshold = 0.5
# Make clusters mask
if type(t_tfce) == dict:
    # If TFCE use p-vaues of voxels directly
    p_tfce = p_tfce.reshape(len(evoked.times), len(evoked.ch_names)).T

    # Reshape to data's shape
    clusters_mask = p_tfce < pval_threshold
else:
    # Get significant clusters
    good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
    significant_clusters = [clusters[idx] for idx in good_clusters_idx]

    # Rehsape to data's shape by adding all clusters into one bool array
    clusters_mask = np.zeros(clusters[0].shape)
    for significant_cluster in significant_clusters:
        clusters_mask += significant_cluster
    clusters_mask = clusters_mask.astype(bool).T

# Plot
fig, ax = plt.subplots(figsize=(14, 5))
if type(t_thresh) == dict:
    title = f'{chs_id}_tthreshTFCE_pthresh{pval_threshold}'
else:
     title = f'{chs_id}_tthresh{round(t_thresh, 2)}_pthresh{pval_threshold}'

fig = grand_avg.plot_image(cmap='jet', mask=clusters_mask, mask_style='mask', mask_alpha=0.5,
                           titles=title, axes=ax, show=display_figs)

if save_fig:
    fig_path = paths().plots_path() + f'Evoked_{data_type}/it_vs_tgt/'
    fname = (save_id + f'_{title}').replace(epoch_id, 'tgt_fix')
    save.fig(fig=fig, path=fig_path, fname=fname)


## plot sensors

chs_id = 'temporal'

# Data filenames
evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'

# Load evoked data
evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, verbose=False)[0]

picks = functions_general.pick_chs(chs_id=chs_id, info=evoked.info)
evoked_sub = evoked.pick(picks)
evoked_sub.info['bads'] = []
fig2 = evoked_sub.plot_sensors()