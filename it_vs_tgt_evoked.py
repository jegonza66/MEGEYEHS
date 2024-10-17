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
run_tfce = True
band_id = None
# Id
epoch_ids = ['it_fix_vs_subsampled', 'tgt_fix_vs']
# Pick MEG chs (Select channels or set picks = 'mag')
chs_id = 'temporal'
chs_ids = [f'{chs_id}_L', f'{chs_id}_R']
labels = ['Distractor', 'Target']
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
baseline = (tmin, -0.05)

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Variable to store data and plot
ga_dict = {}
evokeds = {}
means = {}
stds = {}
bads = []
maxs = {}
mins = {}

# Figure
fig, axs = plt.subplots(ncols=len(chs_ids), figsize=(17, 5))
fig_both, ax_both = plt.subplots(figsize=(10, 5))

for i, epoch_id in enumerate(epoch_ids):

    evokeds[epoch_id] = {}
    means[epoch_id] = {}
    stds[epoch_id] = {}
    maxs[epoch_id] = {}
    mins[epoch_id] = {}

    # Specific run path for saving data and plots
    save_id = f'{epoch_id}_mss{mss}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}'
    run_path = f'Band_{band_id}/{save_id}/'

    # Save data paths
    epochs_save_path = save_path + f'Epochs_{data_type}/' + run_path
    evoked_save_path = save_path + f'Evoked_{data_type}/' + run_path
    grand_avg_data_fname = f'Grand_average_ave.fif'
    # Save figures paths
    epochs_fig_path = plot_path + f'Epochs_{data_type}/' + run_path
    evoked_fig_path = plot_path + f'Evoked_{data_type}/' + run_path

    for j, lat_chs in enumerate(chs_ids):

        ga_dict[epoch_id] = []  # Save evokeds of all channels (not lateralized)
        evokeds[epoch_id][lat_chs] = []

        # Iterate over subjects
        for subject_code in exp_info.subjects_ids:

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

            # Append evoked averaged over all channels
            ga_evoked = evoked.copy()
            ga_dict[epoch_id].append(ga_evoked)

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
        # Separated plots
        axs[j].plot(evoked.times, means[epoch_id][lat_chs], color=f'C{i}', label=labels[i])
        axs[j].fill_between(evoked.times, y1=means[epoch_id][lat_chs]-stds[epoch_id][lat_chs], y2=means[epoch_id][lat_chs]+stds[epoch_id][lat_chs], alpha=0.5, color=f'C{i}', edgecolor=None)
        axs[j].set_xlim(evoked.times[0], evoked.times[-1])

        # Mixed plots
        ax_both.plot(evoked.times, means[epoch_id][lat_chs], color=f'C{i}', label=labels[i])
        ax_both.fill_between(evoked.times, y1=means[epoch_id][lat_chs] - stds[epoch_id][lat_chs], y2=means[epoch_id][lat_chs] + stds[epoch_id][lat_chs], alpha=0.5,
                             color=f'C{i}', edgecolor=None)
        ax_both.set_xlim(evoked.times[0], evoked.times[-1])

if run_tfce:
    # Get subjects trf from both clases split in left and right
    obs_left = [evokeds[epoch_ids[0]][chs_ids[0]], evokeds[epoch_ids[1]][chs_ids[0]]]
    obs_right = [evokeds[epoch_ids[0]][chs_ids[1]], evokeds[epoch_ids[1]][chs_ids[1]]]

    # Permutation cluster test
    pval_threshold = 0.05
    n_permutations = 1024
    degrees_of_freedom = len(exp_info.subjects_ids) - 1
    desired_tval = 0.01
    t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
    # t_thresh = dict(start=0, step=0.2)

    # Left
    t_tfce_left, clusters_left, p_tfce_left, H0_left = permutation_cluster_test(X=obs_left, threshold=t_thresh, n_permutations=n_permutations)

    good_clusters_left_idx = np.where(p_tfce_left < pval_threshold)[0]
    significant_clusters_left = [clusters_left[i][0] for i in good_clusters_left_idx]
    # Plot significant clusters
    if len(significant_clusters_left):
        for cluster in significant_clusters_left:
            xmin = (evoked.times[cluster[0]] - evoked.times[0])/(evoked.times[-1] - evoked.times[0])
            xmax = (evoked.times[cluster[-1]] - evoked.times[0])/(evoked.times[-1] - evoked.times[0])
            xmid = evoked.times[cluster[0]] + (evoked.times[cluster[-1]] - evoked.times[cluster[0]])/2
            axs[0].axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], ymin=0.95, ymax=0.95, color='k')
            # axs[0].axhline(y=axs[0].get_ylim()[1] - 5, xmin=xmin, xmax=xmax, color='k')
            axs[0].text(x=xmid, y=axs[0].get_ylim()[1], s='*')

    # Right
    t_tfce_right, clusters_right, p_tfce_right, H0_right = permutation_cluster_test(X=obs_right, threshold=t_thresh, n_permutations=n_permutations)
    good_clusters_right_idx = np.where(p_tfce_right < 0.05)[0]
    significant_clusters_right = [clusters_right[i][0] for i in good_clusters_right_idx]
    # Plot significant clusters
    if len(significant_clusters_right):
        for cluster in significant_clusters_right:
            xmin = (evoked.times[cluster[0]] - evoked.times[0]) / (evoked.times[-1] - evoked.times[0])
            xmax = (evoked.times[cluster[-1]] - evoked.times[0]) / (evoked.times[-1] - evoked.times[0])
            xmid = evoked.times[cluster[0]] + (evoked.times[cluster[-1]] - evoked.times[cluster[0]]) / 2
            axs[1].axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], ymin=0.95, ymax=0.95, color='k')
            # axs[1].axhline(y=axs[1].get_ylim()[1] - 5, xmin=xmin, xmax=xmax, color='k')
            axs[1].text(x=xmid, y=axs[1].get_ylim()[1], s='*')

    # Both
    if len(significant_clusters_left) == len(significant_clusters_right):
        for cluster_left, cluster_right in zip(significant_clusters_left, significant_clusters_right):
            set_left = set(cluster_left)
            set_right = set(cluster_right)
            # Find intersection of the sets (common elements)
            significant_clusters_both = np.array(list(set_left.intersection(set_right)))

            # Plot overlaped significance
            xmin = (evoked.times[cluster[0]] - evoked.times[0]) / (evoked.times[-1] - evoked.times[0])
            xmax = (evoked.times[cluster[-1]] - evoked.times[0]) / (evoked.times[-1] - evoked.times[0])
            xmid = evoked.times[cluster[0]] + (evoked.times[cluster[-1]] - evoked.times[cluster[0]]) / 2
            ax_both.axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], ymin=0.95, ymax=0.95, color='k')
            ax_both.text(x=xmid, y=ax_both.get_ylim()[1], s='*')

fontsize = 22
# Put legend to 2nd ax
axs[0].legend(loc='lower left', fontsize=fontsize)
axs[1].legend(loc='upper left', fontsize=fontsize)
# Both
handles, labels = ax_both.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_both.legend(by_label.values(), by_label.keys(), fontsize=fontsize)

# Plot vertical lines
axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='gray')
axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='gray')
ax_both.vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='gray')

# Set labels
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('FRF (fT)')
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('FRF (fT)')
ax_both.set_xlabel('time (s)')
ax_both.set_ylabel('FRF (fT)')
axs[0].xaxis.label.set_fontsize(fontsize)
axs[0].yaxis.label.set_fontsize(fontsize)
axs[1].xaxis.label.set_fontsize(fontsize)
axs[1].yaxis.label.set_fontsize(fontsize)
ax_both.xaxis.label.set_fontsize(fontsize)
ax_both.yaxis.label.set_fontsize(fontsize)
axs[0].xaxis.set_tick_params(labelsize=fontsize)
axs[0].yaxis.set_tick_params(labelsize=fontsize)
axs[1].xaxis.set_tick_params(labelsize=fontsize)
axs[1].yaxis.set_tick_params(labelsize=fontsize)
ax_both.xaxis.set_tick_params(labelsize=fontsize)
ax_both.yaxis.set_tick_params(labelsize=fontsize)

# Title
if run_tfce:
    fig.suptitle(f'Target vs. Distractor FRF {chs_id} (t_threshold: {round(t_thresh, 2)} - Significance threshold: {pval_threshold})')
    fig_both.suptitle(f'Target vs. Distractor FRF {chs_id} (t_threshold: {round(t_thresh, 2)} - Significance threshold: {pval_threshold})')
else:
    fig.suptitle(f'Target vs. Distractor FRF {chs_id}')
    fig_both.suptitle(f'Target vs. Distractor FRF {chs_id}')

# Limits
axs[0].set_xlim(left=-0.2, right=0.5)
axs[1].set_xlim(left=-0.2, right=0.5)
ax_both.set_xlim(left=-0.2, right=0.5)

fig.tight_layout()
fig_both.tight_layout()

if save_fig:
    fig_path = paths().plots_path() + f'Evoked_{data_type}/it_vs_tgt/'
    fname = (save_id + f'_{chs_id}').replace(epoch_id, 'Evoked')
    if type(t_thresh) == dict:
        fname += f'_tTFCE_p{pval_threshold}'
    else:
        fname += f'_t{round(t_thresh, 2)}_p{pval_threshold}'
    save.fig(fig=fig, path=fig_path, fname=fname)
    save.fig(fig=fig_both, path=fig_path, fname=fname + '_both')

# Plot topographies
topo_times = [0.1, 0.3]
topo_times_span = [0.005, 0.1]
grand_average = mne.grand_average(ga_dict['tgt_fix_vs'], interpolate_bads=True)

# Mascara en temporales
picks = functions_general.pick_chs(chs_id='temporal', info=grand_average.info)
mask = np.zeros(grand_average.data.shape)
for ch in picks:
    try:
        mask[grand_average.ch_names.index(ch), :] = 1
    except:
        pass
mask = mask.astype(bool)
mask_params = dict(markersize=10, markerfacecolor="gray", alpha=0.65)
fig_topo = grand_average.plot_topomap(times=topo_times, average=topo_times_span, cmap='bwr', show=display_figs, vlim=(-60, 60),
                                      mask=mask, mask_params=mask_params)

if save_fig:
    fig_path = paths().plots_path() + f'Evoked_{data_type}/it_vs_tgt/'
    fname = (save_id + f'_{chs_id}_topomaps').replace(epoch_id, 'Evoked')
    save.fig(fig=fig_topo, path=fig_path, fname=fname)


## it_vs_fix TRF
import load
import save
import mne
import matplotlib.pyplot as plt
import setup
from paths import paths
import functions_general
import numpy as np
import matplotlib
import scipy.stats
from mne.stats import permutation_cluster_test

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
# Reescale and plot with FRF
FRF_TRF = False
# ICA vs raw data
use_ica_data = True
run_tfce = True
standarize = True
band_id = None
# Id
epoch_ids = ['it_fix_vs_subsampled', 'tgt_fix_vs', 'blue', 'red']
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
baseline = (tmin, -0.05)
# TRF hiper-parameter
alpha = None


# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Variable to store data and plot
ga_dict = {}
evokeds = {}
means = {}
stds = {}
bads = []

# Save path
load_path = paths().save_path() + f'TRF_{data_type}/{epoch_ids}_mss{mss}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}' \
                                  f'_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/mag/'
fig_path = load_path.replace(paths().save_path(), paths().plots_path())

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
        ga_dict[epoch_id] = []
        evokeds[epoch_id][lat_chs] = []

        # Iterate over subjects
        for subject_code in exp_info.subjects_ids:

            if use_ica_data:
                # Load subject object
                subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
                meg_data = load.ica_data(subject=subject)
            else:
                # Load subject object
                subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
                meg_data = subject.load_preproc_meg_data()

            # Data filenames
            trf_path = load_path
            trf_fname = f'TRF_{subject_code}.pkl'

            # Load rf data
            rf = load.var(trf_path + trf_fname)
            print('Loaded Receptive Field')

            # Get model coeficients as separate responses to target and items
            # All or multiple regions
            if type(rf) == dict:
                # Define evoked from TRF list to concatenate all
                evoked_list = []

                # iterate over regions
                for chs_idx, chs_subset in enumerate(rf.keys()):

                    # Get TRF coefficients from chs subset
                    trf = rf[chs_subset].coef_[:, i, :]

                    picks = functions_general.pick_chs(chs_id=chs_subset, info=meg_data.info)
                    meg_sub = meg_data.copy().pick(picks)

                    if chs_idx == 0:
                        # Define evoked object from arrays of TRF
                        evoked = mne.EvokedArray(data=trf, info=meg_sub.info, tmin=tmin, baseline=baseline)
                    else:
                        # Append evoked object from arrays of TRF to list, to concatenate all
                        evoked_list.append(mne.EvokedArray(data=trf, info=meg_sub.info, tmin=tmin, baseline=baseline))

                # Concatenate evoked from al regions
                evoked = evoked.add_channels(evoked_list)

            else:
                trf = rf.coef_[:, i, :]
                # Define evoked objects from arrays of TRF
                evoked = mne.EvokedArray(data=trf, info=meg_sub.info, tmin=tmin, baseline=baseline)

            evoked_ga = evoked.copy()
            ga_dict[epoch_id].append(evoked_ga)

            if chs_id != 'mag':
                # Subsample channels
                picks = functions_general.pick_chs(chs_id=lat_chs, info=evoked.info)
                evoked_sub = evoked.pick(picks)
                evoked_data = evoked_sub.get_data()
            else:
                evoked_data = evoked.get_data()

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
degrees_of_freedom = len(exp_info.subjects_ids) - 1
desired_tval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# t_thresh = dict(start=0, step=0.2)
pval_threshold = 0.05

# Left
t_tfce_left, clusters_left, p_tfce_left, H0_left = permutation_cluster_test(X=obs_left, threshold=t_thresh, n_permutations=n_permutations)
good_clusters_left_idx = np.where(p_tfce_left < pval_threshold)[0]
significant_clusters_left = [clusters_left[i][0] for i in good_clusters_left_idx]

# Plot significant clusters
bar_height = axs[0].get_ylim()[1]
if len(significant_clusters_left):
    for cluster in significant_clusters_left:
        # axs[0].axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], color='green', alpha=0.3, label='Signif.')
        axs[0].hlines(y=bar_height, xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], color='gray', alpha=0.7, label='Signif.')

# Right
t_tfce_right, clusters_right, p_tfce_right, H0_right = permutation_cluster_test(X=obs_right, threshold=t_thresh, n_permutations=n_permutations)
good_clusters_right_idx = np.where(p_tfce_right < pval_threshold)[0]
significant_clusters_right = [clusters_right[i][0] for i in good_clusters_right_idx]
# Plot significant clusters
if len(significant_clusters_right):
    for cluster in significant_clusters_right:
        # axs[1].axvspan(xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]], color='green', alpha=0.3, label='Signif.')
        axs[1].hlines(y=bar_height, xmin=evoked.times[cluster[0]], xmax=evoked.times[cluster[-1]],
                      color='gray', alpha=0.7, label='Signif.')

# Drop duplicate labels
for ax, loc in zip([axs[0], axs[1]], ['lower right', 'upper right']):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=15, loc=loc)

# Put legend to 2nd ax
# axs[0].legend(loc='lower right')
# axs[1].legend(loc='upper right')

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
    fig.suptitle(f'Target vs. Distractor FRF-TRF {chs_id} (t_threshold: {round(t_thresh, 2)} - Significance threshold: {pval_threshold})')

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

    if type(t_thresh) == dict:
        fname += f'_tTFCE_p{pval_threshold}'
    else:
        fname += f'_t{round(t_thresh, 2)}_p{pval_threshold}'
    save.fig(fig=fig, path=fig_path, fname=fname)

# Plot topographies
topo_times = [0.1, 0.4]
topo_times_span = [0.01, 0.2]
grand_average = mne.grand_average(ga_dict['tgt_fix'], interpolate_bads=True)


# Mascara en temporales
# picks = functions_general.pick_chs(chs_id='temporal', info=meg_data.info)
# mask = np.zeros(grand_average.data.shape)
# for ch in picks:
#     try:
#         mask[grand_average.ch_names.index(ch), :] = 1
#     except:
#         pass
# mask = mask.astype(bool)
# mask_params = dict(markersize=10, markerfacecolor="g", alpha=0.65)
# fig_topo = grand_average.plot_topomap(times=topo_times, average=topo_times_span, cmap='bwr', show=display_figs,
#                                       mask=mask, mask_params=mask_params, units='a.u.', scalings=1)


fig_topo = grand_average.plot_topomap(times=topo_times, average=topo_times_span, cmap='bwr', show=display_figs,
                                      units='a.u.', scalings=1)

if save_fig:
    fname = f'TRF_{chs_id}_topomaps'
    save.fig(fig=fig_topo, path=fig_path, fname=fname)


## Sources
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
import scipy
from mne.stats import permutation_cluster_test
import itertools

# Load experiment info
exp_info = setup.exp_info()

#--------- Define Parameters ---------#

# Save
save_fig = True
save_data = True
display_figs = True
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
                'evtdur': None,
                'trialdur': None,
                'tmin': -0.3,
                'tmax': 0.6,
                'baseline': (-0.3, -0.05)}

meg_params = {'chs_id': 'mag',
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

# Compare features
run_comparison = True

# Source estimation parameters
force_fsaverage = False
model_name = 'lcmv'
surf_vol = 'volume'
ico = 5
spacing = 5.  # Only for volume source estimation
pick_ori = None  # 'vector' For dipoles, 'max-power' for fixed dipoles in the direction tha maximizes output power
source_power = False
source_estimation = 'evk'  # 'epo' / 'evk' / 'cov' / 'trf'
estimate_source_tf = False
visualize_alignment = False

# Baseline
if source_power or source_estimation == 'cov' or estimate_source_tf:
    bline_mode_subj = 'db'
else:
    bline_mode_subj = 'mean'
bline_mode_ga = 'mean'
plot_edge = 0.15

# Plot
initial_time = 0.1
difference_initial_time = 0.3
positive_cbar = None  # None for free determination, False to include negative values
plot_individuals = True
plot_ga = True

# Permutations test
run_permutations_GA = False
run_permutations_diff = False
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

# Get param to compute difference from params dictionary
param_values = {key: value for key, value in trial_params.items() if type(value) == list}
# Exception in case no comparison
if param_values == {}:
    param_values = {list(trial_params.items())[0][0]: [list(trial_params.items())[0][1]]}

# Paths
run_path = (f"Band_{meg_params['band_id']}/{trial_params['epoch_id']}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_tgtpres{trial_params['tgtpres']}_"
            f"trialdur{trial_params['trialdur']}_evtdur{trial_params['evtdur']}_{trial_params['tmin']}_{trial_params['tmax']}_bline{trial_params['baseline']}/")

# Source plots paths
if source_power or source_estimation == 'cov':
    run_path = run_path.replace(f"{trial_params['epoch_id']}_", f"{trial_params['epoch_id']}_power_")

# Define figure path
if surf_vol == 'volume' or surf_vol == 'mixed':
    fig_path_diff = paths().plots_path() + (f"Source_Space_{meg_params['data_type']}/" + run_path + f"{model_name}_{surf_vol}_ico{ico}_{int(spacing)}_"
                                                                                                    f"{pick_ori}_{bline_mode_subj}_{source_estimation}/")
else:
    fig_path_diff = paths().plots_path() + (f"Source_Space_{meg_params['data_type']}/" + run_path + f"{model_name}_{surf_vol}_ico{ico}_{pick_ori}_"
                                                                                                    f"{bline_mode_subj}_{source_estimation}/")

# Define save path for pre-computed sources
save_path_sources = fig_path_diff.replace(paths().plots_path(), paths().save_path())
fname = 'stcs_mni.pkl'

if os.path.isfile(save_path_sources + fname):
    stcs_default_dict = load.var(file_path=save_path_sources + fname)
else:
    print(f'File not fount at: {save_path_sources + fname}')

    # --------- Run ---------#
    # Save source estimates time courses on FreeSurfer
    stcs_default_dict = {}
    GA_stcs = {}
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
                # Estimate sources from covariance matrix
                if source_estimation == 'cov':
                    # Covariance method
                    cov_method = 'shrunk'

                    # Covariance matrix rank
                    rank = sum([ch_type == 'mag' for ch_type in channel_types]) - len(bad_channels)
                    if meg_params['data_type'] == 'ICA':
                        rank -= len(subject.ex_components)

                    # Define active times
                    active_times = [0, run_params['tmax']]

                    # Covariance fnames
                    cov_baseline_fname = f"Subject_{subject.subject_id}_times{run_params['baseline']}_{cov_method}_{rank}-cov.fif"
                    cov_act_fname = f'Subject_{subject.subject_id}_times{active_times}_{cov_method}_{rank}-cov.fif'

                    stc = functions_analysis.estimate_sources_cov(subject=subject, meg_params=meg_params, trial_params=trial_params, filters=filters, active_times=active_times, rank=rank, bline_mode_subj=bline_mode_subj,
                                                                  save_data=save_data, cov_save_path=cov_save_path, cov_act_fname=cov_act_fname,
                                                                  cov_baseline_fname=cov_baseline_fname, epochs_save_path=epochs_save_path, epochs_data_fname=epochs_data_fname)

                # Estimate sources from epochs
                elif source_estimation == 'epo':
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
                            if meg_params['band_id'] and not meg_params['filter_sensors']:
                                # Filter source data
                                data = functions_general.butter_bandpass_filter(data, band_id=meg_params['band_id'], sfreq=epochs.info['sfreq'], order=3)
                            # Compute envelope
                            analytic_signal = hilbert(data, axis=-1)
                            signal_envelope = np.abs(analytic_signal)
                            # Sum data of every epoch
                            stc.data += signal_envelope

                        else:
                            stc.data += data
                        # Divide by epochs number
                        stc.data /= len(epochs)

                    if source_power:
                        # Drop edges due to artifacts from power computation
                        stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.times.max() - plot_edge)

                # Estimate sources from evoked
                elif source_estimation == 'evk':
                    # Apply filter and get source estimates
                    stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

                    if source_power:
                        # Compute envelope in source space
                        data = stc.data
                        if meg_params['band_id'] and not meg_params['filter_sensors']:
                            # Filter source data
                            data = functions_general.butter_bandpass_filter(data, band_id=meg_params['band_id'], sfreq=evoked.info['sfreq'], order=3)
                        # Compute envelope
                        analytic_signal = hilbert(data, axis=-1)
                        signal_envelope = np.abs(analytic_signal)
                        # Save envelope as data
                        stc.data = signal_envelope

                        # Drop edges due to artifacts from power computation
                        stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.tmax - plot_edge)

                # Estimate sources from evoked
                elif source_estimation == 'trf':

                    # Get trf paths
                    trf_path = paths().save_path() + (f"TRF_{meg_params['data_type']}/Band_{meg_params['band_id']}/{trf_params['input_features']}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_"
                                       f"tgtpres{trial_params['tgtpres']}_trialdur{trial_params['trialdur']}_evtdur{trial_params['evtdur']}_{trf_params['tmin']}_{trf_params['tmax']}_"
                                       f"bline{trf_params['baseline']}_alpha{trf_params['alpha']}_std{trf_params['standarize']}/{meg_params['chs_id']}/")
                    trf_fig_path = trf_path.replace(paths().save_path(), paths().plots_path())
                    trf_fname = f'TRF_{subject.subject_id}.pkl'

                    try:
                        # Load TRF
                        rf = load.var(trf_path + trf_fname)
                        print('Loaded Receptive Field')

                    except:
                        # Compute TRF for defined features
                        rf = functions_analysis.compute_trf(subject=subject, meg_data=meg_data, trial_params=trial_params, trf_params=trf_params, meg_params=meg_params,
                                                            save_data=save_data, trf_path=trf_path, trf_fname=trf_fname)

                    # Get model coeficients as separate responses to each feature
                    subj_evoked, _ = functions_analysis.make_trf_evoked(subject=subject, rf=rf, meg_data=meg_data, trf_params=trf_params, meg_params=meg_params, fig_path=trf_fig_path)

                    # Get evoked from desired feature
                    evoked = subj_evoked[run_params['epoch_id']]

                    # Apply filter and get source estimates
                    stc = beamformer.apply_lcmv(evoked=evoked, filters=filters)

                    if source_power:
                        # Compute envelope in source space
                        data = stc.data
                        if meg_params['band_id'] and not meg_params['filter_sensors']:
                            # Filter source data
                            data = functions_general.butter_bandpass_filter(data, band_id=meg_params['band_id'], sfreq=evoked.info['sfreq'], order=3)
                        # Compute envelope
                        analytic_signal = hilbert(data, axis=-1)
                        signal_envelope = np.abs(analytic_signal)
                        # Save envelope as data
                        stc.data = signal_envelope

                        # Drop edges due to artifacts from power computation
                        stc.crop(tmin=stc.tmin + plot_edge, tmax=stc.tmax - plot_edge)

                else:
                    raise ValueError('No source estimation method was selected. Please select either estimating sources from evoked, epochs or covariance matrix.')

                if bline_mode_subj and not source_estimation == 'cov':
                    # Apply baseline correction
                    print(f"Applying baseline correction: {bline_mode_subj} from {run_params['baseline'][0]} to {run_params['baseline'][1]}")
                    # stc.apply_baseline(baseline=baseline)  # mean
                    if bline_mode_subj == 'db':
                        stc.data = 10 * np.log10(stc.data / np.expand_dims(stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=-1), axis=-1))
                    elif bline_mode_subj == 'ratio':
                        stc.data = stc.data / np.expand_dims(stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=-1), axis=-1)
                    elif bline_mode_subj == 'mean':
                        stc.data = stc.data - np.expand_dims(stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=-1), axis=-1)

                if meg_params['band_id'] and source_power and not source_estimation == ' cov':
                    # Filter higher frequencies than corresponding to nyquist of bandpass filter higher freq
                    l_freq, h_freq = functions_general.get_freq_band(band_id=meg_params['band_id'])
                    stc.data = functions_general.butter_lowpass_filter(data=stc.data, h_freq=h_freq/2, sfreq=evoked.info['sfreq'], order=3)

                # Morph to MNI152 space
                if subject_code != 'fsaverage':

                    # Define morph function
                    morph = mne.compute_source_morph(src=src, subject_from=subject_code, subject_to='fsaverage', src_to=src_default, subjects_dir=subjects_dir)

                    # Apply morph
                    stc_default = morph.apply(stc)

                else:
                    stc_default = stc

                # Append to fs_stcs to make GA
                stcs_default_dict[param][param_value].append(stc_default)

                # Plot
                if plot_individuals:
                    fname = f'{subject.subject_id}'
                    plot_general.sources(stc=stc_default, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=initial_time, surf_vol=surf_vol,
                                         force_fsaverage=force_fsaverage, source_estimation=source_estimation, mask_negatives=mask_negatives,
                                         positive_cbar=positive_cbar, views=['lat', 'med'], save_fig=save_fig, save_vid=False, fig_path=fig_path, fname=fname)

            # Grand Average: Average evoked stcs from this epoch_id
            all_subj_source_data = np.zeros(tuple([len(stcs_default_dict[param][param_value])] + [size for size in stcs_default_dict[param][param_value][0].data.shape]))
            for j, stc in enumerate(stcs_default_dict[param][param_value]):
                all_subj_source_data[j] = stcs_default_dict[param][param_value][j].data
            if mask_negatives:
                all_subj_source_data[all_subj_source_data < 0] = 0

            # Define GA data
            GA_stc_data = all_subj_source_data.mean(0)

            # Copy Source Time Course from default subject morph to define GA STC
            GA_stc = stc_default.copy()

            # Reeplace data
            GA_stc.data = GA_stc_data
            GA_stc.subject = 'fsaverage'

            # Apply baseline on GA data
            if bline_mode_ga and not source_estimation == 'cov':
                print(f"Applying baseline correction: {bline_mode_ga} from {run_params['baseline'][0]} to {run_params['baseline'][1]}")
                # GA_stc.apply_baseline(baseline=baseline)
                if bline_mode_ga == 'db':
                    GA_stc.data = 10 * np.log10(GA_stc.data / GA_stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=1)[:, None])
                elif bline_mode_ga == 'ratio':
                    GA_stc.data = GA_stc.data / GA_stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=1)[:, None]
                elif bline_mode_ga == 'mean':
                    GA_stc.data = GA_stc.data - GA_stc.copy().crop(tmin=run_params['baseline'][0], tmax=run_params['baseline'][1]).data.mean(axis=1)[:, None]

            # Save GA from epoch id
            GA_stcs[param][param_value] = GA_stc

            # --------- Plot GA ---------#
            if plot_ga:
                fname = f'GA'
                brain = plot_general.sources(stc=GA_stc, src=src, subject='fsaverage', subjects_dir=subjects_dir, initial_time=initial_time, surf_vol=surf_vol,
                                             force_fsaverage=force_fsaverage, source_estimation=source_estimation, mask_negatives=mask_negatives,
                                             positive_cbar=positive_cbar, views=['lat', 'med'], save_fig=save_fig, save_vid=False, fig_path=fig_path, fname=fname)

            # --------- Test significance compared to baseline --------- #
            if run_permutations_GA and pick_ori != 'vector':
                stc_all_cluster_vis, significance_voxels, significance_mask, t_thresh_name, time_label, p_threshold = \
                    functions_analysis.run_source_permutations_test(src=src_default, stc=GA_stc, source_data=all_subj_source_data, subject='fsaverage', exp_info=exp_info,
                                                                    save_regions=True, fig_path=fig_path, surf_vol=surf_vol, desired_tval=desired_tval, mask_negatives=mask_negatives,
                                                                    p_threshold=p_threshold)

                # If covariance estimation, no time variable. Clusters are static
                if significance_mask is not None and source_estimation == 'cov':
                    # Mask data
                    GA_stc_sig = GA_stc.copy()
                    GA_stc_sig.data[significance_mask] = 0

                    # --------- Plot GA significant clusters ---------#
                    fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                    brain = plot_general.sources(stc=GA_stc_sig, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0, surf_vol=surf_vol,
                                         time_label=time_label, force_fsaverage=force_fsaverage, source_estimation=source_estimation, views=['lat', 'med'],
                                         mask_negatives=mask_negatives, positive_cbar=positive_cbar, save_vid=False, save_fig=save_fig, fig_path=fig_path, fname=fname)

                # If time variable, visualize clusters using mne's function
                elif significance_mask is not None:
                    fname = f'Clus_t{t_thresh_name}_p{p_threshold}'
                    brain = plot_general.sources(stc=stc_all_cluster_vis, src=src_default, subject='fsaverage', subjects_dir=subjects_dir, initial_time=0,
                                                 surf_vol=surf_vol, time_label=time_label, force_fsaverage=force_fsaverage, source_estimation=source_estimation,
                                                 views=['lat', 'med'], mask_negatives=mask_negatives, positive_cbar=positive_cbar,
                                                 save_vid=False, save_fig=save_fig, fig_path=fig_path, fname=fname)


# TFCE test
n_permutations = 2048
degrees_of_freedom = len(exp_info.subjects_ids) - 1
desired_tval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# t_thresh = dict(start=0, step=0.2)
pval_threshold = 0.05


# Extract time series from mni coords
used_voxels_mm = src_default[0]['rr'][src_default[0]['inuse'].astype(bool)] * 1000

# Plot
selected_box = {0: {'x': (-10, 0), 'y': (9, 19), 'z': (37, 47)},
                1: {'x': (0, 10), 'y': (9, 19), 'z': (37, 47)},
                2: {'x': (-33, -23), 'y': (-11, -1), 'z': (59, 69)},
                3: {'x': (25, 35), 'y': (-5, 5), 'z': (55, 65)}}

titles = ['DACCL', 'DACCR', 'FEFL', 'FEFR']

fontsize = 22

for j in selected_box.keys():
    # Get voxels in box
    voxel_idx = np.where((used_voxels_mm[:, 0] >= selected_box[j]['x'][0]) & (used_voxels_mm[:, 0] <= selected_box[j]['x'][1]) &
                         (used_voxels_mm[:, 1] >= selected_box[j]['y'][0]) & (used_voxels_mm[:, 1] <= selected_box[j]['y'][1]) &
                         (used_voxels_mm[:, 2] >= selected_box[j]['z'][0]) & (used_voxels_mm[:, 2] <= selected_box[j]['z'][1]))[0]

    # Get selected voxels id
    selected_voxels = src_default[0]['vertno'][voxel_idx]

    for param in param_values.keys():
        if len(param_values[param]) > 1 and run_comparison:

            fig, axs = plt.subplots(nrows=len(param_values[param]), figsize=(10, 9))
            test_data = []

            for i, comparison in enumerate(list(itertools.combinations(param_values[param], 2))):

                if all(isinstance(element, int) for element in comparison):
                    comparison = sorted(comparison, reverse=True)

                for value in comparison:
                    # Get data for every subject in the selected voxels
                    voxels_data = []
                    for stc in stcs_default_dict[param][value]:
                        stc_df = stc.to_data_frame()
                        voxels_columns = [f'VOL_{selected_voxel}' for selected_voxel in selected_voxels]
                        voxel_data = stc_df.loc[:, stc_df.columns.isin(voxels_columns)].values
                        voxels_data.append(voxel_data)

                    # Convert data to array
                    voxels_data = np.array(voxels_data)
                    # Average over voxels
                    subects_data = voxels_data.mean(axis=2)

                    test_data.append(subects_data)

                    # Extract mean and std over subjects
                    ga_data = subects_data.mean(axis=0)
                    ga_std = subects_data.std(axis=0) / np.sqrt(len(subects_data)) # Dividir por sqrt(len(subjects))

                    # Plot
                    axs[i].plot(stc.times, ga_data, label=value)
                    axs[i].fill_between(x=stc.times, y1=ga_data - ga_std, y2=ga_data + ga_std, alpha=0.4)

                # TFCE test
                t_tfce, clusters, p_tfce, H0 = permutation_cluster_test(X=test_data, threshold=t_thresh, n_permutations=n_permutations)
                good_clusters_idx = np.where(p_tfce < pval_threshold)[0]
                significant_clusters = [clusters[c][0] for c in good_clusters_idx]

                # Plot significant clusters
                bar_height = axs[i].get_ylim()[1]
                if len(significant_clusters):
                    for cluster in significant_clusters:
                        axs[i].hlines(y=bar_height, xmin=stc.times[cluster[0]], xmax=stc.times[cluster[-1]], color='gray', alpha=0.7)

                # Legend
                handles, labels = axs[i].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axs[i].legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=fontsize)

                # Set labels
                axs[i].set_xlabel('time (s)', fontsize=fontsize)
                axs[i].set_ylabel('Activation', fontsize=fontsize)
                axs[i].xaxis.set_tick_params(labelsize=fontsize)
                axs[i].yaxis.set_tick_params(labelsize=fontsize)

                # Title

                axs[i].set_title(f'{comparison}' + str(p_tfce[good_clusters_idx]), fontsize=fontsize)

                # Remove blank space before and after
                axs[i].set_xlim(-0.2, 0.5)

            # Title
            fig.suptitle(f'{titles[j]}\n{str(selected_box[j])}\n(t_thres: {round(t_thresh, 2)} - p_thresh: {pval_threshold}', fontsize=fontsize)

            fig.tight_layout()

            # Save
            save.fig(fig=fig, path=fig_path_diff, fname=titles[j])


## it vs tgt all channels FRF
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
evt_dur = None

# Window durations
cross1_dur, cross2_dur, mss_duration, vs_dur = functions_general.get_duration()
if 'vs' in epoch_ids:
    trial_dur = vs_dur[mss]  # Edit this to determine the minimum visual search duration for the trial selection (this will only affect vs epoching)
else:
    trial_dur = None

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

    # Get time windows from epoch_id name
    map = dict(tgt_fix={'tmin': -0.3, 'tmax': 0.6, 'plot_xlim': (-0.1, 0.5)},
               it_fix_subsampled={'tmin': -0.5, 'tmax': 3, 'plot_xlim': (-0.1, 0.5)})
    tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, mss=mss, plot_edge=0, map=map)
    # tmin, tmax, plot_xlim = -0.3, 0.6, (-0.1, 0.5)

    # Baseline
    baseline = (tmin, -0.05)

    # list of evoked objects to make GA
    evokeds_ga[epoch_id] = []
    #" List of evoked data as array to compute TFCE
    evokeds_data[epoch_id] = []

    # Specific run path for saving data and plots
    save_id = f'{epoch_id}_mss{mss}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}'
    run_path = f'Band_{band_id}/{save_id}/'

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
grand_avg = mne.grand_average(evokeds_ga['tgt_fix'], interpolate_bads=True)

# Permutation cluster test parameters
n_permutations = 1024
degrees_of_freedom = len(exp_info.subjects_ids) - 1
desired_tval = 0.01
# t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
t_thresh = dict(start=0, step=0.2)
# Get channel adjacency
ch_adjacency_sparse = functions_general.get_channel_adjacency(info=grand_avg.info)
# Clusters out type
if type(t_thresh) == dict:
    out_type = 'indices'
else:
     out_type = 'mask'

# Permutations cluster test (TFCE if t_thresh as dict)
t_tfce, clusters, p_tfce, H0 = permutation_cluster_test(X=observations, threshold=t_thresh, adjacency=ch_adjacency_sparse,
                                                        n_permutations=n_permutations, out_type=out_type, n_jobs=6)

pval_threshold = 0.05
# Make clusters mask
if type(t_thresh) == dict:
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
    title = f'{chs_id}_tTFCE_p{pval_threshold}'
else:
     title = f'{chs_id}_t{round(t_thresh, 2)}_p{pval_threshold}'

fig = grand_avg.plot_image(cmap='bwr', mask=clusters_mask, mask_style='mask', mask_alpha=0.5,
                           titles=title, axes=ax, show=display_figs)

if save_fig:
    fig_path = paths().plots_path() + f'Evoked_{data_type}/it_vs_tgt/'
    fname = (save_id + f'_{title}').replace(epoch_id, 'tgt_fix')
    save.fig(fig=fig, path=fig_path, fname=fname)


## it vs tgt all channels TRF
import load
import mne
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
import save
import setup
from paths import paths
import functions_general
import numpy as np
import scipy
import scipy.stats

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

# ----- Save data and display figures -----#
save_fig = True
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

# -----  Select frequency band -----#
# ICA vs raw data
use_ica_data = True
standarize = False
band_id = None
# Id
epoch_ids = ['it_fix_subsampled', 'tgt_fix', 'blue', 'red']
# Pick MEG chs (Select channels or set picks = 'mag')
chs_id = 'mag'
# Trials
corr_ans = True
tgt_pres = True
mss = None
reject = None
trial_dur = None
evt_dur = None
# TRF hiper-parameter
alpha = 1000

# Get time windows from epoch_id name
tmin, tmax, plot_xlim = -0.3, 0.6, (-0.1, 0.5)
# Baseline
baseline = (tmin, -0.05)

# Data type
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Save path
save_path = paths().save_path() + f'TRF_{data_type}/{epoch_ids}_mss{mss}_corrans{corr_ans}_tgtpres{tgt_pres}_trialdur{trial_dur}' \
                                  f'_evtdur{evt_dur}_{tmin}_{tmax}_bline{baseline}_alpha{alpha}_std{standarize}/{chs_id}/'
fig_path = save_path.replace(paths().save_path(), paths().plots_path()) + 'it_vs_tgt/'

# Variable to store data and plot
evokeds_ga = {}
evokeds_data = {}

# Define observations array to pass to TFCE
observations = []

for i, epoch_id in enumerate(epoch_ids[:2]):
    # list of evoked objects to make GA
    evokeds_ga[epoch_id] = []
    # " List of evoked data as array to compute TFCE
    evokeds_data[epoch_id] = []

    # Iterate over subjects
    for subject_code in exp_info.subjects_ids:

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

        # Get model coeficients as separate responses to target and items
        # All or multiple regions
        if type(rf) == dict:
            # Define evoked from TRF list to concatenate all
            evoked_list = []

            # iterate over regions
            for chs_idx, chs_subset in enumerate(rf.keys()):
                # Get channels subset info
                picks = functions_general.pick_chs(chs_id=chs_subset, info=meg_data.info)
                meg_sub = meg_data.copy().pick(picks)

                # Get TRF coeficients from chs subset
                trf = rf[chs_subset].coef_[:, i, :]

                if chs_idx == 0:
                    # Define evoked object from arrays of TRF
                    evoked = mne.EvokedArray(data=trf, info=meg_sub.info, tmin=tmin, baseline=baseline)
                else:
                    # Append evoked object from arrays of TRF to list, to concatenate all
                    evoked_list.append(mne.EvokedArray(data=trf, info=meg_sub.info, tmin=tmin, baseline=baseline))

            # Concatenate evoked from al regions
            evoked = evoked.add_channels(evoked_list)

        else:
            trf = rf.coef_[:, i, :]
            # Define evoked objects from arrays of TRF
            evoked = mne.EvokedArray(data=trf, info=meg_sub.info, tmin=tmin, baseline=baseline)

        # Extract evoked data as array to use in TFCE
        evoked_data = evoked.get_data()

        # Append subject's evoked to list of evokeds of actual condition to compute Ga
        evokeds_ga[epoch_id].append(evoked)

        # Append subject's evoked to list of evokeds of actual condition to test
        evokeds_data[epoch_id].append(evoked_data.T)

    # Convert to array
    evokeds_epoch_id = np.array(evokeds_data[epoch_id])

    # Append to observations list
    observations.append(evokeds_epoch_id)

# Compute grand average of target fixations
grand_avg = mne.grand_average(evokeds_ga['tgt_fix'], interpolate_bads=True)

# Permutation cluster test parameters
n_permutations = 1024
degrees_of_freedom = len(exp_info.subjects_ids) - 1
desired_tval = 0.01
t_thresh = scipy.stats.t.ppf(1 - desired_tval / 2, df=degrees_of_freedom)
# t_thresh = dict(start=0, step=0.2)
# Get channel adjacency
ch_adjacency_sparse = functions_general.get_channel_adjacency(info=evoked.info)
# Clusters out type
if type(t_thresh) == dict:
    out_type = 'indices'
else:
    out_type = 'mask'

# Permutations cluster test (TFCE if t_thresh as dict)
t_tfce, clusters, p_tfce, H0 = permutation_cluster_test(X=observations, threshold=t_thresh,
                                                        adjacency=ch_adjacency_sparse,
                                                        n_permutations=n_permutations, out_type=out_type, n_jobs=6)

pval_threshold = 0.05
# Make clusters mask
if type(t_thresh) == dict:
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

fig = grand_avg.plot_image(cmap='bwr', mask=clusters_mask, mask_style='mask', mask_alpha=0.5,
                           titles=title, axes=ax, show=display_figs)

if save_fig:
    fname = (f'{title}').replace(epoch_id, 'tgt_fix')
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