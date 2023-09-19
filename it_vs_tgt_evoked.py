import load
import mne
import matplotlib.pyplot as plt

import save
import setup
from paths import paths
import functions_general
import numpy as np

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
fig, axs = plt.subplots(ncols=2, figsize=(17, 5))

for i, epoch_id in enumerate(epoch_ids):
    evokeds[epoch_id] = {}
    means[epoch_id] = {}
    stds[epoch_id] = {}
    maxs[epoch_id] = {}
    mins[epoch_id] = {}
    for j, chs_id in enumerate(chs_ids):
        evokeds[epoch_id][chs_id] = []

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

            # Exclude subjects bad channels
            bad_ch_idx = [i for i, ch_name in enumerate(evoked.info['ch_names']) if ch_name in evoked.info['bads']]
            bads.append(evoked.info['bads'])
            evoked_data = np.delete(evoked_data, bad_ch_idx, axis=0)

            # Append evoked averaged over all channels
            evokeds[epoch_id][chs_id].append(evoked_data.mean(0))

        # Convert to array
        evokeds[epoch_id][chs_id] = np.array(evokeds[epoch_id][chs_id])

        # Take mean and sem over subjects
        means[epoch_id][chs_id] = evokeds[epoch_id][chs_id].mean(0)*1e15
        stds[epoch_id][chs_id] = evokeds[epoch_id][chs_id].std(0)/np.sqrt(len(evokeds[epoch_id][chs_id]))*1e15

        # Get min and max values
        maxs[epoch_id][chs_id] = means[epoch_id][chs_id].max()
        mins[epoch_id][chs_id] = means[epoch_id][chs_id].min()

        # Plot
        axs[j].plot(evoked.times, means[epoch_id][chs_id], color=f'C{i}', label=labels[i])
        axs[j].fill_between(evoked.times, y1=means[epoch_id][chs_id]-stds[epoch_id][chs_id],
                            y2=means[epoch_id][chs_id]+stds[epoch_id][chs_id], alpha=0.5, color=f'C{i}', edgecolor=None)
        axs[j].set_xlim(evoked.times[0], evoked.times[-1])

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
fig.suptitle(f'Target vs. Distractor FRF {chs_id}')

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
    fig_path = paths().plots_path() + f'Evoked_{data_type}/it_vs_fix/'
    fname = save_id.replace(epoch_id, 'Evoked')
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
        if FRF_TRF:
            axs[j].plot(evoked.times, means[epoch_id][lat_chs], color=f'C{i+2}', label=labels[i])
        else:
            axs[j].plot(evoked.times, means[epoch_id][lat_chs], color=f'C{i}', label=labels[i])
            axs[j].fill_between(evoked.times, y1=means[epoch_id][lat_chs]-stds[epoch_id][lat_chs],
                                y2=means[epoch_id][lat_chs]+stds[epoch_id][lat_chs], alpha=0.5, color=f'C{i}', edgecolor=None)
            axs[j].set_xlim(evoked.times[0], evoked.times[-1])

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
    fig.suptitle(f'Target vs. Distractor FRF-TRF {chs_id}')

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
    fig_path = paths().plots_path() + save_path + 'it_vs_fix/'
    if FRF_TRF:
        fname = f'FRF_TRF_{chs_id}'
    else:
        fname = f'TRF_{chs_id}'
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