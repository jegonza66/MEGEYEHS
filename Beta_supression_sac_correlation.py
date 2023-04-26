## Beta supression saccades correlation
import functions_general
import functions_analysis
import load
import mne
import numpy as np
import seaborn as sn
import pandas as pd
import setup
from paths import paths
import matplotlib.pyplot as plt
import save

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
chs_id = 'occipital'
# ICA / RAW
use_ica_data = True
band_id = 'Beta'
corr_ans = None
tgt_pres = None
# mss = None
epoch_id = 'ms'
# epoch_id = 'fix_vs'
# Power frequency range
l_freq = 1
h_freq = 40
log_bands = False

# Baseline method
bline_mode = 'logratio'
#----------#

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 0}
cross1_dur = 0.75
cross2_dur = 1
vs_dur = 4
plot_edge = 0.1

# freqs type
if log_bands:
    freqs_type = 'log'
else:
    freqs_type = 'lin'

if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

sac_ms_num = {}
mean_powers = []
std_powers = {}

for mss in [1, 2, 4]:

    # Define sac number and power variable
    sac_ms_num[mss] = []
    # mean_powers[mss] = []
    # std_powers[mss] = []

    # Duration
    if 'ms' in epoch_id:
        dur = mss_duration[mss] + cross2_dur + vs_dur
    elif 'cross2' in epoch_id:
        dur = cross2_dur + vs_dur  # seconds
    else:
        dur = 0

    # Get time windows from epoch_id name
    map_times = dict(ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
                     cross2={'tmin': -cross1_dur - mss_duration[mss], 'tmax': dur,
                             'plot_xlim': (-cross1_dur - mss_duration[mss] + plot_edge, dur - plot_edge)},
                     sac={'tmin': -0.2, 'tmax': 0.2, 'plot_xlim': (-0.2, 0.2)},
                     fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)})
    tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

    # Baseline duration
    if 'sac' in epoch_id:
        baseline = (tmin, 0)
        # baseline = None
    elif 'fix' in epoch_id or 'fix' in epoch_id:
        baseline = (tmin, -0.05)
    elif 'ms' in epoch_id or 'cross2' in epoch_id and mss:
        baseline = (tmin, 0)
    else:
        baseline = (tmin, 0)

    # Specific run path for saving data and plots
    save_id = f'{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}'
    save_path = f'/{save_id}_{tmin}_{tmax}_bline{baseline}/'
    plot_path = f'/{save_id}_{plot_xlim[0]}_{plot_xlim[1]}_bline{baseline}/'
    if use_ica_data:
        data_type = 'ICA'
    else:
        data_type = 'RAW'

    # Save data paths
    trf_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + save_path
    epochs_save_path = paths().save_path() + f'Epochs_{data_type}/Band_None/' + save_path

    averages_power = []

    for subject_code in exp_info.subjects_ids:
        # Define save path and file name for loading and saving epoched, evoked, and GA data
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

        # Data filenames
        power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

        # MS saccades
        ms_sac = len(subject.saccades.loc[(subject.saccades['screen'] == 'ms') & (subject.saccades['mss'] == mss)])
        sac_ms_num[mss].append(ms_sac)

        try:
            # Load previous data
            power = mne.time_frequency.read_tfrs(trf_save_path + power_data_fname, condition=0)
        except:
            try:
                # Load epoched data
                epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
            except:
                # Get Epochs from Raw data
                if use_ica_data:
                    # Load meg data
                    meg_data = load.ica_data(subject=subject)
                else:
                    # Load meg data
                    meg_data = subject.load_preproc_meg_data()
                # Epoch data
                epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans,
                                                               tgt_pres=tgt_pres, epoch_id=epoch_id, meg_data=meg_data,
                                                               tmin=tmin, tmax=tmax, reject=dict(mag=1),
                                                               save_data=save_data, epochs_save_path=epochs_save_path,
                                                               epochs_data_fname=epochs_data_fname)

            # Compute power and PLI over frequencies
            power = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                           freqs_type=freqs_type, return_itc=False,
                                                           n_cycles_div=2., save_data=save_data,
                                                           trf_save_path=trf_save_path,
                                                           power_data_fname=power_data_fname)

        # Append to GA
        averages_power.append(power)

        # # Pick plot channels
        # picks = functions_general.pick_chs(chs_id=chs_id, info=power.info)
        # power.pick(picks=picks)
        # # Apply baseline correction
        # power.apply_baseline(baseline=baseline, mode=bline_mode)
        # # Crop to Beta power on MS screen
        # fmin, fmax = functions_general.get_freq_band(band_id=band_id)
        # power.crop(tmin=0, tmax=mss_duration[mss], fmin=fmin, fmax=fmax)
        #
        # # Get mean power over time and frequencies and append
        # mean_power = np.mean(power.data)
        # mean_powers.append(mean_power)


    grand_avg_power = mne.grand_average(averages_power)

    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power.info)
    grand_avg_power.pick(picks=picks)
    # Apply baseline correction
    grand_avg_power.apply_baseline(baseline=baseline, mode=bline_mode)
    # Crop to Beta power on MS screen
    fmin, fmax = functions_general.get_freq_band(band_id=band_id)
    grand_avg_power.crop(tmin=0, tmax=mss_duration[mss], fmin=fmin, fmax=fmax)

    # Get mean power over time and frequencies and append
    mean_power = np.mean(grand_avg_power.data)
    std_power = np.std(grand_avg_power.data)

    mean_powers.append(grand_avg_power.data.ravel())
    # std_powers.append(std_power)


sac_num = []
for i in sac_ms_num.keys():
    sac_num.append(int(np.mean(sac_ms_num[i])))

fig, ax = plt.subplots()
bp = ax.boxplot(x=mean_powers, widths=[100, 100, 100], positions=sac_num)
ax.set_ylabel('Power')
ax.set_xlabel('Mean saccades')
plt.title('Beta power and mean saccade number \n'
          'in MS screen for MSS 1, 2 and 4')

plt.tick_params(labelsize=13)
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
fig.tight_layout()


if save_fig:
    fig_path = paths().plots_path()
    fname = f'Beta_supression_{chs_id}'
    save.fig(fig=fig, path=fig_path, fname=fname)


## Beta supression saccades aligned
import functions_general
import functions_analysis
import load
import mne
import numpy as np
import seaborn as sn
import pandas as pd
import setup
from paths import paths
import matplotlib.pyplot as plt
import save

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
chs_id = 'occipital'
# ICA / RAW
use_ica_data = True
band_id = 'Beta'
corr_ans = None
tgt_pres = None
# mss = None
epoch_id = 'sac_ms'
# epoch_id = 'fix_vs'
# Power frequency range
l_freq = 1
h_freq = 40
log_bands = False

# Baseline method
bline_mode = 'logratio'
#----------#

# Duration
mss_duration = {1: 2, 2: 3.5, 4: 5, None: 0}
cross1_dur = 0.75
cross2_dur = 1
vs_dur = 4
plot_edge = 0.1

# freqs type
if log_bands:
    freqs_type = 'log'
else:
    freqs_type = 'lin'

if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

sac_ms_num = {}
mean_powers = []
std_powers = {}

for mss in [1, 2, 4]:

    # Define sac number and power variable
    sac_ms_num[mss] = []
    # mean_powers[mss] = []
    # std_powers[mss] = []

    # Duration
    if 'ms' in epoch_id:
        dur = mss_duration[mss] + cross2_dur + vs_dur
    elif 'cross2' in epoch_id:
        dur = cross2_dur + vs_dur  # seconds
    else:
        dur = 0

    # Get time windows from epoch_id name
    map_times = dict(ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
                     cross2={'tmin': -cross1_dur - mss_duration[mss], 'tmax': dur,
                             'plot_xlim': (-cross1_dur - mss_duration[mss] + plot_edge, dur - plot_edge)},
                     sac={'tmin': -0.25, 'tmax': 0.25, 'plot_xlim': (-0.05, 0.05)},
                     fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)})
    tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

    # Baseline duration
    if 'sac' in epoch_id:
        baseline = (tmin, 0)
        plot_baseline = (plot_xlim[0], 0)
    elif 'fix' in epoch_id or 'fix' in epoch_id:
        baseline = (tmin, -0.05)
    elif 'ms' in epoch_id or 'cross2' in epoch_id and mss:
        baseline = (tmin, 0)
    else:
        baseline = (tmin, 0)

    # Specific run path for saving data and plots
    save_id = f'{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}'
    save_path = f'/{save_id}_{tmin}_{tmax}_bline{baseline}/'
    plot_path = f'/{save_id}_{plot_xlim[0]}_{plot_xlim[1]}_bline{baseline}/'
    if use_ica_data:
        data_type = 'ICA'
    else:
        data_type = 'RAW'

    # Save data paths
    trf_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + save_path
    epochs_save_path = paths().save_path() + f'Epochs_{data_type}/Band_None/' + save_path

    averages_power = []

    for subject_code in exp_info.subjects_ids:
        # Define save path and file name for loading and saving epoched, evoked, and GA data
        if use_ica_data:
            # Load subject object
            subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        else:
            # Load subject object
            subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

        # Data filenames
        power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

        # MS saccades
        ms_sac = len(subject.saccades.loc[(subject.saccades['screen'] == 'ms') & (subject.saccades['mss'] == mss)])
        sac_ms_num[mss].append(ms_sac)

        try:
            # Load previous data
            power = mne.time_frequency.read_tfrs(trf_save_path + power_data_fname, condition=0)
        except:
            try:
                # Load epoched data
                epochs = mne.read_epochs(epochs_save_path + epochs_data_fname)
            except:
                # Get Epochs from Raw data
                if use_ica_data:
                    # Load meg data
                    meg_data = load.ica_data(subject=subject)
                else:
                    # Load meg data
                    meg_data = subject.load_preproc_meg_data()
                # Epoch data
                epochs, events = functions_analysis.epoch_data(subject=subject, mss=mss, corr_ans=corr_ans,
                                                               tgt_pres=tgt_pres, epoch_id=epoch_id, meg_data=meg_data,
                                                               tmin=tmin, tmax=tmax, reject=dict(mag=1),
                                                               save_data=save_data, epochs_save_path=epochs_save_path,
                                                               epochs_data_fname=epochs_data_fname)

            # Compute power and PLI over frequencies
            power = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq,
                                                           freqs_type=freqs_type, return_itc=False,
                                                           n_cycles_div=2., save_data=save_data,
                                                           trf_save_path=trf_save_path,
                                                           power_data_fname=power_data_fname)

        # Append to GA
        averages_power.append(power)

        # # Pick plot channels
        # picks = functions_general.pick_chs(chs_id=chs_id, info=power.info)
        # power.pick(picks=picks)
        # # Apply baseline correction
        # power.apply_baseline(baseline=baseline, mode=bline_mode)
        # # Crop to Beta power on MS screen
        # fmin, fmax = functions_general.get_freq_band(band_id=band_id)
        # power.crop(tmin=0, tmax=mss_duration[mss], fmin=fmin, fmax=fmax)
        #
        # # Get mean power over time and frequencies and append
        # mean_power = np.mean(power.data)
        # mean_powers.append(mean_power)


    grand_avg_power = mne.grand_average(averages_power)

    # Pick plot channels
    picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_power.info)
    grand_avg_power.pick(picks=picks)
    # Apply baseline correction
    grand_avg_power.apply_baseline(baseline=plot_baseline, mode=bline_mode)
    # Crop to Beta power on MS screen
    fmin, fmax = functions_general.get_freq_band(band_id=band_id)
    grand_avg_power.crop(tmin=plot_xlim[0], tmax=plot_xlim[1], fmin=fmin, fmax=fmax)

    # Get mean power over time and frequencies and append
    mean_power = np.mean(grand_avg_power.data)
    std_power = np.std(grand_avg_power.data)

    mean_powers.append(grand_avg_power.data.ravel())
    # std_powers.append(std_power)


sac_num = []
for i in sac_ms_num.keys():
    sac_num.append(int(np.mean(sac_ms_num[i])))

fig, ax = plt.subplots()
bp = ax.boxplot(x=mean_powers, positions=[1, 2, 4])
ax.set_ylabel('Power')
ax.set_xlabel('MSS')
plt.title('Beta power aligned to saccades \n'
          'in MS screen for MSS 1, 2 and 4')

plt.tick_params(labelsize=13)
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
fig.tight_layout()


if save_fig:
    fig_path = paths().plots_path()
    fname = f'Beta_supression_saccades_{chs_id}'
    save.fig(fig=fig, path=fig_path, fname=fname)







