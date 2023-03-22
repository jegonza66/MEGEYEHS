## Beta supression saccades correlation


import functions_general
import functions_analysis
import load
import mne
import numpy as np
import plot_general
import save
import setup
from paths import paths
import matplotlib.pyplot as plt

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
chs_id = 'parietal'
# ICA / RAW
use_ica_data = True
band_id = 'Beta'
corr_ans = None
tgt_pres = None
# mss = None
epoch_id = 'ms'
# epoch_id = 'fix_vs'
# Power frequency range
l_freq=1
h_freq = 100
log_bands = True

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

sac_ms_num = []
mean_powers = []

for mss in [1, 2, 4]:

    # Duration
    if 'cross1' in epoch_id and mss:
        dur = cross1_dur + mss_duration[mss] + cross2_dur + vs_dur  # seconds
    elif 'ms' in epoch_id:
        dur = mss_duration[mss] + cross2_dur + vs_dur
    elif 'cross2' in epoch_id:
        dur = cross2_dur + vs_dur  # seconds
    else:
        dur = 0

    # Get time windows from epoch_id name
    map_times = dict(cross1={'tmin': 0, 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                     ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
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
    elif 'cross1' in epoch_id or 'ms' in epoch_id or 'cross2' in epoch_id and mss:
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
        sac_ms_num.append(ms_sac)

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
                    meg_data = subject.load_preproc_meg()
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

        # Pick plot channels
        picks = functions_general.pick_chs(chs_id=chs_id, info=power.info)
        power.pick(picks=picks)
        # Apply baseline correction
        power.apply_baseline(baseline=baseline, mode=bline_mode)
        # Crop to Beta power on MS screen
        fmin, fmax = functions_general.get_freq_band(band_id=band_id)
        power.crop(tmin=0, tmax=mss_duration[mss], fmin=fmin, fmax=fmax)

        # Get mean power over time and frequencies and append
        mean_power = np.mean(power.data)
        mean_powers.append(mean_power)

# Linear fit and plot
m, b = np.polyfit(sac_ms_num, mean_powers, 1)

plt.figure()
plt.plot(sac_ms_num, mean_powers, '.')
plt.plot(sac_ms_num, m*np.array(sac_ms_num)+b, '--k')
plt.xlabel('Saccades number')
plt.ylabel('Beta power')






