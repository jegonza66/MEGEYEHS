import functions_general
import functions_analysis
import load
import mne
import numpy as np
import os
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
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Parameters -----#
# Select channels
chs_id = 'occipital'
# ICA / RAW
use_ica_data = True
corr_ans = None
tgt_pres = None
mss = None
# epoch_id = 'ms_'
epoch_id = 'fix_vs'
save_id = f'{epoch_id}_mss{mss}_Corr_{corr_ans}_tgt_{tgt_pres}'
# Power frequency range
l_freq = 1
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

# Duration
if 'cross1' in epoch_id and mss:
    dur = cross1_dur + mss_duration[mss] + cross2_dur + vs_dur # seconds
elif 'ms' in epoch_id:
    dur = mss_duration[mss] + cross2_dur + vs_dur
elif 'cross2' in epoch_id:
    dur = cross2_dur + vs_dur  # seconds
else:
    dur = 0

# Get time windows from epoch_id name
map_times = dict(cross1={'tmin': 0, 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                 ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
                 cross2={'tmin': -cross1_dur - mss_duration[mss], 'tmax': dur, 'plot_xlim': (plot_edge, dur - plot_edge)},
                 sac={'tmin': -0.2, 'tmax': 0.2, 'plot_xlim': (-0.2, 0.2)},
                 fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)})
tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id, map=map_times)

# Baseline duration
if 'sac' in epoch_id:
    baseline = (tmin, 0)
    # baseline = None
elif 'fix' in epoch_id or 'fix' in epoch_id:
    baseline = (tmin, -0.05)
elif 'cross1' in epoch_id or 'ms' in epoch_id and mss:
    baseline = (tmin, tmin+cross1_dur)
elif 'cross2' in epoch_id:
    baseline = (tmin, tmin+cross1_dur)
else:
    baseline = (tmin, 0)

# Plot baseline
plot_baseline = (plot_xlim[0], 0)

# freqs type
if log_bands:
    freqs_type = 'log'
else:
    freqs_type = 'lin'

# Specific run path for saving data and plots
save_path = f'/{save_id}_{tmin}_{tmax}_bline{baseline}/'
plot_path = f'/{save_id}_{plot_xlim[0]}_{plot_xlim[1]}_bline{baseline}/'
if use_ica_data:
    data_type = 'ICA'
else:
    data_type = 'RAW'

# Save data paths
trf_save_path = paths().save_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + save_path
epochs_save_path = paths().save_path() + f'Epochs_{data_type}/Band_None/' + save_path
# Save figures paths
trf_fig_path = paths().plots_path() + f'Time_Frequency_{data_type}/{freqs_type}_freqs/' + plot_path + f'{chs_id}/'

# Grand average data variable
grand_avg_power_fname = f'Grand_Average_power_{l_freq}_{h_freq}_tfr.h5'
grand_avg_itc_fname = f'Grand_Average_itc_{l_freq}_{h_freq}_tfr.h5'
averages_power = []
averages_itc = []

for subject_code in exp_info.subjects_ids:

    # Define save path and file name for loading and saving epoched, evoked, and GA data
    if use_ica_data:
        # Load subject object
        subject = load.ica_subject(exp_info=exp_info, subject_code=subject_code)
        # Load meg data
        meg_data = load.ica_data(subject=subject)

    else:
        # Load subject object
        subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
        # Load meg data
        meg_data = subject.load_preproc_meg()

    # Data filenames
    power_data_fname = f'Power_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
    itc_data_fname = f'ITC_{subject.subject_id}_{l_freq}_{h_freq}_tfr.h5'
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'

    try:
        # Load previous data
        power = mne.time_frequency.read_tfrs(trf_save_path + power_data_fname, condition=0)
        itc = mne.time_frequency.read_tfrs(trf_save_path + itc_data_fname, condition=0)
    except:

        # Compute power and PLI over frequencies
        power, itc = functions_analysis.time_frequency(epochs=epochs, l_freq=l_freq, h_freq=h_freq, freqs_type=freqs_type,
                                                       n_cycles_div=4., save_data=save_data, trf_save_path=trf_save_path,
                                                       power_data_fname=power_data_fname, itc_data_fname=itc_data_fname)