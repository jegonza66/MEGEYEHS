import setup
import load
import save
import plot_preproc
import functions_preproc
from paths import paths
import numpy as np
import functions_general
import matplotlib.pyplot as plt
import scipy.signal as sgn

plots_path = paths().plots_path()

subject_code = 12

# ---------------- Load data ----------------#
# Load experiment info
exp_info = setup.exp_info()

# Load run configuration
config = load.config(path=paths().config_path(), fname='config.pkl')

# Define subject
subject = setup.raw_subject(exp_info=exp_info, config=config, subject_code=subject_code)

# Load Meg data
raw = subject.load_raw_meg_data()
raw.pick(exp_info.et_channel_names + [exp_info.trig_ch])
trig_data = raw.get_data(picks=exp_info.trig_ch)[0]
# Get vs start from meg
vs_start_meg = np.where(np.diff(trig_data) == 255)[0][:-1]+1
vs_start_meg_times = raw.times[vs_start_meg]

print('Downsampling MEG data')
raw.resample(1000)

# Get sample num of vs start
vs_start_meg_ds, _ = functions_general.find_nearest(array=raw.times, values=vs_start_meg_times)

# Get et data
et_chs = raw.get_data(picks=exp_info.et_channel_names)
# Get separate data from et channels
meg_gazex_data_raw = et_chs[0]

# Load ET data
et_data = subject.load_raw_et_data()
et_gazex = np.asarray(et_data['samples'][1])
# Define array of times with reset index to map ET time to samples
et_times = np.asarray(et_data['time'])
# Ge vs start from et
vs_start_msg = 'ETSYNC 250'
vs_starttime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(vs_start_msg)][1].str.split(vs_start_msg, expand=True)[0].values.astype(float)
vs_start_et, _ = functions_general.find_nearest(array=et_times, values=vs_starttime_et)

trig_num = 0
samples_lag = []
max_corrs = []

for trig_num in range(len(vs_start_et)-1):
# for trig_num in range(5):

    et_gaze_trial = et_gazex[vs_start_et[trig_num]:vs_start_et[trig_num + 1]]
    meg_gaze_trial = meg_gazex_data_raw[vs_start_meg_ds[trig_num]:vs_start_meg_ds[trig_num+1]]

    et_drop = 500
    max_sample, corrs = functions_general.align_signals(signal_1=meg_gaze_trial, signal_2=et_gaze_trial[et_drop:-et_drop])

    samples_lag_trial = -(max_sample - et_drop)
    samples_lag.append(samples_lag_trial)
    max_corrs.append(np.max(corrs))

    # fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 7))
    # plt.suptitle(f'Samples lag : {samples_lag}')
    # axs[0].plot(et_gaze_trial)
    # axs[0].set_ylabel('ET Gaze x')
    # axs[0].grid()
    # axs[1].plot(meg_gaze_trial)
    # axs[1].set_ylabel('MEG Gaze x')
    # axs[1].set_xlabel('Samples')
    # axs[1].grid()




    # # Positive peaks
    # et_peaks_p = sgn.find_peaks(et_gazex[vs_start_et[trig_num]:vs_start_et[trig_num]+1000], distance=100)[0]
    # meg_peaks_p = sgn.find_peaks(meg_gazex_data_raw[vs_start_meg_ds[trig_num]:vs_start_meg_ds[trig_num+1]], distance=120)[0]
    #
    # et_prominences_p = sgn.peak_prominences(et_gazex[vs_start_et[trig_num]:vs_start_et[trig_num+1]], et_peaks_p)[0]
    # meg_prominences_p = sgn.peak_prominences(meg_gazex_data_raw[vs_start_meg_ds[trig_num]:vs_start_meg_ds[trig_num+1]], meg_peaks_p)[0]
    #
    # # Negative peaks
    # et_peaks_n = sgn.find_peaks(-et_gazex[vs_start_et[trig_num]:vs_start_et[trig_num]+1000])[0]
    # meg_peaks_n = sgn.find_peaks(-meg_gazex_data_raw[vs_start_meg_ds[trig_num]:vs_start_meg_ds[trig_num]+1000])[0]
    #
    # et_prominences_n = sgn.peak_prominences(-et_gazex[vs_start_et[trig_num]:vs_start_et[trig_num]+1000], et_peaks_n)[0]
    # meg_prominences_n = sgn.peak_prominences(-meg_gazex_data_raw[vs_start_meg_ds[trig_num]:vs_start_meg_ds[trig_num]+1000], meg_peaks_n)[0]
    #
    # if np.max(et_prominences_n) > np.max(et_prominences_p):
    #     et_peaks, et_prominences = et_peaks_n, et_prominences_n
    #     meg_peaks, meg_prominences = meg_peaks_n, meg_prominences_n
    # else:
    #     et_peaks, et_prominences = et_peaks_p, et_prominences_p
    #     meg_peaks, meg_prominences = meg_peaks_p, meg_prominences_p
    #
    # et_peak = et_peaks[np.argmax(et_prominences)]
    # meg_peak = meg_peaks[np.argmax(meg_prominences)]
    #
    # et_peak_time = et_peak/1000
    # meg_peak_time = meg_peak/1000
    #
    # time_dif = (et_peak_time - meg_peak_time)*1000
    #
    # # plt.ioff()
    # fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 7))
    # plt.suptitle(f'Time difference: {time_dif}')
    # axs[0].plot(et_times[vs_start_et[trig_num]:vs_start_et[trig_num]+1000]-et_times[vs_start_et[trig_num]], et_gazex[vs_start_et[trig_num]:vs_start_et[trig_num]+1000])
    # axs[0].vlines(x=et_peak_time*1000, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], colors='grey', linestyles='--')
    # axs[0].set_ylabel('ET Gaze x')
    # axs[0].grid()
    # axs[1].plot((raw.times[vs_start_meg_ds[trig_num]:vs_start_meg_ds[trig_num]+1000]-raw.times[vs_start_meg_ds[trig_num]])*1000, meg_gazex_data_raw[vs_start_meg_ds[trig_num]:vs_start_meg_ds[trig_num]+1000])
    # axs[1].vlines(x=meg_peak_time*1000, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], colors='grey', linestyles='--')
    # axs[1].set_ylabel('MEG Gaze x')
    # axs[1].set_xlabel('Time')
    # axs[1].grid()
    # save.fig(fig=fig, path=plots_path + 'DAC delay/', fname=f'trial_{trig_num}.png')


##
import numpy as np

subject_code = 0

#---------------- Load data ----------------#
# Load experiment info
exp_info = setup.exp_info()

# Load run configuration
config = load.config(path=paths().config_path(), fname='config.pkl')

# Define subject
subject = setup.raw_subject(exp_info=exp_info, config=config, subject_code=subject_code)

# Load Meg data
raw = subject.load_raw_meg_data()

# Split chs
right_chs = [ch_name for ch_name in raw.ch_names if 'MR' in ch_name]
left_chs = [ch_name for ch_name in raw.ch_names if 'LR' in ch_name]

# Plot parameters
# Separate R L
group_by = dict(left=left_chs, light=right_chs)

