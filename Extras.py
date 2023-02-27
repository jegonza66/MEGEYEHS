import matplotlib.pyplot as plt
import numpy as np
import mne
import functions_general
import load
from paths import paths
import setup


## Plot channels
save_path = paths().save_path()
ica_path = paths().ica_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

# Subject
subject_code = exp_info.subjects_ids[0]

# Load subject object
subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
raw_data = subject.load_preproc_meg()
info = mne.pick_info(raw_data.info, mne.pick_types(raw_data.info, meg=True, ref_meg=False))

empty_data = np.zeros(info['nchan'])

channels_mask = np.array([True if 'P' in ch_name else False for ch_name in info.ch_names])

mask_params = dict(marker='o', markerfacecolor='orange', markeredgecolor='k', linewidth=0, markersize=6)

mne.viz.plot_topomap(data=empty_data, pos=info, mask=channels_mask, mask_params=mask_params)


## First fixation delay distribution

# Define subject
subject = load.raw_subject()

# Load data
raw = subject.load_raw_meg_data()
bh_data = subject.load_raw_bh_data()

# VS screen start time
vs_time = bh_data['key_resp.started'].values

# Fixatino 1 screen start time
cross1_time = bh_data['fixation_target_5.started'].values
# Fill missing fix 1 screens
cross1_time[np.where(cross1_time =='None')] = bh_data['fixation_target.started'].values[np.where(cross1_time =='None')]
# get times from trial 2 onwards to calculate inter trial time interval
cross1_time = np.append(cross1_time, float('nan'))[1:].astype(float)

delay = cross1_time - vs_time

plt.figure()
plt.title('Delay from VS start to next Trial fixation 1')
plt.plot(delay)
plt.ylim([5, 15])
plt.xlabel('Trial')
plt.ylabel('Time [s]')

##---------------- Reescaling based on eyemap matching ----------------#

# Load edf data
et_data_edf = subject.load_raw_et_data()
et_channels_edf = et_data_edf['samples']
edf_data_srate = 1000.0

# Get x, y and pupil size
edf_time = et_channels_edf[0].values
edf_gazex_data = et_channels_edf[1].values
edf_gazey_data = et_channels_edf[2].values
edf_pupils_data = et_channels_edf[3].values

# Set Gaze "y" Eyemap offsets
# This value is due to the fact that "y" Eyemap goes approximately 5500 samples after the "x" Eyemap
y_offset = 5500
# Due to different sampling rates, computing meg offset based on edf offset for y Eyemap
meg_offset = int(y_offset * raw.info['sfreq'] / edf_data_srate)

Scaled = False
while not Scaled:
    # Plot EDF and MEG signals to select ranges for scaling and get plot lims for scaling signals in those ranges
    fig1, eyemap_interval_edf, eyemap_interval_meg = \
        plot_preproc.get_intervals_signals(reference_signal=edf_gazex_data, signal_to_scale=meg_gazex_data_raw)

    print('Scaling MEG Eye-Tracker data')
    for meg_gaze_data, edf_gaze_data, offset, title in zip((meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw),
                                                           (edf_gazex_data, edf_gazey_data, edf_pupils_data),
                                                           (0, y_offset, 0), ('Gaze x', 'Gaze y', 'Pupils')):
        # Due to different sampling rates, computing meg offset based on edf offset for y Eyemap
        meg_offset_y = int(offset * raw.info['sfreq'] / edf_data_srate)
        eyemap_interval_edf_offset = (eyemap_interval_edf[0] + offset, eyemap_interval_edf[1] + offset)
        eyemap_interval_meg_offset = (eyemap_interval_meg[0] + meg_offset_y, eyemap_interval_meg[1] + meg_offset_y)

        # Re-scale MEG ET data
        functions_general.scale_from_interval(signal_to_scale=meg_gaze_data,
                                              reference_signal=edf_gaze_data,
                                              interval_signal=eyemap_interval_meg_offset,
                                              interval_ref=eyemap_interval_edf_offset)

    # Plot scaled signals
    fig2 = plot_preproc.scaled_signals(time=edf_time, scaled_signals=[meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw],
                               reference_signals=[edf_gazex_data, edf_gazey_data, edf_pupils_data],
                               interval_signal=eyemap_interval_meg, interval_ref=eyemap_interval_edf,
                               ref_offset=[0, y_offset, 0], signal_offset=[0, meg_offset, 0],
                               ylabels=['Gaze x', 'Gaze y', 'Pupil size'])

    # Plotting and choosing different time in signal to check scaling
    fig1, eyemap_interval_edf, eyemap_interval_meg = plot_preproc.get_intervals_signals(reference_signal=edf_gazex_data,
                                                                                signal_to_scale=meg_gazex_data_raw,
                                                                                fig=fig1)

    # Plot scaled signals
    fig2 = plot_preproc.scaled_signals(time=edf_time, scaled_signals=[meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw],
                               reference_signals=[edf_gazex_data, edf_gazey_data, edf_pupils_data],
                               interval_signal=eyemap_interval_meg, interval_ref=eyemap_interval_edf,
                               ref_offset=[0, y_offset, 0], signal_offset=[0, meg_offset, 0],
                               ylabels=['Gaze x', 'Gaze y', 'Pupil size'], fig=fig2)

    Answer = False
    while not Answer:
        Answer = input('Scaling correct?\n'
                       'y/n')
        if Answer == 'y':
            Scaled = True
        elif Answer == 'n':
            Scaled = False
        else:
            print('Please answer y/n')
            Answer = False