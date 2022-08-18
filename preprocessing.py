import mne
import pickle
import os

from paths import paths
import load
import plot
import functions


#---------------- Paths ----------------#
Preproc_data_path = paths().preproc_path()


#---------------- Load data ----------------#
# Define subject
subject = load.subject()

# Load edf data
et_data_edf = subject.et_data()
et_channels_edf = et_data_edf['samples']
edf_data_srate = 1000.0

# Get x, y and pupil size
edf_time = et_channels_edf[0].values
edf_gazex_data = et_channels_edf[1].values
edf_gazey_data = et_channels_edf[2].values
edf_pupils_data = et_channels_edf[3].values

# Load Meg data
raw = subject.ctf_data()

# Get et channels
gazex_ch_name = 'UADC001-4123'
gazey_ch_name = 'UADC002-4123'
pupils_ch_name = 'UADC013-4123'
et_channel_names = [gazex_ch_name, gazey_ch_name, pupils_ch_name]
print('\nGetting ET channels from MEG')
et_channels_meg = raw.get_data(picks=et_channel_names)

# Get x, y and pupil size
meg_gazex_data = et_channels_meg[0]
meg_gazey_data = et_channels_meg[1]
meg_pupils_data = et_channels_meg[2]


#---------------- Reescaling based on eyemap matching ----------------#
# Set Gaze "y" Eyemap offsets
# This value is due to the fact that "y" Eyemap goes approximately 5500 samples after the "x" Eyemap
y_offset = 5500
# Due to different sampling rates, computing meg offset based on edf offset for y Eyemap
meg_offset = int(y_offset * raw.info['sfreq'] / edf_data_srate)

Scaled = False
while not Scaled:
    # Plot EDF and MEG signals to select ranges for scaling and get plot lims for scaling signals in those ranges
    fig1, eyemap_interval_edf, eyemap_interval_meg = \
        plot.get_intervals_signals(reference_signal=edf_gazex_data, signal_to_scale=meg_gazex_data)

    print('Scaling MEG Eye-Tracker data')
    for meg_gaze_data, edf_gaze_data, offset, title in zip((meg_gazex_data, meg_gazey_data, meg_pupils_data),
                                                    (edf_gazex_data, edf_gazey_data, edf_pupils_data),
                                                    (0, y_offset, 0), ('Gaze x', 'Gaze y', 'Pupils')):

        # Due to different sampling rates, computing meg offset based on edf offset for y Eyemap
        meg_offset_y = int(offset * raw.info['sfreq'] / edf_data_srate)
        eyemap_interval_edf_offset = (eyemap_interval_edf[0] + offset, eyemap_interval_edf[1] + offset)
        eyemap_interval_meg_offset = (eyemap_interval_meg[0] + meg_offset_y, eyemap_interval_meg[1] + meg_offset_y)

        # Re-scale MEG ET data
        functions.scale_from_interval(signal_to_scale=meg_gaze_data,
                                      reference_signal=edf_gaze_data,
                                      interval_signal=eyemap_interval_meg_offset,
                                      interval_ref=eyemap_interval_edf_offset)

    # Plot scaled signals
    fig2 = plot.scaled_signals(time=edf_time, scaled_signals=[meg_gazex_data, meg_gazey_data, meg_pupils_data],
                          reference_signals=[edf_gazex_data, edf_gazey_data, edf_pupils_data],
                          interval_signal=eyemap_interval_meg, interval_ref=eyemap_interval_edf,
                          ref_offset=[0, y_offset, 0], signal_offset=[0, meg_offset, 0],
                          ylabels=['Gaze x', 'Gaze y', 'Pupil size'])

    # Plotting and choosing different time in signal to check scaling
    fig1, eyemap_interval_edf, eyemap_interval_meg = plot.get_intervals_signals(reference_signal=edf_gazex_data,
                                                                                signal_to_scale=meg_gazex_data,
                                                                                fig=fig1)

    # Plot scaled signals
    fig2 = plot.scaled_signals(time=edf_time, scaled_signals=[meg_gazex_data, meg_gazey_data, meg_pupils_data],
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

## FIXATIONS
sfixs, ffixs = functions.fixation_detection(x=meg_gazex_data, y=meg_gazey_data, time=raw.times,
                                          missing=1e6, maxdist=50, mindur=100/1000)
efixs = [fix[1] for fix in ffixs]

import numpy as np
import matplotlib.pyplot as plt
plt.figure()
plt.title('Fixations detection')
plt.plot(raw.times, meg_gazex_data)
plt.vlines(x=sfixs, ymin=np.min(meg_gazex_data), ymax=np.max(meg_gazex_data), color='black', linestyles='--', label='Fix. start')
plt.vlines(x=efixs, ymin=np.min(meg_gazex_data), ymax=np.max(meg_gazex_data), color='red', linestyles='--', label='Fix. end')
plt.xlabel('Time [s]')
plt.ylabel('Gaze x')
plt.legend(loc='upper right')

##---------------- Save scaled data to meg data ----------------#
print('Saving scaled et data to meg raw data structure')
# copy raw structure
raw_et = raw.copy()
# make new raw structure from et channels only
raw_et = mne.io.RawArray(et_channels_meg, raw_et.pick(et_channel_names).info)
# change channel names
for ch_name, new_name in zip(raw_et.ch_names, ['ET_gaze_x', 'ET_gaze_y', 'ET_pupils']):
    raw_et.rename_channels({ch_name: new_name})
# save to original raw structure
raw.load_data()
raw.add_channels([raw_et])
del(raw_et)


##---------------- Save preprocesed data ----------------#
print('Saving preprocesed data')
os.makedirs(Preproc_data_path, exist_ok=True)
f = open(Preproc_data_path + f'Subject_{subject.subject_id}.pkl', 'wb')
pickle.dump(raw, f)
f.close()
print(f'Preprocesed data saved to {Preproc_data_path + f"Subject_{subject.subject_id}.pkl"}')