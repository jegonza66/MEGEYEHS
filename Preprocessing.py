import mne
import numpy as np
import pickle
import os

import Load
import Plot


#---------------- Paths ----------------#
Preproc_data_path = 'Save/Preprocesed_Data/'


#---------------- Load data ----------------#
# Define subject
subject = Load.subject(1)

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
et_channels_meg = raw.get_data(picks=et_channel_names)

# Get x, y and pupil size
meg_gazex_data = et_channels_meg[0]
meg_gazey_data = et_channels_meg[1]
meg_pupils_data = et_channels_meg[2]

#---------------- Reescaling based on eyemap matching ----------------#
# Set Gaze y eyemap offsets
# This value is due to the fact that y eyemap goes approximately 5500 samples after the x eyemap
y_offset = 5500

Scaled = False
while not Scaled:
    # Plot EDF and MEG signals to select ranges for scaling
    fig, axs = Plot.signals_to_scale(edf_gazex_data=edf_gazex_data, meg_gazex_data=meg_gazex_data)

    # Get plot lims for scaling signals in those ranges
    eyemap_start_edf, eyemap_end_edf = (int(lim) for lim in axs[0].get_xlim())
    eyemap_start_meg, eyemap_end_meg = (int(lim) for lim in axs[1].get_xlim())

    print('Scaling MEG Eye-Tracker data')
    for meg_gaze_data, edf_gaze_data, offset, title in zip((meg_gazex_data, meg_gazey_data, meg_pupils_data),
                                                    (edf_gazex_data, edf_gazey_data, edf_pupils_data),
                                                    (0, y_offset, 0), ('Gaze x', 'Gaze y', 'Pupils')):
        # Re-scale MEG ET data
        meg_offset = int(offset * raw.info['sfreq'] / edf_data_srate)
        meg_gaze_data -= np.nanmin(meg_gaze_data[eyemap_start_meg + meg_offset: eyemap_end_meg + meg_offset])
        meg_gaze_data /= np.nanmax(meg_gaze_data[eyemap_start_meg + meg_offset: eyemap_end_meg + meg_offset])
        meg_gaze_data *= (
                np.nanmax(edf_gaze_data[eyemap_start_edf + offset: eyemap_end_edf + offset]) - np.nanmin(
            edf_gaze_data[eyemap_start_edf + offset: eyemap_end_edf + offset]))
        meg_gaze_data += np.nanmin(edf_gaze_data[eyemap_start_edf + offset: eyemap_end_edf + offset])

        # Plot scaled signals
        Plot.scaled_et(edf_time=edf_time, meg_data=meg_gaze_data, edf_data=edf_gaze_data,
                       eyemap_start_edf=eyemap_start_edf+offset, eyemap_end_edf=eyemap_end_edf+offset,
                       eyemap_start_meg=eyemap_start_meg+meg_offset, eyemap_end_meg=eyemap_end_meg+meg_offset,
                       title=title, xlabel='Pixels')

    # Check Scaling
    # Plot for choosing different time in signal and check scaling
    fig, axs = Plot.signals_to_scale(edf_gazex_data=edf_gazex_data, meg_gazex_data=meg_gazex_data)

    # Get plot lims for plotting both signals on same plot to check scaling
    eyemap_start_edf, eyemap_end_edf = (int(lim) for lim in axs[0].get_xlim())
    eyemap_start_meg, eyemap_end_meg = (int(lim) for lim in axs[1].get_xlim())

    # Plot scaled signals
    Plot.scaled_et(edf_time=edf_time, meg_data=meg_gazex_data, edf_data=edf_gazex_data,
                   eyemap_start_edf=eyemap_start_edf, eyemap_end_edf=eyemap_end_edf,
                   eyemap_start_meg=eyemap_start_meg, eyemap_end_meg=eyemap_end_meg,
                   title='Gaze x', xlabel='Pixels')

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


#---------------- Save scaled data to meg data ----------------#
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


#---------------- Save preprocesed data ----------------#
print('Saving preprocesed data')
os.makedirs(Preproc_data_path, exist_ok=True)
f = open(Preproc_data_path + f'Subject_{subject.subject_id}.pkl', 'wb')
pickle.dump(raw, f)
f.close()
print(f'Preprocesed data saved to {Preproc_data_path + f"Subject_{subject.subject_id}.pkl"}')




## OLD
# Reescale x data from MEG to match edf min and max values
# edf data has some values of ~1e6. Noise? Blinks? those values are separated from the "actual data"
# meaning there's a big gap between the data values and those higher. Data's highest value = 2028.9

meg_gazey_data -= np.nanmin(meg_gazey_data[eyemap_start_meg:eyemap_end_meg])
meg_gazey_data /= np.nanmax(meg_gazey_data[eyemap_start_meg:eyemap_end_meg])
meg_gazey_data *= (np.nanmax(edf_gazey_data[eyemap_start_edf:eyemap_end_edf]) - np.nanmin(
    edf_gazey_data[eyemap_start_edf:eyemap_end_edf]))
meg_gazey_data += np.nanmin(edf_gazey_data[eyemap_start_edf:eyemap_end_edf])

meg_gazex_data -= np.nanmin(meg_gazex_data)
meg_gazex_data /= np.nanmax(meg_gazex_data)
meg_gazex_data *= (np.nanmax(edf_gazex_data[edf_gazex_data < 3000]) - np.nanmin(edf_gazex_data[edf_gazex_data < 3000]))
meg_gazex_data += np.nanmin(edf_gazex_data[edf_gazex_data < 3000])

# First calibration resampled
# meg_gazex_data -= np.nanmin(meg_gazex_data[135000:142700])
# meg_gazex_data /= np.nanmax(meg_gazex_data[135000:142700])
# meg_gazex_data *= (np.nanmax(edf_gazex_data[7500:14000]) - np.nanmin(edf_gazex_data[7500:14000]))
# meg_gazex_data += np.nanmin(edf_gazex_data[7500:14000])

# First calibration
meg_gazex_data -= np.nanmin(meg_gazex_data[112500:119000])
meg_gazex_data /= np.nanmax(meg_gazex_data[112500:119000])
meg_gazex_data *= (np.nanmax(edf_gazex_data[7500:14000]) - np.nanmin(edf_gazex_data[7500:14000]))
meg_gazex_data += np.nanmin(edf_gazex_data[7500:14000])

# # Plot histograms
# plt.figure()
# edf_x_hist = plt.hist(edf_gazex_data, bins=30, range=(np.nanmin(edf_gazex_data), 2030), label='EDF')
# meg_x_hist = plt.hist(meg_gazex_data, bins=30, label='MEG', alpha=0.7)
# plt.title('Scaling MEG data based on distribution')
# plt.ylabel('All data')
# plt.legend()

# plt.figure()
# edf_x_hist = plt.hist(edf_gazex_data[7500:14000], bins=30, label='EDF')
# meg_x_hist = plt.hist(meg_gazex_data[135000:142700], bins=30, label='MEG', alpha=0.7)
# plt.title('Scaling MEG data based on eyemap matching')
# plt.ylabel('Eyemap data')
# plt.legend()
#
# plt.figure()
# edf_x_hist = plt.hist(edf_gazex_data, bins=30, label='EDF')
# meg_x_hist = plt.hist(meg_gazex_data, bins=30, label='MEG', alpha=0.7)
# plt.title('Scaling MEG data based on eyemap matching')
# plt.ylabel('Eyemap data')
# plt.legend()
