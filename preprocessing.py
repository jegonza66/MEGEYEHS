import mne
import pickle
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from paths import paths
import load
import plot
import functions

#---------------- Paths ----------------#
Preproc_data_path = paths().preproc_path()
Results_path = paths().results_path()

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
print('\nGetting ET channels data from MEG')
et_channels_meg = raw.get_data(picks=et_channel_names)
print('Done')

# Get x, y and pupil size
meg_gazex_data = et_channels_meg[0]
meg_gazey_data = et_channels_meg[1]
meg_pupils_data = et_channels_meg[2]

#---------------- Reescaling based on conversion parameters ----------------#

# Define Parameters
minvoltage = -5  # from analog.ini analog_dac_range
maxvoltage = 5  # from analog.ini analog_dac_range
minrange = -0.2  # from analog.ini analog_x_range to allow for +/- 20# outside display
maxrange = 1.2  # from analog.ini analog_x_range to allow for +/- 20# outside display
screenright = 1919  # OPM lab
screenleft = 0
screentop = 0
screenbottom = 1079  # OPM lab

# Scale
R_h = (meg_gazex_data - minvoltage) / (maxvoltage - minvoltage)  # voltage range proportion
S_h = R_h * (maxrange - minrange) + minrange  # proportion of screen width or height
R_v = (meg_gazey_data - minvoltage) / (maxvoltage - minvoltage)
S_v = R_v * (maxrange - minrange) + minrange
meg_gazex_data = S_h * (screenright - screenleft + 1) + screenleft
meg_gazey_data = S_v * (screenbottom - screentop + 1) + screentop

# now overwrites output in Volts to give the output in pixels
et_channels_meg = [meg_gazex_data, meg_gazey_data, meg_pupils_data]


##---------------- Blinks removal and missing signal interpolation ----------------#

# Copy pupils data to detect blinks from
meg_pupils_data_blinks = copy.copy(meg_pupils_data)

# Define blinks as 1 and non blinks as 0 instead of True False
blinks = (meg_pupils_data_blinks < -4.75).astype(int)

# Minimum blink duration
blink_min_time = 50 # ms
blink_min_samples = blink_min_time / 1000 * raw.info['sfreq']

# Get blinks start/end samples and duration
blinks_start = np.where(np.diff(blinks) == 1)[0]
blinks_end = np.where(np.diff(blinks) == -1)[0]
blinks_dur = blinks_end - blinks_start

# Get actual and fake blinks based on duration condition
actual_blinks = np.where(blinks_dur > blink_min_samples)[0]
fake_blinks = np.where(blinks_dur <= blink_min_samples)[0]

# Samples before and after the threshold as true blink start to remove
start_interval_samples = 8
end_interval_samples = 16

# Remove blinks
meg_gazex_data_blinks = copy.copy(meg_gazex_data)
meg_gazey_data_blinks = copy.copy(meg_gazey_data)

for actual_blink_idx in actual_blinks:
    blink_interval = np.arange(blinks_start[actual_blink_idx] - start_interval_samples, blinks_end[actual_blink_idx] + end_interval_samples)
    meg_gazex_data_blinks[blink_interval] = float('nan')
    meg_gazey_data_blinks[blink_interval] = float('nan')
    meg_pupils_data_blinks[blink_interval] = float('nan')

# Remove fake blinks by interpolation
for fake_blink_idx in fake_blinks:
    blink_interval = np.arange(blinks_start[fake_blink_idx] - start_interval_samples, blinks_end[fake_blink_idx] + end_interval_samples)
    interpolation_x = np.linspace(meg_gazex_data_blinks[blink_interval[0]], meg_gazex_data_blinks[blink_interval[-1]], len(blink_interval))
    interpolation_y = np.linspace(meg_gazey_data_blinks[blink_interval[0]], meg_gazey_data_blinks[blink_interval[-1]], len(blink_interval))
    interpolation_pupil = np.linspace(meg_pupils_data_blinks[blink_interval[0]], meg_pupils_data_blinks[blink_interval[-1]], len(blink_interval))
    meg_gazex_data_blinks[blink_interval] = interpolation_x
    meg_gazey_data_blinks[blink_interval] = interpolation_y
    meg_pupils_data_blinks[blink_interval] = interpolation_pupil

##---------------- Defining response events and trials ----------------#

# Get events in meg data (only red blue and green)
evt_buttons = raw.annotations.description
evt_times = raw.annotations.onset[(evt_buttons == 'red') | (evt_buttons == 'blue') | (evt_buttons == 'green')]
evt_buttons = evt_buttons[(evt_buttons == 'red') | (evt_buttons == 'blue') | (evt_buttons == 'green')]

# Check for first trial after 3 green presses
first_trial = functions.first_trial(evt_buttons)

# Derop events before 1st trial
evt_buttons = evt_buttons[first_trial:]
evt_times = evt_times[first_trial:]

# Split events into blocks by green press at begening of block
blocks_start_end = np.where(evt_buttons == 'green')[0]
# Delete succesive presses of green
blocks_start_end = list(np.delete(blocks_start_end, np.where(np.diff(blocks_start_end) < 2)))
# Add first trial idx
blocks_start_end.insert(0, -1)
# Add las trial idx
blocks_start_end.append(len(evt_buttons))
# Define start ing and ending trial of each block
blocks_bounds = [(blocks_start_end[i] + 1, blocks_start_end[i+1]) for i in range(len(blocks_start_end)-1)]

# Load behavioural data
bh_data = subject.beh_data()
# Get only trial data rows
bh_data = bh_data.loc[~pd.isna(bh_data['target.started'])].reset_index(drop=True)
# Get Visual search start time
search_start_key_idx = ['search' in key and 'started' in key for key in bh_data.keys()]
search_start_key = bh_data.keys()[search_start_key_idx]
search_start = bh_data[search_start_key].values.ravel()
# Get response time
rt = np.array([value.replace('[]', 'nan') for value in bh_data['rt'].values]).astype(float)
# Get response
responses = bh_data['key_resp.keys'].values

# Define varialbes to store data
no_answer = []
buttons = []
button_times = []
button_trials = []

# Define trials block by block
for block_num, block_bounds in enumerate(blocks_bounds):
    print(f'Block: {block_num + 1}')
    block_start = block_bounds[0]
    block_end = block_bounds[1]
    block_trials = 30

    # Get events in block from MEG data
    meg_evt_times = copy.copy(evt_times[block_start:block_end])
    meg_evt_buttons = copy.copy(evt_buttons[block_start:block_end])
    # Get events in block from BH data
    bh_evt_times = (search_start + rt)[block_num * block_trials:(block_num + 1) * block_trials]
    responses_block = responses[block_num * block_trials:(block_num + 1) * block_trials]

    # Align MEG and BH data in time
    time_diff = search_start[block_num*block_trials] + rt[block_num*block_trials] - meg_evt_times[0]
    meg_evt_times += time_diff

    # Iterate over trials
    for trial in range(block_trials):
        if not np.isnan(bh_evt_times[trial]):
            idx = functions.find_nearest(meg_evt_times, bh_evt_times[trial])
            buttons.append((meg_evt_buttons[idx]))
            button_times.append(meg_evt_times[idx] - time_diff)
            button_trials.append(block_num*block_trials+trial+1)

            if (meg_evt_buttons[idx] == 'blue' and responses_block[trial] != subject.map['blue']) or (meg_evt_buttons[idx] == 'red' and responses_block[trial] != subject.map['red']):
                raise ValueError(f'Different answer in MEG and BH data in trial: {trial}')

            if abs(meg_evt_times[idx] - bh_evt_times[trial]) > 0.05:
                print(f'Over 50ms difference in Trial: {trial}')

        else:
            no_answer.append(block_num*block_trials+trial)

# Save clean events to MEG data
raw.annotations.description = np.array(buttons)
raw.annotations.onset = np.array(button_times)
raw.annotations.trials = np.array(button_trials)


##---------------- Fixations and saccades detection ----------------#

# Define data to save to excel file needed to run the saccades detection program Remodnav
eye_data = {'x': meg_gazex_data_blinks, 'y': meg_gazey_data_blinks}
df = pd.DataFrame(eye_data)

# Remodnav parameters
fname = f'eye_data_{subject.subject_id}.csv'
out_fname = f'Fix_Sac_detection_{subject.subject_id}.tsv'
screen_size = 38
screen_distance = 58
screen_resolution = 1920
px2deg = math.degrees(math.atan2(.5 * screen_size, screen_distance)) / (.5 * screen_resolution)
sfreq = raw.info['sfreq']

# Save csv file
df.to_csv(fname, sep='\t', header=False, index=False)

# Run Remodnav not considering pursuit class and min fixations 100 ms
command = f'remodnav {fname} {out_fname} {px2deg} {sfreq} --savgol-length {0.0195} --min-pursuit-duration {2} ' \
          f'--min-fixation-duration {0.1}'
os.system(command)

# Read results file with detections
results = pd.read_csv(out_fname, sep='\t')

# Move eye data, detections file and image to subject results directory
out_folder = Results_path + 'Preprocessing/' + subject.subject_id + '/'
os.makedirs(out_folder, exist_ok=True)
# Move et data file
os.rename(fname, out_folder + fname)
# Move results file
os.rename(out_fname, out_folder + out_fname)
# Move results image
out_fname = out_fname.replace('tsv', 'png')
os.rename(out_fname, out_folder + out_fname)

# Get saccades
saccades = results.loc[results['label'] == 'SACC'] # Incomplete


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
del (raw_et)

##---------------- Save preprocesed data ----------------#

# Pick data from MEG channels and other channels of interest
# But not the 422 channels ~12Gb

print('Saving preprocesed data')
os.makedirs(Preproc_data_path, exist_ok=True)
f = open(Preproc_data_path + f'Subject_{subject.subject_id}.pkl', 'wb')
pickle.dump(raw, f)
f.close()
print(f'Preprocesed data saved to {Preproc_data_path + f"Subject_{subject.subject_id}.pkl"}')










##---------------- Reescaling based on eyemap matching ----------------#

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



## Plot data with and without blinks to compare

plt.figure()
plt.title('MEG')
plt.plot(meg_gazex_data, label='Blinks')
plt.plot(meg_gazex_data_blinks, label='Clean')
plt.legend()

plt.figure()
plt.title('MEG')
plt.plot(meg_gazey_data, label='Blinks')
plt.plot(meg_gazey_data_blinks, label='Clean')
plt.legend()

plt.figure()
plt.plot(meg_pupils_data)
plt.plot(meg_pupils_data_blinks)

plt.figure()
plt.title('EDF')
plt.plot(edf_gazey_data)


## PLOT evts

plt.figure()
plt.title('Behavioural and MEG data mapping')
plt.plot(meg_evt_times, meg_evt_times, 'o', label='MEG evts')
plt.plot(bh_evt_times, bh_evt_times, '.', label='BH evts')
plt.xlabel('time [s]')
plt.ylabel('time [s]')
plt.legend()
plt.savefig('Good_scale.png')
