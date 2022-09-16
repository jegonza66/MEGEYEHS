import mne
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time

from paths import paths
import load
import plot
import functions

#---------------- Paths ----------------#
preproc_data_path = paths().preproc_path()
results_path = paths().results_path()
plots_path = paths().plots_path()
items_pos_path = paths().item_pos_path()

#---------------- Load data ----------------#
# Define subject
subject = load.subject()

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
meg_gazex_data_raw = et_channels_meg[0]
meg_gazey_data_raw = et_channels_meg[1]
meg_pupils_data_raw = et_channels_meg[2]

#---------------- Reescaling based on conversion parameters ----------------#

print('Rescaling')
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
R_h = (meg_gazex_data_raw - minvoltage) / (maxvoltage - minvoltage)  # voltage range proportion
S_h = R_h * (maxrange - minrange) + minrange  # proportion of screen width or height
R_v = (meg_gazey_data_raw - minvoltage) / (maxvoltage - minvoltage)
S_v = R_v * (maxrange - minrange) + minrange
meg_gazex_data_scaled = S_h * (screenright - screenleft + 1) + screenleft
meg_gazey_data_scaled = S_v * (screenbottom - screentop + 1) + screentop

# now overwrites output in Volts to give the output in pixels
et_channels_meg = [meg_gazex_data_scaled, meg_gazey_data_scaled, meg_pupils_data_raw]


#---------------- Blinks removal and missing signal interpolation ----------------#

print('Removing blinks')

# Copy pupils data to detect blinks from
meg_pupils_data_clean = copy.copy(meg_pupils_data_raw)
meg_gazex_data_clean = copy.copy(meg_gazex_data_scaled)
meg_gazey_data_clean = copy.copy(meg_gazey_data_scaled)

# Define missing values as 1 and non missing as 0 instead of True False
missing = (meg_pupils_data_clean < -4.6).astype(int)

# Samples before and after the threshold as true blink start to remove
start_interval_samples = 12
end_interval_samples = 24

# Get missing start/end samples and duration
missing_start = np.where(np.diff(missing) == 1)[0]
missing_end = np.where(np.diff(missing) == -1)[0]
missing_dur = missing_end - missing_start

# Remove blinks
# First, fill missing values (and suroundings) with nan
for i in range(len(missing_start)):
    blink_interval = np.arange(missing_start[i] - start_interval_samples, missing_end[i] + end_interval_samples)
    meg_gazex_data_clean[blink_interval] = float('nan')
    meg_gazey_data_clean[blink_interval] = float('nan')
    meg_pupils_data_clean[blink_interval] = float('nan')

# Redefine missing values. This way, if multiple missing intervals were close to one an other, they're now one big missing interval
missing = np.isnan(meg_pupils_data_clean).astype(int)

# Get missing start/end samples and duration
missing_start = np.where(np.diff(missing) == 1)[0]
missing_end = np.where(np.diff(missing) == -1)[0]
missing_dur = missing_end - missing_start

# Minimum blink duration
blink_min_time = 70  # ms
# Consider we enlarged the intervals when classifying for real and fake blinks
blink_min_samples = blink_min_time / 1000 * raw.info['sfreq'] + start_interval_samples + end_interval_samples

# Get actual and fake blinks based on duration condition
actual_blinks = np.where(missing_dur > blink_min_samples)[0]
fake_blinks = np.where(missing_dur <= blink_min_samples)[0]

# Interpolate fake blinks
for fake_blink_idx in fake_blinks:
    blink_interval = np.arange(missing_start[fake_blink_idx] - start_interval_samples, missing_end[fake_blink_idx] + end_interval_samples)

    interpolation_x = np.linspace(meg_gazex_data_clean[blink_interval[0]], meg_gazex_data_clean[blink_interval[-1]], len(blink_interval))
    interpolation_y = np.linspace(meg_gazey_data_clean[blink_interval[0]], meg_gazey_data_clean[blink_interval[-1]], len(blink_interval))
    interpolation_pupil = np.linspace(meg_pupils_data_clean[blink_interval[0]], meg_pupils_data_clean[blink_interval[-1]], len(blink_interval))
    meg_gazex_data_clean[blink_interval] = interpolation_x
    meg_gazey_data_clean[blink_interval] = interpolation_y
    meg_pupils_data_clean[blink_interval] = interpolation_pupil

# plt.figure()
# plt.title('Gaze x')
# plt.plot(meg_gazex_data_scaled, label='Blinks')
# plt.plot(meg_gazex_data_clean, label='Clean')
# plt.legend()
#
# plt.figure()
# plt.title('Gaze y')
# plt.plot(meg_gazey_data_scaled, label='Blinks')
# plt.plot(meg_gazey_data_clean, label='Clean')
# plt.legend()
#
# plt.figure()
# plt.title('Pupils')
# plt.plot(meg_pupils_data_raw, label='Blinks')
# plt.plot(meg_pupils_data_clean, label='Clean')
# plt.grid()
# plt.legend()


#---------------- Defining response events and trials ----------------#

print('Detecting events and defining trials')
# Get events in meg data (only red blue and green)
evt_buttons = raw.annotations.description
evt_times = raw.annotations.onset[(evt_buttons == 'red') | (evt_buttons == 'blue') | (evt_buttons == 'green')]
evt_buttons = evt_buttons[(evt_buttons == 'red') | (evt_buttons == 'blue') | (evt_buttons == 'green')]

# Check for first trial when first response is not green
first_trial = functions.first_trial(evt_buttons)

# Drop events before 1st trial
evt_buttons = evt_buttons[first_trial:]
evt_times = evt_times[first_trial:]

# Split events into blocks by green press at begening of block
blocks_start_end = np.where(evt_buttons == 'green')[0]
# Delete succesive presses of green
blocks_start_end = list(np.delete(blocks_start_end, np.where(np.diff(blocks_start_end) < 2)))
# Add first trial idx (0) on first sample (0), as we're gonna
blocks_start_end.insert(0, -1)
# Add las trial idx
blocks_start_end.append(len(evt_buttons))
# Define starting and ending trial of each block
blocks_bounds = [(blocks_start_end[i] + 1, blocks_start_end[i+1]) for i in range(len(blocks_start_end)-1)]

# Load behavioural data
bh_data = subject.beh_data()
# Get only trial data rows
bh_data = bh_data.loc[~pd.isna(bh_data['target.started'])].reset_index(drop=True)
# Get MS start time
ms_start = bh_data['target.started'].values.ravel().astype(float)
# Get fix 1 start time
fix1_start_key_idx = ['fixation_target.' in key and 'started' in key for key in bh_data.keys()]
fix1_start_key = bh_data.keys()[fix1_start_key_idx]
fix1_start = np.array([value.replace('None', f'{ms_start[i]}') for i, value in enumerate(bh_data[fix1_start_key].values.ravel())]).astype(float)
# Get fix 2 start time
fix2_start_key_idx = ['fixation_target_2' in key and 'started' in key for key in bh_data.keys()]
fix2_start_key = bh_data.keys()[fix2_start_key_idx]
fix2_start = bh_data[fix2_start_key].values.ravel().astype(float)
# Get Visual search start time
search_start_key_idx = ['search' in key and 'started' in key for key in bh_data.keys()]
search_start_key = bh_data.keys()[search_start_key_idx]
search_start = bh_data[search_start_key].values.ravel().astype(float)
# Get response time
rt = np.array([value.replace('[]', 'nan') for value in bh_data['rt'].values]).astype(float)
# Get response
responses = bh_data['key_resp.keys'].values

# Define variables to store data
no_answer = []
fix1_times_meg = []
ms_times_meg = []
fix2_times_meg = []
vs_times_meg = []
buttons_meg = []
button_times_meg = []
button_trials_meg = []

# Define trials block by block
for block_num, block_bounds in enumerate(blocks_bounds):
    print(f'\nBlock: {block_num + 1}')
    block_start = block_bounds[0]
    block_end = block_bounds[1]
    block_trials = 30

    # Get events in block from MEG data
    meg_evt_block_times = copy.copy(evt_times[block_start:block_end])
    meg_evt_block_buttons = copy.copy(evt_buttons[block_start:block_end])

    # Get events in block from BH data
    bh_evt_block_times = (search_start + rt)[block_num * block_trials:(block_num + 1) * block_trials]
    responses_block = responses[block_num * block_trials:(block_num + 1) * block_trials]

    # Get durations in block from BH data
    fix1_block_dur = (ms_start - fix1_start)[block_num * block_trials:(block_num + 1) * block_trials]
    ms_block_dur = (fix2_start - ms_start)[block_num * block_trials:(block_num + 1) * block_trials]
    fix2_block_dur = (search_start - fix2_start)[block_num * block_trials:(block_num + 1) * block_trials]
    rt_block_times = rt[block_num * block_trials:(block_num + 1) * block_trials]

    # Align MEG and BH data in time
    time_diff = search_start[block_num*block_trials] + rt[block_num*block_trials] - meg_evt_block_times[0]
    bh_evt_block_times -= time_diff

    # Iterate over trials
    for trial in range(block_trials):
        if not np.isnan(bh_evt_block_times[trial]):
            idx, meg_evt_time = functions.find_nearest(meg_evt_block_times, bh_evt_block_times[trial])

            trial_search_time = meg_evt_time - rt_block_times[trial]
            vs_times_meg.append(trial_search_time)

            trial_fix2_times = trial_search_time - fix2_block_dur[trial]
            fix2_times_meg.append(trial_fix2_times)

            trial_ms_time = trial_fix2_times - ms_block_dur[trial]
            ms_times_meg.append(trial_ms_time)

            trial_fix1_time = trial_ms_time - fix1_block_dur[trial]
            fix1_times_meg.append(trial_fix1_time)

            buttons_meg.append((meg_evt_block_buttons[idx]))
            button_times_meg.append(meg_evt_time)
            button_trials_meg.append(int(block_num*block_trials + trial + 1))

            if (meg_evt_block_buttons[idx] == 'blue' and responses_block[trial] != subject.map['blue']) or (meg_evt_block_buttons[idx] == 'red' and responses_block[trial] != subject.map['red']):
                raise ValueError(f'Different answer in MEG and BH data in trial: {trial}')

            if abs(meg_evt_time - bh_evt_block_times[trial]) > 0.05:
                print(f'Over 50ms difference in Trial: {trial}')

        else:
            print(f'No answer in Trial: {trial}')
            no_answer.append(block_num*block_trials+trial)

# Save clean events to MEG data

raw.annotations.trials = np.array(button_trials_meg)
raw.annotations.fix1 = np.array(fix1_times_meg)
raw.annotations.ms = np.array(ms_times_meg)
raw.annotations.fix2 = np.array(fix2_times_meg)
raw.annotations.vs = np.array(vs_times_meg)
raw.annotations.buttons = np.array(buttons_meg)
raw.annotations.rt = np.array(button_times_meg)

# # Using actual raw data times
# button_times_true = np.array([functions.find_nearest(raw.times, button_time)[1] for button_time in button_times_meg])
# fix1_times_true = np.array([functions.find_nearest(raw.times, fix1_time)[1] for fix1_time in fix1_times_meg])
# fix2_times_true = np.array([functions.find_nearest(raw.times, fix2_time)[1] for fix2_time in fix2_times_meg])
# ms_times_true = np.array([functions.find_nearest(raw.times, ms_time)[1] for ms_time in ms_times_meg])
# vs_times_true = np.array([functions.find_nearest(raw.times, vs_time)[1] for vs_time in vs_times_meg])
#
# # Save clean events to MEG data
# raw.annotations.description = np.array(buttons_meg)
# raw.annotations.onset = np.array(button_times_true)
# raw.annotations.trials = np.array(button_trials_meg)
# raw.annotations.fix1 = np.array(fix1_times_true)
# raw.annotations.ms = np.array(ms_times_true)
# raw.annotations.fix2 = np.array(fix2_times_true)
# raw.annotations.vs = np.array(vs_times_true)


#---------------- Fixations and saccades detection ----------------#

out_fname = f'Fix_Sac_detection_{subject.subject_id}.tsv'
out_folder = results_path + 'Preprocessing/' + subject.subject_id + '/'

try:
    # Load pre run saccades and fixation detection
    print('Loading saccades and fixations detection')
    results = pd.read_csv(out_folder + out_fname, sep='\t')
except:
    #If not pre run data, run
    print('Running saccades and fixations detection')

    # Define data to save to excel file needed to run the saccades detection program Remodnav
    eye_data = {'x': meg_gazex_data_clean, 'y': meg_gazey_data_clean}
    df = pd.DataFrame(eye_data)

    # Remodnav parameters
    fname = f'eye_data_{subject.subject_id}.csv'
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
    os.makedirs(out_folder, exist_ok=True)
    # Move et data file
    os.replace(fname, out_folder + fname)
    # Move results file
    os.replace(out_fname, out_folder + out_fname)
    # Move results image
    out_fname = out_fname.replace('tsv', 'png')
    os.replace(out_fname, out_folder + out_fname)

# Get saccades and fixations
saccades = copy.copy(results.loc[results['label'] == 'SACC'])
fixations = copy.copy(results.loc[results['label'] == 'FIXA'])

# Get mss and target pres/abs from bh data
mss = bh_data['Nstim'].astype(int)
pres_abs = bh_data['Tpres'].astype(int)
corr_ans = bh_data['corr'].astype(int)

# Get MSS, present/absent for every trial
fix_trial = []
fix_screen = []
trial_mss = []
tgt_pres_abs = []
trial_correct = []

# Iterate over fixations to classify them
print('Classifying fixations')

for fix_time in fixations['onset'].values:
    # find fixation's trial
    for i, fixation_cross_time in enumerate(fix1_times_meg):
        if fix_time < fixation_cross_time:
            i -= 1
            break

    trial = button_trials_meg[i]
    trial_idx = trial - 1

    # First fixation cross
    if fix1_times_meg[i] < fix_time < ms_times_meg[i]:
        fix_trial.append(trial)
        fix_screen.append('fix1')
        trial_mss.append(mss[trial_idx])
        tgt_pres_abs.append(pres_abs[trial_idx])
        trial_correct.append(corr_ans[trial_idx])

    # MS
    elif ms_times_meg[i] < fix_time < fix2_times_meg[i]:
        fix_trial.append(trial)
        fix_screen.append('ms')
        trial_mss.append(mss[trial_idx])
        tgt_pres_abs.append(pres_abs[trial_idx])
        trial_correct.append(corr_ans[trial_idx])

    # Second fixations corss
    elif fix2_times_meg[i] < fix_time < vs_times_meg[i]:
        fix_trial.append(trial)
        fix_screen.append('fix2')
        trial_mss.append(mss[trial_idx])
        tgt_pres_abs.append(pres_abs[trial_idx])
        trial_correct.append(corr_ans[trial_idx])

    # VS
    elif vs_times_meg[i] < fix_time < button_times_meg[i]:
        fix_trial.append(trial)
        fix_screen.append('vs')
        trial_mss.append(mss[trial_idx])
        tgt_pres_abs.append(pres_abs[trial_idx])
        trial_correct.append(corr_ans[trial_idx])

    # No trial identified
    else:
        fix_trial.append(None)
        fix_screen.append(None)
        trial_mss.append(None)
        tgt_pres_abs.append(None)
        trial_correct.append(None)

fixations['Trial'] = fix_trial
fixations['Screen'] = fix_screen
fixations['MSS'] = trial_mss
fixations['Target'] = tgt_pres_abs
fixations['Correct'] = trial_correct
fixations = fixations.astype({'Trial': float, 'MSS': float, 'Target': float})


## Target vs distractor
import scipy.io
items_pos = scipy.io.loadmat(items_pos_path)
items_pos_data = items_pos['pos']

items_pos_type = items_pos_data.dtype  # dtypes of structures are "unsized objects"

# * SciPy reads in structures as structured NumPy arrays of dtype object
# * The size of the array is the size of the structure array, not the number
#   elements in any particular field. The shape defaults to 2-dimensional.
# * For convenience make a dictionary of the data using the names from dtypes
# * Since the structure has only one element, but is 2-D, index it at [0, 0]
ndata = {n: items_pos_data[n] for n in items_pos_type.names}
# Reconstruct the columns of the data table from just the time series
# Use the number of intervals to test if a field is a column or metadata
columns = items_pos_type.names
# now make a data frame, setting the time stamps as the index
df = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1), columns=columns)



##---------------- Save scaled data to meg data ----------------#

print('Saving scaled et data to meg raw data structure')
# copy raw structure
raw_et = raw.copy()
# make new raw structure from et channels only
raw_et = mne.io.RawArray(et_channels_meg, raw_et.pick(et_channel_names).info)
# change channel names
for ch_name, new_name in zip(raw_et.ch_names, ['ET_gaze_x', 'ET_gaze_y', 'ET_pupils']):
    raw_et.rename_channels({ch_name: new_name})

# Pick data from MEG channels and other channels of interest
# But not the 422 channels ~12Gb
channel_indices = mne.pick_types(raw.info, meg=True)
channel_indices = np.append(channel_indices, mne.pick_channels(raw.info['ch_names'], ['UPPT001']))
raw.pick(channel_indices)

# save to original raw structure (requiers to load data)
print('Loading MEG data')
raw.load_data()
print('Adding new ET channels')
raw.add_channels([raw_et])
del (raw_et)

##---------------- Save preprocesed data ----------------#

print('Saving preprocesed data')
preproc_save_path = preproc_data_path + subject.subject_id +'/'
os.makedirs(preproc_save_path, exist_ok=True)
preproc_meg_data_fname = f'Subject_{subject.subject_id}_meg.fif'
raw.save(preproc_save_path + preproc_meg_data_fname)
print(f'Preprocesed data saved to {preproc_save_path + preproc_meg_data_fname}')












##---------------- Reescaling based on eyemap matching ----------------#

# Load edf data
et_data_edf = subject.et_data()
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
        plot.get_intervals_signals(reference_signal=edf_gazex_data, signal_to_scale=meg_gazex_data_raw)

    print('Scaling MEG Eye-Tracker data')
    for meg_gaze_data, edf_gaze_data, offset, title in zip((meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw),
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
    fig2 = plot.scaled_signals(time=edf_time, scaled_signals=[meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw],
                               reference_signals=[edf_gazex_data, edf_gazey_data, edf_pupils_data],
                               interval_signal=eyemap_interval_meg, interval_ref=eyemap_interval_edf,
                               ref_offset=[0, y_offset, 0], signal_offset=[0, meg_offset, 0],
                               ylabels=['Gaze x', 'Gaze y', 'Pupil size'])

    # Plotting and choosing different time in signal to check scaling
    fig1, eyemap_interval_edf, eyemap_interval_meg = plot.get_intervals_signals(reference_signal=edf_gazex_data,
                                                                                signal_to_scale=meg_gazex_data_raw,
                                                                                fig=fig1)

    # Plot scaled signals
    fig2 = plot.scaled_signals(time=edf_time, scaled_signals=[meg_gazex_data_raw, meg_gazey_data_raw, meg_pupils_data_raw],
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
plt.title('Gaze x')
plt.plot(meg_gazex_data_scaled, label='Blinks')
plt.plot(meg_gazex_data_clean, label='Clean')
plt.legend()

plt.figure()
plt.title('Gaze y')
plt.plot(meg_gazey_data_scaled, label='Blinks')
plt.plot(meg_gazey_data_clean, label='Clean')
plt.legend()

plt.figure()
plt.title('Pupils')
plt.plot(meg_pupils_data_raw, label='Blinks')
plt.plot(meg_pupils_data_clean, label='Clean')
plt.grid()
plt.legend()

plt.figure()
plt.title('EDF')
plt.plot(edf_gazey_data)


## PLOTS

# Evts
plt.figure()
plt.title('Behavioural and MEG data mapping')
plt.plot(meg_evt_block_times, meg_evt_block_times, 'o', label='MEG evts')
plt.plot(bh_evt_block_times, bh_evt_block_times, '.', label='BH evts')
plt.xlabel('time [s]')
plt.ylabel('time [s]')
plt.legend()
plt.savefig('Good_scale.png')

# Scanpaths and screens Trials
save_fig = True
plt.ioff()
time.sleep(1)
for trial in range(len(raw.annotations.trials)):
    print(f'Trial: {raw.annotations.trials[trial]}')
    pres_abs_trial = 'Present' if pres_abs[trial] == 1 else 'Absent'
    correct_trial = 'Correct' if pres_abs[trial] == 1 else 'Incorrect'

    plt.figure(figsize=(15, 5))
    plt.title(f'Trial {raw.annotations.trials[trial]} - {pres_abs_trial} - {correct_trial}')
    trial_start_idx = functions.find_nearest(raw.times, raw.annotations.fix1[trial])[0] - 120 * 2
    trial_end_idx = functions.find_nearest(raw.times, raw.annotations.rt[trial])[0] + 120 * 6

    plt.plot(raw.times[trial_start_idx:trial_end_idx], meg_gazex_data_clean[trial_start_idx:trial_end_idx], label='Gaze x')
    plt.plot(raw.times[trial_start_idx:trial_end_idx], meg_gazey_data_clean[trial_start_idx:trial_end_idx] - 1000, 'black', label='Gaze y')
    ymin = plt.gca().get_ylim()[0]
    ymax = plt.gca().get_ylim()[1] + 400

    plt.axvspan(ymin=0, ymax=1, xmin=raw.annotations.fix1[trial], xmax=raw.annotations.ms[trial], color='grey',
                alpha=0.4, label='Fix')
    plt.axvspan(ymin=0, ymax=1, xmin=raw.annotations.ms[trial], xmax=raw.annotations.fix2[trial], color='red',
                alpha=0.4, label='MS')
    plt.axvspan(ymin=0, ymax=1, xmin=raw.annotations.fix2[trial], xmax=raw.annotations.vs[trial], color='grey',
                alpha=0.4, label='Fix')
    plt.axvspan(ymin=0, ymax=1, xmin=raw.annotations.vs[trial], xmax=raw.annotations.rt[trial], color='green',
                alpha=0.4, label='VS')

    plt.xlabel('time [s]')
    plt.ylabel('Gaze')
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    if save_fig:
        save_path = plots_path + f'Gaze_Trials/{subject.subject_id}/'
        os.makedirs(save_path, exist_ok=True)

        plt.savefig(save_path + f'Trial {raw.annotations.trials[trial]}.png')
