import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from paths import paths
import load
import plot
import functions
import preproc_functions

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

# Get et channels by name [Gaze x, Gaze y, Pupils]
et_channel_names = ['UADC001-4123', 'UADC002-4123', 'UADC013-4123']

print('\nGetting ET channels data from MEG')
et_channels_meg = raw.get_data(picks=et_channel_names)

# Get separate data from et channels
meg_gazex_data_raw = et_channels_meg[0]
meg_gazey_data_raw = et_channels_meg[1]
meg_pupils_data_raw = et_channels_meg[2]

#---------------- Reescaling based on conversion parameters ----------------#
meg_gazex_data_scaled, meg_gazey_data_scaled = preproc_functions.reescale_et_channels(meg_gazex_data_raw=meg_gazex_data_raw,
                                                                                      meg_gazey_data_raw=meg_gazey_data_raw)

#---------------- Blinks removal ----------------#
# Define intervals around blinks to also fill with nan. Due to conversion noice from square signal
blink_min_dur = 70
start_interval_samples = 12
end_interval_samples = 24

meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean = preproc_functions.blinks_to_nan(meg_gazex_data_scaled=meg_gazex_data_scaled,
                                                                                                    meg_gazey_data_scaled=meg_gazey_data_scaled,
                                                                                                    meg_pupils_data_raw=meg_pupils_data_raw,
                                                                                                    start_interval_samples=start_interval_samples,
                                                                                                    end_interval_samples=end_interval_samples)

#---------------- Missing signal interpolation ----------------#
et_channels_meg = preproc_functions.fake_blink_interpolate(meg_gazex_data_clean=meg_gazex_data_clean,
                                                           meg_gazey_data_clean=meg_gazey_data_clean,
                                                           meg_pupils_data_clean=meg_pupils_data_clean,
                                                           blink_min_dur=blink_min_dur,
                                                           start_interval_samples=start_interval_samples,
                                                           end_interval_samples=start_interval_samples,
                                                           sfreq=raw.info['sfreq'])

#---------------- Defining response events and trials ----------------#
bh_data, response_trials_meg, fix1_times_meg, ms_times_meg, fix2_times_meg, vs_times_meg, buttons_meg, response_times_meg = \
    preproc_functions.define_events_trials(raw=raw, subject=subject)

#---------------- Fixations and saccades detection ----------------#
fixations, saccades = preproc_functions.fixations_saccades_detection(raw=raw, meg_gazex_data_clean=meg_gazex_data_clean,
                                                                     meg_gazey_data_clean=meg_gazey_data_clean,
                                                                     subject=subject)

#---------------- Fixations classification ----------------#
fixations = preproc_functions.fixation_classification(bh_data=bh_data, fixations=fixations, fix1_times_meg=fix1_times_meg,
                                                      response_trials_meg=response_trials_meg, ms_times_meg=ms_times_meg,
                                                      fix2_times_meg=fix2_times_meg, vs_times_meg=vs_times_meg,
                                                      response_times_meg=response_times_meg,
                                                      meg_pupils_data_clean=meg_pupils_data_clean, times=raw.times)

# First fixation delay distribution
fixations1_fix_screen = fixations.loc[(fixations['screen'].isin(['fix1', 'fix2'])) & (fixations['n_fix'] == 1)]
plt.hist(fixations1_fix_screen['delay'], bins=15)
plt.title('1st fixation delay distribution')
plt.xlabel('Time [s]')
plt.savefig(plots_path + '1st fix delay dist.png')


fixations_pupil_s = fixations.loc[(fixations['screen'].isin(['fix1', 'ms', 'fix2'])) & (fixations['n_fix'] == 1)]

pupil_diffs = []
mss = []
for trial in response_trials_meg:
    trial_data = fixations_pupil_s.loc[fixations_pupil_s['trial'] == trial]

    if 'fix1' in trial_data['screen'].values:
        pupil_diff = trial_data[trial_data['screen'] == 'fix2']['pupil'].values[0] - trial_data[trial_data['screen'] == 'fix1']['pupil'].values[0]
    else:
        pupil_diff = trial_data[trial_data['screen'] == 'fix2']['pupil'].values[0] - \
                     trial_data[trial_data['screen'] == 'ms']['pupil'].values[0]
    pupil_diffs.append(pupil_diff)
    mss.append(trial_data['mss'].values[0])

plt.plot(mss, pupil_diffs, '.')
plt.title('1st fixation delay distribution')
plt.xlabel('MSS')
plt.ylabel('Pupil size increase (fix point 2 - 1)')



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
items_pos = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1), columns=columns)

for key in items_pos.keys():
    key_values = [value[0] for value in items_pos[key].values]
    items_pos[key] = key_values

double_list_keys = list(items_pos.keys())
remove_keys = ['folder', 'item', 'cmp', 'trialabsent']
for key in remove_keys:
    double_list_keys.remove(key)

for key in double_list_keys:
    key_values = [value[0] for value in items_pos[key].values]
    items_pos[key] = key_values


# iterate over fixations checking for trial number, then check image used, then check in item_pos the position of items and mesure distance
fixations_vs = fixations.loc[fixations['screen'] == 'vs']

fix_item_distance = []
fix_target = []
trial_images = []
item_images = []

for fix_idx, fix in fixations_vs.iterrows():
    trial = fix['trial']
    trial_idx = np.where(np.array(response_trials_meg) == trial)[0]

    fix_x = np.mean([fix['start_x'], fix['end_x']])
    fix_y = np.mean([fix['start_y'], fix['end_y']])

    trial_image = bh_data['searchimage'][trial_idx].values[0].split('cmp_')[-1].split('.jpg')[0]

    trial_images.append(trial_image)

    # Find item position information for such image
    trial_items = items_pos.loc[items_pos['folder'] == trial_image]
    item_images.append(trial_items['folder'])


    distances = []
    targets = []
    for item_idx, item in trial_items.iterrows():

        item_x = item['center_x']
        item_y = item['center_y']

        x_dist = abs(fix_x - item_x)
        y_dist = abs(fix_y - item_y)

        distance = np.sqrt(x_dist**2 + y_dist**2)

        distances.append(distance)
        targets.append(item['istarget'])

    fixated_item = np.argmin(np.array(distances))
    item_distance = distances[fixated_item]
    istarget = targets[fixated_item]

    fix_item_distance.append(item_distance)
    fix_target.append(istarget)

fixations_vs['distance'] = fix_item_distance
fixations_vs['fix_target'] = fix_target
fixations_vs['trial_image'] = trial_images
fixations_vs['item_image'] = item_images


fixations_target = fixations_vs.loc[fixations_vs['fix_target'] == 1]




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
    pres_abs_trial = 'Present' if bh_data['Tpres'].astype(int)[trial] == 1 else 'Absent'
    correct_trial = 'Correct' if bh_data['corr'].astype(int)[trial] == 1 else 'Incorrect'
    mss = bh_data['Nstim'][trial]

    plt.figure(figsize=(15, 5))
    plt.title(f'Trial {raw.annotations.trials[trial]} - {pres_abs_trial} - {correct_trial} - MSS: {int(mss)}')
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
