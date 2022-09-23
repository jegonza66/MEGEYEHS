import copy
import numpy as np
import pandas as pd
import os
import math

import functions
from paths import paths


def reescale_et_channels(meg_gazex_data_raw, meg_gazey_data_raw, minvoltage=-5, maxvoltage=5, minrange=-0.2, maxrange=1.2,
                         screenright=1919, screenleft=0, screentop=0, screenbottom=1079):
    """
    Reescale Gaze data from MEG Eye Tracker channels to correct digital to analog conversion.

    Parameters
    ----------
    meg_gazex_data_raw: ndarray
        Raw gaze x data to rescale
    meg_gazey_data_raw: ndarray
        Raw gaze x data to rescale
    minvoltage: int
        The minimum voltage value from the digital-analog conversion.
        Default to -5 from analog.ini analog_dac_range.
    maxvoltage: int
        The maximum voltage value from the digital-analog conversion.
        Default to 5 from analog.ini analog_dac_range.
    minrange: int
        The minimum gaze position tracked by the eye tracker outside the screen.
        Default to -0.2 from analog.ini analog_x_range to allow for +/- 20% outside display
    maxrange: int
        The maximum gaze position tracked by the eye tracker outside the screen.
        Default to 1.2 from analog.ini analog_x_range to allow for +/- 20% outside display
    screenright: int
        Pixel number of the right side of the screen. Default to 1919.
    screenleft: int
        Pixel number of the left side of the screen. Default to 0.
    screentop: int
        Pixel number of the top side of the screen. Default to 0.
    screenbottom: int
        Pixel number of the bottom side of the screen. Default to 1079.

    Returns
    -------
    meg_gazex_data_scaled: ndarray
        The scaled gaze x data
    meg_gazey_data_scaled: ndarray
        The scaled gaze y data
    """

    print('Rescaling')
    # Scale
    R_h = (meg_gazex_data_raw - minvoltage) / (maxvoltage - minvoltage)  # voltage range proportion
    S_h = R_h * (maxrange - minrange) + minrange  # proportion of screen width or height
    R_v = (meg_gazey_data_raw - minvoltage) / (maxvoltage - minvoltage)
    S_v = R_v * (maxrange - minrange) + minrange
    meg_gazex_data_scaled = S_h * (screenright - screenleft + 1) + screenleft
    meg_gazey_data_scaled = S_v * (screenbottom - screentop + 1) + screentop

    return meg_gazex_data_scaled, meg_gazey_data_scaled


def blinks_to_nan(meg_pupils_data_raw, meg_gazex_data_scaled, meg_gazey_data_scaled, start_interval_samples, end_interval_samples):
    print('Removing blinks')

    # Copy pupils data to detect blinks from
    meg_gazex_data_clean = copy.copy(meg_gazex_data_scaled)
    meg_gazey_data_clean = copy.copy(meg_gazey_data_scaled)
    meg_pupils_data_clean = copy.copy(meg_pupils_data_raw)

    # Check for peaks in pupils signal
    pupils_diff = np.concatenate((np.array([float('nan')]), np.diff(meg_pupils_data_clean)))

    # Define missing values as 1 and non missing as 0 instead of True False
    missing = ((meg_pupils_data_clean < -4.6) | (abs(pupils_diff) > 0.1)).astype(int)

    # Get missing start/end samples and duration
    missing_start = np.where(np.diff(missing) == 1)[0]
    missing_end = np.where(np.diff(missing) == -1)[0]

    # Take samples according to start or end with missing data (missing_start and end would have different sizes)
    if missing[0] == 1:
        missing_end = missing_end[1:]
    if missing[-1] == 1:
        missing_start = missing_start[:-1]

    # Remove blinks
    # First, fill missing values (and suroundings) with nan
    for i in range(len(missing_start)):
        blink_interval = np.arange(missing_start[i] - start_interval_samples, missing_end[i] + end_interval_samples)
        meg_gazex_data_clean[blink_interval] = float('nan')
        meg_gazey_data_clean[blink_interval] = float('nan')
        meg_pupils_data_clean[blink_interval] = float('nan')

    return meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean


def fake_blink_interpolate(meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean, sfreq, blink_min_dur=70,
                           start_interval_samples=12, end_interval_samples=24):

    print('Interpolating fake blinks')

    # Missing signal
    missing = np.isnan(meg_pupils_data_clean).astype(int)

    # Get missing start/end samples and duration
    missing_start = np.where(np.diff(missing) == 1)[0]
    missing_end = np.where(np.diff(missing) == -1)[0]
    missing_dur = missing_end - missing_start

    # Consider we enlarged the intervals when classifying for real and fake blinks
    blink_min_samples = blink_min_dur / 1000 * sfreq + start_interval_samples + end_interval_samples

    # Get fake blinks based on duration condition (actual blinks were already filled with nan
    fake_blinks = np.where(missing_dur <= blink_min_samples)[0]

    # Interpolate fake blinks
    for fake_blink_idx in fake_blinks:
        blink_interval = np.arange(missing_start[fake_blink_idx] - start_interval_samples,
                                   missing_end[fake_blink_idx] + end_interval_samples)

        interpolation_x = np.linspace(meg_gazex_data_clean[blink_interval[0]], meg_gazex_data_clean[blink_interval[-1]],
                                      len(blink_interval))
        interpolation_y = np.linspace(meg_gazey_data_clean[blink_interval[0]], meg_gazey_data_clean[blink_interval[-1]],
                                      len(blink_interval))
        interpolation_pupil = np.linspace(meg_pupils_data_clean[blink_interval[0]],
                                          meg_pupils_data_clean[blink_interval[-1]], len(blink_interval))
        meg_gazex_data_clean[blink_interval] = interpolation_x
        meg_gazey_data_clean[blink_interval] = interpolation_y
        meg_pupils_data_clean[blink_interval] = interpolation_pupil

    et_channels_meg = [meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean]

    return et_channels_meg


def define_events_trials(raw, subject):
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

    # Add first trial idx (0) on first sample (0) and Add last trial idx in the end
    blocks_start_end = np.concatenate((np.array([-1]), blocks_start_end, np.array([len(evt_buttons)])))

    # Delete succesive presses of green
    blocks_start_end = list(np.delete(blocks_start_end, np.where(np.diff(blocks_start_end) < 2)).astype(int))

    # Delete blocks shorter than 20 trials
    blocks_start_end = list(np.delete(blocks_start_end, np.where(np.diff(blocks_start_end) < 20)).astype(int))

    # Define starting and ending trial of each block
    blocks_bounds = [(blocks_start_end[i] + 1, blocks_start_end[i + 1]) for i in range(len(blocks_start_end) - 1)]

    # Load behavioural data
    bh_data = subject.bh_data()

    # Get only trial data rows
    bh_data = bh_data.loc[~pd.isna(bh_data['target.started'])].reset_index(drop=True)

    # Get MS start time
    ms_start = bh_data['target.started'].values.ravel().astype(float)

    # Get fix 1 start time
    fix1_start_key_idx = ['fixation_target.' in key and 'started' in key for key in bh_data.keys()]
    fix1_start_key = bh_data.keys()[fix1_start_key_idx][0]

    # If there's a missing fix1 screen, the time for that screen would be None, and the column type would be string.
    # If there's not, the type would be float. Change type to string and replace None by the mss screen start time
    if type(bh_data[fix1_start_key][0]) == str:
        fix1_start = np.array([value.replace('None', f'{ms_start[i]}') for i, value in
                               enumerate(bh_data[fix1_start_key].values.ravel())]).astype(float)
    else:
        fix1_start = bh_data[fix1_start_key].values

    # Get fix 2 start time
    fix2_start_key_idx = ['fixation_target_2' in key and 'started' in key for key in bh_data.keys()]
    fix2_start_key = bh_data.keys()[fix2_start_key_idx]
    fix2_start = bh_data[fix2_start_key].values.ravel().astype(float)

    # Get Visual search start time
    search_start_key_idx = ['search' in key and 'started' in key for key in bh_data.keys()]
    search_start_key = bh_data.keys()[search_start_key_idx]
    search_start = bh_data[search_start_key].values.ravel().astype(float)

    # Get responses
    responses = bh_data['key_resp.keys'].values

    # Get response time
    rt = copy.copy(bh_data['key_resp.rt'].values)
    # completed_responses_idx = np.where(pd.isna(bh_data['key_resp.rt']) & (responses != 'None'))[0]
    # rt.loc[completed_responses_idx] = 10
    # rt = rt.values

    # Define variables to store data
    no_answer = []
    fix1_times_meg = []
    ms_times_meg = []
    fix2_times_meg = []
    vs_times_meg = []
    buttons_meg = []
    response_times_meg = []
    response_trials_meg = []
    time_differences = []

    # Define trials block by block
    for block_num, block_bounds in enumerate(blocks_bounds):
        print(f'\nBlock: {block_num + 1}')
        block_start = block_bounds[0]
        block_end = block_bounds[1]
        block_trials = 30
        block_idxs = np.arange(block_num * block_trials, (block_num + 1) * block_trials)

        # Get events in block from MEG data
        meg_evt_block_times = copy.copy(evt_times[block_start:block_end])
        meg_evt_block_buttons = copy.copy(evt_buttons[block_start:block_end])

        # Get durations in block from BH data
        fix1_block_dur = (ms_start - fix1_start)[block_idxs]
        ms_block_dur = (fix2_start - ms_start)[block_idxs]
        fix2_block_dur = (search_start - fix2_start)[block_idxs]
        rt_block_times = rt[block_idxs]

        # Align MEG and BH data in time
        block_data_aligned = False
        attempts = 0
        align_sample = 0
        while not block_data_aligned and attempts < 5:
            # Save block variables resetting them every attempt
            no_answer_block = []
            fix1_times_meg_block = []
            ms_times_meg_block = []
            fix2_times_meg_block = []
            vs_times_meg_block = []
            buttons_meg_block = []
            response_times_meg_block = []
            response_trials_meg_block = []
            time_diff_block = []

            try:
                # Get events in block from BH data
                bh_evt_block_times = (search_start + rt)[block_idxs]
                responses_block = responses[block_idxs]

                # Realign bh and meg block timelines
                block_time_realign = search_start[block_num*block_trials] + rt[block_num*block_trials] - meg_evt_block_times[align_sample]
                bh_evt_block_times = bh_evt_block_times - block_time_realign

                # Iterate over trials
                for trial in range(block_trials):
                    if not np.isnan(bh_evt_block_times[trial]):
                        idx, meg_evt_time = functions.find_nearest(meg_evt_block_times, bh_evt_block_times[trial])

                        time_diff = bh_evt_block_times[trial] - meg_evt_time
                        time_diff_block.append(time_diff)

                        if (meg_evt_block_buttons[idx] == 'blue' and int(responses_block[trial]) != int(subject.map['blue'])) or (
                                meg_evt_block_buttons[idx] == 'red' and int(responses_block[trial]) != int(subject.map['red'])):
                            print(f'Different answer in MEG and BH data in trial: {trial}\n'
                                  f'Discarding and realingning on following sample\n')
                            raise ValueError(f'Different answer in MEG and BH data in trial: {trial}')

                        if abs(time_diff) > 0.02:# and block_num * block_trials + trial not in completed_responses_idx:
                            print(f'{round(abs(meg_evt_time - bh_evt_block_times[trial])*1000,1)} ms difference in Trial: {trial}')

                        # Define screen times from MEG response
                        trial_search_time = meg_evt_time - rt_block_times[trial]
                        vs_times_meg_block.append(trial_search_time)

                        trial_fix2_times = trial_search_time - fix2_block_dur[trial]
                        fix2_times_meg_block.append(trial_fix2_times)

                        trial_ms_time = trial_fix2_times - ms_block_dur[trial]
                        ms_times_meg_block.append(trial_ms_time)

                        trial_fix1_time = trial_ms_time - fix1_block_dur[trial]
                        fix1_times_meg_block.append(trial_fix1_time)

                        buttons_meg_block.append((meg_evt_block_buttons[idx]))
                        response_times_meg_block.append(meg_evt_time)
                        response_trials_meg_block.append(int(block_num * block_trials + trial + 1))

                    # Consider manually completed responses
                    # elif np.isnan(bh_evt_block_times[trial]) and responses_block[trial] != 'None':

                    else:
                        print(f'No answer in Trial: {trial}')
                        no_answer_block.append(block_num * block_trials + trial)

                # Define variable of not completed responses times to assess the real time difference
                # (completed responses might artificially increase this value)
                # real_time_diff = [element for trial, element in zip(block_idxs, time_diff_block) if trial not in completed_responses_idx]
                # if np.mean(real_time_diff) > 0.2:
                if np.mean(abs(np.array(time_diff_block))) > 0.2:
                    print(f'Average time difference for this block: {np.mean(abs(np.array(time_diff_block)))} s\n'
                          f'Discarding and realingning on following sample\n')
                    raise ValueError(f'Average time difference for this block over 200 ms')

                else:
                    block_data_aligned = True
                    # Append block data to overall data
                    no_answer.append(no_answer_block)
                    fix1_times_meg.append(fix1_times_meg_block)
                    ms_times_meg.append(ms_times_meg_block)
                    fix2_times_meg.append(fix2_times_meg_block)
                    vs_times_meg.append(vs_times_meg_block)
                    buttons_meg.append(buttons_meg_block)
                    response_times_meg.append(response_times_meg_block)
                    response_trials_meg.append(response_trials_meg_block)
                    time_differences.append(time_diff_block)
            except:
                align_sample += 1
                attempts += 1

        if not block_data_aligned:
            raise ValueError(f'Could not align MEG and BH responses in block {block_num + 1}')

    # flatten variables over blocks
    response_trials_meg = functions.flatten_list(response_trials_meg)
    fix1_times_meg = functions.flatten_list(fix1_times_meg)
    ms_times_meg = functions.flatten_list(ms_times_meg)
    fix2_times_meg = functions.flatten_list(fix2_times_meg)
    vs_times_meg = functions.flatten_list(vs_times_meg)
    buttons_meg = functions.flatten_list(buttons_meg)
    response_times_meg = functions.flatten_list(response_times_meg)
    time_differences = functions.flatten_list(time_differences)
    no_answer = functions.flatten_list((no_answer))

    # Save clean events to MEG data
    raw.annotations.trial = np.array(response_trials_meg)
    raw.annotations.fix1 = np.array(fix1_times_meg)
    raw.annotations.ms = np.array(ms_times_meg)
    raw.annotations.fix2 = np.array(fix2_times_meg)
    raw.annotations.vs = np.array(vs_times_meg)
    raw.annotations.description = np.array(buttons_meg)
    raw.annotations.onset = np.array(response_times_meg)
    raw.annotations.time_differences = np.array(time_differences)
    raw.annotations.no_answer = np.array(no_answer)
    
    # Save data to subject class
    subject.trial = np.array(response_trials_meg)
    subject.fix1 = np.array(fix1_times_meg)
    subject.ms = np.array(ms_times_meg)
    subject.fix2 = np.array(fix2_times_meg)
    subject.vs = np.array(vs_times_meg)
    subject.description = np.array(buttons_meg)
    subject.onset = np.array(response_times_meg)
    subject.time_differences = np.array(time_differences)
    subject.no_answer = np.array(no_answer)

    return bh_data, raw, subject


def fixations_saccades_detection(raw, meg_gazex_data_clean, meg_gazey_data_clean, subject, screen_size=38,
                                 screen_distance=58, screen_resolution=1920, force_run=False):

    out_fname = f'Fix_Sac_detection_{subject.subject_id}.tsv'
    out_folder = paths().results_path() + 'Preprocessing/' + subject.subject_id + '/'

    if not force_run:
        try:
            # Load pre run saccades and fixation detection
            results = pd.read_csv(out_folder + out_fname, sep='\t')
            print('Detected saccades and fixations loaded')
        except:
            force_run = True

    if force_run:
            # If not pre run data, run
            print('Running saccades and fixations detection')

            # Define data to save to excel file needed to run the saccades detection program Remodnav
            eye_data = {'x': meg_gazex_data_clean, 'y': meg_gazey_data_clean}
            df = pd.DataFrame(eye_data)

            # Remodnav parameters
            fname = f'eye_data_{subject.subject_id}.csv'
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

    return fixations, saccades


def fixation_classification(bh_data, fixations, raw, meg_pupils_data_clean):

    fix1_times_meg = raw.annotations.fix1
    response_trials_meg = raw.annotations.trial
    ms_times_meg = raw.annotations.ms
    fix2_times_meg = raw.annotations.fix2
    vs_times_meg = raw.annotations.vs
    response_times_meg = raw.annotations.onset
    times = raw.times

    # Get mss and target pres/abs from bh data
    mss = bh_data['Nstim'].astype(int)
    pres_abs = bh_data['Tpres'].astype(int)
    corr_ans = bh_data['key_resp.corr'].astype(int)

    # Get MSS, present/absent for every trial
    fix_trial = []
    fix_screen = []
    trial_mss = []
    tgt_pres_abs = []
    trial_correct = []
    n_fixs = []
    fix_delay = []
    pupil_size = []

    # Iterate over fixations to classify them
    print('Classifying fixations')

    # Define dict to store screen-trial fixation number
    fix_numbers = {}
    previous_trial = 0

    for fix_time in fixations['onset'].values:
        # find fixation's trial
        for response_idx, trial_end_time in enumerate(response_times_meg):
            if fix_time < trial_end_time:
                break

        # Define trial to store screen fixation number
        trial = response_trials_meg[response_idx]
        trial_idx = trial - 1
        fix_trial.append(trial)

        if trial != previous_trial:
            fix_numbers[trial] = {'fix1': 0, 'ms': 0, 'fix2': 0, 'vs': 0}

        # First fixation cross
        if fix1_times_meg[response_idx] < fix_time < ms_times_meg[response_idx]:
            fix_screen.append('fix1')
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - fix1_times_meg[response_idx])

            screen_time_idx = \
                np.where(np.logical_and(fix1_times_meg[response_idx] < times, times < ms_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # MS
        elif ms_times_meg[response_idx] < fix_time < fix2_times_meg[response_idx]:
            fix_screen.append('ms')
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - ms_times_meg[response_idx])

            screen_time_idx = \
                np.where(np.logical_and(ms_times_meg[response_idx] < times, times < fix2_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # Second fixations corss
        elif fix2_times_meg[response_idx] < fix_time < vs_times_meg[response_idx]:
            fix_screen.append('fix2')
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - fix2_times_meg[response_idx])

            screen_time_idx = \
                np.where(np.logical_and(fix2_times_meg[response_idx] < times, times < vs_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # VS
        elif vs_times_meg[response_idx] < fix_time < response_times_meg[response_idx]:
            fix_screen.append('vs')
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - vs_times_meg[response_idx])

            screen_time_idx = \
                np.where(np.logical_and(vs_times_meg[response_idx] < times, times < response_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # No screen identified
        else:
            fix_screen.append(None)
            trial_mss.append(None)
            tgt_pres_abs.append(None)
            trial_correct.append(None)
            fix_delay.append(None)
            n_fixs.append(None)
            pupil_size.append(None)

        if fix_screen[-1]:
            fix_numbers[trial][fix_screen[-1]] += 1
            n_fixs.append(fix_numbers[trial][fix_screen[-1]])

        previous_trial = trial

    fixations['pupil'] = pupil_size
    fixations['trial'] = fix_trial
    fixations['mss'] = trial_mss
    fixations['screen'] = fix_screen
    fixations['target_pres'] = tgt_pres_abs
    fixations['correct'] = trial_correct
    fixations['n_fix'] = n_fixs
    fixations['time'] = fix_delay

    fixations = fixations.astype({'trial': float, 'mss': float, 'target_pres': float, 'time': float, 'n_fix': float, 'pupil': float})

    return fixations


def target_vs_distractor(fixations, bh_data, distance_threshold=100, screen_res_x=1920, screen_res_y=1080,
                         img_res_x=1280, img_res_y=1024):

    print('Identifying fixated items')

    # Load items data
    items_pos_path = paths().item_pos_path()
    items_pos = pd.read_csv(items_pos_path)

    # Rescale images from original resolution to screen resolution to match fixations scale
    items_pos['pos_x_corr'] = items_pos['pos_x'] + (screen_res_x - img_res_x) / 2
    items_pos['center_x_corr'] = items_pos['center_x'] + (screen_res_x - img_res_x) / 2
    items_pos['pos_y_corr'] = items_pos['pos_y'] + (screen_res_y - img_res_y) / 2
    items_pos['center_y_corr'] = items_pos['center_y'] + (screen_res_y - img_res_y) / 2

    # iterate over fixations checking for trial number, then check image used, then check in item_pos the position of items and mesure distance
    fixations_vs = copy.copy(fixations.loc[fixations['screen'] == 'vs'])

    # Define save data variables
    items = []
    fix_item_distance = []
    fix_target = []
    trials_image = []

     # Iterate over fixations
    for fix_idx, fix in fixations_vs.iterrows():

        # Get fixation trial
        trial = fix['trial']
        trial_idx = trial - 1

        # Get fixations x and y
        fix_x = np.mean([fix['start_x'], fix['end_x']])
        fix_y = np.mean([fix['start_y'], fix['end_y']])

        # Get trial image
        trial_image = bh_data['searchimage'][trial_idx].split('cmp_')[-1].split('.jpg')[0]
        trials_image.append(trial_image)

        # Find item position information for such image
        trial_items = items_pos.loc[items_pos['folder'] == trial_image]

        # Define trial save variables
        distances = []
        target = []

        # Iterate over trial items
        for item_idx, item in trial_items.iterrows():
            # Item position
            item_x = item['center_x_corr']
            item_y = item['center_y_corr']

            # Fixations to item distance
            x_dist = abs(fix_x - item_x)
            y_dist = abs(fix_y - item_y)
            distance = np.sqrt(x_dist ** 2 + y_dist ** 2)

            distances.append(distance)
            target.append(item['istarget'])

        # Closest item to fixation
        min_distance = np.min(np.array(distances))
        min_distance_idx = np.argmin(np.array(distances))

        if min_distance < distance_threshold:
            item = min_distance_idx
            item_distance = min_distance
            istarget = target[min_distance_idx]
        else:
            item = None
            item_distance = None
            istarget = None

        # Save trial data
        items.append(item)
        fix_item_distance.append(item_distance)
        fix_target.append(istarget)

    # Save to fixations_vs df
    fixations_vs['item'] = items
    fixations_vs['fix_target'] = fix_target
    fixations_vs['distance'] = fix_item_distance
    fixations_vs['trial_image'] = trials_image

    return fixations_vs, items_pos