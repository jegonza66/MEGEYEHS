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

    # Define missing values as 1 and non missing as 0 instead of True False
    missing = (meg_pupils_data_clean < -4.6).astype(int)

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
    # Delete succesive presses of green
    blocks_start_end = list(np.delete(blocks_start_end, np.where(np.diff(blocks_start_end) < 2)))
    # Add first trial idx (0) on first sample (0), as we're gonna
    blocks_start_end.insert(0, -1)
    # Add las trial idx
    blocks_start_end.append(len(evt_buttons))
    # Define starting and ending trial of each block
    blocks_bounds = [(blocks_start_end[i] + 1, blocks_start_end[i + 1]) for i in range(len(blocks_start_end) - 1)]

    # Load behavioural data
    bh_data = subject.beh_data()
    # Get only trial data rows
    bh_data = bh_data.loc[~pd.isna(bh_data['target.started'])].reset_index(drop=True)
    # Get MS start time
    ms_start = bh_data['target.started'].values.ravel().astype(float)
    # Get fix 1 start time
    fix1_start_key_idx = ['fixation_target.' in key and 'started' in key for key in bh_data.keys()]
    fix1_start_key = bh_data.keys()[fix1_start_key_idx]
    fix1_start = np.array([value.replace('None', f'{ms_start[i]}') for i, value in
                           enumerate(bh_data[fix1_start_key].values.ravel())]).astype(float)
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
    response_times_meg = []
    response_trials_meg = []

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
        time_diff = search_start[block_num * block_trials] + rt[block_num * block_trials] - meg_evt_block_times[0]
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
                response_times_meg.append(meg_evt_time)
                response_trials_meg.append(int(block_num * block_trials + trial + 1))

                if (meg_evt_block_buttons[idx] == 'blue' and responses_block[trial] != subject.map['blue']) or (
                        meg_evt_block_buttons[idx] == 'red' and responses_block[trial] != subject.map['red']):
                    raise ValueError(f'Different answer in MEG and BH data in trial: {trial}')

                if abs(meg_evt_time - bh_evt_block_times[trial]) > 0.05:
                    print(f'Over 50ms difference in Trial: {trial}')

            else:
                print(f'No answer in Trial: {trial}')
                no_answer.append(block_num * block_trials + trial)

    # Save clean events to MEG data
    raw.annotations.trials = np.array(response_trials_meg)
    raw.annotations.fix1 = np.array(fix1_times_meg)
    raw.annotations.ms = np.array(ms_times_meg)
    raw.annotations.fix2 = np.array(fix2_times_meg)
    raw.annotations.vs = np.array(vs_times_meg)
    raw.annotations.buttons = np.array(buttons_meg)
    raw.annotations.rt = np.array(response_times_meg)

    return bh_data, response_trials_meg, fix1_times_meg, ms_times_meg, fix2_times_meg, vs_times_meg, buttons_meg, response_times_meg


def fixations_saccades_detection(raw, meg_gazex_data_clean, meg_gazey_data_clean, subject, screen_size=38,
                                 screen_distance=58, screen_resolution=1920, force_run=False):

    out_fname = f'Fix_Sac_detection_{subject.subject_id}.tsv'
    out_folder = paths().results_path() + 'Preprocessing/' + subject.subject_id + '/'

    if not force_run:
        try:
            # Load pre run saccades and fixation detection
            print('Loading saccades and fixations detection')
            results = pd.read_csv(out_folder + out_fname, sep='\t')
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


def fixation_classification(bh_data, fixations, fix1_times_meg, response_trials_meg, ms_times_meg, fix2_times_meg,
                            vs_times_meg, response_times_meg, meg_pupils_data_clean, times):

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
    n_fixs = []
    fix_delay = []
    pupil_size = []

    # Agregar proemdio de tamaño de pupila en c fijacion

    # Iterate over fixations to classify them
    print('Classifying fixations')

    # Define dict to store screen-trial fixation number
    fix_numbers = {}
    previous_trial = 0

    for fix_time in fixations['onset'].values:
        # find fixation's trial
        for trial_idx, trial_end_time in enumerate(response_times_meg):
            if fix_time < trial_end_time:
                break

        # Define trial to store screen fixation number
        trial = response_trials_meg[trial_idx]
        fix_trial.append(trial)

        if trial != previous_trial:
            fix_numbers[trial] = {'fix1': 0, 'ms': 0, 'fix2': 0, 'vs': 0}

        # First fixation cross
        if fix1_times_meg[trial_idx] < fix_time < ms_times_meg[trial_idx]:
            fix_screen.append('fix1')
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - fix1_times_meg[trial_idx])

            screen_time_idx = \
                np.where(np.logical_and(fix1_times_meg[trial_idx] < times, times < ms_times_meg[trial_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # MS
        elif ms_times_meg[trial_idx] < fix_time < fix2_times_meg[trial_idx]:
            fix_screen.append('ms')
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - ms_times_meg[trial_idx])

            screen_time_idx = \
                np.where(np.logical_and(ms_times_meg[trial_idx] < times, times < fix2_times_meg[trial_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # Second fixations corss
        elif fix2_times_meg[trial_idx] < fix_time < vs_times_meg[trial_idx]:
            fix_screen.append('fix2')
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - fix2_times_meg[trial_idx])

            screen_time_idx = \
                np.where(np.logical_and(fix2_times_meg[trial_idx] < times, times < vs_times_meg[trial_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # VS
        elif vs_times_meg[trial_idx] < fix_time < response_times_meg[trial_idx]:
            fix_screen.append('vs')
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - vs_times_meg[trial_idx])

            screen_time_idx = \
                np.where(np.logical_and(vs_times_meg[trial_idx] < times, times < response_times_meg[trial_idx]))[0]
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

    fixations['trial'] = fix_trial
    fixations['screen'] = fix_screen
    fixations['mss'] = trial_mss
    fixations['target_pres'] = tgt_pres_abs
    fixations['correct'] = trial_correct
    fixations['delay'] = fix_delay
    fixations['n_fix'] = n_fixs
    fixations['pupil'] = pupil_size
    fixations = fixations.astype({'trial': float, 'mss': float, 'target_pres': float, 'delay': float, 'n_fix': float, 'pupil': float})

    return fixations