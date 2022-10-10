import copy
import numpy as np
import pandas as pd
import os
import math
import mne

import functions
import preproc_plot
from paths import paths


def bh_emap_dur(bh_data_eyemap):

    # Get eyemap durations from BH
    # HL emap trial start
    hl_start_bh = bh_data_eyemap['emap_stim_L.started'].loc[bh_data_eyemap['emap_st_tags'] == 'HL']. \
        astype(float).reset_index(drop=True)

    # VL emap trial start
    vl_start_bh = bh_data_eyemap['emap_stim_L.started'].loc[bh_data_eyemap['emap_st_tags'] == 'VL']. \
        astype(float).reset_index(drop=True)

    # HS emap trial start
    hs_start_bh = bh_data_eyemap['emap_stim_L.started'].loc[bh_data_eyemap['emap_st_tags'] == 'HS']. \
        astype(float).reset_index(drop=True)

    # VS emap trial start
    vs_start_bh = bh_data_eyemap['emap_stim_L.started'].loc[bh_data_eyemap['emap_st_tags'] == 'VS']. \
        astype(float).reset_index(drop=True)

    # BL emap trial start
    bl_start_bh = bh_data_eyemap['emap_stim_L.started'].loc[bh_data_eyemap['emap_st_tags'] == 'BL']. \
        astype(float).reset_index(drop=True)
    bl_end_bh = bh_data_eyemap['emap_stim_L.stopped'].loc[bh_data_eyemap['emap_st_tags'] == 'BL']. \
        astype(float).reset_index(drop=True)

    # End-msg end
    emap_endmsg_end_bh = bh_data_eyemap['key_resp_3.stopped'].loc[bh_data_eyemap['emap_st_tags'] == 'BL']. \
        astype(float).reset_index(drop=True)

    # Get emap screen durations
    end_msg_dur = emap_endmsg_end_bh - bl_end_bh
    bl_dur = bl_end_bh - bl_start_bh
    vs_dur = bl_start_bh - vs_start_bh
    hs_dur = vs_start_bh - hs_start_bh
    vl_dur = hs_start_bh - vl_start_bh
    hl_dur = vl_start_bh - hl_start_bh

    return end_msg_dur, bl_dur, vs_dur, hs_dur, vl_dur, hl_dur


def et_screen_samples(et_data, subject, et_times, exp_info):
    block_start_msg = '!MODE RECORD'
    eyemap_start_msg = 'ETSYNC 152'
    eyemap_end_msg = 'ETSYNC 151'
    cross1_msg = 'ETSYNC 50'
    ms_start_msg = 'ETSYNC 200'
    ms_end_msg = 'ETSYNC 201'
    vs_start_msg = 'ETSYNC 250'
    vs_end_msg = 'ETSYNC 251'

    # ET screen times
    blocks_starttime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(block_start_msg)][1].str.split(block_start_msg, expand=True)[2:][0].values.astype(float)
    eyemap_starttime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(eyemap_start_msg)][1].str.split(eyemap_start_msg, expand=True)[0].values.astype(float)
    eyemap_endtime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(eyemap_end_msg)][1].str.split(eyemap_end_msg, expand=True)[0].values.astype(float)
    ms_starttime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(ms_start_msg)][1].str.split(ms_start_msg, expand=True)[0].values.astype(float)
    ms_endtime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(ms_end_msg)][1].str.split(ms_end_msg, expand=True)[0].values.astype(float)
    cross1_starttime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(cross1_msg)][1].str.split(cross1_msg, expand=True)[0].values.astype(float)
    vs_starttime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(vs_start_msg)][1].str.split(vs_start_msg, expand=True)[0].values.astype(float)
    vs_endtime_et = et_data['msg'].loc[et_data['msg'][1].str.contains(vs_end_msg)][1].str.split(vs_end_msg, expand=True)[0].values.astype(float)

    block_trials = 30
    if subject.subject_id in exp_info.trials_loop_subjects:
        # Define eyemap indexes to keep given that every trial has 5 eyemaps. We want to keep only the 5 eyemaps of the first trial of each block
        good_idx = [element for block_num in range(7) for element in np.arange(block_num*block_trials*5, block_num*block_trials*5+5)]
        eyemap_starttime_et = eyemap_starttime_et[good_idx]
        eyemap_endtime_et = eyemap_endtime_et[good_idx]

    # Add last block end sample to delimit final block end
    blocks_starttime_et = np.concatenate((blocks_starttime_et, np.array([vs_endtime_et[-1]])))

    # ET screen samples
    print('Mapping ET times to samples')
    blocks_start_et, _ = functions.find_nearest(array=et_times, values=blocks_starttime_et)
    eyemap_start_et, _ = functions.find_nearest(array=et_times, values=eyemap_starttime_et)
    eyemap_end_et, _ = functions.find_nearest(array=et_times, values=eyemap_endtime_et)
    ms_start_et, _ = functions.find_nearest(array=et_times, values=ms_starttime_et)
    ms_end_et, _ = functions.find_nearest(array=et_times, values=ms_endtime_et)
    cross1_start_et, _ = functions.find_nearest(array=et_times, values=cross1_starttime_et)
    vs_start_et, _ = functions.find_nearest(array=et_times, values=vs_starttime_et)
    vs_end_et, _ = functions.find_nearest(array=et_times, values=vs_endtime_et)

    return blocks_start_et, eyemap_start_et, eyemap_end_et, ms_start_et, ms_end_et, cross1_start_et, vs_start_et, vs_end_et


def meg_blocks_bounds_evt(raw, et_channel_names):

    # copy raw structure
    print('Extracting ET data and downsampling')
    raw_et = raw.copy()
    # Make new raw structure from et channels only
    raw_et.pick(et_channel_names)
    # Downsample
    raw_et.resample(1000)
    # Get gazex data
    meg_et = raw_et.get_data(et_channel_names)
    meg_gazex = meg_et[0]

    # Get events in meg data (only red blue and green)
    evt_buttons = raw_et.annotations.description
    evt_times = raw_et.annotations.onset[(evt_buttons == 'red') | (evt_buttons == 'blue') | (evt_buttons == 'green')]
    evt_buttons = evt_buttons[(evt_buttons == 'red') | (evt_buttons == 'blue') | (evt_buttons == 'green')]

    # Split events into blocks by green press at begening of block
    blocks_start_end = np.where(evt_buttons == 'green')[0]

    # Add first trial idx (0) on first sample (0) and Add last trial idx in the end
    blocks_start_end = np.concatenate((blocks_start_end, np.array([len(evt_buttons) - 1])))

    # Delete blocks shorter than 20 trials or consecutive green presses
    blocks_start_end = list(np.delete(blocks_start_end, np.where(np.diff(blocks_start_end) < 20)).astype(int))

    # Define starting and ending trial of each block by annotations indexes
    blocks_bounds_evt = [(blocks_start_end[i] - 1, blocks_start_end[i + 1]) for i in range(len(blocks_start_end) - 1)]

    # Get MEG times for block bounds
    blocks_times = [(evt_times[blocks_bounds_evt[i][0]], evt_times[blocks_bounds_evt[i][1]]) for i in
                    range(len(blocks_bounds_evt))]

    # Map MEG times to MEG samples
    blocks_bounds_meg = [(functions.find_nearest(raw_et.times, blocks_times[i][0])[0],
                          functions.find_nearest(raw_et.times, blocks_times[i][1])[0])
                         for i in range(len(blocks_bounds_evt))]

    return raw_et, meg_gazex, evt_buttons, evt_times, blocks_bounds_evt, blocks_bounds_meg


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


def blinks_to_nan(meg_pupils_data_raw, meg_gazex_data_scaled, meg_gazey_data_scaled, config):
    print('Removing blinks')

    # Get configuration
    pupil_size_thresh = config.pupil_thresh
    start_interval_samples = config.start_interval_samples
    end_interval_samples = config.end_interval_samples

    # Copy pupils data to detect blinks from
    meg_gazex_data_clean = copy.copy(meg_gazex_data_scaled)
    meg_gazey_data_clean = copy.copy(meg_gazey_data_scaled)
    meg_pupils_data_clean = copy.copy(meg_pupils_data_raw)

    # Check for peaks in pupils signal
    pupils_diff = np.concatenate((np.array([float('nan')]), np.diff(meg_pupils_data_clean)))

    # Define missing values as 1 and non missing as 0 instead of True False
    missing = ((meg_pupils_data_clean < pupil_size_thresh) | (abs(pupils_diff) > 0.1)).astype(int)

    # Get missing start/end samples and duration
    missing_start = np.where(np.diff(missing) == 1)[0]
    missing_end = np.where(np.diff(missing) == -1)[0]

    # Take samples according to start or end with missing data (missing_start and end would have different sizes)
    if missing[0] == 1:
        missing_end = missing_end[1:]
    if missing[-1] == 1:
        missing_start = missing_start[:-1]

    # Remove blinks by filling missing values (and suroundings) with nan
    for i in range(len(missing_start)):
        blink_interval = np.arange(missing_start[i] - start_interval_samples, missing_end[i] + end_interval_samples)
        meg_gazex_data_clean[blink_interval] = float('nan')
        meg_gazey_data_clean[blink_interval] = float('nan')
        meg_pupils_data_clean[blink_interval] = float('nan')

    return meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean


def fake_blink_interpolate(meg_gazex_data_clean, meg_gazey_data_clean, meg_pupils_data_clean, sfreq, config):

    print('Interpolating fake blinks')

    # Get interpolation configuration
    blink_min_dur = config.blink_min_dur
    start_interval_samples = config.start_interval_samples
    end_interval_samples = config.end_interval_samples

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


def define_events_trials(raw, subject, config, exp_info, et_channel_names, force_realign=False):
    print('Detecting events and defining trials by matching signals')

    #----- Behavioural data -----#
    # Load behavioural data
    bh_data_raw = subject.bh_data()

    # Get bh data from eyemap
    bh_data_eyemap = bh_data_raw.loc[np.logical_and(pd.notnull(bh_data_raw['emap_stim_L.started']),
                                                    bh_data_raw['emap_stim_L.started'] != 'None')].reset_index(drop=True)

    # Get only trial data rows
    bh_data = bh_data_raw.loc[~pd.isna(bh_data_raw['target.started'])].reset_index(drop=True)

    end_msg_dur, bl_dur, vs_dur, hs_dur, vl_dur, hl_dur = bh_emap_dur(bh_data_eyemap=bh_data_eyemap)

    # Load ET data
    et_data = subject.et_data()
    et_gazex = np.asarray(et_data['samples'][1])

    # Define array of times with reset index to map ET time to samples
    et_times = np.asarray(et_data['time'])

    # Get screen start samples from ET
    blocks_start_et, eyemap_start_et, eyemap_end_et, ms_start_et, ms_end_et, cross1_start_et, vs_start_et, vs_end_et = \
        et_screen_samples(et_data=et_data, subject=subject, et_times=et_times, exp_info=exp_info)

    # Downsample raw data and extract gazex channel, evt buttons, times and blocks bounds
    raw_et, meg_gazex, evt_buttons, evt_times, blocks_bounds_evt, blocks_bounds_meg = \
        meg_blocks_bounds_evt(raw=raw, et_channel_names=et_channel_names)

    # Define variables to store data
    no_answer = []
    emap_times_meg = []
    cross1_times_meg = []
    ms_times_meg = []
    cross2_times_meg = []
    vs_times_meg = []
    vsend_times_meg = []
    buttons_meg = []
    response_times_meg = []
    response_trials_meg = []
    time_differences = []
    description = []
    onset = []

    # Save samples shift for each block if no previous data
    if 'et_samples_shift' not in subject.config.preproc.__dict__.keys() or force_realign:
        subject.config.preproc.et_samples_shift = {}
        config.preprocessing.et_samples_shift = {}
        config.update_config = True

    # Define trials block by block
    for block_num in range(len(blocks_bounds_evt)):
        print(f'\nBlock: {block_num + 1}')
        block_bounds_evt = blocks_bounds_evt[block_num]
        block_start_evt = block_bounds_evt[0]
        block_end_evt = block_bounds_evt[1]
        block_trials = 30
        block_idxs = np.arange(block_num * block_trials, (block_num + 1) * block_trials)

        # Get events in block from MEG data
        meg_evt_block_times = copy.copy(evt_times[block_start_evt:block_end_evt])
        meg_evt_block_buttons = copy.copy(evt_buttons[block_start_evt:block_end_evt])

        # Align signals
        if block_num not in subject.config.preproc.et_samples_shift.keys() or force_realign:
            # Drop parts of the signal that don't match to shorten the correlation values search
            et_block_start = blocks_start_et[block_num]
            et_block_end = blocks_start_et[block_num+1]
            et_drop_start = 35000
            et_drop_end = 150000

            meg_block_start = blocks_bounds_meg[block_num][0]
            meg_block_end = blocks_bounds_meg[block_num][1]
            meg_drop_start = 0
            meg_drop_end = 75000

            et_gazex_block = et_gazex[et_block_start+et_drop_start:et_block_end-et_drop_end]
            meg_gazex_block = meg_gazex[meg_block_start+meg_drop_start:meg_block_end-meg_drop_end]

            print('Finding optimal alignment')
            if len(et_gazex_block) > len(meg_gazex_block):
                raise ValueError('ET data is longer than MEG data. Please change the meg_drop_start parameter to a smaller one')
            else:
                max_sample, corrs = functions.align_signals(signal_1=meg_gazex_block, signal_2=et_gazex_block)
                samples_shift = meg_block_start + meg_drop_start + max_sample - et_drop_start - et_block_start
                max_sample = samples_shift - meg_block_start - meg_drop_start + et_drop_start + et_block_start

            # Save samples shift
            subject.config.preproc.et_samples_shift[block_num] = samples_shift

            preproc_plot.alignment(subject=subject, et_gazex=et_gazex, meg_gazex=meg_gazex, corrs=corrs,
                                   et_block_start=et_block_start, meg_block_start=meg_block_start, max_sample=max_sample,
                                   et_block_end=et_block_end, meg_block_end=meg_block_end, et_drop_start=et_drop_start,
                                   meg_drop_start=meg_drop_start, block_num=block_num, block_trials=block_trials,
                                   block_idxs=block_idxs, cross1_start_et=cross1_start_et,
                                   eyemap_start_et=eyemap_start_et, eyemap_end_et=eyemap_end_et)

        # If already previous samples shift data, use that
        else:
            samples_shift = subject.config.preproc.et_samples_shift[block_num]

        # Save block variables
        no_answer_block = []
        buttons_meg_block = []
        response_times_meg_block = []
        response_trials_meg_block = []
        time_diff_block = []
        description_block = []
        onset_block = []

        # # Get emap MEG end time from ET data triggers
        emap_end_block = eyemap_end_et[4 + 5*block_num] + samples_shift
        emap_end_meg_block = raw_et.times[emap_end_block]

        # Get emap trials MEG time from bh duration
        emap_bl_end_meg_block = emap_end_meg_block - end_msg_dur[block_num]
        emap_bl_start_meg_block = emap_bl_end_meg_block - bl_dur[block_num]
        emap_vs_start_meg_block = emap_bl_start_meg_block - vs_dur[block_num]
        emap_hs_start_meg_block = emap_vs_start_meg_block - hs_dur[block_num]
        emap_vl_start_meg_block = emap_hs_start_meg_block - vl_dur[block_num]
        emap_hl_start_meg_block = emap_vl_start_meg_block - hl_dur[block_num]

        # Append to onset and description
        emap_times_meg_block = [emap_hl_start_meg_block, emap_vl_start_meg_block, emap_hs_start_meg_block,
                                emap_vs_start_meg_block, emap_bl_start_meg_block, emap_bl_end_meg_block]
        emap_desc_meg_block = ['hl_start', 'vl_start', 'hs_start', 'vs_start', 'bl_start', 'bl_end']

        onset_block.append(emap_times_meg_block)
        description_block.append(emap_desc_meg_block)

        # Get cross1 MEG start time from ET data triggers and samples shift
        cross1_start_block = cross1_start_et[block_idxs] + samples_shift
        cross1_times_meg_block = raw_et.times[cross1_start_block]
        onset_block.append(list(cross1_times_meg_block))
        description_block.append([f'cross1_t{total_trial}' for total_trial in range(int(block_num * block_trials + 1),
                                                                                    int((block_num + 1) * block_trials + 1))])

        # Get ms start time from ET data triggers and samples shift
        ms_start_block = ms_start_et[block_idxs] + samples_shift
        ms_times_meg_block = raw_et.times[ms_start_block]
        onset_block.append(list(ms_times_meg_block))
        description_block.append([f'ms_t{total_trial}' for total_trial in range(int(block_num * block_trials + 1),
                                                                                int((block_num + 1) * block_trials + 1))])

        # Get cross2 MEG start time from ET data triggers and samples shift
        cross2_start_block = ms_end_et[block_idxs] + samples_shift
        cross2_times_meg_block = raw_et.times[cross2_start_block]
        onset_block.append(list(cross2_times_meg_block))
        description_block.append([f'cross2_t{total_trial}' for total_trial in range(int(block_num * block_trials + 1),
                                                                                    int((block_num + 1) * block_trials + 1))])

        # Get ms MEG start time from ET data triggers and samples shift
        vs_start_block = vs_start_et[block_idxs] + samples_shift
        vs_times_meg_block = raw_et.times[vs_start_block]
        onset_block.append(list(vs_times_meg_block))
        description_block.append([f'vs_t{total_trial}' for total_trial in range(int(block_num * block_trials + 1),
                                                                                int((block_num + 1) * block_trials + 1))])

        # Get vs MEG start time from ET data triggers and samples shift
        vs_end_block = vs_end_et[block_idxs] + samples_shift
        vs_end_times_meg_block = raw_et.times[vs_end_block]
        onset_block.append(list(vs_end_times_meg_block))
        description_block.append([f'vsend_t{total_trial}' for total_trial in range(int(block_num * block_trials + 1),
                                                                                int((block_num + 1) * block_trials + 1))])

        # Flatten lists to append trial data
        onset_block = functions.flatten_list(onset_block)
        description_block = functions.flatten_list(description_block)

        # Iterate over trials
        print('Identifying responses in trials')
        for trial in range(block_trials):
            total_trial = int(block_num * block_trials + trial + 1)

            idx, meg_evt_time = functions.find_nearest(array=meg_evt_block_times, values=vs_end_times_meg_block[trial])
            time_diff = vs_end_times_meg_block[trial] - meg_evt_time

            # Answer after a 10 search screen or time difference of 100 ms -> No answer
            if (time_diff < 0 and vs_end_times_meg_block[trial] - vs_times_meg_block[trial] >= 10) or abs(time_diff) > 0.1:
                print(f'No answer in Trial: {total_trial}')
                no_answer_block.append(total_trial)

            else:
                if abs(time_diff) > 0.02:
                    print(f'{round((vs_end_times_meg_block[trial] - meg_evt_time) * 1000, 1)} ms difference in Trial: {total_trial}')

                # Append time and button to annotations
                onset_block.append(meg_evt_time)
                description_block.append(meg_evt_block_buttons[idx])

                # Append to subject data
                time_diff_block.append(time_diff)
                buttons_meg_block.append((meg_evt_block_buttons[idx]))
                response_times_meg_block.append(meg_evt_time)
                response_trials_meg_block.append(total_trial)

        # Append block data to overall data
        emap_times_meg.append(emap_times_meg_block)
        cross1_times_meg.append(cross1_times_meg_block)
        ms_times_meg.append(ms_times_meg_block)
        cross2_times_meg.append(cross2_times_meg_block)
        vs_times_meg.append(vs_times_meg_block)
        vsend_times_meg.append(vs_end_times_meg_block)
        buttons_meg.append(buttons_meg_block)
        response_times_meg.append(response_times_meg_block)
        response_trials_meg.append(response_trials_meg_block)
        no_answer.append(no_answer_block)

        # Save annotations from block
        description.append(description_block)
        onset.append(onset_block)

    # Flatten variables over blocks
    emap_times_meg = functions.flatten_list(emap_times_meg)
    cross1_times_meg = functions.flatten_list(cross1_times_meg)
    ms_times_meg = functions.flatten_list(ms_times_meg)
    cross2_times_meg = functions.flatten_list(cross2_times_meg)
    vs_times_meg = functions.flatten_list(vs_times_meg)
    vsend_times_meg = functions.flatten_list(vsend_times_meg)
    buttons_meg = functions.flatten_list(buttons_meg)
    response_times_meg = functions.flatten_list(response_times_meg)
    time_differences = functions.flatten_list(time_differences)
    response_trials_meg = functions.flatten_list(response_trials_meg)
    no_answer = functions.flatten_list(no_answer)

    description = functions.flatten_list(description)
    onset = functions.flatten_list(onset)

    # Save clean events to MEG data
    raw.annotations.description = np.array(description)
    raw.annotations.onset = np.array(onset)

    # Save data to subject class
    subject.emap = np.array(emap_times_meg)
    subject.cross1 = np.array(cross1_times_meg)
    subject.ms = np.array(ms_times_meg)
    subject.cross2 = np.array(cross2_times_meg)
    subject.vs = np.array(vs_times_meg)
    subject.vsend = np.array(vsend_times_meg)
    subject.description = np.array(buttons_meg)
    subject.onset = np.array(response_times_meg)
    subject.time_differences = np.array(time_differences)
    subject.trial = np.array(response_trials_meg)
    subject.no_answer = np.array(no_answer)

    # Save updated configuration
    if config.update_config:
        config.preprocessing.et_samples_shift[subject.subject_id] = subject.config.preproc.et_samples_shift
        config.save_config(config_path=paths().config_path())

    return bh_data, raw, subject


def fixations_saccades_detection(raw, meg_gazex_data_clean, meg_gazey_data_clean, subject, screen_size=38,
                                 screen_distance=58, screen_resolution=1920, force_run=False):

    out_fname = f'Fix_Sac_detection_{subject.subject_id}.tsv'
    out_folder = paths().save_path() + 'Preprocesed_Data/' + subject.subject_id + '/Sac-Fix_detection/'

    if not force_run:
        try:
            # Load pre run saccades and fixation detection
            results = pd.read_csv(out_folder + out_fname, sep='\t')
            print('Saccades and fixations loaded')
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


def saccades_classification(subject, bh_data, saccades, raw, exp_info):

    print('Classifying saccades')

    # Eyemap trials start and end indexes
    hl_start_idx = np.where(raw.annotations.description == 'hl_start')[0]
    vl_start_idx = np.where(raw.annotations.description == 'vl_start')[0]
    hs_start_idx = np.where(raw.annotations.description == 'hs_start')[0]
    vs_start_idx = np.where(raw.annotations.description == 'vs_start')[0]
    bl_start_idx = np.where(raw.annotations.description == 'bl_start')[0]
    bl_end_idx = np.where(raw.annotations.description == 'bl_end')[0]

    # Eyemap trials start and end times
    hl_start_times = raw.annotations.onset[hl_start_idx]
    vl_start_times = raw.annotations.onset[vl_start_idx]
    hs_start_times = raw.annotations.onset[hs_start_idx]
    vs_start_times = raw.annotations.onset[vs_start_idx]
    bl_start_times = raw.annotations.onset[bl_start_idx]
    bl_end_times = raw.annotations.onset[bl_end_idx]

    response_trials_meg = subject.trial
    cross1_times_meg = subject.cross1
    ms_times_meg = subject.ms
    cross2_times_meg = subject.cross2
    vs_times_meg = subject.vs
    vsend_times_meg = subject.vsend

    # Get mss and target pres/abs from bh data
    mss = bh_data['Nstim'].astype(int)
    pres_abs = bh_data['Tpres'].astype(int)
    corr_ans = bh_data['key_resp.corr'].astype(int)

    # Get correct_answers for subjects with no BH
    if subject.subject_id in exp_info.missing_bh_subjects:
        corr_ans = np.zeros(len(bh_data)).astype(int)
        corr_answers = bh_data['corrAns']
        actual_answers = subject.description
        for i, trial in enumerate(response_trials_meg):
            trial_idx = trial-1
            if int(subject.map[actual_answers[i]]) != corr_answers[trial_idx]: # Check if mapping of button corresponds to correct answer
                corr_ans[trial_idx] = 0
            else:
                corr_ans[trial_idx] = 1

    # Get MSS, present/absent for every trial
    sac_trial = []
    sac_screen = []
    trial_mss = []
    tgt_pres_abs = []
    trial_correct = []
    n_sacs = []
    sac_delay = []
    description = []
    onset = []

    # Define dict to store screen-trial fixation number
    sac_numbers = {}

    for i, sac_time in enumerate(saccades['onset'].values):
        emap_sac = False

        # find saccades's block
        for (block_num, emap_start_time), emap_end_time in zip(enumerate(hl_start_times), bl_end_times):
            if emap_start_time < sac_time < emap_end_time:
                emap_sac = True
                break

        # Save sac number within block
        if block_num not in sac_numbers.keys():
            sac_numbers[block_num] = {'emap_hl': 0, 'emap_vl': 0, 'emap_hs': 0, 'emap_vs': 0, 'emap_bl': 0}
        
        if emap_sac:
            # Find saccades emap trial
            sac_trial.append(None)
            trial_mss.append(None)
            tgt_pres_abs.append(None)
            trial_correct.append(None)

            # hl
            if hl_start_times[block_num] < sac_time < vl_start_times[block_num]:
                # Saccade data
                screen = 'emap_hl'
                sac_screen.append(screen)
                sac_delay.append(sac_time - hl_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                description.append(f'sac_{screen}_{n_sacs[-1]}')
                onset.append([sac_time])
            # vl
            elif vl_start_times[block_num] < sac_time < hs_start_times[block_num]:
                # Saccade data
                screen = 'emap_vl'
                sac_screen.append(screen)
                sac_delay.append(sac_time - vl_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                description.append(f'sac_{screen}_{n_sacs[-1]}')
                onset.append([sac_time])
            # hs
            elif hs_start_times[block_num] < sac_time < vs_start_times[block_num]:
                # Saccade data
                screen = 'emap_hs'
                sac_screen.append(screen)
                sac_delay.append(sac_time - hs_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                description.append(f'sac_{screen}_{n_sacs[-1]}')
                onset.append([sac_time])
            # vs
            elif vs_start_times[block_num] < sac_time < bl_start_times[block_num]:
                # Saccade data
                screen = 'emap_vs'
                sac_screen.append(screen)
                sac_delay.append(sac_time - vs_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                description.append(f'sac_{screen}_{n_sacs[-1]}')
                onset.append([sac_time])
            # bl
            elif bl_start_times[block_num] < sac_time < bl_end_times[block_num]:
                # Saccade data
                screen = 'emap_bl'
                sac_screen.append(screen)
                sac_delay.append(sac_time - bl_start_times[block_num])
                sac_numbers[block_num][screen] += 1
                n_sacs.append(sac_numbers[block_num][screen])

                # Save to raw
                description.append(f'sac_{screen}_{n_sacs[-1]}')
                onset.append([sac_time])

            else:
                sac_screen.append(None)
                sac_delay.append(None)
                n_sacs.append(None)

        # No emap saccade
        else:
            trial_found = False
            for (trial_idx, trial_start_time), trial_end_time in zip(enumerate(cross1_times_meg), vsend_times_meg):
                if trial_start_time < sac_time < trial_end_time:
                    trial_found = True
                    break
            if trial_found:
                # Define trial to store screen sacation number. if sac_time > all trials end time, sac gets stored as last trial under screen None.
                trial = trial_idx + 1

                sac_trial.append(trial)
                trial_mss.append(mss[trial_idx])
                tgt_pres_abs.append(pres_abs[trial_idx])
                trial_correct.append(corr_ans[trial_idx])

                if f'trial_{trial}' not in sac_numbers[block_num].keys():
                    sac_numbers[block_num][f'trial_{trial}'] = {'cross1': 0, 'ms': 0, 'cross2': 0, 'vs': 0}

                # First sacation cross
                if cross1_times_meg[trial_idx] < sac_time < ms_times_meg[trial_idx]:
                    # Saccade data
                    screen = 'cross1'
                    sac_screen.append(screen)
                    sac_delay.append(sac_time - cross1_times_meg[trial_idx])
                    sac_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_sacs.append(sac_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    description.append(f'sac_{screen}_{n_sacs[-1]}')
                    onset.append([sac_time])

                # MS
                elif ms_times_meg[trial_idx] < sac_time < cross2_times_meg[trial_idx]:
                    # Saccade data
                    screen = 'ms'
                    sac_screen.append(screen)
                    sac_delay.append(sac_time - ms_times_meg[trial_idx])
                    sac_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_sacs.append(sac_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    description.append(f'sac_{screen}_{n_sacs[-1]}')
                    onset.append([sac_time])

                # Second sacations corss
                elif cross2_times_meg[trial_idx] < sac_time < vs_times_meg[trial_idx]:
                    # Saccade data
                    screen = 'cross2'
                    sac_screen.append(screen)
                    sac_delay.append(sac_time - ms_times_meg[trial_idx])
                    sac_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_sacs.append(sac_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    description.append(f'sac_{screen}_{n_sacs[-1]}')
                    onset.append([sac_time])

                # VS
                elif vs_times_meg[trial_idx] < sac_time < vsend_times_meg[trial_idx]:
                    # Saccade data
                    screen = 'vs'
                    sac_screen.append(screen)
                    sac_delay.append(sac_time - ms_times_meg[trial_idx])
                    sac_numbers[block_num][f'trial_{trial}'][screen] += 1
                    n_sacs.append(sac_numbers[block_num][f'trial_{trial}'][screen])

                    # Save to raw
                    description.append(f'sac_{screen}_{n_sacs[-1]}')
                    onset.append([sac_time])

                # No screen identified
                else:
                    sac_screen.append(None)
                    sac_delay.append(None)
                    n_sacs.append(None)

            # No trial identified
            else:
                sac_trial.append(None)
                trial_mss.append(None)
                tgt_pres_abs.append(None)
                trial_correct.append(None)
                sac_screen.append(None)
                sac_delay.append(None)
                n_sacs.append(None)

        print("\rProgress: {}%".format(int((i + 1) * 100 / len(saccades))), end='')

    print()
    saccades['trial'] = sac_trial
    saccades['mss'] = trial_mss
    saccades['screen'] = sac_screen
    saccades['target_pres'] = tgt_pres_abs
    saccades['correct'] = trial_correct
    saccades['n_sac'] = n_sacs
    saccades['time'] = sac_delay

    saccades = saccades.astype({'trial': 'Int64', 'mss': 'Int64', 'target_pres': 'Int64', 'time': float, 'correct': 'Int64',
                                'n_sac': 'Int64'})

    # Add vs fixations data to raw annotations
    raw.annotations.description = np.concatenate((raw.annotations.description, np.array(description)))
    raw.annotations.onset = np.concatenate((raw.annotations.onset, np.array(functions.flatten_list(onset))))

    # Order raw.annotations chronologically
    raw_annot = np.array([raw.annotations.onset, raw.annotations.description])
    raw_annot = raw_annot[:, raw_annot[0].astype(float).argsort()]
    raw.annotations.onset = raw_annot[0].astype(float)
    raw.annotations.description = raw_annot[1]

    # Set durations and ch_names length to match the annotations length
    raw.annotations.duration = np.zeros(len(raw.annotations.description))

    return saccades, raw


def fixation_classification(subject, bh_data, fixations, raw, meg_pupils_data_clean, exp_info):

    cross1_times_meg = subject.cross1
    response_trials_meg = subject.trial
    ms_times_meg = subject.ms
    cross2_times_meg = subject.cross2
    vs_times_meg = subject.vs
    vsend_times_meg = subject.vsend
    times = raw.times

    # Get mss and target pres/abs from bh data
    mss = bh_data['Nstim'].astype(int)
    pres_abs = bh_data['Tpres'].astype(int)
    corr_ans = bh_data['key_resp.corr'].astype(int)

    if subject.subject_id in exp_info.missing_bh_subjects:
        corr_ans = np.zeros(len(bh_data)).astype(int)
        corr_answers = bh_data['corrAns']
        actual_answers = subject.description
        for i, trial in enumerate(response_trials_meg):
            trial_idx = trial-1
            if int(subject.map[actual_answers[i]]) != corr_answers[trial_idx]: # Check if mapping of button corresponds to correct answer
                corr_ans[trial_idx] = 0
            else:
                corr_ans[trial_idx] = 1

    # Differences only in no answer trials. We're recovering the answers!
    # diff = (corr_ans1-corr_ans).values
    # np.where(diff!=0)[0]+1

    # Get MSS, present/absent for every trial
    fix_trial = []
    fix_screen = []
    trial_mss = []
    tgt_pres_abs = []
    trial_correct = []
    n_fixs = []
    fix_delay = []
    pupil_size = []
    description = []
    onset = []

    # Iterate over fixations to classify them
    print('Classifying fixations')

    # Define dict to store screen-trial fixation number
    fix_numbers = {}

    for i, fix_time in enumerate(fixations['onset'].values):
        # find fixation's trial
        for trial_idx, trial_end_time in enumerate(vsend_times_meg):
            if fix_time < trial_end_time:
                break

        # Define trial to store screen fixation number
        trial = trial_idx + 1

        fix_trial.append(trial)
        trial_mss.append(mss[trial_idx])
        tgt_pres_abs.append(pres_abs[trial_idx])
        trial_correct.append(corr_ans[trial_idx])

        if trial not in fix_numbers.keys():
            fix_numbers[trial] = {'cross1': 0, 'ms': 0, 'cross2': 0, 'vs': 0}

        # First fixation cross
        if cross1_times_meg[trial_idx] < fix_time < ms_times_meg[trial_idx]:
            screen = 'cross1'
            fix_screen.append(screen)
            fix_delay.append(fix_time - cross1_times_meg[trial_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(cross1_times_meg[trial_idx] < times, times < ms_times_meg[trial_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # MS
        elif ms_times_meg[trial_idx] < fix_time < cross2_times_meg[trial_idx]:
            screen = 'ms'
            fix_screen.append(screen)
            fix_delay.append(fix_time - ms_times_meg[trial_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(ms_times_meg[trial_idx] < times, times < cross2_times_meg[trial_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # Second fixations corss
        elif cross2_times_meg[trial_idx] < fix_time < vs_times_meg[trial_idx]:
            screen = 'cross2'
            fix_screen.append(screen)
            fix_delay.append(fix_time - cross2_times_meg[trial_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(cross2_times_meg[trial_idx] < times, times < vs_times_meg[trial_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # VS
        elif vs_times_meg[trial_idx] < fix_time < vsend_times_meg[trial_idx]:
            screen = 'vs'
            fix_screen.append(screen)
            fix_delay.append(fix_time - vs_times_meg[trial_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(vs_times_meg[trial_idx] < times, times < vsend_times_meg[trial_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # No screen identified
        else:
            fix_screen.append(None)
            fix_delay.append(None)
            n_fixs.append(None)
            pupil_size.append(None)

        previous_trial = trial
        print("\rProgress: {}%".format(int((i + 1) * 100 / len(fixations))), end='')

    print()
    fixations['pupil'] = pupil_size
    fixations['trial'] = fix_trial
    fixations['mss'] = trial_mss
    fixations['screen'] = fix_screen
    fixations['target_pres'] = tgt_pres_abs
    fixations['correct'] = trial_correct
    fixations['n_fix'] = n_fixs
    fixations['time'] = fix_delay

    fixations = fixations.astype({'trial': int, 'mss': 'Int64', 'target_pres': 'Int64', 'time': float, 'correct': 'Int64',
                                  'n_fix': 'Int64', 'pupil': float})

    # Add vs fixations data to raw annotations
    raw.annotations.description = np.concatenate((raw.annotations.description, np.array(description)))
    raw.annotations.onset = np.concatenate((raw.annotations.onset, np.array(functions.flatten_list(onset))))

    # Order raw.annotations chronologically
    raw_annot = np.array([raw.annotations.onset, raw.annotations.description])
    raw_annot = raw_annot[:, raw_annot[0].astype(float).argsort()]
    raw.annotations.onset = raw_annot[0].astype(float)
    raw.annotations.description = raw_annot[1]

    # Set durations and ch_names length to match the annotations length
    raw.annotations.duration = np.zeros(len(raw.annotations.description))

    return fixations, raw


def target_vs_distractor(fixations, bh_data, subject, distance_threshold=100, screen_res_x=1920, screen_res_y=1080,
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
    fixations.loc[fixations['screen'] == 'vs', 'item'] = items
    fixations.loc[fixations['screen'] == 'vs', 'fix_target'] = fix_target
    fixations.loc[fixations['screen'] == 'vs', 'distance'] = fix_item_distance
    fixations.loc[fixations['screen'] == 'vs', 'trial_image'] = trials_image

    return fixations, items_pos


def add_et_channels(raw, et_channels_meg, et_channel_names):
    #---------------- Add scaled data to meg data ----------------#
    print('\nSaving scaled et data to meg raw data structure')
    # copy raw structure
    raw_et = raw.copy()
    # make new raw structure from et channels only
    raw_et = mne.io.RawArray(et_channels_meg, raw_et.pick(et_channel_names).info)
    # change channel names
    for ch_name, new_name in zip(raw_et.ch_names, ['ET_gaze_x', 'ET_gaze_y', 'ET_pupils']):
        raw_et.rename_channels({ch_name: new_name})

    # Pick data from MEG channels and other channels of interest
    channel_idx = mne.pick_types(raw.info, meg=True)
    channel_idx = np.append(channel_idx, mne.pick_channels(raw.info['ch_names'], ['UPPT001', 'UADC001-4123', 'UADC002-4123', 'UADC013-4123']))
    raw.pick(channel_idx)

    # save to original raw structure (requires to load data)
    print('Loading MEG data')
    raw.load_data()
    print('Adding new ET channels')
    raw.add_channels([raw_et], force_update_info=True)

    return raw



## OLD out of use


def define_events_trials_BH(raw, subject):
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
    cross1_start_key_idx = ['fixation_target.' in key and 'started' in key for key in bh_data.keys()]
    cross1_start_key = bh_data.keys()[cross1_start_key_idx][0]

    # If there's a missing cross1 screen, the time for that screen would be None, and the column type would be string.
    # If there's not, the type would be float. Change type to string and replace None by the mss screen start time
    if type(bh_data[cross1_start_key][0]) == str:
        cross1_start = np.array([value.replace('None', f'{ms_start[i]}') for i, value in
                                 enumerate(bh_data[cross1_start_key].values.ravel())]).astype(float)
    else:
        cross1_start = bh_data[cross1_start_key].values

    # Get fix 2 start time
    cross2_start_key_idx = ['fixation_target_2' in key and 'started' in key for key in bh_data.keys()]
    cross2_start_key = bh_data.keys()[cross2_start_key_idx]
    cross2_start = bh_data[cross2_start_key].values.ravel().astype(float)

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
    cross1_times_meg = []
    ms_times_meg = []
    cross2_times_meg = []
    vs_times_meg = []
    buttons_meg = []
    response_times_meg = []
    response_trials_meg = []
    time_differences = []
    description = []
    onset = []

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
        cross1_block_dur = (ms_start - cross1_start)[block_idxs]
        ms_block_dur = (cross2_start - ms_start)[block_idxs]
        cross2_block_dur = (search_start - cross2_start)[block_idxs]
        rt_block_times = rt[block_idxs]

        # Align MEG and BH data in time
        block_data_aligned = False
        attempts = 0
        align_sample = 0
        while not block_data_aligned and attempts < 5:
            # Save block variables resetting them every attempt
            no_answer_block = []
            cross1_times_meg_block = []
            ms_times_meg_block = []
            cross2_times_meg_block = []
            vs_times_meg_block = []
            buttons_meg_block = []
            response_times_meg_block = []
            response_trials_meg_block = []
            time_diff_block = []
            description_block = []
            onset_block = []

            try:
                # Get events in block from BH data
                bh_evt_times_block = (search_start + rt)[block_idxs]
                responses_block = responses[block_idxs]

                # Realign bh and meg block timelines
                block_time_realign = search_start[block_num * block_trials] + rt[block_num * block_trials] - \
                                     meg_evt_block_times[align_sample]
                bh_evt_times_block = bh_evt_times_block - block_time_realign

                # Iterate over trials
                for trial in range(block_trials):
                    if not np.isnan(bh_evt_times_block[trial]):
                        total_trial = int(block_num * block_trials + trial + 1)

                        idx, meg_evt_time = functions.find_nearest(meg_evt_block_times, bh_evt_times_block[trial])
                        onset_block.append(meg_evt_time)
                        description_block.append(meg_evt_block_buttons[idx])

                        time_diff = bh_evt_times_block[trial] - meg_evt_time
                        time_diff_block.append(time_diff)

                        if (meg_evt_block_buttons[idx] == 'blue' and int(responses_block[trial]) != int(
                                subject.map['blue'])) or (
                                meg_evt_block_buttons[idx] == 'red' and int(responses_block[trial]) != int(
                            subject.map['red'])):
                            print(f'Different answer in MEG and BH data in trial: {trial + 1}\n'
                                  f'Discarding and realingning on following sample\n')
                            raise ValueError(f'Different answer in MEG and BH data in trial: {trial + 1}')

                        if abs(time_diff) > 0.02:  # and block_num * block_trials + trial not in completed_responses_idx:
                            print(
                                f'{round(abs(meg_evt_time - bh_evt_times_block[trial]) * 1000, 1)} ms difference in Trial: {trial + 1}')

                        # Define screen times from MEG response
                        trial_search_time = meg_evt_time - rt_block_times[trial]
                        vs_times_meg_block.append(trial_search_time)
                        onset_block.append(trial_search_time)
                        description_block.append(f'vs_t{total_trial}')

                        trial_cross2_time = trial_search_time - cross2_block_dur[trial]
                        cross2_times_meg_block.append(trial_cross2_time)
                        onset_block.append(trial_cross2_time)
                        description_block.append(f'cross2_t{total_trial}')

                        trial_ms_time = trial_cross2_time - ms_block_dur[trial]
                        ms_times_meg_block.append(trial_ms_time)
                        onset_block.append(trial_ms_time)
                        description_block.append(f'ms_t{total_trial}')

                        trial_cross1_time = trial_ms_time - cross1_block_dur[trial]
                        cross1_times_meg_block.append(trial_cross1_time)

                        # Save cross1 onset only if there was cross 1 present in that trial
                        if cross1_block_dur[trial]:
                            onset_block.append(trial_cross1_time)
                            description_block.append(f'cross1_t{total_trial}')

                        buttons_meg_block.append((meg_evt_block_buttons[idx]))
                        response_times_meg_block.append(meg_evt_time)
                        response_trials_meg_block.append(int(block_num * block_trials + trial + 1))

                    # Consider manually completed responses
                    # elif np.isnan(bh_evt_times_block[trial]) and responses_block[trial] != 'None':

                    else:
                        print(f'No answer in Trial: {trial + 1}')
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
                    cross1_times_meg.append(cross1_times_meg_block)
                    ms_times_meg.append(ms_times_meg_block)
                    cross2_times_meg.append(cross2_times_meg_block)
                    vs_times_meg.append(vs_times_meg_block)
                    buttons_meg.append(buttons_meg_block)
                    response_times_meg.append(response_times_meg_block)
                    response_trials_meg.append(response_trials_meg_block)
                    time_differences.append(time_diff_block)

                    # Save annotations from block
                    description.append(description_block)
                    onset.append(onset_block)
            except:
                align_sample += 1
                attempts += 1

        if not block_data_aligned:
            raise ValueError(f'Could not align MEG and BH responses in block {block_num + 1}')

    # flatten variables over blocks
    response_trials_meg = functions.flatten_list(response_trials_meg)
    cross1_times_meg = functions.flatten_list(cross1_times_meg)
    ms_times_meg = functions.flatten_list(ms_times_meg)
    cross2_times_meg = functions.flatten_list(cross2_times_meg)
    vs_times_meg = functions.flatten_list(vs_times_meg)
    buttons_meg = functions.flatten_list(buttons_meg)
    response_times_meg = functions.flatten_list(response_times_meg)
    time_differences = functions.flatten_list(time_differences)
    no_answer = functions.flatten_list(no_answer)

    description = functions.flatten_list(description)
    onset = functions.flatten_list(onset)

    # Save clean events to MEG data
    raw.annotations.description = np.array(description)
    raw.annotations.onset = np.array(onset)

    # Save data to subject class
    subject.trial = np.array(response_trials_meg)
    subject.cross1 = np.array(cross1_times_meg)
    subject.ms = np.array(ms_times_meg)
    subject.cross2 = np.array(cross2_times_meg)
    subject.vs = np.array(vs_times_meg)
    subject.description = np.array(buttons_meg)
    subject.onset = np.array(response_times_meg)
    subject.time_differences = np.array(time_differences)
    subject.no_answer = np.array(no_answer)

    return bh_data, raw, subject



def fixation_classification_BH(subject, bh_data, fixations, raw, meg_pupils_data_clean, exp_info):

    cross1_times_meg = subject.cross1
    response_trials_meg = subject.trial
    ms_times_meg = subject.ms
    cross2_times_meg = subject.cross2
    vs_times_meg = subject.vs
    response_times_meg = subject.onset
    times = raw.times

    # Get mss and target pres/abs from bh data
    mss = bh_data['Nstim'].astype(int)
    pres_abs = bh_data['Tpres'].astype(int)
    corr_ans = bh_data['key_resp.corr'].astype(int)

    if subject.subject_id in exp_info.missing_bh_subjects:
        corr_ans = np.zeros(len(bh_data)).astype(int)
        corr_answers = bh_data['corrAns']
        actual_answers = subject.description
        for i, trial in enumerate(response_trials_meg):
            trial_idx = trial-1
            if int(subject.map[actual_answers[i]]) != corr_answers[trial_idx]: # Check if mapping of button corresponds to correct answer
                corr_ans[trial_idx] = 0
            else:
                corr_ans[trial_idx] = 1

    # Differences only in no answer trials. We're recovering the answers!
    # diff = (corr_ans1-corr_ans).values
    # np.where(diff!=0)[0]+1

    # Get MSS, present/absent for every trial
    fix_trial = []
    fix_screen = []
    trial_mss = []
    tgt_pres_abs = []
    trial_correct = []
    n_fixs = []
    fix_delay = []
    pupil_size = []
    description = []
    onset = []

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
            fix_numbers[trial] = {'cross1': 0, 'ms': 0, 'cross2': 0, 'vs': 0}

        # First fixation cross
        if cross1_times_meg[response_idx] < fix_time < ms_times_meg[response_idx]:
            screen = 'cross1'
            fix_screen.append(screen)
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - cross1_times_meg[response_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(cross1_times_meg[response_idx] < times, times < ms_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # MS
        elif ms_times_meg[response_idx] < fix_time < cross2_times_meg[response_idx]:
            screen = 'ms'
            fix_screen.append(screen)
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - ms_times_meg[response_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(ms_times_meg[response_idx] < times, times < cross2_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # Second fixations corss
        elif cross2_times_meg[response_idx] < fix_time < vs_times_meg[response_idx]:
            screen = 'cross2'
            fix_screen.append(screen)
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - cross2_times_meg[response_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
            screen_time_idx = \
                np.where(np.logical_and(cross2_times_meg[response_idx] < times, times < vs_times_meg[response_idx]))[0]
            pupil_data_screen = meg_pupils_data_clean[screen_time_idx]
            pupil_size.append(np.nanmean(pupil_data_screen))

        # VS
        elif vs_times_meg[response_idx] < fix_time < response_times_meg[response_idx]:
            screen = 'vs'
            fix_screen.append(screen)
            trial_mss.append(mss[trial_idx])
            tgt_pres_abs.append(pres_abs[trial_idx])
            trial_correct.append(corr_ans[trial_idx])
            fix_delay.append(fix_time - vs_times_meg[response_idx])
            fix_numbers[trial][screen] += 1
            n_fixs.append(fix_numbers[trial][screen])

            description.append(f'fix_{screen}_{n_fixs[-1]}')
            onset.append([fix_time])

            # Average pupil size
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

    # Add vs fixations data to raw annotations
    raw.annotations.description = np.concatenate((raw.annotations.description, np.array(description)))
    raw.annotations.onset = np.concatenate((raw.annotations.onset, np.array(functions.flatten_list(onset))))

    # Order raw.annotations chronologically
    raw_annot = np.array([raw.annotations.onset, raw.annotations.description])
    raw_annot = raw_annot[:, raw_annot[0].astype(float).argsort()]
    raw.annotations.onset = raw_annot[0].astype(float)
    raw.annotations.description = raw_annot[1]

    # Set durations and ch_names length to match the annotations length
    raw.annotations.duration = np.zeros(len(raw.annotations.description))

    return fixations, raw
