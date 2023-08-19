import numpy as np
import pandas as pd
import mne


def scale_from_interval(signal_to_scale, reference_signal, interval_signal=None, interval_ref=None):
    """
    Scale signal based on matching signal with different scale.

    Parameters
    ----------
    signal_to_scale: ndarray
        The 1D signal you wish to re-scale.
    reference_signal: ndarray
        The 1D reference signal with propper scaling.
    interval_signal: {'list', 'tuple'}, optional
        The signal interval to use for scaling. if None, the whole signal is used. Default to None.
    interval_ref: {'list', 'tuple'}, optional
        The reference signal interval to use for scaling. if None, the whole signal is used. Default to None.

    Returns
    -------
    signal_to_scale: ndarray
      The re-scaled signal.
    """

    signal_to_scale -= np.nanmin(signal_to_scale[interval_signal[0]:interval_signal[1]])
    signal_to_scale /= np.nanmax(signal_to_scale[interval_signal[0]:interval_signal[1]])
    signal_to_scale *= (
            np.nanmax(reference_signal[interval_ref[0]:interval_ref[1]])
            - np.nanmin(reference_signal[interval_ref[0]:interval_ref[1]]))
    signal_to_scale += np.nanmin(reference_signal[interval_ref[0]:interval_ref[1]])

    return signal_to_scale


def remove_missing(x, y, time, missing):
    mx = np.array(x == missing, dtype=int)
    my = np.array(y == missing, dtype=int)
    x = x[(mx + my) != 2]
    y = y[(mx + my) != 2]
    time = time[(mx + my) != 2]
    return x, y, time


def get_buttons_and_times(raw, exp_info):

    # # Get data from buttons channel
    # buttons_channel = raw.get_data(picks=exp_info.button_ch).ravel()
    # # Compute difference
    # buttons_diff = np.diff(buttons_channel, prepend=buttons_channel[0])
    # # Get onset from button events
    # button_evt_idx = np.where(buttons_diff > 0)[0]
    # # Get values from buttons channel in events
    # button_values = buttons_channel[button_evt_idx]
    # # Map from values to colors
    # button_colors = np.vectorize(exp_info.buttons_ch_map.get)(button_values)

    # Get events from buttons channel
    button_events = mne.find_events(raw=raw, stim_channel=exp_info.button_ch)
    # Get events index
    button_evt_idx = button_events[:, 0]
    # Get events channel values
    button_values = button_events[:, 2]
    # Map channel values to colors
    button_colors = np.vectorize(exp_info.buttons_ch_map.get)(button_values)

    # Define variables as needed
    evt_buttons = button_colors
    evt_times = raw.times[button_evt_idx]

    return evt_buttons, evt_times


def find_nearest(array, values):
    """
    Find the nearest element in an array to a given value

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    values: int, float, list or 1D array
        If int or float, use that value to find neares elemen in array and return index and element as int and array.dtype
        If list or array, iterate over values and return arrays of indexes and elements nearest to each value.

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    element: int float
        The nearest element to the specified value
    """
    array = np.asarray(array)

    if isinstance(values, float) or isinstance(values, int):
        idx = (np.abs(array - values)).argmin()
        return idx, array[idx]

    elif len(values):
        idxs = []
        elements = []
        for value in values:
            idx = (np.abs(array - value)).argmin()
            idxs.append(idx)
            elements.append(array[idx])
        return np.asarray(idxs), np.asarray(elements)


def find_previous(array, value):
    """
    Find the nearest element in an array to a given value

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    """

    array = np.asarray(array)
    idx = np.max(np.where(array - value <= 0)[0])
    return idx, array[idx]



def find_first_within(array, low_bound, up_bound):
    """
    Find the first element from an array in a certain interval

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    low_bound: float
        the lower boundary of the search interval
    up_bound: float
        the upper boundary of the search interval

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    value: float
      The value of the array in the found index.
    """

    array = np.asarray(array)
    elements = np.where(np.logical_and((array > low_bound), (array < up_bound)))[0]
    try:
        idx = np.min(elements)
        return idx, array[idx]
    except:
        return False, False


def find_last_within(array, low_bound, up_bound):
    """
    Find the first element from an array in a certain interval

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    low_bound: float
        the lower boundary of the search interval
    up_bound: float
        the upper boundary of the search interval

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    value: float
      The value of the array in the found index.
    """

    array = np.asarray(array)
    elements = np.where(np.logical_and((array > low_bound), (array < up_bound)))[0]
    try:
        idx = np.max(elements)
        return idx, array[idx]
    except:
        return False, False



def find_all_within(array, low_bound, up_bound):
    """
    Find the first element from an array in a certain interval

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the values.
    low_bound: float
        the lower boundary of the search interval
    up_bound: float
        the upper boundary of the search interval

    Returns
    -------
    idx: int
      The index of the elements in the array that are cotained by the given bounds.
    value: float
      The values of the array in the found indexes.
    """

    array = np.asarray(array)
    elements = np.where(np.logical_and((array > low_bound), (array < up_bound)))[0]
    try:
        idx = np.max(elements)
        return idx, array[idx]
    except:
        return False, False


def first_trial(evt_buttons):
    """
    Get event corresponding to last green button press before 1st trial begins

    Parameters
    ----------
    evt_buttons: ndarray
        The 1D array containing the responses from the MEG data.

    Returns
    -------
    first_trial: int
      The number of trial corresponding to the last green button press before the 1st trial begins.
    """

    for i, button in enumerate(evt_buttons):
        if button != 'green':
            return i


def flatten_list(ls):
    flat_list = [element for sublist in ls for element in sublist]
    return flat_list


def align_signals(signal_1, signal_2):
    '''
    Find samples shift that aligns two matching signals by Pearson correlation.

    Parameters
    ----------
    signal_1: ndarray
        1D array signal. Must be longer than signal_2, if not, the samples shift will be referenced to signal_2.

    signal_2: ndarray
        1D array signal.

    Returns
    -------
    max_sample: int
      Sample shift of maximum correlation between signals.
    '''

    invert = False

    start_samples = len(signal_1) - len(signal_2)
    # invert signals to return samples shift referenced to the longer signal
    if start_samples < 0:
        print('Signal_2 is longer. Inverting reference.')
        save_signal = signal_1
        signal_1 = signal_2
        signal_2 = save_signal
        invert = True

    corrs = []
    for i in range(start_samples):
        print("\rProgress: {}%".format(int((i + 1) * 100 / start_samples)), end='')
        df = pd.DataFrame({'x1': signal_1[i:i + len(signal_2)], 'x2': signal_2})
        corrs.append(df.corr()['x1']['x2'])
        # if df.corr()['x1']['x2'] > 0.5 and all(np.diff(corrs[-50:]) < 0): # Original parameters
        if any(np.array(corrs) > 0.9) and all(np.diff(corrs[-100:]) < 0):
            print(f'\nMaximal correlation sample shift found in sample {i}')
            break
    max_sample = np.argmax(corrs)
    print(f'Maximum correlation of {np.max(corrs)}')

    if invert:
        max_sample = -max_sample

    return max_sample, corrs


def ch_name_map(orig_ch_name):
    if orig_ch_name[-5:] == '-4123':
        new_ch_name = orig_ch_name[:-5]
    else:
        new_ch_name = orig_ch_name
    return new_ch_name


def pick_chs(chs_id, info):
    '''
    :param chs_id: 'mag'/'LR'/'parietal/occipital/'frontal'/sac_chs/parietal+'
        String identifying the channels to pick.
    :param info: class attribute
        info attribute from the evoked data.
    :return: picks: list
        List of chosen channel names.
    '''

    if chs_id == 'mag':
        picks = 'mag'
    elif chs_id == 'sac_chs':
        picks = ['MLF14', 'MLF13', 'MLF12', 'MLF11', 'MRF11', 'MRF12', 'MRF13', 'MRF14', 'MZF01']
    elif chs_id == 'LR':
        right_chs = ['MRT51', 'MRT52', 'MRT53']
        left_chs = ['MLT51', 'MLT52', 'MLT53']
        picks = right_chs + left_chs

    else:
        ids = chs_id.split('_')
        all_chs = info.ch_names
        picks = []
        for id in ids:
            if id == 'parietal':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'P' in ch_name]
            elif id == 'parietal+':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'P' in ch_name]
                picks += ['MLT25', 'MLT26', 'MLT27', 'MLO24', 'MLO23', 'MLO22', 'MLO21', 'MLT15', 'MLT16',
                                 'MLO14', 'MLO13', 'MLO12', 'MLO11',
                                 'MZO01',
                                 'MRT25', 'MRT26', 'MRT27', 'MRO24', 'MRO23', 'MRO22', 'MRO21', 'MRT15', 'MRT16',
                                 'MRO14', 'MRO13', 'MRO12', 'MRO11']
            elif id == 'occipital':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'O' in ch_name]
            elif id == 'frontal':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'F' in ch_name]
            elif id == 'temporal':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'T' in ch_name]
            elif id == 'central':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'C' in ch_name]

            # Subset from picked chanels
            elif id == 'L':
                picks = [ch_name for ch_name in picks if 'M' in ch_name and 'L' in ch_name]
            elif id == 'R':
                picks = [ch_name for ch_name in picks if 'M' in ch_name and 'R' in ch_name]

    return picks


def get_freq_band(band_id):
    '''
    :param band_id: str ('Delta/Theta/Alpha/Beta/Gamma
        String determining the frequency bands to get.

    :return: l_freq: int
        Lower edge of frequency band.
    :return: h_freq: int
        High edge of frequency band.
    '''
    if type(band_id) == str:

        # Get multiple frequency bands
        bands = band_id.split('_')
        l_freqs = []
        h_freqs = []

        for band in bands:
            if band == 'Delta':
                l_freq = 1
                h_freq = 4
            elif band == 'Theta':
                l_freq = 4
                h_freq = 8
            elif band == 'Alpha':
                l_freq = 8
                h_freq = 12
            elif band == 'Beta':
                l_freq = 12
                h_freq = 30
            elif band == 'Gamma':
                l_freq = 30
                h_freq = 45
            elif band == 'HGamma':
                l_freq = 45
                h_freq = 100
            elif band == 'Broad':
                l_freq = 0.5
                h_freq = 100

            l_freqs.append(l_freq)
            h_freqs.append(h_freq)

        l_freq = np.min(l_freqs)
        h_freq = np.max(h_freqs)

    elif type(band_id) == tuple:
        l_freq = band_id[0]
        h_freq = band_id[1]

    elif band_id == None:

        l_freq = None
        h_freq = None

    return l_freq, h_freq


def get_time_lims(epoch_id, mss=None, plot_edge=0.1, map=None):
    '''
    :param epoch_id: str
        String with the name of the epochs to select.
    :param map: dict
        Dictionary of dictionaries indicating the times associated to each type of epoch id.
        Keys should be 'fix', 'sac', and within those keys, a dictionary with keys 'tmin', 'tmax', 'plot_xlim' with their corresponding values.

    :return: tmin: float
        time corresponding to time start of the epochs.
    :return: tmax: float
        time corresponding to time end of the epochs.
    :return: plot_xlim: tuple of float
        time start and end to plot.

    '''
    if map and epoch_id in map.keys():
        tmin = map[epoch_id]['tmin']
        tmax = map[epoch_id]['tmax']
        plot_xlim = map[epoch_id]['plot_xlim']

    else:
        try:
            dur, cross1_dur, cross2_dur, mss_duration, vs_dur = get_duration(epoch_id=epoch_id, mss=mss)
            print('Using default time values')
            map = dict(ms={'tmin': -cross1_dur, 'tmax': dur, 'plot_xlim': (-cross1_dur + plot_edge, dur - plot_edge)},
                       cross2={'tmin': -cross1_dur - mss_duration[mss], 'tmax': dur,
                               'plot_xlim': (-cross1_dur - mss_duration[mss] + plot_edge, dur - plot_edge)},
                       vs={'tmin': -cross1_dur - mss_duration[mss] - cross2_dur, 'tmax': dur,
                           'plot_xlim': (-cross1_dur - mss_duration[mss] - cross2_dur + plot_edge, dur - plot_edge)},
                       sac={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)},
                       fix={'tmin': -0.2, 'tmax': 0.3, 'plot_xlim': (-0.1, 0.25)})
            if 'fix' in epoch_id:
                tmin = map['fix']['tmin']
                tmax = map['fix']['tmax']
                plot_xlim = map['fix']['plot_xlim']
            elif 'sac' in epoch_id:
                tmin = map['sac']['tmin']
                tmax = map['sac']['tmax']
                plot_xlim = map['sac']['plot_xlim']
            else:
                tmin = map[epoch_id]['tmin']
                tmax = map[epoch_id]['tmax']
                plot_xlim = map[epoch_id]['plot_xlim']
        except:
            raise ValueError('Epoch id not in default map keys.')

    return tmin, tmax, plot_xlim


def get_duration(epoch_id,  mss, vs_dur=None):
    mss_duration = {1: 2, 2: 3.5, 4: 5, None: 2}
    if not vs_dur:
        vs_dur = {1: (2, 9.8), 2: (3, 9.8), 4: (3.5, 9.8), None: (2, 9.8)}
    cross2_dur = 1
    cross1_dur = 0.75

    # Duration
    if 'ms' in epoch_id:
        dur = mss_duration[mss] + cross2_dur + vs_dur[mss][0]
    elif 'cross2' in epoch_id:
        dur = cross2_dur + vs_dur[mss][-0]  # seconds
    elif 'vs' in epoch_id:
        dur = vs_dur[mss][0]  # seconds
    else:
        dur = 0

    return dur, cross1_dur, cross2_dur, mss_duration, vs_dur


def get_baseline_duration(epoch_id, mss, tmin, tmax, plot_xlim, cross1_dur, mss_duration, cross2_dur, map=None):
    # Baseline duration
    if map and epoch_id in map.keys():
        baseline = (map[epoch_id]['baseline'][0], map[epoch_id]['baseline'][1])
        plot_baseline = (map[epoch_id]['plot_baseline'][0], map[epoch_id]['plot_baseline'][1])

    elif 'sac' in epoch_id:
        baseline = (tmin, 0)
        plot_baseline = (plot_xlim[0], 0)
        # baseline = None
    elif 'fix' in epoch_id or 'fix' in epoch_id:
        baseline = (tmin, -0.05)
        plot_baseline = (plot_xlim[0], 0)
    elif 'ms' in epoch_id:
        baseline = (-cross1_dur, 0)
        plot_baseline = baseline
    elif 'cross2' in epoch_id and mss:
        baseline = (-mss_duration[mss] - cross2_dur, -mss_duration[mss])
        plot_baseline = baseline
    elif 'vs' in epoch_id and mss:
        baseline = (-cross1_dur -cross2_dur - mss_duration[mss], -cross2_dur - mss_duration[mss])
        plot_baseline = baseline
    else:
        tmax = tmin + cross1_dur
        if tmax.is_integer():
            tmax = int(tmax)
        baseline = (tmin, tmax)
        plot_baseline = baseline

    if baseline[0] < tmin:
        baseline = (tmin, baseline[1])
    if baseline[1] > tmax:
        baseline = (baseline[1], tmax)
    if baseline[0] > baseline[1]:
        print('Baseline start is greater than end. Setting to (0, 0)')
        baseline = (0, 0)

    if plot_baseline[0] < tmin:
        plot_baseline = (tmin, plot_baseline[1])
    if plot_baseline[1] > tmax:
        plot_baseline = (plot_baseline[0], tmax)
    if plot_baseline[0] > plot_baseline[1]:
        print('Plot_baseline start is greater than end. Setting to (0, 0)')
        plot_baseline = (0, 0)

    return baseline, plot_baseline


def get_plots_timefreqs(epoch_id, mss, cross2_dur, mss_duration, topo_bands, plot_xlim, timefreqs_joint=None):
    '''
    :param epoch_id:
    :param timefreqs_joint: list of tuples. Each tuple represents the time and frequency of the topoplot.
    :param mss:
    :param cross2_dur:
    :param mss_duration:
    :param topo_bands:
    :return:
    '''
    # Plot_joint topoplots time frequencies
    if epoch_id == 'ms':
        if not timefreqs_joint:
            timefreqs_joint = [(0.55, 10)]

            if mss:
                vs_timefreq = {1: [(2.5, 10), (3.15, 7), (3.75, 10)],
                           2: [(4, 10), (4.65, 7), (5.25, 10)],
                           4: [(5.5, 10), (6.15, 7), (6.75, 10)]}

                timefreqs_joint += vs_timefreq[mss]

            # Check that time freqs are contained in plot times
            timefreqs_joint = [timefreq for timefreq in timefreqs_joint if timefreq[0] > plot_xlim[0] and timefreq[0] < plot_xlim[1]]

        timefreqs_tfr = {}
        for i, time_freq in enumerate(timefreqs_joint):
            # Get plot time from time defined for plot_joint
            time = time_freq[0]
            # Get frequencies from frequency id previously defined
            fmin, fmax = get_freq_band(band_id=topo_bands[i])
            timefreqs_tfr[f'topo{i}'] = dict(title=topo_bands[i], tmin=time, tmax=time, fmin=fmin, fmax=fmax)

        #TFR vlines
        vlines_times = [0, mss_duration[mss], mss_duration[mss] + 1]

    elif epoch_id == 'vs':
        vs_timefreqs = [(-0.7, 8), (0.15, 6), (0.75, 10)]
        if not timefreqs_joint:
            if mss:
                ms_timefreq = {1: [(-2.45, 10)],
                               2: [(-3.95, 10)],
                               4: [(-5.45, 10)]}
                timefreqs_joint = ms_timefreq[mss] + vs_timefreqs
            else:
                timefreqs_joint = vs_timefreqs

            # Check that time freqs are contained in plot times
            timefreqs_joint = [timefreq for timefreq in timefreqs_joint if timefreq[0] > plot_xlim[0] and timefreq[0] < plot_xlim[1]]

        timefreqs_tfr = {}
        for i, time_freq in enumerate(timefreqs_joint):
            # Get plot time from time defined for plot_joint
            time = time_freq[0]
            # Get frequencies from frequency id previously defined
            fmin, fmax = get_freq_band(band_id=topo_bands[i])
            timefreqs_tfr[f'topo{i}'] = dict(title=topo_bands[i], tmin=time, tmax=time, fmin=fmin, fmax=fmax)

        # TFR vlines
        vlines_times = [- cross2_dur - mss_duration[mss], -cross2_dur, 0]

    elif 'fix' in epoch_id:
        timefreqs_joint = [(0.095, 10)]
        vlines_times = None
        timefreqs_tfr = timefreqs_joint
    else:
        timefreqs_joint = None
        vlines_times = None
        timefreqs_tfr = timefreqs_joint

    return timefreqs_joint, timefreqs_tfr, vlines_times


def get_item(epoch_id):

    if 'tgt' in epoch_id:  # 1 for target, 0 for item, None for none
        tgt = 1
    elif 'it' in epoch_id:
        tgt = 0
    else:
        tgt = None

    return tgt


def get_dir(epoch_id):

    if '_sac' in epoch_id:
        dir = epoch_id.split('_sac')[0]
    else:
        dir = None

    return dir


def get_screen(epoch_id):

    screens = ['emap', 'cross1', 'ms', 'vs', 'cross2']
    screen = epoch_id.split('_')[-1]

    if screen not in screens:
        screen = None

    return screen


def get_mss(epoch_id):

    if 'mss' in epoch_id:
        mss = epoch_id.split('mss')[-1][0]
    else:
        mss = None

    return mss


def get_condition_trials(subject, mss=None, trial_dur=None, corr_ans=None, tgt_pres=None):
    '''

    :param subject:
    :param mss:
    :param trial_dur: tuple. Minimum and maximun duration of visual search.
    :param corr_ans:
    :param tgt_pres:
    :return:
    '''
    bh_data = subject.bh_data
    if corr_ans:
        bh_data = bh_data.loc[subject.corr_ans == 1]
    elif corr_ans == False:
        bh_data = bh_data.loc[subject.corr_ans == 0]
    if mss:
        bh_data = bh_data.loc[bh_data['Nstim'] == mss]
    if trial_dur:
        rt = subject.rt
        good_trials = np.where((rt > trial_dur[0]) & (rt < trial_dur[1]))[0]
        matching_trials = list(set(good_trials) & set(bh_data.index))
        matching_trials.sort()
        bh_data = bh_data.loc[matching_trials]
    if tgt_pres:
        bh_data = bh_data.loc[bh_data['Tpres'] == 1]
    elif tgt_pres == False:
        bh_data = bh_data.loc[bh_data['Tpres'] == 0]

    trials = list((bh_data.index + 1).astype(str))  # +1 for 0th index

    return trials, bh_data