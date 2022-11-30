import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    elif chs_id == 'parietal':
        ch_names = info.ch_names
        picks = [ch_name for ch_name in ch_names if 'M' in ch_name and 'P' in ch_name]
    elif chs_id == 'occipital':
        ch_names = info.ch_names
        picks = [ch_name for ch_name in ch_names if 'M' in ch_name and 'O' in ch_name]
    elif chs_id == 'frontal':
        ch_names = info.ch_names
        picks = [ch_name for ch_name in ch_names if 'M' in ch_name and 'F' in ch_name]
    elif chs_id == 'sac_chs':
        picks = ['MLF14', 'MLF13', 'MLF12', 'MLF11', 'MRF11', 'MRF12', 'MRF13', 'MRF14', 'MZF01']
    elif chs_id == 'LR':
        right_chs = ['MRT51', 'MRT52', 'MRT53']
        left_chs = ['MLT51', 'MLT52', 'MLT53']
        picks = right_chs + left_chs

    return picks