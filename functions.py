import numpy as np


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


def find_nearest(array, value):
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
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


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