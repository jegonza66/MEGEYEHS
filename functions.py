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


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50):
    """Detects fixations, defined as consecutive samples with an inter-sample
    distance of less than a set amount of pixels (disregarding missing data)

    arguments
    x		-	numpy array of x positions
    y		-	numpy array of y positions
    time		-	numpy array of EyeTribe timestamps
    keyword arguments
    missing	-	value to be used for missing data (default = 0.0)
    maxdist	-	maximal inter sample distance in pixels (default = 25)
    mindur	-	minimal duration of a fixation in milliseconds; detected
                fixation cadidates will be disregarded if they are below
                this duration (default = 50)

    returns
    Sfix, Efix
                Sfix	-	list of lists, each containing [starttime]
                Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
    """

    x, y, time = remove_missing(x, y, time, missing)

    # empty list to contain data
    sfix = []
    efix = []

    # loop through all coordinates
    si = 0
    fixstart = False
    for i in range(1, len(x)):
        # calculate Euclidean distance from the current fixation coordinate
        # to the next coordinate
        squared_distance = ((x[si] - x[i]) ** 2 + (y[si] - y[i]) ** 2)
        dist = 0.0
        if squared_distance > 0:
            dist = squared_distance ** 0.5
        # check if the next coordinate is below maximal distance
        if dist <= maxdist and not fixstart:
            # start a new fixation
            si = 0 + i
            fixstart = True
            sfix.append(time[i])
        elif dist > maxdist and fixstart:
            # end the current fixation
            fixstart = False
            # only store the fixation if the duration is ok
            if time[i - 1] - sfix[-1] >= mindur:
                efix.append([sfix[-1], time[i - 1], time[i - 1] - sfix[-1], x[si], y[si]])
            # delete the last fixation start if it was too short
            else:
                sfix.pop(-1)
            si = 0 + i
        elif not fixstart:
            si += 1
    # add last fixation end (we can lose it if dist > maxdist is false for the last point)
    if len(sfix) > len(efix):
        efix.append([sfix[-1], time[len(x) - 1], time[len(x) - 1] - sfix[-1], x[si], y[si]])
    return sfix, efix
