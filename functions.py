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
    interval_scaled: {'list', 'tuple'}, optional
        The scaled signal interval to use for scaling. if None, the whole signal is used. Default to None.

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



