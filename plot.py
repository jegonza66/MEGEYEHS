import matplotlib.pyplot as plt
import numpy as np


def get_intervals_signals(reference_signal, signal_to_scale):
    """
    Get the intervals of interest for scaling signals.
    Plot the two signals to scale in 2 subplots and interactively zoom in/out to the matching
    regions of interest of each signal to get the corresponding intervals.
    When ready, press Enter to continue.

    Parameters
    ----------
    reference_signal: ndarray
        The 1D reference signal with proper scaling.
    signal_to_scale: ndarray
        The 1D signal you wish to re-scale.

    Returns
    -------
    fig, axs:
      The figure and axis instances from which you can get the intervals of interest for scaling.
    """

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(reference_signal)
    axs[0].set_title('EDF')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Pixels')
    axs[1].plot(signal_to_scale)
    axs[1].set_title('MEG')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Pixels')
    fig.tight_layout()

    # Time to move around the plot to match the signals for scaling
    print(f'\rPlease arange the plots to matching parts of the signals. When ready, press Enter')
    while not plt.waitforbuttonpress():
        pass

    return fig, axs


def scaled_et(time, scaled_signal, reference_signal, title=None, xlabel=None, interval_signal=None, interval_ref=None):
    """
    Plot scaled signals in selected interval into one plot for comparisson.

    Parameters
    ----------
    time: ndarray
        1D array of time for plot
    scaled_signal: ndarray
        The 1D scaled signal.
    reference_signal: ndarray
        The 1D reference signal with proper original scaling.
    title: str, optional
        The plot title. Default None
    xlabel: str, optional
        The label for the x axis. Default None
    interval_signal: {'list', 'tuple'}, optional
        The signal interval to use for scaling. if None, the whole signal is used. Default to None.
    interval_ref: {'list', 'tuple'}, optional
        The scaled signal interval to use for scaling. if None, the whole signal is used. Default to None.

    """
    # Plot scaled signals
    plt.figure()
    plt.suptitle(f'{title}')
    plt.plot(np.linspace(time[interval_ref[0]]/1000, time[interval_ref[1]]/1000,
                         interval_signal[1] - interval_signal[0]),
             scaled_signal[interval_signal[0]:interval_signal[1]],
             label='MEG')
    plt.plot(time[interval_ref[0]:interval_ref[1]]/1000,
             reference_signal[interval_ref[0]:interval_ref[1]],
             label='EDF')
    plt.xlabel('Time [s]')
    plt.ylabel(xlabel)
    plt.legend()
    plt.tight_layout()
    plt.pause(0.5)