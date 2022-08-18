import matplotlib.pyplot as plt
import numpy as np


def get_intervals_signals(reference_signal, signal_to_scale, fig=None):
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
    fig: instance of figure, default None
        Figure to use for the plots. if None, figure is created.

    Returns
    -------
    axs0_start, axs0_end, axs1_start, axs1_end: int
      The axis start and end samples.
    """

    if fig == None:
        fig, axs = plt.subplots(2, 1)
    else:
        plt.close(fig)
        fig, axs = plt.subplots(2, 1)

    axs[0].plot(reference_signal)
    axs[0].set_title('EDF')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Pixels')
    axs[1].plot(signal_to_scale)
    axs[1].set_title('MEG')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Volts [\mu V]')
    fig.tight_layout()
    plt.pause(0.5)

    # Time to move around the plot to match the signals for scaling
    print('\nPlease arange the plots to matching parts of the signals. When ready, press Enter')
    while not plt.waitforbuttonpress():
        pass

    # Get plot limits for scaling signals in those ranges
    axs0_interval = [int(lim) for lim in axs[0].get_xlim()]
    axs1_interval = [int(lim) for lim in axs[1].get_xlim()]

    return fig, axs0_interval, axs1_interval


def scaled_signals(time, scaled_signals, reference_signals, interval_signal=None, interval_ref=None,
                   ref_offset=[0, 5500, 0], signal_offset=[0, 5500*1.2, 0], ylabels=['Gaze x', 'Gaze y', 'Pupil size'],
                   fig=None):
    """
    Plot scaled signals in selected interval into one plot for comparison and check scaling.

    Parameters
    ----------
    time: ndarray
        1D array of time for plot
    scaled_signal: list
        list of 1D scaled signals.
    reference_signal: list
        List of 1D reference signals with proper original scaling.
    interval_signal: {'list', 'tuple'}, default None
        The signal interval to use for scaling. if None, the whole signal is used. Default to None.
    interval_ref: {'list', 'tuple'}, default None
        The scaled signal interval to use for scaling. if None, the whole signal is used. Default to None.
    ref_offset: list, default [0, 5500, 0]
        List of offsets for the reference signal. Usually it is going to be 0 for Gaze x and Pupils size and an integer
        fot Gaze y, accounting for the offset between x and y Eyemaps.
    signal_offset: list, default [0, int(5500 * 1.2), 0]
        List of offsets for the scaled signal. Usually it is going to be 0 for Gaze x and Pupils size and an integer
        fot Gaze y, accounting for the offset between x and y Eyemaps.It differs from ref offset in the fact that
        signals might have different sampling rates.
    ylabels: list, default ['Gaze x', 'Gaze y', 'Pupil size']
        List of ylables to use in each subplot.
    fig: instance of figure, default None
        Figure to use for the plots. if None, figure is created.

    Returns
    ----------
    fig: instance of matplotlib figure
        The resulting figure
    """

    # Check inputs
    if len(scaled_signals) == len(reference_signals) == len(ref_offset) == len(signal_offset) == len(ylabels):
        num_subplots = len(scaled_signals)
    # If scaled and reference signals match in length, raise warning on the rest of the arguments
    elif len(scaled_signals) == len(reference_signals):
        num_subplots = len(scaled_signals)
        print(f'Lists: ref_offset, signal_offset, ylabels should have the same size, but have sizes:'
              f' {len(ref_offset)}, {len(signal_offset)}, {len(ylabels)}.\n'
              f'Using default values.')
        ref_offset = [0, 5500, 0][:num_subplots]
        signal_offset = [0, int(5500 * 1.2), 0][:num_subplots]
        ylabels = ['Gaze x', 'Gaze y', 'Pupil size'][:num_subplots]
    # If scaled and reference signals do not match in length, raise error
    else:
        raise ValueError(f'Lists: scaled_signal, reference_signal must have the same size, but have sizes: '
                         f'{len(scaled_signals)}, {len(reference_signals)}')

    # Make intervals to list because of indexing further ahead
    if not interval_signal:
        interval_signal = [None, None]
    if not interval_ref:
        interval_ref = [None, None]

    # If figure not provided, create instance of figure
    if not fig:
        fig, axs = plt.subplots(num_subplots, 1)
    else:
        plt.close(fig)
        fig, axs = plt.subplots(num_subplots, 1)

    # Set plot title
    plt.suptitle('Scaled and reference signals')
    # Iterate over signals ploting separately in subplots.
    for i, ax in enumerate(fig.axes):
        ax.plot(np.linspace(time[interval_ref[0]+ref_offset[i]]/1000, time[interval_ref[1]+ref_offset[i]]/1000,
                             interval_signal[1] - interval_signal[0]),
                 scaled_signals[i][interval_signal[0]+signal_offset[i]:interval_signal[1]+signal_offset[i]],
                 label='MEG')
        ax.plot(time[interval_ref[0]+ref_offset[i]:interval_ref[1]+ref_offset[i]]/1000,
                 reference_signals[i][interval_ref[0]+ref_offset[i]:interval_ref[1]+ref_offset[i]],
                 label='EDF')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabels[i])
        if i==0:
            ax.legend(loc='upper right')

    fig.tight_layout()
    plt.pause(0.5)

    return fig