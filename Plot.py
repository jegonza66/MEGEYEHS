import matplotlib.pyplot as plt
import numpy as np


def signals_to_scale(edf_gazex_data, meg_gazex_data):
    # Plot both signals in different subplots
    fig, axs = plt.subplots(2, 1)
    axs[0].cla()
    axs[1].cla()
    axs[0].plot(edf_gazex_data)
    axs[0].set_title('EDF')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Pixels')
    axs[1].plot(meg_gazex_data)
    axs[1].set_title('MEG')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Pixels')
    fig.tight_layout()

    # Time to move around the plot to match the signals for scaling
    print(f'\rPlease arange the plots to matching parts of the signals. When ready, press Enter')
    while not plt.waitforbuttonpress():
        pass

    return fig, axs


def scaled_et(edf_time, meg_data, edf_data, eyemap_start_edf, eyemap_end_edf, eyemap_start_meg,
              eyemap_end_meg, title, xlabel):
    # Plot scaled signals
    plt.figure()
    plt.suptitle(f'{title}')
    plt.plot(np.linspace(edf_time[eyemap_start_edf]/1000, edf_time[eyemap_end_edf]/1000,
                         eyemap_end_meg - eyemap_start_meg),
             meg_data[eyemap_start_meg:eyemap_end_meg],
             label='MEG')
    plt.plot(edf_time[eyemap_start_edf:eyemap_end_edf]/1000,
             edf_data[eyemap_start_edf:eyemap_end_edf],
             label='EDF')
    plt.xlabel('Time [s]')
    plt.ylabel(xlabel)
    plt.legend()
    plt.tight_layout()
    plt.pause(0.5)