import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib as mpl
import os
import seaborn as sn

import functions
from paths import paths

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


def scanpath(fixations_vs, items_pos, bh_data, raw, gazex, gazey, subject, trial,
             screen_res_x=1920, screen_res_y=1080, img_res_x=1280, img_res_y=1024, display_fig=False, save=True):
    plt.clf()
    plt.close('all')

    # Path to psychopy data
    exp_path = paths().experiment_path()

    # Get trial
    fixations_t = fixations_vs.loc[fixations_vs['trial'] == trial]
    item_pos_t = items_pos.loc[items_pos['folder'] == fixations_t['trial_image'].values[0]]

    # Get vs from trial
    vs_start_idx = functions.find_nearest(raw.times, raw.annotations.vs[np.where(raw.annotations.trial == trial)[0]])[0]
    vs_end_idx = functions.find_nearest(raw.times, raw.annotations.onset[np.where(raw.annotations.trial == trial)[0]])[0]

    # Load search image
    img = mpimg.imread(exp_path + 'cmp_' + fixations_t['trial_image'].values[0] + '.jpg')

    # Load targets
    bh_data_trial = bh_data.loc[bh_data['searchimage'] == 'cmp_' + fixations_t['trial_image'].values[0] + '.jpg']
    target_keys = ['st1', 'st2', 'st3', 'st4', 'st5']
    targets = bh_data_trial[target_keys].values[0]
    st1 = mpimg.imread(exp_path + targets[0])
    st2 = mpimg.imread(exp_path + targets[1])
    st3 = mpimg.imread(exp_path + targets[2])
    st4 = mpimg.imread(exp_path + targets[3])
    st5 = mpimg.imread(exp_path + targets[4])

    # Load correct vs incorrect
    correct_ans = bh_data_trial['key_resp.corr'].values

    # Colormap: Get fixation durations for scatter circle size
    sizes = fixations_t['duration'] * 100
    # Define rainwbow cmap for fixations
    cmap = plt.cm.rainbow
    # define the bins and normalize
    fix_num = fixations_t['n_fix'].values.astype(int)
    bounds = np.linspace(1, fix_num[-1]+1, fix_num[-1]+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Display image True or False
    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    plt.figure(figsize=(10, 9))
    plt.suptitle(f'Subject {subject.subject_id} - Trial {trial}')

    # Items axes
    ax1 = plt.subplot2grid((5, 5), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((5, 5), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((5, 5), (0, 2), colspan=1)
    ax4 = plt.subplot2grid((5, 5), (0, 3), colspan=1)
    ax5 = plt.subplot2grid((5, 5), (0, 4), colspan=1)

    # Image axis
    ax6 = plt.subplot2grid((5, 5), (1, 0), colspan=5, rowspan=3)

    # Remove ticks from items and image axes
    for ax in plt.gcf().get_axes():
        ax.set_xticks([])
        ax.set_yticks([])

    # Gaze axis
    ax7 = plt.subplot2grid((5, 5), (4, 0), colspan=5)

    # Targets
    for ax, st in zip([ax1, ax2, ax3, ax4, ax5], [st1, st2, st3, st4, st5]):
        ax.imshow(st)

    # Colour
    if correct_ans:
        for spine in ax5.spines.values():
            spine.set_color('green')
            spine.set_linewidth(3)
    else:
        for spine in ax5.spines.values():
            spine.set_color('red')
            spine.set_linewidth(3)

    # Fixations
    ax6.scatter(fixations_t['start_x'] - (screen_res_x - img_res_x) / 2,
                fixations_t['start_y'] - (screen_res_y - img_res_y) / 2,
                c=fix_num, s=sizes, cmap=cmap, norm=norm, zorder=3)

    # Image
    ax6.imshow(img, zorder=0)

    # Items circles
    ax6.scatter(item_pos_t['center_x'], item_pos_t['center_y'], s=1000, color='grey', alpha=0.5, zorder=1)
    target = item_pos_t.loc[item_pos_t['istarget'] == 1]

    # Target green/red
    if len(target):
        if correct_ans:
            ax6.scatter(target['center_x'], target['center_y'], s=1000, color='green', alpha=0.3, zorder=1)
        else:
            ax6.scatter(target['center_x'], target['center_y'], s=1000, color='red', alpha=0.3, zorder=1)

    # Scanpath
    ax6.plot(gazex[vs_start_idx:vs_end_idx] - (1920 - 1280) / 2,
             gazey[vs_start_idx:vs_end_idx] - (1080 - 1024) / 2,
             '--', color='black', zorder=2)


    PCM = ax6.get_children()[0]  # When the fixations dots for color mappable were ploted (first)
    cb = plt.colorbar(PCM, ax=ax6, ticks=[fix_num[0] + 1/2, fix_num[int(len(fix_num)/2)]+1/2, fix_num[-1]+1/2])
    cb.ax.set_yticklabels([fix_num[0], fix_num[int(len(fix_num)/2)], fix_num[-1]])
    cb.ax.tick_params(labelsize=10)
    cb.set_label('# of fixation', fontsize=13)

    # Gaze
    ax7.plot(raw.times[vs_start_idx:vs_end_idx], gazex[vs_start_idx:vs_end_idx], label='X')
    ax7.plot(raw.times[vs_start_idx:vs_end_idx], gazey[vs_start_idx:vs_end_idx], 'black', label='Y')
    ax7.legend(fontsize=8)
    ax7.set_ylabel('Gaze')
    ax7.set_xlabel('Time [s]')

    if save:
        save_path = paths().plots_path() + f'Scanpaths/{subject.subject_id}/'
        os.makedirs(save_path + 'svg/', exist_ok=True)
        plt.savefig(save_path + f'Trial{trial}.png')
        plt.savefig(save_path + f'svg/Trial{trial}.svg')



def trial_gaze(raw, bh_data, gazex, gazey, subject, trial, display_fig=False, save=True):
    plt.clf()
    plt.close('all')

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    # get trial info from bh data
    trial_idx_bh = trial - 1
    pres_abs_trial = 'Present' if bh_data['Tpres'].astype(int)[trial_idx_bh] == 1 else 'Absent'
    correct_trial = 'Correct' if bh_data['key_resp.corr'].astype(int)[trial_idx_bh] == 1 else 'Incorrect'
    mss = bh_data['Nstim'][trial_idx_bh]

    # Get trial start and end samples
    trial_idx_annot = np.where(raw.annotations.trial == trial)[0]
    trial_start_idx = \
    functions.find_nearest(raw.times, raw.annotations.fix1[trial_idx_annot])[0] - 120 * 2
    trial_end_idx = functions.find_nearest(raw.times, raw.annotations.onset[trial_idx_annot])[0] + 120 * 6

    # Plot
    plt.figure(figsize=(15, 5))
    plt.title(f'Trial {trial} - {pres_abs_trial} - {correct_trial} - MSS: {int(mss)}')

    # Gazes
    plt.plot(raw.times[trial_start_idx:trial_end_idx], gazex[trial_start_idx:trial_end_idx], label='X')
    plt.plot(raw.times[trial_start_idx:trial_end_idx], gazey[trial_start_idx:trial_end_idx] - 1000, 'black', label='Y')

    # Screens
    plt.axvspan(ymin=0, ymax=1, xmin=raw.annotations.fix1[trial_idx_annot][0], xmax=raw.annotations.ms[trial_idx_annot][0], color='grey',
                alpha=0.4, label='Fix')
    plt.axvspan(ymin=0, ymax=1, xmin=raw.annotations.ms[trial_idx_annot][0], xmax=raw.annotations.fix2[trial_idx_annot][0], color='red',
                alpha=0.4, label='MS')
    plt.axvspan(ymin=0, ymax=1, xmin=raw.annotations.fix2[trial_idx_annot][0], xmax=raw.annotations.vs[trial_idx_annot][0], color='grey',
                alpha=0.4, label='Fix')
    plt.axvspan(ymin=0, ymax=1, xmin=raw.annotations.vs[trial_idx_annot][0], xmax=raw.annotations.onset[trial_idx_annot][0], color='green',
                alpha=0.4, label='VS')

    plt.xlabel('time [s]')
    plt.ylabel('Gaze')
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    if save:
        save_path = paths().plots_path() + f'Gaze_Trials/{subject.subject_id}/'
        os.makedirs(save_path + 'svg/', exist_ok=True)
        plt.savefig(save_path + f'Trial {trial}.png')
        plt.savefig(save_path + f'svg/Trial {trial}.svg')


def first_fixation_delay(fixations, subject, display_fig=True, save=True):

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    fixations1_fix_screen = fixations.loc[(fixations['screen'].isin(['fix1', 'fix2'])) & (fixations['n_fix'] == 1)]
    plt.figure()
    plt.hist(fixations1_fix_screen['time'], bins=40)
    plt.title('1st fixation delay distribution')
    plt.xlabel('Time [s]')

    if save:
        save_path = paths().plots_path()
        save_path += '1st fixation/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f'{subject.subject_id} 1st fix delay dist.png')


def pupil_size_increase(fixations, response_trials_meg, subject, display_fig=True, save=True):

    if display_fig:
        plt.ion()
    else:
        plt.ioff()

    fixations_pupil_s = fixations.loc[(fixations['screen'].isin(['fix1', 'ms', 'fix2'])) & (fixations['n_fix'] == 1)]

    pupil_diffs = []
    mss = []
    for trial in response_trials_meg:
        trial_data = fixations_pupil_s.loc[fixations_pupil_s['trial'] == trial]

        try:
            if 'fix1' in trial_data['screen'].values:
                pupil_diff = trial_data[trial_data['screen'] == 'fix2']['pupil'].values[0] - trial_data[trial_data['screen'] == 'fix1']['pupil'].values[0]
            else:
                pupil_diff = trial_data[trial_data['screen'] == 'fix2']['pupil'].values[0] - \
                             trial_data[trial_data['screen'] == 'ms']['pupil'].values[0]
            pupil_diffs.append(pupil_diff)
            mss.append(trial_data['mss'].values[0])
        except:
            print(f'No fix or mss data in trial {trial}')

    plt.figure()
    sn.boxplot(x=mss, y=pupil_diffs)
    plt.title('Pupil size')
    plt.xlabel('MSS')
    plt.ylabel('Pupil size increase (fix point 2 - 1)')

    if save:
        save_path = paths().plots_path()
        save_path += '1st fixation/'
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f'{subject.subject_id} Pupil size increase.png')