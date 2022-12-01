import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from paths import paths
import save

save_path = paths().save_path()
plot_path = paths().plots_path()


def epochs(subject, epochs, picks, order=None, overlay=None, combine='mean', sigma=5, display_figs=False,
           save_fig=None, fig_path=None, fname=None, group_by=None):

    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    fig_ep = epochs.plot_image(picks=picks, order=order, sigma=sigma, cmap='jet', overlay_times=overlay, combine=combine,
                               title=subject.subject_id, show=display_figs)

    # Save figure
    if save_fig:
        if len(fig_ep) == 1:
            fig = fig_ep[0]
            save.fig(fig=fig, path=fig_path, fname=fname)
        else:
            for i in range(len(fig_ep)):
                fig = fig_ep[i]
                group = group_by.keys()[i]
                fname += f'{group}'
                save.fig(fig=fig, path=fig_path, fname=fname)


def evoked(evoked_meg, evoked_misc, picks, plot_gaze=False, fig=None,
           axes=None, plot_xlim=None, display_figs=False, save_fig=True, fig_path=None, fname=None):

    # Sanity check
    if save_fig and (not fname or not fig_path):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if axes:
        plot_gaze = False

        if save_fig and not fig:
            raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if plot_gaze:
        # Get Gaze x ch
        gaze_x_ch_idx = np.where(np.array(evoked_misc.ch_names) == 'ET_gaze_x')[0][0]
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        axs[1].plot(evoked_misc.times, evoked_misc.data[gaze_x_ch_idx, :])
        axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
        axs[1].set_ylabel('Gaze x')
        axs[1].set_xlabel('Time')

        evoked_meg.plot(picks=picks, gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim, show=display_figs)
        axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    else:
        evoked_meg.plot(picks=picks, gfp=True, axes=axes, time_unit='s', spatial_colors=True, xlim=plot_xlim,
                        show=display_figs)
        axes.vlines(x=0, ymin=axes.get_ylim()[0], ymax=axes.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)


def evoked_topo(evoked_meg, picks, topo_times, title=None, fig=None, axes_ev=None, axes_topo=None, xlim=None, ylim=None,
                display_figs=False, save_fig=False, fig_path=None, fname=None):

    # Sanity check
    if save_fig and (not fig_path or not fname):
        raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

    if axes_ev and axes_topo:
        if save_fig and not fig:
            raise ValueError('Please provide path and filename to save figure. Else, set save_fig to false.')

        display_figs = True

        evoked_meg.plot_joint(times=topo_times, title=title, picks=picks, show=display_figs,
                              ts_args={'axes': axes_ev, 'xlim': xlim, 'ylim': ylim},
                              topomap_args={'axes': axes_topo})

        axes_ev.vlines(x=0, ymin=axes_ev.get_ylim()[0], ymax=axes_ev.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)

    else:
        fig = evoked_meg.plot_joint(times=topo_times, title=title, picks=picks, show=display_figs,
                                    ts_args={'xlim': xlim, 'ylim': ylim})

        all_axes = plt.gcf().get_axes()
        axes_ev = all_axes[0]
        axes_ev.vlines(x=0, ymin=axes_ev.get_ylim()[0], ymax=axes_ev.get_ylim()[1], color='grey', linestyles='--')

        if save_fig:
            save.fig(fig=fig, path=fig_path, fname=fname)


def fig_vs_ms():

    fig = plt.figure(figsize=(10, 9))

    # 1st row Topoplots
    # VS
    ax1 = fig.add_axes([0.12, 0.88, 0.08, 0.09])
    ax2 = fig.add_axes([0.3, 0.88, 0.08, 0.09])
    ax3 = fig.add_axes([0.42, 0.88, 0.013, 0.09])

    #MS
    ax4 = fig.add_axes([0.7, 0.88, 0.08, 0.09])
    ax5 = fig.add_axes([0.82, 0.88, 0.013, 0.09])

    # 2nd row Evokeds
    ax6 = fig.add_axes([0.07, 0.71, 0.4, 0.15])
    ax7 = fig.add_axes([0.57, 0.71, 0.4, 0.15])

    # 3 row Topoplots
    # VS
    ax8 = fig.add_axes([0.2, 0.54, 0.08, 0.09])
    ax9 = fig.add_axes([0.32, 0.54, 0.013, 0.09])
    # MS
    ax10 = fig.add_axes([0.7, 0.54, 0.08, 0.09])
    ax11 = fig.add_axes([0.82, 0.54, 0.013, 0.09])

    # 4 row Evokeds
    ax12 = fig.add_axes([0.07, 0.38, 0.4, 0.15])
    ax13 = fig.add_axes([0.57, 0.38, 0.4, 0.15])

    # 5 row Topoplot Difference
    ax14 = fig.add_axes([0.4, 0.22, 0.08, 0.09])
    ax15 = fig.add_axes([0.52, 0.22, 0.013, 0.09])

    # 6th row Evoked Diference
    ax16 = fig.add_axes([0.1, 0.05, 0.8, 0.15])

    # groups
    ax_evoked_vs_1 = ax6
    ax_topo_vs_1 = [ax1, ax2, ax3]

    ax_evoked_ms_1 = ax7
    ax_topo_ms_1 = [ax4, ax5]

    ax_evoked_vs_2 = ax12
    ax_topo_vs_2 = [ax8, ax9]

    ax_evoked_ms_2 = ax13
    ax_topo_ms_2 = [ax10, ax11]

    ax_evoked_diff = ax16
    ax_topo_diff = [ax14, ax15]

    return fig, ax_evoked_vs_1, ax_topo_vs_1, ax_evoked_ms_1, ax_topo_ms_1, ax_evoked_vs_2, ax_topo_vs_2, \
           ax_evoked_ms_2, ax_topo_ms_2, ax_evoked_diff, ax_topo_diff


def fig_psd():

    fig = plt.figure(figsize=(15, 5))

    # 1st row Topoplots
    ax1 = fig.add_axes([0.05, 0.6, 0.15, 0.3])
    ax2 = fig.add_axes([0.225, 0.6, 0.15, 0.3])
    ax3 = fig.add_axes([0.4, 0.6, 0.15, 0.3])
    ax4 = fig.add_axes([0.575, 0.6, 0.15, 0.3])
    ax5 = fig.add_axes([0.75,  0.6, 0.15, 0.3])

    # 2nd row PSD
    ax6 = fig.add_axes([0.15,  0.1, 0.7, 0.4])

    # Group axes
    axs_topo = [ax1, ax2, ax3, ax4, ax5]
    ax_psd = ax6

    return fig, axs_topo, ax_psd


def fig_time_frequency(fontsize=None):

    if fontsize:
        matplotlib.rcParams.update({'font.size': fontsize})

    fig, axes_topo = plt.subplots(3, 3, figsize=(15, 8), gridspec_kw={'width_ratios': [5, 1, 1]})

    for ax in axes_topo[:, 0]:
        ax.remove()
    axes_topo = [ax for ax_arr in axes_topo[:, 1:] for ax in ax_arr ]

    ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.8])

    return fig, axes_topo, ax1