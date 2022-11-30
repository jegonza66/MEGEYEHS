import functions_general
import setup
import load
import save
import mne
import os
from paths import paths
import matplotlib.pyplot as plt
import numpy as np
import plot_general

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

save_data = False
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

# Filter evoked
filter_evoked = False
l_freq = 0.5
h_freq = 100

fig, ax_evoked_vs_1, ax_topo_vs_1, ax_evoked_ms_1, ax_topo_ms_1, ax_evoked_vs_2, ax_topo_vs_2, ax_evoked_ms_2, \
ax_topo_ms_2, ax_evoked_diff, ax_topo_diff = plot_general.fig_vs_ms()

epoch_ids = ['it_fix_vs', 'fix_ms', 'it_fix_vs', 'fix_ms', ]

# Pick MEG chs (Select channels or set picks = 'mag')
chs_id_list = ['mag', 'mag', 'parietal', 'parietal']  # ('mag'/'LR'/'parietal/occipital/'frontal'/sac_chs')

ax_evoked_list = [ax_evoked_vs_1, ax_evoked_ms_1, ax_evoked_vs_2, ax_evoked_ms_2]
ax_topo_list = [ax_topo_vs_1, ax_topo_ms_1, ax_topo_vs_2, ax_topo_ms_2]

topo_times_list = [[0.088, 0.11], [0.105], [0.11], [0.105]]
ylim_list = [dict(mag=[-110, 110]), dict(mag=[-110, 110]), dict(mag=[-50, 60]), dict(mag=[-50, 60])]


for epoch_id, chs_id, ax_evoked, ax_topo, topo_times, ylim in zip(epoch_ids, chs_id_list, ax_evoked_list,
                                                                    ax_topo_list, topo_times_list, ylim_list):

    #----- Load Grand Average data -----#
    ga_save_path = save_path + f'Evoked/{epoch_id}/'
    grand_avg_data_fname = f'Grand_average_ave.fif'
    grand_avg = mne.read_evokeds(ga_save_path + grand_avg_data_fname, condition=0)

    # Pick channels
    picks = functions_general.pick_chs(chs_id)

    # Separate meg and misc channels
    grand_avg_meg = grand_avg.copy().pick('meg')

    # Filter MEG data
    if filter_evoked:
        grand_avg_meg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

    # Plot evoked
    save_fig = False
    fig_path = plot_path + f'Evoked/{epoch_id}/'
    fname = f'Grand_average_{chs_id}'

    plot_general.evoked_topo(evoked_meg=grand_avg_meg, picks=picks,
                             filter_evoked=filter_evoked, l_freq=l_freq, h_freq=h_freq,
                             axes_ev=ax_evoked, axes_topo=ax_topo, topo_times=topo_times, ylim=ylim,
                             display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=fname)

# Compute and plot difference between VS to MS

#----- Load Grand Average data -----#
epoch_id = 'it_fix_vs'
ga_save_path = save_path + f'Evoked/{epoch_id}/'
grand_avg_data_fname = f'Grand_average_ave.fif'
grand_avg_vs = mne.read_evokeds(ga_save_path + grand_avg_data_fname, condition=0)
grand_avg_vs_meg = grand_avg_vs.copy().pick('meg')

epoch_id = 'fix_ms'
ga_save_path = save_path + f'Evoked/{epoch_id}/'
grand_avg_data_fname = f'Grand_average_ave.fif'
grand_avg_ms = mne.read_evokeds(ga_save_path + grand_avg_data_fname, condition=0)
grand_avg_ms_meg = grand_avg_ms.copy().pick('meg')


# Make evoked data from difference
difference = grand_avg_vs_meg.get_data() - grand_avg_ms_meg.get_data()
evoked_diff = mne.EvokedArray(data=difference, info=grand_avg_vs_meg.info, tmin=grand_avg_vs_meg.times[0])

# Topoplot times
topo_times = [0.115]
ylim = dict(mag=[-80, 70])

# Pick channels
chs_id = 'parietal'
picks = functions_general.pick_chs(chs_id)

# Plot
save_fig = True
fig_path = plot_path + f'Evoked/'
fname = f'Grand_Average_VS_MS_{chs_id}'

plot_general.evoked_topo(evoked_meg=evoked_diff, picks=picks, filter_evoked=filter_evoked, l_freq=l_freq, h_freq=h_freq,
                         axes_ev=ax_evoked_diff, axes_topo=ax_topo_diff, topo_times=topo_times, ylim=ylim,
                         display_figs=display_figs, save_fig=False)

if save_fig:
    save.fig(fig=fig, path=fig_path, fname=fname)

