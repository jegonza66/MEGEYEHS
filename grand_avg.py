import functions_general
import setup
import mne
from paths import paths
import matplotlib.pyplot as plt
import plot_general

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

#----- Save data and display figures -----#
display_figs = True
if display_figs:
    plt.ion()
else:
    plt.ioff()

#-----  Select frequency band -----#
band_id = None

fig, ax_evoked_vs_1, ax_topo_vs_1, ax_evoked_ms_1, ax_topo_ms_1, ax_evoked_vs_2, ax_topo_vs_2, ax_evoked_ms_2, \
ax_topo_ms_2, ax_evoked_diff, ax_topo_diff = plot_general.fig_vs_ms()

epoch_ids = ['it_fix_vs', 'fix_ms', 'it_fix_vs', 'fix_ms', ]

# Pick MEG chs (Select channels or set picks = 'mag')
chs_id_list = ['mag', 'mag', 'parietal', 'parietal']  # ('mag'/'LR'/'parietal/occipital/'frontal'/sac_chs')

ax_evoked_list = [ax_evoked_vs_1, ax_evoked_ms_1, ax_evoked_vs_2, ax_evoked_ms_2]
ax_topo_list = [ax_topo_vs_1, ax_topo_ms_1, ax_topo_vs_2, ax_topo_ms_2]

topo_times_list = [[0.09, 0.11], [0.105], [0.11], [0.11]]
ylim_list = [dict(mag=[-110, 110]), dict(mag=[-110, 110]), dict(mag=[-50, 60]), dict(mag=[-50, 60])]


for epoch_id, chs_id, ax_evoked, ax_topo, topo_times, ylim in zip(epoch_ids, chs_id_list, ax_evoked_list,
                                                                    ax_topo_list, topo_times_list, ylim_list):

    # Get time windows from epoch_id name
    tmin, tmax, plot_xlim = functions_general.get_time_lims(epoch_id=epoch_id)

    # Specific run path for saving data and plots
    run_path = f'/Band_{band_id}/{epoch_id}_{tmin}_{tmax}/'

    #----- Load Grand Average data -----#
    ga_save_path = save_path + f'Evoked/' + run_path
    grand_avg_data_fname = f'Grand_average_ave.fif'
    grand_avg = mne.read_evokeds(ga_save_path + grand_avg_data_fname, condition=0)

    # Pick channels
    picks = functions_general.pick_chs(chs_id, info=grand_avg.info)

    # Separate meg and misc channels
    grand_avg_meg = grand_avg.copy().pick('meg')

    # Plot evoked
    save_fig = False
    fig_path = plot_path + f'Evoked/' + f'{band_id}/'
    fname = f'Grand_average_{chs_id}'
    plot_general.evoked_topo(evoked_meg=grand_avg_meg, picks=picks, fig=fig,
                             axes_ev=ax_evoked, axes_topo=ax_topo, topo_times=topo_times, ylim=ylim,
                             display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=fname)

# Compute and plot difference between VS to MS

#----- Load Grand Average data -----#
epoch_id = 'it_fix_vs'
run_path = f'/Band_{band_id}/{epoch_id}_{tmin}_{tmax}/'
ga_save_path = save_path + f'Evoked/' + run_path
grand_avg_data_fname = f'Grand_average_ave.fif'
grand_avg_vs = mne.read_evokeds(ga_save_path + grand_avg_data_fname, condition=0)
grand_avg_vs_meg = grand_avg_vs.copy().pick('meg')

epoch_id = 'fix_ms'
run_path = f'/Band_{band_id}/{epoch_id}_{tmin}_{tmax}/'
ga_save_path = save_path + f'Evoked/' + run_path
grand_avg_data_fname = f'Grand_average_ave.fif'
grand_avg_ms = mne.read_evokeds(ga_save_path + grand_avg_data_fname, condition=0)
grand_avg_ms_meg = grand_avg_ms.copy().pick('meg')

# Make evoked data from difference
difference = grand_avg_vs_meg.get_data() - grand_avg_ms_meg.get_data()
evoked_diff = mne.EvokedArray(data=difference, info=grand_avg_vs_meg.info, tmin=grand_avg_vs_meg.times[0])

# Topoplot times
topo_times = [0.113]
ylim = dict(mag=[-80, 70])

# Pick channels
chs_id = 'mag'
picks = functions_general.pick_chs(chs_id=chs_id, info=grand_avg_vs_meg.info)

# Plot
save_fig = True
fig_path = plot_path + f'Evoked/' + f'Band_{band_id}/'
fname = f'Grand_Average_VS_MS_{chs_id}'

plot_general.evoked_topo(evoked_meg=evoked_diff, picks=picks, fig=fig,
                         axes_ev=ax_evoked_diff, axes_topo=ax_topo_diff, topo_times=topo_times, ylim=ylim,
                         display_figs=display_figs, save_fig=save_fig, fig_path=fig_path, fname=fname)


##
import functions_general
import setup
import mne
from paths import paths
import matplotlib.pyplot as plt
import plot_general
import save
import numpy as np


save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

#----- Save data and display figures -----#
display_figs = True
save_fig = False

if display_figs:
    plt.ion()
else:
    plt.ioff()


trial_params = {'epoch_id': 'it_fix_vs+tgt_fix_vs',  # use'+' to mix conditions (red+blue)
                'corrans': None,
                'tgtpres': None,
                'mss': None,
                'reject': None,  # None to use default {'mag': 5e-12} / False for no rejection / 'subject' to use subjects predetermined rejection value
                'evtdur': None,
                'trialdur': None,
                'tmin': -0.3,
                'tmax': 0.6,
                'baseline': (-0.3, -0.05)
                }

meg_params = {'chs_id': 'temporal',
              'band_id': None,
              'filter_sensors': False,
              'filter_method': 'iir',
              'data_type': 'ICA'
              }

# Paths
run_path = (f"Band_{meg_params['band_id']}/{trial_params['epoch_id']}_mss{trial_params['mss']}_corrans{trial_params['corrans']}_tgtpres{trial_params['tgtpres']}_"
            f"trialdur{trial_params['trialdur']}_evtdur{trial_params['evtdur']}_{trial_params['tmin']}_{trial_params['tmax']}_bline{trial_params['baseline']}/")

evoked_save_path = paths().save_path() + f"Evoked_{meg_params['data_type']}/" + run_path

#----- Load Grand Average data -----#
ga_save_path = save_path + f'Evoked_{meg_params["data_type"]}/' + run_path
grand_avg_data_fname = f'Grand_average_ave.fif'
grand_avg = mne.read_evokeds(ga_save_path + grand_avg_data_fname, condition=0)

# Separate MEG and misc channels
grand_avg_meg = grand_avg.copy().pick('mag')
grand_avg_misc = grand_avg.copy().pick('misc')

# Pick channels
picks = functions_general.pick_chs(chs_id=[meg_params['chs_id']], info=grand_avg.info)

# Plot times
topo_times = [0.09, 0.4]
topo_times_span = [0.01, 0.2]

# mask
mask = np.zeros(grand_avg.data.shape)
for ch in picks:
    try:
        mask[grand_avg.ch_names.index(ch), :] = 1
    except:
        pass
mask = mask.astype(bool)
mask_params = dict(markersize=10, markerfacecolor="grey", alpha=0.65)

fig_topo = grand_avg.plot_topomap(times=topo_times, average=topo_times_span, cmap='bwr', show=display_figs, mask=mask, mask_params=mask_params, vlim=(-60, 60))
if save_fig:
    fig_path = ga_save_path.replace(paths().save_path(), paths().plots_path())
    fname = f'{chs_id}_topomaps'
    save.fig(fig=fig_topo, path=fig_path, fname=fname)
