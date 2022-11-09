import setup
import load
import save
import mne
import os
from paths import paths
import matplotlib.pyplot as plt

save_path = paths().save_path()
plot_path = paths().plots_path()
exp_info = setup.exp_info()

save_data = False
display_figs = False
if display_figs:
    plt.ion()
else:
    plt.ioff()

# Pick MEG chs (Select channels or set picks = 'mag')
pick_chs = 'LR'
if pick_chs == 'mag':
    picks = 'mag'
elif pick_chs == 'LR':
    right_chs = ['MRT51', 'MRT52', 'MRT53']
    left_chs = ['MLT51', 'MLT52', 'MLT53']
    picks = right_chs + left_chs

# Filter evoked
filter_evoked = False
l_freq = 0.5
h_freq = 100

epoch_ids = ['l_sac']

if any('fix' in id for id in epoch_ids):
    tmin = -0.1
    tmax = 0.4
    plot_xlim = (tmin, tmax)
elif any('sac' in id for id in epoch_ids):
    tmin = -0.1
    tmax = 0.1
    plot_xlim = (-0.05, 0.1)

evokeds = []
for subject_code in exp_info.subjects_ids:

    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)

    evoked_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}/' + subject.subject_id + '/'
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
    evoked = mne.read_evokeds(evoked_save_path + evoked_data_fname, condition=0)
    evokeds.append(evoked)

    # Separate MEG and misc channels
    evoked_meg = evoked.copy().pick('meg')
    evoked_misc = evoked.copy().pick('misc')

    # Filter MEG data
    if filter_evoked:
        evoked_meg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    axs[1].plot(evoked_misc.times, evoked_misc.data[-7, :])
    axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
    axs[1].set_ylabel('Gaze x')
    axs[1].set_xlabel('Time')
    evoked_meg.plot(gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim,
                    titles=f'Subject {subject.subject_id}', show=display_figs)
    axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')
    fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
    fname = 'Evoked_' + subject.subject_id + f'_{pick_chs}'
    if filter_evoked:
        fname += f'_lfreq{l_freq}_hfreq{h_freq}.png'
    else:
        fname += '.png'
    save.fig(fig, fig_path, fname)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    ga_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}/'
    os.makedirs(ga_save_path, exist_ok=True)
    grand_avg_data_fname = f'Grand_average_ave.fif'
    grand_avg.save(ga_save_path + grand_avg_data_fname, overwrite=True)

# Separate meg and misc channels
grand_avg_meg = grand_avg.copy().pick('meg')
grand_avg_misc = grand_avg.copy().pick('misc')

# Filter MEG data
if filter_evoked:
    grand_avg_meg.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

# Plot
fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
axs[1].plot(grand_avg_misc.times, grand_avg_misc.data[-7, :])
axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
axs[1].set_ylabel('Gaze x')
axs[1].set_xlabel('Time')
grand_avg_meg.plot(gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim,
                   titles=f'Grand average', show=display_figs)
axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')
fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}/'
fname = f'Grand_average_{pick_chs}'
if filter_evoked:
    fname += f'_lfreq{l_freq}_hfreq{h_freq}.png'
else:
    fname += '.png'
save.fig(fig, fig_path, fname)

# Plot Saccades frontal channels
if any('sac' in id for id in epoch_ids):
    sac_chs = ['MLF14-4123', 'MLF13-4123', 'MLF12-4123', 'MLF11-4123', 'MRF11-4123', 'MRF12-4123', 'MRF13-4123',
               'MRF14-4123', 'MZF01-4123']

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    axs[1].plot(grand_avg_misc.times, grand_avg_misc.data[-7, :])
    axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
    axs[1].set_ylabel('Gaze x')
    axs[1].set_xlabel('Time')
    grand_avg_meg.plot(picks=sac_chs, gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim,
                       titles=f'Grand average', show=display_figs)
    axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')
    fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
    fname = 'Grand_average_ch_sel.png'
    if filter_evoked:
        fname += f'_lfreq{l_freq}_hfreq{h_freq}.png'
    else:
        fname += '.png'
    save.fig(fig, fig_path, fname)