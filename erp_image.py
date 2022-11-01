import pandas as pd
import numpy as np
import functions
import load
import mne
import save
import matplotlib.pyplot as plt
import setup
from paths import paths

save_path = paths().save_path()
plot_path = paths().plots_path()

save_data = True
display_figs = True

if display_figs:
    plt.ion()
else:
    plt.ioff()

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

for subject_code in range(13):

    exp_info = setup.exp_info()
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg()

    # Exclude bad channels
    bads = subject.bad_channels
    meg_data.info['bads'].extend(bads)

    # Get events from annotations
    events, event_id = mne.events_from_annotations(meg_data)

    # Reject based on channel amplitude
    reject = dict(mag=4e-1)
    # Epoch data
    epochs = mne.Epochs(meg_data, events,  tmin=tmin, tmax=tmax, event_id=event_id, reject=reject, event_repeated='merge')
    # Drop bad epochs
    epochs.drop_bad()

    # Select epochs
    epoch_keys = [key for epoch_id in epoch_ids for key in event_id.keys() if epoch_id in key]
    epochs = mne.concatenate_epochs([epochs[key] for key in epoch_keys if len(epochs[key])])

    # Get saccades info for every trial
    saccades = subject.saccades
    left_saccades = saccades.loc[(saccades['dir'] == 'l') &
                                 (~pd.isna(saccades['screen'])) &
                                 (left_saccades['id'].isin(epoch_keys))].reset_index(drop=True)
    event_saccades = left_saccades.loc[left_saccades['id'].isin(epoch_keys)].reset_index(drop=True)
    left_sac_dur = left_saccades['duration']

    # Parameters for plotting
    order = left_sac_dur.argsort()  # Sorting from longer to shorter

    # Just for now, after re running preprocessing delete
    epochs.rename_channels(functions.ch_name_map)

    # Split chs
    right_chs = ['MRT12', 'MRT13', 'MRT14', 'MRT22', 'MRT23', 'MRT24', 'MRT33', 'MRT34', 'MRT42', 'MRT43', 
                 'MRT51', 'MRT52', 'MRT53']
    right_idx = [idx for ch_name in right_chs for idx in range(len(epochs.ch_names)) if ch_name == epochs.ch_names[idx]]

    left_chs = ['MLT12', 'MLT13', 'MLT14', 'MLT22', 'MLT23', 'MLT24', 'MLT33', 'MLT34', 'MLT42', 'MLT43',
                'MLT51', 'MLT52', 'MLT53']
    left_idx = [idx for ch_name in left_chs for idx in range(len(epochs.ch_names)) if ch_name == epochs.ch_names[idx]]

    picks = right_chs + left_chs
    picks = 'mag'


    def epoch_std(epoch_data):
        epoch_std = np.std(epoch_data, axis=1)
        return epoch_std


    def right_left(epoch_subset_data):
        epoch_right = epoch_subset_data[:, :13, :]
        epoch_left = epoch_subset_data[:, 13:, :]

        mean_left = np.abs(np.mean(epoch_left, axis=1))
        mean_right = np.abs(np.mean(epoch_right, axis=1))

        diff = mean_left + mean_right

        return diff

    epoch_subset = epochs.copy()
    epoch_subset = epoch_subset.pick(picks)
    epoch_subset_data = epoch_subset.get_data()

    epoch_subset_std = epoch_std(epoch_subset_data)


    epoch_subset.plot_image(picks=picks, order=order, sigma=0, cmap='jet', overlay_times=left_sac_dur, combine=right_left,
                            title=subject.subject_id)

    # Plot Trials
    epochs.plot_image(picks=picks, order=order, sigma=0, cmap='jet', overlay_times=left_sac_dur, combine='std',
                      title=subject.subject_id)

    # Plot evoked
    evoked = epochs.average(picks=['meg', 'misc'])
    # Filter evoked
    evoked.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

    epoch_data = epochs.get_data()
    epoch_all_std = epoch_std(epoch_data)

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    axs[1].plot(evoked.times, evoked.data[-7, :])
    axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
    axs[1].set_ylabel('Gaze x')
    axs[1].set_xlabel('Time')
    evoked.pick(picks)
    evoked.plot(gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim, show=display_figs)
    axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')

    fig_path = plot_path + f'ERP_image/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
    fname = subject.subject_id + '.png'



