import load
import os
import mne
import save
import matplotlib.pyplot as plt
import setup
from paths import paths

save_path = paths().save_path()
plot_path = paths().plots_path()

save_data = True
display_figs = False

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


evokeds = []
for subject_code in range(13):

    exp_info = setup.exp_info()
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.load_preproc_meg()

    # PICK MEG AND STIM CHS
    meg_data.pick('meg')

    # Exclude bad channels
    bads = subject.bad_channels
    meg_data.info['bads'].extend(bads)

    # Reject based on channel amplitude
    reject = dict(mag=4e-12)

    # Get events from annotations
    events, event_id = mne.events_from_annotations(meg_data)

    # Epoch data
    epochs = mne.Epochs(meg_data, events,  tmin=tmin, tmax=tmax, event_id=event_id, reject=reject, event_repeated='merge')
    # Drop bad epochs
    epochs.drop_bad()

    # Select epochs
    epoch_keys = [key for epoch_id in epoch_ids for key in event_id.keys() if epoch_id in key]
    epochs = mne.concatenate_epochs([epochs[key] for key in epoch_keys if len(epochs[key])])
    # AVERAGE EPOCHS TO GET EVOKED
    evoked = epochs.average()
    # GET MEG CHS ONLY
    evoked.pick('meg')
    # FILTER
    evoked.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    # Apend to evokeds list to pass to grand average
    evokeds.append(evoked)

    # Save data
    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        epoch_save_path = save_path + f'Epochs/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}_tmin{tmin}_tmax{tmax}/' + subject.subject_id + '/'
        os.makedirs(epoch_save_path, exist_ok=True)
        epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
        epochs.save(epoch_save_path + epochs_data_fname, overwrite=True)

        # Save evoked data
        evoked_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}_tmin{tmin}_tmax{tmax}/' + subject.subject_id + '/'
        os.makedirs(evoked_save_path, exist_ok=True)
        evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
        evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

    fig = evoked.plot(gfp=True, time_unit='s', spatial_colors=True, xlim=plot_xlim,
                      titles=f'Subject {subject.subject_id}', show=display_figs)
    fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}_tmin{tmin}_tmax{tmax}/'
    fname = subject.subject_id + '.png'
    save.fig(fig, fig_path, fname)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)

# Save grand average
if save_data:
    ga_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}_tmin{tmin}_tmax{tmax}/'
    os.makedirs(ga_save_path, exist_ok=True)
    grand_avg_data_fname = f'Grand_average_ave.fif'
    grand_avg.save(ga_save_path + grand_avg_data_fname, overwrite=True)

# PLOT
fig = grand_avg.plot(gfp=True, spatial_colors=True, time_unit='s', xlim=plot_xlim, titles='Grand average', show=display_figs)
fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}_tmin{tmin}_tmax{tmax}/'
fname = 'Grand_average.png'
save.fig(fig, fig_path, fname)

# Saccades channels
if any('sac' in id for id in epoch_ids):
    sac_chs = ['MLF14-4123', 'MLF13-4123', 'MLF12-4123', 'MLF11-4123', 'MRF11-4123', 'MRF12-4123', 'MRF13-4123', 'MRF14-4123', 'MZF01-4123']
    fig = grand_avg.plot(picks=sac_chs, gfp=True, spatial_colors=True, time_unit='s', xlim=plot_xlim, titles='Grand average', show=display_figs)
    fname = 'Grand_average_ch_sel.png'
    save.fig(fig, fig_path, fname)

## ET subplot

meg_data.pick(['meg', 'misc'])
evoked = epochs.average(picks=['meg', 'misc'])
# GET MEG CHS ONLY
# evoked.pick(['meg', 'misc'])

fig, axs = plt.subplots(2, 1, sharex=True)
axs[1].plot(grand_avg.times, grand_avg.data[-7, :])
axs[1].vlines(x=0, ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1], color='grey', linestyles='--')
axs[1].set_ylabel('Gaze x')
axs[1].set_xlabel('Time')
grand_avg.pick('mag')
grand_avg.plot(gfp=True, axes=axs[0], time_unit='s', spatial_colors=True, xlim=plot_xlim,
            titles=f'Grand average', show=display_figs)
axs[0].vlines(x=0, ymin=axs[0].get_ylim()[0], ymax=axs[0].get_ylim()[1], color='grey', linestyles='--')
fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}_tmin{tmin}_tmax{tmax}/'
fname = 'Grand_average_gazex.png'
save.fig(fig, fig_path, fname)