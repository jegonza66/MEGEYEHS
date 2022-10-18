import load
import os
import mne
import save

import setup
from paths import paths

save_path = paths().save_path()
plot_path = paths().plots_path()
l_freq = 0.5
h_freq = 100
epoch_ids = ['fix_vs']
plot_xlim = (-0.05, 0.1)

evokeds = []

for subject_code in range(9):

    exp_info = setup.exp_info()
    subject = load.preproc_subject(exp_info=exp_info, subject_code=subject_code)
    meg_data = subject.preproc_meg()

    # PICK MEG AND STIM CHS
    meg_data.pick(['meg'])
    # Exclude bad channels
    bads = subject.bad_channels
    meg_data.info['bads'].extend(bads)

    # Reject based on channel amplitude
    reject = dict(mag=4e-12)

    # Get events from annotations
    events, event_id = mne.events_from_annotations(meg_data)

    # Epoch data
    epochs = mne.Epochs(meg_data, events, event_id=event_id, reject=reject, event_repeated='merge')
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

    # Save epoched data
    epochs.reset_drop_log_selection()
    epoch_save_path = save_path + f'Epochs/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/' + subject.subject_id + '/'
    os.makedirs(epoch_save_path, exist_ok=True)
    epochs_data_fname = f'Subject_{subject.subject_id}_epo.fif'
    epochs.save(epoch_save_path + epochs_data_fname, overwrite=True)

    # Save evoked data
    evoked_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/' + subject.subject_id + '/'
    os.makedirs(evoked_save_path, exist_ok=True)
    evoked_data_fname = f'Subject_{subject.subject_id}_ave.fif'
    evoked.save(evoked_save_path + evoked_data_fname, overwrite=True)

    fig = evoked.plot(gfp=True, time_unit='s', spatial_colors=True, xlim=plot_xlim)
    fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
    fname = subject.subject_id + '.png'
    save.fig(fig, fig_path, fname)

# Compute grand average
grand_avg = mne.grand_average(evokeds, interpolate_bads=False)
# Save grand average
ga_save_path = save_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
os.makedirs(ga_save_path, exist_ok=True)
grand_avg_data_fname = f'Grand_average_ave.fif'
grand_avg.save(ga_save_path + grand_avg_data_fname, overwrite=True)

# PLOT
fig = grand_avg.plot(gfp=True, spatial_colors=True, time_unit='s', xlim=plot_xlim, window_title=f'Grand average {"-".join(epoch_ids)}')
fig_path = plot_path + f'Evoked/{"-".join(epoch_ids)}_lfreq{l_freq}_hfreq{h_freq}/'
fname = 'Grand_average.png'
save.fig(fig, fig_path, fname)

# Saccades channels
if any('sac' in id for id in epoch_ids):
    sac_chs = ['MLF14-4123', 'MLF13-4123', 'MLF12-4123', 'MLF11-4123', 'MRF11-4123', 'MRF12-4123', 'MRF13-4123', 'MRF14-4123', 'MZF01-4123']
    fig = grand_avg.plot(picks=sac_chs, gfp=True, spatial_colors=True, time_unit='s', xlim=plot_xlim, window_title=f'Grand average {"-".join(epoch_ids)}')
    fname = 'Grand_average_ch_sel.png'
    save.fig(fig, fig_path, fname)
